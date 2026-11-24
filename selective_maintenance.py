from __future__ import annotations

import heapq
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.lines import Line2D
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaincc


# 1. Inputs: the five-state model in the paper.
@dataclass(frozen=True)
class Settings:
    shapes: tuple[float, ...] = (1.5, 2.0, 2.5, 1.5)
    means: tuple[float, ...] = (2.0, 2.5, 3.0, 2.0)
    repair_start: int = 4                    # Repair set = {4, 5}.
    cycle_length: float = 3.0
    cycles: int = 28
    units: int = 200_000
    random_seed: int = 20260903
    population_step: float = 0.02
    max_age: float = 35.0
    age_bin_width: float = 0.2
    observations_per_cycle: int = 30
    convolution_step: float = 0.002
    convolution_horizon: float = 50.0
    series_tolerance: float = 1e-13
    policy_tolerance: float = 1e-10
    max_policy_iterations: int = 1_200
    cost_tolerance: float = 1e-7
    fixed_cost: float = 1.0
    repair_cost: float = 2.0
    loss_rate: float = 1.0
    additional_loss_rates: tuple[float, ...] = (0.3, 0.4)
    output_directory: str = "outputs"

    def validate(self):
        # Figure 2 and the initial distributions in Figure 3 use this structure.
        if len(self.shapes) != 4 or len(self.means) != 4 or self.repair_start != 4:
            raise ValueError("This paper script requires five states and repair set {4, 5}.")
        positive = (*self.shapes, *self.means, self.cycle_length,
                    self.population_step, self.max_age, self.age_bin_width,
                    self.convolution_step, self.convolution_horizon,
                    self.series_tolerance, self.policy_tolerance,
                    self.cost_tolerance, self.fixed_cost)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Model and numerical settings must be finite and positive.")
        counts = (self.cycles, self.units, self.observations_per_cycle,
                  self.max_policy_iterations)
        if any(not isinstance(x, int) or x < 1 for x in counts):
            raise ValueError("Counts must be positive integers.")
        if min(self.repair_cost, self.loss_rate, *self.additional_loss_rates) < 0:
            raise ValueError("Cost rates cannot be negative.")
        for duration in (self.cycle_length / self.observations_per_cycle,
                         self.cycle_length / 2, self.age_bin_width, self.max_age):
            if not np.isclose(duration / self.population_step,
                              round(duration / self.population_step), atol=1e-9, rtol=0):
                raise ValueError("Observation times and age bins must align with the population grid.")
        if any(shape < 1 for shape in self.shapes[:self.repair_start - 1]):
            raise ValueError("The paper's uniform reachability assumption requires shapes >= 1 here.")


class WeibullModel:
    def __init__(self, settings):
        self.shapes = np.asarray(settings.shapes)
        self.means = np.asarray(settings.means)
        self.scales = self.means / gamma(1 + 1 / self.shapes)
        self.n_states = len(self.shapes) + 1
        self.transient_count = self.n_states - 1
        self.repair_start = settings.repair_start
        self.outside_count = self.repair_start - 1
        self.mean_entry_time = float(self.means[:self.outside_count].sum())

    def cumulative_hazard(self, state, age):
        return (np.maximum(np.asarray(age), 0) / self.scales[state]) ** self.shapes[state]

    def cdf(self, state, age):
        return -np.expm1(-self.cumulative_hazard(state, age))

    def step_survival(self, state, ages, step):
        # Conditional survival avoids division by extremely small probabilities.
        return np.exp(-(self.cumulative_hazard(state, ages + step)
                        - self.cumulative_hazard(state, ages)))

    def sample_sojourn(self, state, size, rng):
        return self.scales[state] * rng.weibull(self.shapes[state], size=size)

    def parameter_table(self):
        states = np.arange(1, self.n_states)
        return pd.DataFrame({
            "state": states,
            "repair_set_membership": np.where(states < self.repair_start, "Outside", "Inside"),
            "shape": self.shapes, "scale": self.scales, "mean": self.means,
        })


# 2. Individual-unit simulation. Residual times are retained until a transition.
@dataclass
class IndividualUnits:
    model: WeibullModel
    rng: np.random.Generator
    state: np.ndarray                       # Zero-based state index.
    age: np.ndarray
    residual: np.ndarray


    @classmethod
    def fresh(
        cls,
        model: WeibullModel,
        size: int,
        rng: np.random.Generator,
    ) -> "IndividualUnits":
        if size <= 0:
            raise ValueError("size must be positive")
        state = np.zeros(size, dtype=np.int8)
        age = np.zeros(size, dtype=float)
        residual = model.sample_sojourn(0, size, rng)
        return cls(model, rng, state, age, residual)

    @property
    def size(self) -> int:
        return int(self.state.size)

    def advance(self, duration: float) -> None:
        """Advance all units exactly over ``duration`` using residual event times."""

        if duration < 0:
            raise ValueError("duration must be nonnegative")
        if duration == 0:
            return

        time_left = np.full(self.size, float(duration))
        transient = self.state < self.model.transient_count
        # A unit can cross multiple state boundaries during one observation
        # interval.  The loop processes all such events without time stepping.
        while True:
            crossing = transient & (self.residual <= time_left + 2.0e-15)
            if not np.any(crossing):
                break

            indices = np.flatnonzero(crossing)
            elapsed = self.residual[indices]
            time_left[indices] -= elapsed
            self.age[indices] = 0.0
            self.state[indices] += 1

            entered_transient = indices[
                self.state[indices] < self.model.transient_count
            ]
            for current in np.unique(self.state[entered_transient]):
                selected = entered_transient[self.state[entered_transient] == current]
                self.residual[selected] = self.model.sample_sojourn(
                    int(current), selected.size, self.rng
                )

            entered_absorbing = indices[
                self.state[indices] >= self.model.transient_count
            ]
            self.residual[entered_absorbing] = np.inf
            transient = self.state < self.model.transient_count

        active = self.state < self.model.transient_count
        self.age[active] += time_left[active]
        self.residual[active] -= time_left[active]
        if np.any(self.residual[active] < -1.0e-10):
            raise RuntimeError("a negative residual time remained after event processing")

    def maintain(self) -> float:
        """Apply selective maintenance and return the repaired population fraction."""

        in_repair_set = self.state >= self.model.outside_count
        repaired = int(np.sum(in_repair_set))
        self.state[in_repair_set] = 0
        self.age[in_repair_set] = 0.0
        self.residual[in_repair_set] = self.model.sample_sojourn(0, repaired, self.rng)
        return repaired / self.size

    def state_ratios(self) -> np.ndarray:
        return np.bincount(self.state, minlength=self.model.n_states) / self.size

    def full_histogram(self, bin_width: float, max_age: float) -> tuple[np.ndarray, float]:
        bins = int(np.ceil(max_age / bin_width))
        histogram = np.zeros((self.model.transient_count, bins), dtype=float)
        clipped = np.minimum((self.age / bin_width).astype(int), bins - 1)
        for current in range(self.model.transient_count):
            mask = self.state == current
            histogram[current] = np.bincount(clipped[mask], minlength=bins) / self.size
        absorbing = float(np.mean(self.state == self.model.n_states - 1))
        return histogram, absorbing


# 3. Population model on the state-age grid.
def distance(first, second):
    """Total variation between two probability arrays on the same grid."""
    return float(0.5 * np.abs(first - second).sum())


class PopulationGrid:
    def __init__(self, model, settings):
        self.model = model
        self.dt = settings.population_step
        self.n_bins = int(np.ceil(settings.max_age / self.dt))
        ages = np.arange(self.n_bins) * self.dt
        self.survival = np.vstack([
            model.step_survival(i, ages, self.dt)
            for i in range(model.transient_count)
        ])

    def initial(self, mixture=((1, 0.0, 1.0),)):
        """Mixture entries are (paper state number, age, probability)."""
        mass = np.zeros((self.model.outside_count, self.n_bins))
        for state, age, weight in mixture:
            index = int(round(age / self.dt))
            if not 1 <= state <= self.model.outside_count or not 0 <= index < self.n_bins:
                raise ValueError("Initial state-age pair lies outside the post-maintenance grid.")
            if weight < 0 or not np.isclose(index * self.dt, age):
                raise ValueError("Initial weights must be nonnegative and ages must align with the grid.")
            mass[state - 1, index] += weight
        if not np.isclose(mass.sum(), 1.0):
            raise ValueError("Initial weights must sum to one.")
        return mass

    def step(self, mass):
        """Advance one step and return mass exiting the last represented state.

        Three rows represent states outside the repair set; four rows represent
        all nonabsorbing states. The same update is used in both cases.
        The last age cell retains any overflow, as in the manuscript.
        """
        survived = mass * self.survival[:len(mass)]
        exits = (mass - survived).sum(axis=1)
        updated = np.zeros_like(mass)
        updated[:, 1:] = survived[:, :-1]
        updated[:, -1] += survived[:, -1]
        updated[1:, 0] += exits[:-1]
        return updated, float(exits[-1])

    def advance(self, mass, steps):
        current = mass.copy()
        exited = 0.0
        for _ in range(steps):
            current, increment = self.step(current)
            exited += increment
        if abs(current.sum() + exited - mass.sum()) > 2e-12:
            raise RuntimeError("Population evolution did not conserve mass.")
        return current, exited

    def cycle(self, mass, cycle_length):
        """Return map on states outside the repair set, with off-grid interpolation."""
        steps = int(np.floor(cycle_length / self.dt + 1e-12))
        fraction = cycle_length / self.dt - steps
        survivors, repaired = self.advance(mass, steps)
        if fraction > 1e-12:
            following, increment = self.step(survivors)
            survivors = (1 - fraction) * survivors + fraction * following
            repaired += fraction * increment
        survivors[0, 0] += repaired
        return survivors, repaired

    def threshold_cycle(self, mass, threshold, maximum_time):
        current = mass.copy()
        repair_mass = 0.0
        for step in range(1, int(np.ceil(maximum_time / self.dt)) + 1):
            previous = current
            before = repair_mass
            current, increment = self.step(previous)
            repair_mass += increment
            if repair_mass + 2e-12 >= threshold:
                fraction = 1.0 if increment <= 2e-15 else (threshold - before) / increment
                fraction = float(np.clip(fraction, 0, 1))
                post = (1 - fraction) * previous + fraction * current
                post[0, 0] += threshold
                post /= post.sum()
                return post, (step - 1 + fraction) * self.dt
        raise RuntimeError("Threshold was not reached within the time limit.")

    def stationary(self, cycle_length, tolerance):
        """Normalize the renewal series; bound its omitted mass on this grid."""
        steps = int(round(cycle_length / self.dt))
        if steps < 1 or not np.isclose(steps * self.dt, cycle_length, rtol=0, atol=1e-10):
            raise ValueError("The renewal calculation requires a grid-aligned cycle length.")
        count = self.model.outside_count
        # Backward recursion bounds one-cycle survival over all initial states
        # and ages. It gives a geometric bound for the remaining series mass.
        survival = np.ones((count, self.n_bins))
        for _ in range(steps):
            updated = np.empty_like(survival)
            for state in range(count):
                conditional = self.survival[state]
                following = np.r_[survival[state, 1:], survival[state, -1]]
                updated[state] = conditional * following
                if state + 1 < count:
                    updated[state] += (1 - conditional) * survival[state + 1, 0]
            survival = updated
        contraction = float(survival.max())
        if not 0 <= contraction < 1:
            raise RuntimeError("Cannot bound the stationary-series tail on this grid.")
        current = self.initial()
        total = np.zeros_like(current)
        for _ in range(100_000):
            total += current
            current, _ = self.advance(current, steps)
            if current.sum() / (1 - contraction) <= tolerance:
                theta = float(1 / total.sum())
                return theta * total, theta
        raise RuntimeError("Stationary series did not reach the specified tolerance.")

    def coarsen(self, mass, bin_width):
        group = int(round(bin_width / self.dt))
        groups = int(np.ceil(self.n_bins / group))
        padded = np.zeros((len(mass), groups * group))
        padded[:, :self.n_bins] = mass
        return padded.reshape(len(mass), groups, group).sum(axis=2)


# 4. Independent calculation of T_r = X_1 + ... + X_(r-1).
class RepairEntryDistribution:
    def __init__(self, model, settings):
        self.model = model
        self.dt = settings.convolution_step
        self.tolerance = settings.series_tolerance
        edges = np.arange(int(np.ceil(settings.convolution_horizon / self.dt)) + 1) * self.dt
        probabilities = np.array([1.0])
        for state in range(model.outside_count):
            masses = np.maximum(np.diff(model.cdf(state, edges)), 0.0)
            omitted = max(0.0, 1.0 - float(masses.sum()))
            if omitted > 1e-10:
                raise ValueError("Increase convolution_horizon: a Weibull tail is too large.")
            masses[-1] += omitted
            masses /= masses.sum()
            probabilities = np.maximum(fftconvolve(probabilities, masses), 0.0)
        probabilities /= probabilities.sum()
        self.times = (np.arange(len(probabilities)) + 0.5 * model.outside_count) * self.dt
        self.cumulative = np.minimum(np.cumsum(probabilities), 1.0)

    def tail_bound(self, cycle_length, last_index):
        """Bound the omitted continuous survival series, excluding grid error."""
        count = self.model.outside_count
        shapes, scales = self.model.shapes[:count], self.model.scales[:count]
        lower = last_index * cycle_length / count
        integrated = scales / shapes * gamma(1 / shapes) * gammaincc(
            1 / shapes, (lower / scales) ** shapes
        )
        return float(count * integrated.sum() / cycle_length)

    def repair_fraction(self, cycle_length):
        if cycle_length <= 0:
            raise ValueError("Cycle length must be positive.")
        last = int(np.ceil(self.times[-1] / cycle_length)) + 1
        times = cycle_length * np.arange(last + 1)
        terms = 1 - np.interp(times, self.times, self.cumulative, left=0.0, right=1.0)
        terms[0] = 1.0
        if self.tail_bound(cycle_length, last) > self.tolerance:
            raise ValueError("Increase convolution_horizon to control the survival-series tail.")
        return float(1 / terms.sum())

    def calibrated_cycle(self, threshold):
        upper = self.model.mean_entry_time
        while self.repair_fraction(upper) < threshold:
            upper *= 2
        return float(brentq(lambda tau: self.repair_fraction(tau) - threshold,
                           self.dt, upper, xtol=1e-11, rtol=1e-11))


# 5. Experiments for Figures 1, 2 and 3.
def append_ratios(records, time_value, cycle, phase, source, ratios):
    for state, fraction in enumerate(ratios, 1):
        records.append(dict(time=time_value, cycle=cycle, phase=phase,
                            source=source, state=state, fraction=float(fraction)))


def fixed_cycle_experiments(settings, model, grid):
    tau = settings.cycle_length
    tau_steps = int(round(tau / grid.dt))
    observation_steps = tau_steps // settings.observations_per_cycle
    observation_time = observation_steps * grid.dt
    units = IndividualUnits.fresh(model, settings.units, np.random.default_rng(settings.random_seed))
    population = np.zeros((model.transient_count, grid.n_bins))
    population[0, 0] = 1.0
    absorbing = 0.0
    trajectories = []
    max_error = 0.0

    def observe(time_value, cycle, phase):
        nonlocal max_error
        individual = units.state_ratios()
        deterministic = np.r_[population.sum(axis=1), absorbing]
        append_ratios(trajectories, time_value, cycle, phase, "individual unit simulation", individual)
        append_ratios(trajectories, time_value, cycle, phase, "population", deterministic)
        max_error = max(max_error, float(np.abs(individual - deterministic).max()))

    observe(0.0, 0, "post-maintenance")
    for cycle in range(1, settings.cycles + 1):
        for observation in range(1, settings.observations_per_cycle + 1):
            units.advance(observation_time)
            population, exited = grid.advance(population, observation_steps)
            absorbing += exited
            phase = "pre-maintenance" if observation == settings.observations_per_cycle else "free evolution"
            observe((cycle - 1) * tau + observation * observation_time, cycle, phase)
        units.maintain()
        repaired = population[model.outside_count:].sum() + absorbing
        population[model.outside_count:] = 0.0
        population[0, 0] += repaired
        absorbing = 0.0
        observe(cycle * tau, cycle, "post-maintenance")

    # One further cycle provides the three state-age snapshots in Figure 2.
    snapshots = []
    previous_step = 0
    for phase, phase_time in [("post-maintenance", 0.0), ("mid-cycle", tau / 2), ("pre-maintenance", tau)]:
        target_step = int(round(phase_time / grid.dt))
        steps = target_step - previous_step
        if steps:
            units.advance(steps * grid.dt)
            population, exited = grid.advance(population, steps)
            absorbing += exited
        previous_step = target_step
        histograms = [("individual unit simulation", units.full_histogram(settings.age_bin_width, settings.max_age)[0]),
                      ("population", grid.coarsen(population, settings.age_bin_width))]
        for source, histogram in histograms:
            for state, row in enumerate(histogram, 1):
                for age_bin, mass in enumerate(row):
                    snapshots.append(dict(phase=phase, phase_time=phase_time, source=source,
                                          state=state, age_left=age_bin * settings.age_bin_width,
                                          age_right=(age_bin + 1) * settings.age_bin_width,
                                          mass=float(mass), density=float(mass / settings.age_bin_width)))

    # Figure 3 compares three post-maintenance laws with the renewal-series law.
    stationary, _ = grid.stationary(tau, settings.series_tolerance)
    reference = grid.coarsen(stationary, settings.age_bin_width)
    starts = {
        "state 1, age 0": ((1, 0.0, 1.0),),
        "state 3, age 1": ((3, 1.0, 1.0),),
        "mixed": ((1, 0.0, 0.40), (2, 0.6, 0.35), (3, 1.0, 0.25)),
    }
    convergence = []
    for name, mixture in starts.items():
        current = grid.initial(mixture)
        for cycle in range(settings.cycles + 1):
            value = distance(grid.coarsen(current, settings.age_bin_width), reference)
            convergence.append(dict(initial_distribution=name, cycle=cycle, tv_to_stationary=value))
            if cycle < settings.cycles:
                current, _ = grid.cycle(current, tau)
    return pd.DataFrame(trajectories), pd.DataFrame(snapshots), pd.DataFrame(convergence), max_error


# 6. Figure 4: iterate both policies, then compare with independent convolution.
def stationary_policy(grid, settings, *, threshold=None, cycle_length=None):
    current = grid.initial()
    previous_time = np.nan
    maximum_time = max(4 * grid.model.mean_entry_time, 19.0)
    for _ in range(settings.max_policy_iterations):
        if threshold is not None:
            updated, elapsed = grid.threshold_cycle(current, threshold, maximum_time)
            repaired = threshold
            time_error = abs(elapsed - previous_time) if np.isfinite(previous_time) else np.inf
        else:
            elapsed = cycle_length
            updated, repaired = grid.cycle(current, elapsed)
            time_error = 0.0
        error = distance(current, updated)
        current = updated
        if error <= settings.policy_tolerance and time_error <= max(settings.policy_tolerance, 1e-11):
            return elapsed, repaired
        previous_time = elapsed
    raise RuntimeError(f"Policy iteration did not converge: threshold={threshold}, cycle_length={cycle_length}.")


def correspondence_experiment(settings, grid, distribution):
    rows = []
    for target in np.linspace(0.05, 0.95, 19):
        tau, _ = stationary_policy(grid, settings, threshold=float(target))
        _, recovered = stationary_policy(grid, settings, cycle_length=tau)
        independent = distribution.repair_fraction(tau)
        rows.append(dict(threshold=float(target), threshold_policy_tau=tau,
                         fixed_policy_recovered_threshold=recovered,
                         calibrated_cycle_length=distribution.calibrated_cycle(float(target)),
                         independent_repair_fraction=independent,
                         independent_calibration_error=abs(independent - target)))
    fixed_rows = []
    # Match the 70 cycle lengths used in the manuscript's middle panel.
    for tau in np.linspace(0.2, 14.0, 70):
        _, theta = grid.stationary(float(tau), settings.series_tolerance)
        fixed_rows.append(dict(tau=float(tau), fixed_policy_threshold=theta))
    return pd.DataFrame(rows), pd.DataFrame(fixed_rows)


# 7. Figure 5 and Table 2: long-run cost and a bounded global search.
def cost_components(distribution, settings, tau, loss_rate=None):
    cd = settings.loss_rate if loss_rate is None else loss_rate
    theta = distribution.repair_fraction(float(tau))
    setup = settings.fixed_cost / tau
    repair = settings.repair_cost * theta / tau
    loss = cd * (1 - distribution.model.mean_entry_time * theta / tau)
    return dict(tau=float(tau), theta=theta, setup=setup, repair=repair,
                performance_loss=loss, total=setup + repair + loss)


def optimize_cost(distribution, settings):
    """Bound the minimum of the discretized cost without assuming unimodality.

    The upper bound is an evaluated cost. Monotonicity of Theta gives a lower
    bound on every interval; split the most promising interval until the gap
    is small enough. This gap excludes convolution and interpolation error.
    """
    K = settings.fixed_cost
    cd = settings.loss_rate
    coefficient = cd * distribution.model.mean_entry_time - settings.repair_cost
    if coefficient <= K:
        raise ValueError("The baseline loss rate does not admit a finite cost minimizer.")

    def cost(tau):
        return cost_components(distribution, settings, tau)["total"]

    trial = settings.cycle_length
    for _ in range(100):
        incumbent = cost(trial)
        if incumbent < cd:
            break
        trial *= 2
    else:
        raise RuntimeError("Could not find a cycle length below the no-maintenance cost.")
    # J(tau) >= K/tau, and J(tau) >= cd - (coefficient-K)/tau.
    # These inequalities exclude both ends of the positive half-line.
    lower = K / incumbent
    upper = (coefficient - K) / (cd - incumbent)
    grid = np.linspace(lower, upper, 1201)
    values = np.array([cost(tau) for tau in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    local_minima = np.where((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:]))[0] + 1
    for index in local_minima:
        result = minimize_scalar(cost, bounds=(grid[index - 1], grid[index + 1]),
                                 method="bounded", options={"xatol": 1e-11})
        candidates.append((float(result.fun), float(result.x)))
    best_cost, best_tau = min(candidates)
    theta_cache = {}

    def interval_bound(left, right):
        if right not in theta_cache:
            theta_cache[right] = distribution.repair_fraction(right)
        numerator = K - coefficient * theta_cache[right]
        return cd + min(numerator / left, numerator / right)

    intervals = []
    coarse = np.linspace(lower, upper, 101)
    for left, right in zip(coarse[:-1], coarse[1:]):
        heapq.heappush(intervals, (interval_bound(left, right), left, right))
    for _ in range(250_000):
        if best_cost - intervals[0][0] <= settings.cost_tolerance:
            break
        _, left, right = heapq.heappop(intervals)
        middle = (left + right) / 2
        value = cost(middle)
        if value < best_cost:
            best_cost, best_tau = value, middle
        for a, b in [(left, middle), (middle, right)]:
            heapq.heappush(intervals, (interval_bound(a, b), a, b))
    else:
        raise RuntimeError("The numerical cost bounds did not reach the tolerance.")
    return {**cost_components(distribution, settings, best_tau),
            "lower_bound": float(intervals[0][0]),
            "upper_bound": float(best_cost),
            "cost_gap": float(best_cost - intervals[0][0])}


def cost_experiment(settings, distribution):
    optimum = optimize_cost(distribution, settings)
    rows = []
    for loss_rate in (*settings.additional_loss_rates, settings.loss_rate):
        for tau in np.linspace(1.0, 100.0, 1981):
            rows.append({**cost_components(distribution, settings, tau, loss_rate), "loss_rate": loss_rate})
    selected = [cost_components(distribution, settings, tau)
                for tau in (1.0, 3.0, optimum["tau"], 12.0, 20.0)]
    return pd.DataFrame(rows), pd.DataFrame(selected), optimum


# 8. Plotting. Layouts and styles follow the manuscript figures.
COLORS = ("#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
LINESTYLES = ("-", ":", "--", "-.", (0, (5, 1, 1, 1)))


def save_figure(fig, directory, stem):
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{stem}.{extension}")
    plt.close(fig)


def configure_matplotlib() -> None:
    """Use the Times family preferred by the paper for every plot element."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.0,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.65,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def plot_state_proportions(
    trajectory: pd.DataFrame,
    tau: float,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    max_time = min(float(trajectory["time"].max()), 14.0 * tau)
    selected = trajectory[(trajectory["time"] >= 0.0) & (trajectory["time"] <= max_time)]
    fig, axis = plt.subplots(1, 1, figsize=(7.2, 3.35))

    for state in sorted(selected["state"].unique()):
        population = selected[
            (selected["source"] == "population") & (selected["state"] == state)
        ]
        individual = selected[
            (selected["source"] == "individual unit simulation")
            & (selected["state"] == state)
        ]
        axis.plot(
            population["time"],
            population["fraction"],
            color=COLORS[state - 1],
            linestyle=LINESTYLES[state - 1],
            linewidth=1.25,
            zorder=2,
        )
        stride = max(1, len(individual) // 95)
        axis.plot(
            individual["time"].iloc[::stride],
            individual["fraction"].iloc[::stride],
            linestyle="none",
            marker="o",
            markersize=2.0,
            markerfacecolor="white",
            markeredgewidth=0.55,
            color=COLORS[state - 1],
            zorder=3,
        )
    for maintenance_time in np.arange(tau, max_time + 1.0e-9, tau):
        axis.axvline(maintenance_time, color="#bdbdbd", lw=0.4, zorder=0)

    axis.set_xlim(0.0, max_time)
    axis.set_ylim(-0.015, 1.015)
    axis.set_xlabel("Time")
    axis.set_ylabel("Fraction of units")

    source_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markersize=3,
            label="Individual unit simulation",
        ),
        Line2D([0], [0], color="#555555", lw=1.3, label="Population model"),
    ]
    state_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            label=f"State {i + 1}",
        )
        for i in range(5)
    ]
    axis.legend(
        handles=source_handles + state_handles,
        ncol=1,
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir, "state_proportions")

def plot_distribution_snapshots(
    snapshots: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    phases = ["post-maintenance", "mid-cycle", "pre-maintenance"]
    titles = [r"$s=0$", r"$s=\tau/2$", r"$s=\tau^{-}$"]
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 5.35), sharex=True, sharey=False)

    for row, (phase, title) in enumerate(zip(phases, titles)):
        for state in range(1, 5):
            axis = axes[row, state - 1]
            selected = snapshots[(snapshots["phase"] == phase) & (snapshots["state"] == state)]
            population = selected[selected["source"] == "population"]
            population_centers = 0.5 * (
                population["age_left"] + population["age_right"]
            )
            axis.plot(
                population_centers,
                population["density"],
                linestyle="-",
                linewidth=1.25,
                color=COLORS[state - 1],
                zorder=2,
            )

            individual = selected[
                selected["source"] == "individual unit simulation"
            ]
            individual_centers = 0.5 * (
                individual["age_left"] + individual["age_right"]
            )
            nonzero = individual["density"].to_numpy() > 2.0e-5
            axis.plot(
                individual_centers.to_numpy()[nonzero][::2],
                individual["density"].to_numpy()[nonzero][::2],
                marker="o",
                linestyle="none",
                markersize=2.0,
                markerfacecolor="white",
                markeredgewidth=0.5,
                color=COLORS[state - 1],
                zorder=3,
            )
            axis.set_xlim(0, 10)
            if state == 1:
                axis.set_ylim(0, 2.0)
                axis.set_yticks((0.0, 0.5, 1.0, 1.5, 2.0))
            else:
                axis.set_ylim(0, 0.2)
                axis.set_yticks((0.0, 0.1, 0.2))
            axis.tick_params(axis="y", labelleft=True)
            if row == 0:
                axis.set_title(f"State {state}")
            if state == 1:
                axis.set_ylabel(f"{title}\nMass / bin width")
            if row == 2:
                axis.set_xlabel("Age")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            ls="none",
            markerfacecolor="white",
            markersize=3,
            label="Individual unit simulation",
        ),
        Line2D([0], [0], color="#333333", lw=1.25, label="Population model"),
    ]
    # State labels and the shared legend identify the panels and data sources.
    fig.legend(
        handles=legend_handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        fontsize=11.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.55, h_pad=0.55, w_pad=0.7)
    save_figure(fig, figure_dir, "state_age")

def plot_convergence(
    deterministic_starts: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    fig, axis = plt.subplots(1, 1, figsize=(5.9, 3.2))
    ordered = (
        ("state 1, age 0", "State 1, age 0", COLORS[0], LINESTYLES[0]),
        ("state 3, age 1", "State 3, age 1", COLORS[2], LINESTYLES[2]),
        ("mixed", "Mixed", COLORS[1], LINESTYLES[1]),
    )
    for name, label, color, linestyle in ordered:
        data = deterministic_starts[
            deterministic_starts["initial_distribution"] == name
        ]
        axis.plot(
            data["cycle"],
            data["tv_to_stationary"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
        )
    axis.set_xlabel("Maintenance cycle $k$")
    axis.set_ylabel(r"Projected TV distance $d_h$")
    axis.set_xlim(0.0, 15.0)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, figure_dir, "tv_convergence")

def plot_threshold_comparison(
    threshold_to_cycle: pd.DataFrame,
    fixed_to_threshold: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35))

    axes[0].plot(
        threshold_to_cycle["threshold"],
        threshold_to_cycle["threshold_policy_tau"],
        color="#0072B2",
        marker="o",
        markersize=3.0,
        linewidth=1.25,
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlabel(r"Threshold $\theta$")
    axes[0].set_ylabel(r"Corresponding cycle length $\tau_\theta$")
    axes[0].set_title("Threshold policy", loc="center", pad=7)

    axes[1].plot(
        fixed_to_threshold["tau"],
        fixed_to_threshold["fixed_policy_threshold"],
        color="#D55E00",
        linewidth=1.25,
    )
    axes[1].set_xlim(0.0, float(fixed_to_threshold["tau"].max()))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel(r"Fixed cycle length $\tau$")
    axes[1].set_ylabel(r"Corresponding threshold $\theta$")
    axes[1].set_title("Fixed-cycle policy", loc="center", pad=7)

    axes[2].plot(
        (0.0, 1.0),
        (0.0, 1.0),
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label=r"$y=x$",
    )
    axes[2].plot(
        threshold_to_cycle["threshold"],
        threshold_to_cycle["fixed_policy_recovered_threshold"],
        color="#009E73",
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        label="Round-trip values",
    )
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel(r"Input threshold $\theta$")
    axes[2].set_ylabel(r"Recovered threshold $\widehat\theta$")
    axes[2].set_title("Round-trip correspondence", loc="center", pad=7)
    axes[2].legend(frameon=True)

    fig.tight_layout()
    save_figure(fig, figure_dir, "policy_correspondence")

def plot_maintenance_cost(scenarios, optimum, figure_dir):
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"],
            "mathtext.fontset": "stix", "font.size": 13, "axes.titlesize": 15,
            "axes.labelsize": 14, "legend.fontsize": 12, "xtick.labelsize": 12,
            "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
            "pdf.fonttype": 42, "savefig.dpi": 320,
            "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        })
        fig, axis = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        colors = ("#0072B2", "#009E73", "#111111")
        styles = ("-.", "--", "-")
        for index, (loss_rate, data) in enumerate(scenarios.groupby("loss_rate", sort=False)):
            color, style = colors[index % len(colors)], styles[index % len(styles)]
            axis.plot(data.tau, data.total, color=color, linestyle=style, linewidth=1.6,
                      label=rf"$c_D={loss_rate:g}$")
            axis.axhline(loss_rate, color=color, linestyle=":", linewidth=0.85, alpha=0.7)
        axis.scatter([optimum["tau"]], [optimum["total"]], s=26, color="#D55E00", zorder=5)
        axis.annotate(r"$\tau^*\approx " + f'{optimum["tau"]:.3f}' + "$",
                      xy=(optimum["tau"], optimum["total"]), xytext=(23, 0.67),
                      arrowprops=dict(arrowstyle="-", color="#D55E00"), color="#B54700", fontsize=12)
        axis.set(xlim=(1, 100), ylim=(0.25, 1.4),
                 xlabel=r"Cycle length $\tau$", ylabel="Long-run average cost")
        axis.set_xticks([1, 20, 40, 60, 80, 100])
        axis.legend(loc="upper right")
        axis.grid(alpha=0.22)
        save_figure(fig, figure_dir, "long_run_average")


# 9. Export only the paper's tables and figure data.
def write_tables(parameters, selected_costs, table_dir):
    rows = [f"{row.state} & {row.repair_set_membership} & {row.shape:.2f} & "
            f"{row.scale:.4f} & {row.mean:.2f}" + r" \\"
            for row in parameters.itertuples(index=False)]
    parameter_tex = (r"\begin{tabular}{ccccc}" + "\n" + r"\hline" + "\n"
                     + r"State & Repair-set membership & Shape $\beta_i$ & Scale $\eta_i$ & Mean $m_i$ \\"
                     + "\n" + r"\hline" + "\n" + "\n".join(rows)
                     + "\n" + r"\hline" + "\n" + r"\end{tabular}" + "\n")
    (table_dir / "table01_parameters.tex").write_text(parameter_tex, encoding="utf-8")
    columns = ("tau", "theta", "setup", "repair", "performance_loss", "total")
    rows = [" & ".join(f"{row[column]:.4f}" for column in columns) + r" \\"
            for _, row in selected_costs.iterrows()]
    costs_tex = (r"\begin{tabular}{cccccc}" + "\n" + r"\hline" + "\n"
                 + r"$\tau$ & $\Theta(\tau)$ & Fixed cost & Repair cost & Performance loss & Total cost \\"
                 + "\n" + r"\hline" + "\n" + "\n".join(rows)
                 + "\n" + r"\hline" + "\n" + r"\end{tabular}" + "\n")
    (table_dir / "table02_selected_costs.tex").write_text(costs_tex, encoding="utf-8")


def main():
    settings = Settings()
    settings.validate()
    output = Path(settings.output_directory)
    data_dir, figure_dir, table_dir = [output / name for name in ("data", "figures", "tables")]
    for directory in (data_dir, figure_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = WeibullModel(settings)
    grid = PopulationGrid(model, settings)
    distribution = RepairEntryDistribution(model, settings)

    print("[1/4] State proportions, state-age snapshots, and convergence...", flush=True)
    trajectory, snapshots, convergence, state_error = fixed_cycle_experiments(settings, model, grid)
    print("[2/4] Correspondence between periodic and threshold-based policies...", flush=True)
    threshold, fixed = correspondence_experiment(settings, grid, distribution)
    print("[3/4] Long-run cost and optimal cycle length...", flush=True)
    cost_curve, selected, optimum = cost_experiment(settings, distribution)
    parameters = model.parameter_table()

    frames = {
        "figure1_state_proportions": trajectory,
        "figure2_state_age": snapshots,
        "figure3_convergence": convergence,
        "figure4_threshold_to_cycle": threshold,
        "figure4_cycle_to_threshold": fixed,
        "figure5_cost": cost_curve,
        "table1_parameters": parameters,
        "table2_selected_costs": selected,
    }
    for name, frame in frames.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")

    print("[4/4] Saving the five figures and two tables...", flush=True)
    plot_state_proportions(trajectory, settings.cycle_length, figure_dir)
    plot_distribution_snapshots(snapshots, figure_dir)
    plot_convergence(convergence, figure_dir)
    plot_threshold_comparison(threshold, fixed, figure_dir)
    plot_maintenance_cost(cost_curve, optimum, figure_dir)
    write_tables(parameters, selected, table_dir)
    round_trip = float(np.abs(threshold.fixed_policy_recovered_threshold - threshold.threshold).max())
    summary = (
        f"Mean time for an as-new unit to enter the repair set: {model.mean_entry_time:.10g}\n"
        f"Fixed maintenance cycle length: {settings.cycle_length:g}\n"
        f"Maximum state-proportion error over {settings.cycles} cycles: {state_error:.10g}\n"
        f"Maximum independent calibration error: {threshold.independent_calibration_error.max():.10g}\n"
        f"Maximum round-trip threshold error on the same grid: {round_trip:.10g}\n"
        f"Optimal cycle length: {optimum['tau']:.10g}\n"
        f"Minimum average cost: {optimum['total']:.12g}\n"
        f"Numerical lower bound: {optimum['lower_bound']:.12g}\n"
        f"Numerical upper bound: {optimum['upper_bound']:.12g}\n"
        f"Numerical cost gap: {optimum['cost_gap']:.6g}\n"
        "The cost gap bounds the discretized objective; it excludes convolution/interpolation error.\n"
        f"Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, "
        f"SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}\n"
    )
    (output / "results.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Completed in {time.perf_counter() - start:.1f} seconds. Results: {output.resolve()}")


if __name__ == "__main__":
    main()
