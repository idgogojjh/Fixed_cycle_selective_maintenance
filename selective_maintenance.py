#!/usr/bin/env python3
"""Reproduce the selective-maintenance numerical experiments.

Run: python selective_maintenance.py
See README.txt for installation, commands, methods, and output descriptions.
All inputs are generated from the model specified below; no data files are needed.
"""
from __future__ import annotations

import argparse
import heapq
import io
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

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


# 1. Experiment settings

@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible configuration for all numerical experiments.

    State indices in the paper are one-based.  Python arrays are zero-based.
    States 1, ..., r-1 are non-degraded, states r, ..., n are degraded, and
    state n is absorbing during free evolution.
    """

    n_states: int = 5
    degraded_threshold: int = 4
    weibull_shapes: tuple[float, ...] = (1.5, 2.0, 2.5, 1.5)
    sojourn_means: tuple[float, ...] = (2.0, 2.5, 3.0, 2.0)

    # Deterministic age-grid solver.
    population_dt: float = 0.02
    max_age: float = 35.0
    stationary_mass_tolerance: float = 1.0e-13

    # High-accuracy deterministic convolution for T_deg.
    survival_dt: float = 0.002
    survival_individual_horizon: float = 50.0
    survival_tolerance: float = 1.0e-13

    # Main fixed-cycle experiment.
    baseline_tau: float = 3.0
    fixed_cycles: int = 28
    observations_per_cycle: int = 30
    microscopic_units: int = 200_000
    coarse_age_bin_width: float = 0.20

    # Threshold/fixed-cycle correspondence.  The curve is evaluated on the
    # interior of (0, 1), while the plot axes display the full [0, 1] range.
    threshold_curve_min: float = 0.05
    threshold_curve_max: float = 0.95
    threshold_curve_points: int = 19
    reported_thresholds: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
    threshold_max_cycles: int = 1_200
    threshold_convergence_tolerance: float = 1.0e-10
    fixed_mapping_tau_min: float = 0.20
    fixed_mapping_tau_max: float = 14.0
    fixed_mapping_points: int = 70

    random_seed: int = 20260903
    output_root: str = "outputs"

    @property
    def nondegraded_count(self) -> int:
        return self.degraded_threshold - 1

    @property
    def transient_count(self) -> int:
        return self.n_states - 1

    @property
    def mean_degradation_time(self) -> float:
        return float(sum(self.sojourn_means[: self.nondegraded_count]))

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)

    def validate(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be at least two")
        if not 2 <= self.degraded_threshold <= self.n_states:
            raise ValueError("degraded_threshold must belong to {2, ..., n_states}")
        if len(self.weibull_shapes) != self.transient_count:
            raise ValueError("one Weibull shape is required for each nonabsorbing state")
        if len(self.sojourn_means) != self.transient_count:
            raise ValueError("one sojourn mean is required for each nonabsorbing state")
        if any(x <= 0 for x in self.weibull_shapes + self.sojourn_means):
            raise ValueError("all Weibull parameters and means must be positive")
        if self.population_dt <= 0 or self.max_age <= 0:
            raise ValueError("population_dt and max_age must be positive")
        steps = self.baseline_tau / self.population_dt
        if abs(steps - round(steps)) > 1.0e-10:
            raise ValueError("baseline_tau must be an integer multiple of population_dt")
        if round(steps) % self.observations_per_cycle != 0:
            raise ValueError(
                "the number of population steps per baseline cycle must be divisible "
                "by observations_per_cycle"
            )
        if not 0 < self.threshold_curve_min < self.threshold_curve_max < 1:
            raise ValueError("threshold curve endpoints must lie strictly inside (0, 1)")
        if self.threshold_curve_points < 2:
            raise ValueError("threshold_curve_points must be at least two")
        if not all(0 < theta < 1 for theta in self.reported_thresholds):
            raise ValueError("all reported thresholds must lie in (0, 1)")
        if self.threshold_max_cycles < 1 or self.threshold_convergence_tolerance <= 0:
            raise ValueError("threshold convergence controls must be positive")
        if not 0 < self.fixed_mapping_tau_min < self.fixed_mapping_tau_max:
            raise ValueError("fixed-cycle mapping limits must be positive and ordered")
        if self.fixed_mapping_points < 2:
            raise ValueError("fixed_mapping_points must be at least two")
    def quick(self) -> "ExperimentConfig":
        """Return a fast smoke-test configuration with the same model."""

        return replace(
            self,
            population_dt=0.05,
            survival_dt=0.01,
            survival_individual_horizon=35.0,
            fixed_cycles=12,
            observations_per_cycle=20,
            microscopic_units=15_000,
            threshold_max_cycles=700,
            threshold_convergence_tolerance=5.0e-9,
            stationary_mass_tolerance=1.0e-10,
            survival_tolerance=1.0e-10,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mean_degradation_time"] = self.mean_degradation_time
        return data


# 2. Sequential Weibull degradation model

@dataclass(frozen=True)
class SequentialWeibullModel:
    """Sequential semi-Markov degradation model with Weibull sojourn times."""

    n_states: int
    degraded_threshold: int
    shapes: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "SequentialWeibullModel":
        config.validate()
        shapes = np.asarray(config.weibull_shapes, dtype=float)
        means = np.asarray(config.sojourn_means, dtype=float)
        scales = means / gamma(1.0 + 1.0 / shapes)
        return cls(
            n_states=config.n_states,
            degraded_threshold=config.degraded_threshold,
            shapes=shapes,
            means=means,
            scales=scales,
        )

    @property
    def transient_count(self) -> int:
        return self.n_states - 1

    @property
    def nondegraded_count(self) -> int:
        return self.degraded_threshold - 1

    @property
    def mean_degradation_time(self) -> float:
        return float(np.sum(self.means[: self.nondegraded_count]))

    def cumulative_hazard(self, state: int, age: np.ndarray | float) -> np.ndarray:
        age_array = np.asarray(age, dtype=float)
        return np.power(np.maximum(age_array, 0.0) / self.scales[state], self.shapes[state])

    def survival(self, state: int, age: np.ndarray | float) -> np.ndarray:
        return np.exp(-self.cumulative_hazard(state, age))

    def cdf(self, state: int, age: np.ndarray | float) -> np.ndarray:
        return -np.expm1(-self.cumulative_hazard(state, age))

    def density(self, state: int, age: np.ndarray | float) -> np.ndarray:
        x = np.asarray(age, dtype=float)
        beta = self.shapes[state]
        eta = self.scales[state]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            prefactor = (beta / eta) * np.power(np.maximum(x, 0.0) / eta, beta - 1.0)
            value = prefactor * np.exp(-np.power(np.maximum(x, 0.0) / eta, beta))
        if beta > 1.0:
            value = np.where(x == 0.0, 0.0, value)
        elif beta == 1.0:
            value = np.where(x == 0.0, 1.0 / eta, value)
        return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def conditional_step_survival(
        self, state: int, ages: np.ndarray, increment: float
    ) -> np.ndarray:
        """P(X(a)>increment) evaluated without dividing tiny survivals."""

        h0 = self.cumulative_hazard(state, ages)
        h1 = self.cumulative_hazard(state, ages + increment)
        return np.exp(-(h1 - h0))

    def residual_cdf(self, state: int, age: float, horizon: float) -> float:
        increment = self.cumulative_hazard(state, age + horizon) - self.cumulative_hazard(
            state, age
        )
        return float(-np.expm1(-increment))

    def residual_transition_infimum(self, state: int, horizon: float) -> float:
        """Return inf_a P(X_i(a) <= horizon) for a Weibull sojourn law.

        This is the residual-life quantity that appears directly in the
        general uniform-degradation bound.  No secondary constant is
        introduced.  For a Weibull shape below one the infimum over all
        ages is zero; otherwise it is attained at age zero.
        """

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.shapes[state] < 1.0:
            return 0.0
        return float(self.cdf(state, horizon))

    def sample_sojourn(
        self, state: int, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.scales[state] * rng.weibull(self.shapes[state], size=size)

    def sample_residual(
        self, state: int, ages: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample the residual Weibull life conditional on survival to each age."""

        ages = np.asarray(ages, dtype=float)
        exponential = rng.exponential(scale=1.0, size=ages.shape)
        beta = self.shapes[state]
        eta = self.scales[state]
        total = eta * np.power(np.power(ages / eta, beta) + exponential, 1.0 / beta)
        return total - ages

    def parameter_frame(self) -> pd.DataFrame:
        states = np.arange(1, self.n_states)
        return pd.DataFrame(
            {
                "state": states,
                "classification": [
                    "non-degraded" if i < self.degraded_threshold else "degraded"
                    for i in states
                ],
                "weibull_shape": self.shapes,
                "weibull_scale": self.scales,
                "sojourn_mean": self.means,
            }
        )


# 3. Individual-unit simulation

@dataclass
class MicroPopulation:
    """Exact event-driven simulation of independent physical units.

    ``state`` is zero-based.  States ``0, ..., n-2`` have an age and a
    residual sojourn time; state ``n-1`` is absorbing during free evolution.
    Selective maintenance resets units in states ``r, ..., n`` (paper
    notation) to the fresh state ``(1, 0)``.
    """

    model: SequentialWeibullModel
    rng: np.random.Generator
    state: np.ndarray
    age: np.ndarray
    residual: np.ndarray

    @classmethod
    def fresh(
        cls,
        model: SequentialWeibullModel,
        size: int,
        rng: np.random.Generator,
    ) -> "MicroPopulation":
        if size <= 0:
            raise ValueError("size must be positive")
        state = np.zeros(size, dtype=np.int8)
        age = np.zeros(size, dtype=float)
        residual = model.sample_sojourn(0, size, rng)
        return cls(model, rng, state, age, residual)

    @classmethod
    def from_state_age_mixture(
        cls,
        model: SequentialWeibullModel,
        size: int,
        state_weights: tuple[float, ...],
        state_ages: tuple[float, ...],
        rng: np.random.Generator,
    ) -> "MicroPopulation":
        """Sample a non-degraded state-age mixture with correct residual lives."""

        count = model.nondegraded_count
        if len(state_weights) != count or len(state_ages) != count:
            raise ValueError("one weight and age are required per non-degraded state")
        weights = np.asarray(state_weights, dtype=float)
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("state weights must be nonnegative and sum to one")
        state = rng.choice(count, size=size, p=weights).astype(np.int8)
        age = np.asarray(state_ages, dtype=float)[state]
        residual = np.empty(size, dtype=float)
        for current in range(count):
            mask = state == current
            residual[mask] = model.sample_residual(
                current,
                np.full(int(np.sum(mask)), float(state_ages[current])),
                rng,
            )
        return cls(model, rng, state, age, residual)

    @property
    def size(self) -> int:
        return int(self.state.size)

    def copy(self) -> "MicroPopulation":
        return MicroPopulation(
            self.model,
            self.rng,
            self.state.copy(),
            self.age.copy(),
            self.residual.copy(),
        )

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

        degraded = self.state >= self.model.nondegraded_count
        repaired = int(np.sum(degraded))
        self.state[degraded] = 0
        self.age[degraded] = 0.0
        self.residual[degraded] = self.model.sample_sojourn(0, repaired, self.rng)
        return repaired / self.size

    def state_ratios(self) -> np.ndarray:
        return np.bincount(self.state, minlength=self.model.n_states) / self.size

    def nondegraded_histogram(self, bin_width: float, max_age: float) -> np.ndarray:
        if bin_width <= 0 or max_age <= 0:
            raise ValueError("bin_width and max_age must be positive")
        bins = int(np.ceil(max_age / bin_width))
        histogram = np.zeros((self.model.nondegraded_count, bins), dtype=float)
        clipped = np.minimum((self.age / bin_width).astype(int), bins - 1)
        for current in range(self.model.nondegraded_count):
            mask = self.state == current
            histogram[current] = np.bincount(clipped[mask], minlength=bins) / self.size
        return histogram

    def full_histogram(self, bin_width: float, max_age: float) -> tuple[np.ndarray, float]:
        bins = int(np.ceil(max_age / bin_width))
        histogram = np.zeros((self.model.transient_count, bins), dtype=float)
        clipped = np.minimum((self.age / bin_width).astype(int), bins - 1)
        for current in range(self.model.transient_count):
            mask = self.state == current
            histogram[current] = np.bincount(clipped[mask], minlength=bins) / self.size
        absorbing = float(np.mean(self.state == self.model.n_states - 1))
        return histogram, absorbing


# 4. Age-grid population solver

@dataclass
class FullPopulationState:
    """Age-grid masses in states 1, ..., n-1 and mass in absorbing state n."""

    age_mass: np.ndarray
    absorbing_mass: float = 0.0

    def copy(self) -> "FullPopulationState":
        return FullPopulationState(self.age_mass.copy(), float(self.absorbing_mass))

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.age_mass) + self.absorbing_mass)


@dataclass
class StationaryGridResult:
    distribution: np.ndarray
    repair_fraction: float
    cycle_count_terms: int
    omitted_survival_mass: float
    invariant_residual_tv: float
    omitted_series_tail_bound: float
    grid_survival_contraction: float


class AgeGridSolver:
    """Deterministic finite-volume/cohort approximation on a common age grid.

    During a small time step, a cohort either survives its current state and
    moves to the next age cell or exits and enters the next state at age zero.
    The scheme is mass-conservative.  It is first-order in ``dt``; the driver
    performs a grid-refinement check for the reported baseline results.
    """

    def __init__(self, model: SequentialWeibullModel, dt: float, max_age: float):
        if dt <= 0 or max_age <= 0:
            raise ValueError("dt and max_age must be positive")
        self.model = model
        self.dt = float(dt)
        self.max_age = float(max_age)
        # Cells represent [j dt, (j+1) dt); the final cell is also used as a
        # conservative overflow cell.  This convention makes aggregation into
        # wider reporting bins identical to the microscopic histogram.
        self.n_bins = int(np.ceil(max_age / dt))
        self.ages = np.arange(self.n_bins, dtype=float) * dt
        self._conditional_survival = np.vstack(
            [
                model.conditional_step_survival(state, self.ages, self.dt)
                for state in range(model.transient_count)
            ]
        )

    @property
    def nondegraded_count(self) -> int:
        return self.model.nondegraded_count

    @property
    def transient_count(self) -> int:
        return self.model.transient_count

    def steps_for_time(self, time: float) -> int:
        value = time / self.dt
        steps = int(round(value))
        if abs(value - steps) > 1.0e-9:
            raise ValueError(f"time {time} is not an integer multiple of dt={self.dt}")
        return steps

    def fresh_nondegraded(self) -> np.ndarray:
        mass = np.zeros((self.nondegraded_count, self.n_bins), dtype=float)
        mass[0, 0] = 1.0
        return mass

    def initial_nondegraded(
        self,
        state_weights: tuple[float, ...],
        state_ages: tuple[float, ...],
    ) -> np.ndarray:
        if len(state_weights) != self.nondegraded_count or len(state_ages) != self.nondegraded_count:
            raise ValueError("weights and ages must have one entry per non-degraded state")
        weights = np.asarray(state_weights, dtype=float)
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("state_weights must be nonnegative and sum to one")
        mass = np.zeros((self.nondegraded_count, self.n_bins), dtype=float)
        for state, (weight, age) in enumerate(zip(weights, state_ages)):
            index = int(round(age / self.dt))
            if not 0 <= index < self.n_bins:
                raise ValueError("initial age lies outside the age grid")
            mass[state, index] += weight
        return mass

    def embed_nondegraded(self, distribution: np.ndarray) -> FullPopulationState:
        self._validate_nondegraded(distribution)
        age_mass = np.zeros((self.transient_count, self.n_bins), dtype=float)
        age_mass[: self.nondegraded_count] = distribution
        return FullPopulationState(age_mass=age_mass, absorbing_mass=0.0)

    def step_full(self, state: FullPopulationState) -> FullPopulationState:
        if state.age_mass.shape != (self.transient_count, self.n_bins):
            raise ValueError("invalid full-population array shape")
        new_mass = np.zeros_like(state.age_mass)
        new_absorbing = float(state.absorbing_mass)

        for current in range(self.transient_count):
            mass = state.age_mass[current]
            survived = mass * self._conditional_survival[current]
            exited = mass - survived
            new_mass[current, 1:] += survived[:-1]
            # The last cell is an overflow cell.  With max_age=35 its mass is
            # numerically negligible for the reported baseline.
            new_mass[current, -1] += survived[-1]
            exit_mass = float(np.sum(exited))
            if current + 1 < self.transient_count:
                new_mass[current + 1, 0] += exit_mass
            else:
                new_absorbing += exit_mass

        result = FullPopulationState(new_mass, new_absorbing)
        if abs(result.total_mass - state.total_mass) > 2.0e-12:
            raise RuntimeError("the full population step did not conserve mass")
        return result

    def propagate_full(self, state: FullPopulationState, steps: int) -> FullPopulationState:
        result = state.copy()
        for _ in range(steps):
            result = self.step_full(result)
        return result

    def state_ratios(self, state: FullPopulationState) -> np.ndarray:
        return np.concatenate([np.sum(state.age_mass, axis=1), [state.absorbing_mass]])

    def degraded_mass(self, state: FullPopulationState) -> float:
        degraded_rows = state.age_mass[self.model.degraded_threshold - 1 :]
        return float(np.sum(degraded_rows) + state.absorbing_mass)

    def maintain_full(self, state: FullPopulationState) -> FullPopulationState:
        repaired = self.degraded_mass(state)
        new_mass = state.age_mass.copy()
        new_mass[self.model.degraded_threshold - 1 :] = 0.0
        new_mass[0, 0] += repaired
        result = FullPopulationState(new_mass, 0.0)
        if abs(result.total_mass - state.total_mass) > 2.0e-12:
            raise RuntimeError("maintenance did not conserve mass")
        return result

    def extract_nondegraded(self, state: FullPopulationState) -> np.ndarray:
        return state.age_mass[: self.nondegraded_count].copy()

    def step_killed(self, distribution: np.ndarray) -> tuple[np.ndarray, float]:
        self._validate_nondegraded(distribution)
        new_mass = np.zeros_like(distribution)
        killed = 0.0
        for current in range(self.nondegraded_count):
            mass = distribution[current]
            survived = mass * self._conditional_survival[current]
            exited = mass - survived
            new_mass[current, 1:] += survived[:-1]
            new_mass[current, -1] += survived[-1]
            exit_mass = float(np.sum(exited))
            if current + 1 < self.nondegraded_count:
                new_mass[current + 1, 0] += exit_mass
            else:
                killed += exit_mass
        if abs(float(np.sum(distribution)) - float(np.sum(new_mass)) - killed) > 2.0e-12:
            raise RuntimeError("the killed step did not account for all mass")
        return new_mass, killed

    def propagate_killed(self, distribution: np.ndarray, steps: int) -> tuple[np.ndarray, float]:
        current = distribution.copy()
        killed_total = 0.0
        for _ in range(steps):
            current, killed = self.step_killed(current)
            killed_total += killed
        return current, killed_total

    def return_map(self, distribution: np.ndarray, steps: int) -> tuple[np.ndarray, float]:
        survivors, killed = self.propagate_killed(distribution, steps)
        result = survivors
        result[0, 0] += killed
        if abs(float(np.sum(result)) - float(np.sum(distribution))) > 2.0e-12:
            raise RuntimeError("return map did not preserve probability mass")
        return result, killed

    def return_map_at_time(
        self, distribution: np.ndarray, cycle_time: float
    ) -> tuple[np.ndarray, float]:
        """Apply a fixed-cycle return map at an off-grid cycle time.

        The final fractional step uses the same first-order interpolation as
        ``threshold_return_map``.  Consequently, the forward and reverse
        policy mappings are compared under one consistent discretization.
        """

        self._validate_nondegraded(distribution)
        if cycle_time <= 0:
            raise ValueError("cycle_time must be positive")
        complete_steps = int(np.floor(cycle_time / self.dt + 1.0e-12))
        alpha = float(cycle_time / self.dt - complete_steps)
        if alpha < 1.0e-12:
            alpha = 0.0
        survivors, killed = self.propagate_killed(distribution, complete_steps)
        if alpha > 0.0:
            next_survivors, killed_in_step = self.step_killed(survivors)
            survivors = (1.0 - alpha) * survivors + alpha * next_survivors
            killed += alpha * killed_in_step
        result = survivors.copy()
        result[0, 0] += killed
        if abs(float(np.sum(result)) - float(np.sum(distribution))) > 2.0e-12:
            raise RuntimeError("off-grid return map did not preserve probability mass")
        return result, float(killed)

    def stationary_renewal(
        self,
        tau_steps: int,
        tolerance: float = 1.0e-13,
        maximum_terms: int = 100_000,
    ) -> StationaryGridResult:
        if tau_steps < 1:
            raise ValueError("tau_steps must be positive")
        # Backward survival probabilities give the induced mass norm of the
        # killed, discretized one-cycle kernel.  This is a bound for this grid
        # kernel, not an assumed transfer of the continuous-model bound.
        survival = np.ones((self.nondegraded_count, self.n_bins))
        for _ in range(tau_steps):
            updated = np.empty_like(survival)
            for state in range(self.nondegraded_count):
                conditional = self._conditional_survival[state]
                following = np.r_[survival[state, 1:], survival[state, -1]]
                updated[state] = conditional * following
                if state + 1 < self.nondegraded_count:
                    updated[state] += (1.0 - conditional) * survival[state + 1, 0]
            survival = updated
        contraction = float(np.max(survival))
        if not 0 <= contraction < 1:
            raise RuntimeError("grid survival contraction is not strictly below one")
        current = self.fresh_nondegraded()
        potential = np.zeros_like(current)
        terms = 0
        while terms < maximum_terms:
            potential += current
            terms += 1
            current, _ = self.propagate_killed(current, tau_steps)
            tail_bound = float(np.sum(current)) / (1.0 - contraction)
            if tail_bound <= tolerance:
                break
        else:
            raise RuntimeError("stationary renewal series did not reach the tolerance")

        total_potential = float(np.sum(potential))
        repair_fraction = 1.0 / total_potential
        stationary = repair_fraction * potential
        mapped, _ = self.return_map(stationary, tau_steps)
        residual = self.tv_distance(stationary, mapped)
        return StationaryGridResult(
            distribution=stationary,
            repair_fraction=repair_fraction,
            cycle_count_terms=terms,
            omitted_survival_mass=float(np.sum(current)),
            invariant_residual_tv=residual,
            omitted_series_tail_bound=tail_bound,
            grid_survival_contraction=contraction,
        )

    def threshold_return_map(
        self,
        distribution: np.ndarray,
        threshold: float,
        maximum_time: float,
    ) -> tuple[np.ndarray, float, float]:
        self._validate_nondegraded(distribution)
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie in (0, 1)")
        current = distribution.copy()
        degraded = 0.0
        maximum_steps = int(np.ceil(maximum_time / self.dt))
        for step in range(1, maximum_steps + 1):
            previous = current
            degraded_before = degraded
            current, killed = self.step_killed(previous)
            degraded = degraded_before + killed
            # The stationary renewal series is truncated at a much smaller
            # tolerance, so allow roundoff at an equality threshold.  This
            # prevents an artificial extra time step when D(t; mu*)=theta.
            if degraded + 2.0e-12 >= threshold:
                # Locate the threshold inside the final time step.  Linear
                # interpolation is consistent with the first-order cohort
                # scheme and avoids a spurious limit cycle caused solely by
                # restricting inspection times to the time grid.
                alpha = 1.0 if killed <= 2.0e-15 else (threshold - degraded_before) / killed
                alpha = float(np.clip(alpha, 0.0, 1.0))
                survivors = (1.0 - alpha) * previous + alpha * current
                post = survivors.copy()
                post[0, 0] += threshold
                post /= np.sum(post)
                stopping_time = (step - 1 + alpha) * self.dt
                return post, stopping_time, threshold
        raise RuntimeError("threshold was not reached within maximum_time")

    def coarse_nondegraded(self, distribution: np.ndarray, bin_width: float) -> np.ndarray:
        self._validate_nondegraded(distribution)
        group = max(1, int(round(bin_width / self.dt)))
        groups = int(np.ceil(self.n_bins / group))
        padded = np.zeros((self.nondegraded_count, groups * group), dtype=float)
        padded[:, : self.n_bins] = distribution
        return padded.reshape(self.nondegraded_count, groups, group).sum(axis=2)

    def full_age_histogram(self, state: FullPopulationState, bin_width: float) -> tuple[np.ndarray, float]:
        group = max(1, int(round(bin_width / self.dt)))
        groups = int(np.ceil(self.n_bins / group))
        padded = np.zeros((self.transient_count, groups * group), dtype=float)
        padded[:, : self.n_bins] = state.age_mass
        histogram = padded.reshape(self.transient_count, groups, group).sum(axis=2)
        return histogram, float(state.absorbing_mass)

    @staticmethod
    def tv_distance(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            raise ValueError("TV distance requires arrays with equal shape")
        return float(0.5 * np.sum(np.abs(first - second)))

    def overflow_mass(self, distribution: np.ndarray) -> float:
        return float(np.sum(distribution[:, -1]))

    def _validate_nondegraded(self, distribution: np.ndarray) -> None:
        if distribution.shape != (self.nondegraded_count, self.n_bins):
            raise ValueError("invalid non-degraded distribution shape")
        if np.any(distribution < -1.0e-15):
            raise ValueError("population mass cannot be negative")


# 5. Degradation-time distribution and stationary repair fraction

@dataclass
class DegradationTimeDistribution:
    """Deterministic convolution approximation of the fresh degradation time.

    Each Weibull law is discretized into probability masses on intervals of
    width ``dt``.  A mass is assigned to the midpoint of its interval.  FFT
    convolution then gives the distributions of the remaining sums
    X_i + ... + X_{r-1}.  The approximation is deterministic and independent
    of the microscopic Monte Carlo simulation.
    """

    model: SequentialWeibullModel
    dt: float
    individual_horizon: float
    remaining_sum_grids: dict[int, np.ndarray]
    remaining_sum_cdfs: dict[int, np.ndarray]
    remaining_sum_masses: dict[int, np.ndarray]

    @classmethod
    def build(
        cls,
        model: SequentialWeibullModel,
        dt: float,
        individual_horizon: float,
    ) -> "DegradationTimeDistribution":
        if dt <= 0 or individual_horizon <= 0:
            raise ValueError("dt and individual_horizon must be positive")

        edge_count = int(np.ceil(individual_horizon / dt))
        edges = np.arange(edge_count + 1, dtype=float) * dt
        component_masses: dict[int, np.ndarray] = {}
        for state in range(model.nondegraded_count):
            cdf_edges = model.cdf(state, edges)
            masses = np.maximum(np.diff(cdf_edges), 0.0)
            omitted = max(0.0, 1.0 - float(np.sum(masses)))
            if omitted > 1.0e-10:
                raise ValueError(
                    f"individual_horizon truncates {omitted:.3e} probability "
                    f"from state {state + 1}; increase the horizon"
                )
            masses[-1] += omitted
            component_masses[state] = masses / np.sum(masses)

        grids: dict[int, np.ndarray] = {}
        cdfs: dict[int, np.ndarray] = {}
        sums: dict[int, np.ndarray] = {}
        for start in range(model.nondegraded_count):
            conv = np.array([1.0])
            component_count = 0
            for state in range(start, model.nondegraded_count):
                conv = fftconvolve(conv, component_masses[state])
                conv = np.maximum(conv, 0.0)
                component_count += 1
            conv /= np.sum(conv)
            # The convolution index records the sum of interval indices.
            # Each of the component masses is represented at its midpoint.
            grid = (np.arange(conv.size, dtype=float) + 0.5 * component_count) * dt
            cumulative = np.minimum(np.cumsum(conv), 1.0)
            grids[start] = grid
            cdfs[start] = cumulative
            sums[start] = conv

        return cls(
            model=model,
            dt=dt,
            individual_horizon=individual_horizon,
            remaining_sum_grids=grids,
            remaining_sum_cdfs=cdfs,
            remaining_sum_masses=sums,
        )

    def cdf(self, t: np.ndarray | float, start_state: int = 0) -> np.ndarray:
        values = np.asarray(t, dtype=float)
        grid = self.remaining_sum_grids[start_state]
        cdf_values = self.remaining_sum_cdfs[start_state]
        out = np.interp(values, grid, cdf_values, left=0.0, right=1.0)
        return np.clip(out, 0.0, 1.0)

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        return np.clip(1.0 - self.cdf(t, start_state=0), 0.0, 1.0)

    @property
    def support_max(self) -> float:
        return float(self.remaining_sum_grids[0][-1])

    def survival_series_terms(self, tau: float, tolerance: float = 1.0e-14) -> np.ndarray:
        if tau <= 0:
            raise ValueError("tau must be positive")
        maximum_k = int(np.ceil(self.support_max / tau)) + 1
        times = tau * np.arange(maximum_k + 1, dtype=float)
        terms = self.survival(times)
        terms[0] = 1.0
        # Retain all terms over the convolution support.  A small individual
        # survival term alone does not bound the sum of the omitted terms.
        if self.continuous_tail_bound(tau, maximum_k) > tolerance:
            raise ValueError("increase individual_horizon to control the series tail")
        return terms

    def continuous_tail_bound(self, tau: float, last_index: int) -> float:
        """Bound sum_{k>M} P(T_deg > k*tau) for the original Weibull sum.

        Monotonicity bounds the sum by an integrated survival divided by tau.
        The union bound P(sum X_i > t) <= sum P(X_i > t/d) then yields the
        incomplete-gamma expression.  This controls truncation only; CDF grid
        and interpolation errors are assessed by refinement separately.
        """
        count = self.model.nondegraded_count
        shapes = self.model.shapes[:count]
        scales = self.model.scales[:count]
        lower = last_index * tau / count
        integrated = scales / shapes * gamma(1.0 / shapes) * gammaincc(
            1.0 / shapes, (lower / scales) ** shapes
        )
        return float(count * np.sum(integrated) / tau)

    def stationary_repair_fraction(self, tau: float, tolerance: float = 1.0e-14) -> float:
        terms = self.survival_series_terms(tau, tolerance=tolerance)
        return float(1.0 / np.sum(terms))

    def calibrate_cycle(
        self,
        target: float,
        lower: float | None = None,
        upper: float | None = None,
        tolerance: float = 1.0e-11,
    ) -> float:
        if not 0 < target < 1:
            raise ValueError("target must lie in (0, 1)")
        lo = self.dt if lower is None else lower
        hi = self.model.mean_degradation_time if upper is None else upper
        while self.stationary_repair_fraction(hi) < target:
            hi *= 2.0
            if hi > self.support_max:
                raise RuntimeError("failed to bracket the calibrated cycle")
        return float(
            brentq(
                lambda tau: self.stationary_repair_fraction(tau) - target,
                lo,
                hi,
                xtol=tolerance,
                rtol=tolerance,
            )
        )

    def uniform_epsilon(self, tau0: float) -> float:
        """Uniform degradation probability used in the general proof.

        For a unit currently in state i with age a, the first factor is the
        infimum over a of the residual probability of leaving state i during
        h_i=tau0/(r-i).  Subsequent states begin at age zero, which gives the
        remaining product of ordinary sojourn-time distribution functions.
        """

        if tau0 <= 0:
            raise ValueError("tau0 must be positive")
        probabilities: list[float] = []
        for state in range(self.model.nondegraded_count):
            transition_count = self.model.nondegraded_count - state
            h_i = tau0 / transition_count
            probability = self.model.residual_transition_infimum(state, h_i)
            for future_state in range(state + 1, self.model.nondegraded_count):
                probability *= float(self.model.cdf(future_state, h_i))
            probabilities.append(probability)
        epsilon = float(min(probabilities))
        if not 0.0 < epsilon < 1.0:
            raise ValueError(
                "the residual-life lower bound does not provide a positive epsilon"
            )
        return epsilon

    @staticmethod
    def geometric_tail_bound(epsilon: float, m: np.ndarray | int) -> np.ndarray:
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must lie in (0, 1)")
        indices = np.asarray(m, dtype=float)
        log_bound = (indices + 1.0) * np.log1p(-epsilon) - np.log(epsilon)
        return np.exp(np.clip(log_bound, -745.0, 709.0))

    def actual_tail(self, tau: float, m: int, tolerance: float = 1.0e-14) -> float:
        terms = self.survival_series_terms(tau, tolerance=tolerance)
        start = m + 1
        if start >= terms.size:
            return 0.0
        return float(np.sum(terms[start:]))

    def certified_interval(
        self,
        tau: float,
        m: int,
        epsilon: float,
        tolerance: float = 1.0e-14,
    ) -> tuple[float, float, float, float]:
        """Series-truncation interval; excludes CDF discretization error."""
        terms = self.survival_series_terms(tau, tolerance=tolerance)
        if m >= terms.size:
            partial = float(np.sum(terms))
        else:
            partial = float(np.sum(terms[: m + 1]))
        remainder_bound = float(self.geometric_tail_bound(epsilon, m))
        lower = 1.0 / (partial + remainder_bound)
        upper = 1.0 / partial
        actual = float(1.0 / np.sum(terms))
        return lower, upper, actual, remainder_bound


# 6. Fixed-cycle and threshold-policy experiments

@dataclass
class FixedCycleResults:
    trajectory: pd.DataFrame
    cycle_convergence: pd.DataFrame
    deterministic_starts: pd.DataFrame
    snapshots: pd.DataFrame
    snapshot_atoms: pd.DataFrame
    stationary_state_ratios: pd.DataFrame
    validation_errors: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class ThresholdResults:
    threshold_to_cycle: pd.DataFrame
    fixed_to_threshold: pd.DataFrame
    correspondence_summary: pd.DataFrame


def _append_ratios(
    records: list[dict[str, float | int | str]],
    time: float,
    cycle: int,
    phase: str,
    source: str,
    ratios: np.ndarray,
) -> None:
    for state, value in enumerate(ratios, start=1):
        records.append(
            {
                "time": time,
                "cycle": cycle,
                "phase": phase,
                "source": source,
                "state": state,
                "fraction": float(value),
            }
        )


def _histogram_records(
    histogram: np.ndarray,
    source: str,
    phase_name: str,
    phase_time: float,
    bin_width: float,
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for state in range(histogram.shape[0]):
        for age_bin, probability in enumerate(histogram[state]):
            records.append(
                {
                    "phase": phase_name,
                    "phase_time": phase_time,
                    "source": source,
                    "state": state + 1,
                    "age_left": age_bin * bin_width,
                    "age_right": (age_bin + 1) * bin_width,
                    "mass": float(probability),
                    "density": float(probability / bin_width),
                }
            )
    return records


def run_fixed_cycle_experiments(
    config: ExperimentConfig,
    model: SequentialWeibullModel,
    solver: AgeGridSolver,
    degradation: DegradationTimeDistribution,
) -> FixedCycleResults:
    tau = config.baseline_tau
    tau_steps = solver.steps_for_time(tau)
    observation_steps = tau_steps // config.observations_per_cycle
    observation_dt = observation_steps * solver.dt
    rng = np.random.default_rng(config.random_seed)

    stationary = solver.stationary_renewal(
        tau_steps,
        tolerance=config.stationary_mass_tolerance,
    )
    stationary_coarse = solver.coarse_nondegraded(
        stationary.distribution, config.coarse_age_bin_width
    )

    micro = MicroPopulation.fresh(model, config.microscopic_units, rng)
    full = solver.embed_nondegraded(solver.fresh_nondegraded())
    trajectory_records: list[dict[str, float | int | str]] = []
    convergence_records: list[dict[str, float | int | str]] = []

    _append_ratios(
        trajectory_records,
        0.0,
        0,
        "post-maintenance",
        "individual unit simulation",
        micro.state_ratios(),
    )
    _append_ratios(
        trajectory_records, 0.0, 0, "post-maintenance", "population", solver.state_ratios(full)
    )

    micro_hist = micro.nondegraded_histogram(config.coarse_age_bin_width, config.max_age)
    pop_hist = solver.coarse_nondegraded(solver.extract_nondegraded(full), config.coarse_age_bin_width)
    convergence_records.append(
        {
            "cycle": 0,
            "micro_vs_population": solver.tv_distance(micro_hist, pop_hist),
            "population_vs_stationary": solver.tv_distance(pop_hist, stationary_coarse),
            "micro_vs_stationary": solver.tv_distance(micro_hist, stationary_coarse),
        }
    )

    for cycle in range(1, config.fixed_cycles + 1):
        for observation in range(1, config.observations_per_cycle + 1):
            micro.advance(observation_dt)
            full = solver.propagate_full(full, observation_steps)
            time = (cycle - 1) * tau + observation * observation_dt
            phase = "pre-maintenance" if observation == config.observations_per_cycle else "free evolution"
            _append_ratios(
                trajectory_records,
                time,
                cycle,
                phase,
                "individual unit simulation",
                micro.state_ratios(),
            )
            _append_ratios(
                trajectory_records, time, cycle, phase, "population", solver.state_ratios(full)
            )

        micro.maintain()
        full = solver.maintain_full(full)
        time = cycle * tau
        _append_ratios(
            trajectory_records,
            time,
            cycle,
            "post-maintenance",
            "individual unit simulation",
            micro.state_ratios(),
        )
        _append_ratios(
            trajectory_records, time, cycle, "post-maintenance", "population", solver.state_ratios(full)
        )

        micro_hist = micro.nondegraded_histogram(
            config.coarse_age_bin_width, config.max_age
        )
        pop_hist = solver.coarse_nondegraded(
            solver.extract_nondegraded(full), config.coarse_age_bin_width
        )
        convergence_records.append(
            {
                "cycle": cycle,
                "micro_vs_population": solver.tv_distance(micro_hist, pop_hist),
                "population_vs_stationary": solver.tv_distance(pop_hist, stationary_coarse),
                "micro_vs_stationary": solver.tv_distance(micro_hist, stationary_coarse),
            }
        )

    # Compare convergence from deliberately different initial state-age laws.
    starts = {
        "state 1, age 0": solver.fresh_nondegraded(),
        "state 3, age 1": solver.initial_nondegraded((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        "mixed": solver.initial_nondegraded((0.40, 0.35, 0.25), (0.0, 0.6, 1.0)),
    }
    start_records: list[dict[str, float | int | str]] = []
    for name, initial in starts.items():
        current = initial
        for cycle in range(config.fixed_cycles + 1):
            current_coarse = solver.coarse_nondegraded(
                current, config.coarse_age_bin_width
            )
            start_records.append(
                {
                    "initial_distribution": name,
                    "cycle": cycle,
                    "tv_to_stationary": solver.tv_distance(current_coarse, stationary_coarse),
                }
            )
            if cycle < config.fixed_cycles:
                current, _ = solver.return_map(current, tau_steps)

    # One additional stationary-regime cycle supplies comparable state-age
    # snapshots.  The individual-unit and population calculations have already
    # passed through the configured burn-in.
    snapshot_records: list[dict[str, float | int | str]] = []
    atom_records: list[dict[str, float | str]] = []
    snapshot_specs = [
        ("post-maintenance", 0.0, 0),
        ("mid-cycle", tau / 2.0, solver.steps_for_time(tau / 2.0)),
        ("pre-maintenance", tau, tau_steps),
    ]
    last_step = 0
    for phase_name, phase_time, target_step in snapshot_specs:
        increment = target_step - last_step
        if increment:
            micro.advance(increment * solver.dt)
            full = solver.propagate_full(full, increment)
        last_step = target_step
        for source, pair in (
            (
                "individual unit simulation",
                micro.full_histogram(config.coarse_age_bin_width, config.max_age),
            ),
            (
                "population",
                solver.full_age_histogram(full, config.coarse_age_bin_width),
            ),
        ):
            histogram, absorbing = pair
            snapshot_records.extend(
                _histogram_records(
                    histogram,
                    source,
                    phase_name,
                    phase_time,
                    config.coarse_age_bin_width,
                )
            )
            atom_records.append(
                {
                    "phase": phase_name,
                    "phase_time": phase_time,
                    "source": source,
                    "absorbing_state_mass": absorbing,
                }
            )

    phase_ratio_records: list[dict[str, float | int | str]] = []
    for phase_name, phase_time, target_step in snapshot_specs:
        phase_state = solver.propagate_full(
            solver.embed_nondegraded(stationary.distribution), target_step
        )
        _append_ratios(
            phase_ratio_records,
            phase_time,
            0,
            phase_name,
            "stationary theory",
            solver.state_ratios(phase_state),
        )

    trajectory_frame = pd.DataFrame(trajectory_records)
    ratio_wide = trajectory_frame.pivot_table(
        index=["time", "cycle", "phase", "state"],
        columns="source",
        values="fraction",
    ).reset_index()
    ratio_wide["absolute_error"] = np.abs(
        ratio_wide["individual unit simulation"] - ratio_wide["population"]
    )
    validation_records: list[dict[str, float | int | str]] = []
    for window, selected in (
        ("all observations", ratio_wide),
        (
            f"cycles {config.fixed_cycles - 4}--{config.fixed_cycles}",
            ratio_wide[ratio_wide["time"] >= config.fixed_cycles * tau - 5.0 * tau],
        ),
    ):
        for state, data in selected.groupby("state"):
            validation_records.append(
                {
                    "comparison": "state proportions: individual unit simulation vs population",
                    "window_or_phase": window,
                    "state": int(state),
                    "mean_absolute_error": float(data["absolute_error"].mean()),
                    "root_mean_squared_error": float(
                        np.sqrt(np.mean(np.square(data["absolute_error"])))
                    ),
                    "maximum_absolute_error": float(data["absolute_error"].max()),
                    "binned_tv_distance": np.nan,
                }
            )

    snapshot_frame = pd.DataFrame(snapshot_records)
    atom_frame = pd.DataFrame(atom_records)
    for phase in snapshot_frame["phase"].drop_duplicates():
        phase_data = snapshot_frame[snapshot_frame["phase"] == phase]
        mass_wide = phase_data.pivot_table(
            index=["state", "age_left"], columns="source", values="mass"
        )
        phase_atoms = atom_frame[atom_frame["phase"] == phase].set_index("source")
        for first, second in (("individual unit simulation", "population"),):
            atom_difference = abs(
                float(phase_atoms.loc[first, "absorbing_state_mass"])
                - float(phase_atoms.loc[second, "absorbing_state_mass"])
            )
            tv = 0.5 * (
                float(np.sum(np.abs(mass_wide[first] - mass_wide[second])))
                + atom_difference
            )
            validation_records.append(
                {
                    "comparison": f"state-age snapshot: {first} vs {second}",
                    "window_or_phase": phase,
                    "state": 0,
                    "mean_absolute_error": np.nan,
                    "root_mean_squared_error": np.nan,
                    "maximum_absolute_error": np.nan,
                    "binned_tv_distance": tv,
                }
            )

    summary = pd.DataFrame(
        [
            {
                "quantity": "mean degradation time",
                "value": model.mean_degradation_time,
                "unit": "time",
            },
            {"quantity": "fixed cycle", "value": tau, "unit": "time"},
            {
                "quantity": "stationary repair fraction (population grid)",
                "value": stationary.repair_fraction,
                "unit": "probability",
            },
            {
                "quantity": "stationary repair fraction (survival series)",
                "value": degradation.stationary_repair_fraction(tau),
                "unit": "probability",
            },
            {
                "quantity": "stationary invariant residual TV",
                "value": stationary.invariant_residual_tv,
                "unit": "probability",
            },
            {
                "quantity": "stationary renewal first omitted survival term",
                "value": stationary.omitted_survival_mass,
                "unit": "probability",
            },
            {
                "quantity": "stationary renewal omitted-series bound (grid kernel)",
                "value": stationary.omitted_series_tail_bound,
                "unit": "mass",
            },
            {
                "quantity": "stationary renewal retained terms",
                "value": stationary.cycle_count_terms,
                "unit": "terms",
            },
            {
                "quantity": "microscopic units",
                "value": float(config.microscopic_units),
                "unit": "units",
            },
        ]
    )
    return FixedCycleResults(
        trajectory=trajectory_frame,
        cycle_convergence=pd.DataFrame(convergence_records),
        deterministic_starts=pd.DataFrame(start_records),
        snapshots=snapshot_frame,
        snapshot_atoms=atom_frame,
        stationary_state_ratios=pd.DataFrame(phase_ratio_records),
        validation_errors=pd.DataFrame(validation_records),
        summary=summary,
    )


def run_threshold_experiments(
    config: ExperimentConfig,
    solver: AgeGridSolver,
    degradation: DegradationTimeDistribution,
) -> ThresholdResults:
    target_values = np.linspace(
        config.threshold_curve_min,
        config.threshold_curve_max,
        config.threshold_curve_points,
    )
    maximum_time = max(
        4.0 * model_time_scale(degradation), config.fixed_mapping_tau_max + 5.0
    )
    correspondence_records: list[dict[str, float | int]] = []
    for target in target_values:
        threshold_distribution, threshold_tau, threshold_cycles, threshold_residual = (
            _stationary_threshold_policy(
                solver,
                float(target),
                maximum_time,
                config.threshold_max_cycles,
                config.threshold_convergence_tolerance,
            )
        )
        fixed_distribution, recovered, fixed_cycles, fixed_residual = (
            _stationary_fixed_policy(
                solver,
                threshold_tau,
                config.threshold_max_cycles,
                config.threshold_convergence_tolerance,
            )
        )
        independent_tau = degradation.calibrate_cycle(float(target))
        independent_threshold = degradation.stationary_repair_fraction(threshold_tau)
        correspondence_records.append(
            {
                "threshold": float(target),
                "threshold_policy_tau": threshold_tau,
                "fixed_policy_recovered_threshold": recovered,
                "round_trip_absolute_error": abs(recovered - target),
                "between_stationary_policies_tv": solver.tv_distance(
                    threshold_distribution, fixed_distribution
                ),
                "threshold_cycles_to_convergence": threshold_cycles,
                "fixed_cycles_to_convergence": fixed_cycles,
                "threshold_invariance_residual_tv": threshold_residual,
                "fixed_invariance_residual_tv": fixed_residual,
                "survival_series_calibrated_tau": independent_tau,
                "survival_series_threshold_at_iterated_tau": independent_threshold,
                "independent_calibration_error": abs(independent_threshold - target),
                "independent_cycle_error": abs(threshold_tau - independent_tau),
            }
        )

    # The fixed-cycle mapping is computed independently through the stationary
    # renewal series on cycle lengths aligned with the population grid.
    tau_values = np.linspace(
        config.fixed_mapping_tau_min,
        config.fixed_mapping_tau_max,
        config.fixed_mapping_points,
    )
    fixed_records: list[dict[str, float]] = []
    for tau in tau_values:
        steps = solver.steps_for_time(float(tau))
        stationary = solver.stationary_renewal(
            steps, tolerance=config.stationary_mass_tolerance
        )
        fixed_records.append(
            {
                "tau": float(tau),
                "fixed_policy_threshold": stationary.repair_fraction,
            }
        )

    correspondence = pd.DataFrame(correspondence_records)
    reported = correspondence[
        correspondence["threshold"].round(10).isin(
            [round(value, 10) for value in config.reported_thresholds]
        )
    ].reset_index(drop=True)
    return ThresholdResults(
        threshold_to_cycle=correspondence,
        fixed_to_threshold=pd.DataFrame(fixed_records),
        correspondence_summary=reported,
    )


def model_time_scale(degradation: DegradationTimeDistribution) -> float:
    return degradation.model.mean_degradation_time


def _stationary_threshold_policy(
    solver: AgeGridSolver,
    target: float,
    maximum_time: float,
    maximum_cycles: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int, float]:
    current = solver.fresh_nondegraded()
    previous_tau = np.nan
    for cycle in range(1, maximum_cycles + 1):
        mapped, cycle_time, _ = solver.threshold_return_map(
            current, target, maximum_time=maximum_time
        )
        residual = solver.tv_distance(current, mapped)
        time_change = (
            np.inf if not np.isfinite(previous_tau) else abs(cycle_time - previous_tau)
        )
        current = mapped
        if residual <= tolerance and time_change <= max(tolerance, 1.0e-11):
            return current, float(cycle_time), cycle, float(residual)
        previous_tau = cycle_time
    raise RuntimeError(f"threshold policy did not converge for theta={target:g}")


def _stationary_fixed_policy(
    solver: AgeGridSolver,
    cycle_time: float,
    maximum_cycles: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int, float]:
    current = solver.fresh_nondegraded()
    for cycle in range(1, maximum_cycles + 1):
        mapped, repaired = solver.return_map_at_time(current, cycle_time)
        residual = solver.tv_distance(current, mapped)
        current = mapped
        if residual <= tolerance:
            return current, float(repaired), cycle, float(residual)
    raise RuntimeError(f"fixed-cycle policy did not converge for tau={cycle_time:g}")


# 7. Cost optimization and renewal-reward validation

K, CR, CD = 1.0, 2.0, 1.0

def components(dist, tau, cd=CD):
    th = dist.stationary_repair_fraction(float(tau))
    m = dist.model.mean_degradation_time
    setup, repair, loss = K/tau, CR*th/tau, cd*(1-m*th/tau)
    return dict(tau=float(tau), theta=th, setup=setup, repair=repair,
                degraded_loss=loss, total=setup+repair+loss)

def optimize(dist, tol=1e-7):
    """Bracket the entire positive domain and bound the interpolated objective.

    Interval bounds use only monotonicity of the computed theta. The numerical
    bound is for this discretized objective, not an error certificate for the
    exact continuous Weibull model. Grid refinement is reported separately.
    """
    m=dist.model.mean_degradation_time; A=CD*m-CR
    f=lambda t:components(dist,t)['total']
    trial=3.0; incumbent=f(trial)
    assert incumbent<CD and A>K
    lo=K/incumbent; hi=(A-K)/(CD-incumbent)
    grid=np.linspace(lo,hi,1201); vals=np.array([f(t) for t in grid])
    candidates=[(vals[0],grid[0]),(vals[-1],grid[-1])]
    for j in np.where((vals[1:-1]<=vals[:-2])&(vals[1:-1]<=vals[2:]))[0]+1:
        r=minimize_scalar(f,bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':1e-11})
        candidates.append((float(r.fun),float(r.x)))
    best,xbest=min(candidates)
    cache={}
    def theta(t):
        if t not in cache:cache[t]=dist.stationary_repair_fraction(t)
        return cache[t]
    def lb(l,u):
        n=K-A*theta(u)
        return CD+min(n/l,n/u)
    heap=[]; seq=itertools.count()
    coarse=np.linspace(lo,hi,101)
    for l,u in zip(coarse[:-1],coarse[1:]):heapq.heappush(heap,(lb(l,u),next(seq),l,u))
    splits=0
    while heap and best-heap[0][0]>tol:
        low,_,l,u=heapq.heappop(heap)
        mid=(l+u)/2; v=f(mid)
        if v<best:best,xbest=v,mid
        for a,b in [(l,mid),(mid,u)]:heapq.heappush(heap,(lb(a,b),next(seq),a,b))
        splits+=1
        if splits>250000:raise RuntimeError('Numerical global bound did not converge')
    return dict(**components(dist,xbest), search_lower=lo,search_upper=hi,
                numerical_lower_bound=heap[0][0],numerical_upper_bound=best,
                numerical_cost_gap=best-heap[0][0],interval_splits=splits)

def run_cost_experiments(
    cfg: ExperimentConfig,
    degradation: DegradationTimeDistribution,
    out: Path,
) -> dict:
    """Compute cost results; plotting is handled by the unified runner."""
    (out / "data").mkdir(parents=True, exist_ok=True)
    model = degradation.model
    results=[];base_dist=None
    for factor in (1.0, 0.5, 0.25):
        dt = cfg.survival_dt * factor
        dist = degradation if factor == 1.0 else DegradationTimeDistribution.build(
            model, dt, cfg.survival_individual_horizon
        )
        opt=optimize(dist)
        opt['survival_dt']=dt;results.append(opt)
        if factor == 1.0:base_dist=dist
        print(f"  Cost grid dt={dt:g}: tau*={opt['tau']:.9f}, cost={opt['total']:.12f}", flush=True)
    dist=base_dist;opt=results[0];tstar=opt['tau'];baseline=components(dist,cfg.baseline_tau)
    pd.DataFrame(results).to_csv(out/'data/grid_refinement.csv',index=False)
    tau=np.unique(np.r_[np.linspace(.1,25,2491),np.geomspace(25,1000,400),tstar,3.])
    curve=pd.DataFrame([components(dist,t) for t in tau]);curve.to_csv(out/'data/cost_curve.csv',index=False)
    selected=[1,2,3,5,7.5,tstar,12,20,50]
    table=pd.DataFrame([components(dist,t) for t in selected]);table.to_csv(out/'data/selected_costs.csv',index=False)

    # Independent renewal-reward validation from 200,000 fresh degradation lives.
    # Fresh degradation lifetimes use the configured distribution, sample count
    # and seed for a renewal-reward calculation.
    rng=np.random.default_rng(cfg.random_seed)
    sample=sum(model.sample_sojourn(i,cfg.microscopic_units,rng) for i in range(model.nondegraded_count))
    validation=[]
    # Independent numerical validation, retained in the CSV only.
    for t in sorted(set([*np.arange(1.0,26.0,1.0),tstar])):
        n=np.ceil(sample/t);L=t*n;R=CR+CD*(L-sample)
        ratio=R.mean()/L.mean(); estimate=K/t+ratio
        influence=(R-ratio*L)/L.mean();se=influence.std(ddof=1)/np.sqrt(sample.size)
        exact=components(dist,t)['total']
        validation.append(dict(tau=t,deterministic_cost=exact,mc_cost=estimate,mc_se=se,
                               ci_lower=estimate-1.96*se,ci_upper=estimate+1.96*se,
                               deterministic_in_95ci=bool(abs(exact-estimate)<=1.96*se)))
    mc=pd.DataFrame(validation);mc.to_csv(out/'data/renewal_monte_carlo.csv',index=False)
    scenarios=[]
    for cd in [.3,.4,1.]:
        for t in np.linspace(1.0,100.0,1981):
            row=components(dist,t,cd);row['loss_rate']=cd;scenarios.append(row)
    pd.DataFrame(scenarios).to_csv(out/'data/existence_boundary_curves.csv',index=False)
    summary=dict(model=cfg.to_dict(),cost_parameters=dict(K=K,c_R=CR,c_D=CD),
                 mean_degradation_time=model.mean_degradation_time,
                 existence_left=CD*model.mean_degradation_time,existence_right=K+CR,
                 optimal=opt,baseline=baseline,
                 saving_vs_baseline_pct=100*(baseline['total']-opt['total'])/baseline['total'],
                 saving_vs_no_maintenance_pct=100*(CD-opt['total'])/CD,
                 monte_carlo_n=sample.size,monte_carlo_mean_T=float(sample.mean()),
                 simulation_cycle_count=len(mc),simulation_cycle_lengths=mc.tau.tolist(),
                 refinement=results,primary_result=f'Deterministic approximation with survival_dt={cfg.survival_dt:g}',
                 policy_scope='Fixed-cycle cost optimization and stationary threshold/cycle correspondence.')
    (out/'cost_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary


# 8. Numerical refinement and independent reference checks

def run_reference_checks(config: ExperimentConfig, output: Path) -> None:
    model = SequentialWeibullModel.from_config(config)
    data = output / "data"
    refined = DegradationTimeDistribution.build(
        model, config.survival_dt / 4, config.survival_individual_horizon
    )
    reference_theta = refined.stationary_repair_fraction(config.baseline_tau)
    population_records = []
    solutions = []
    settings = [
        (config.population_dt, config.max_age, config.stationary_mass_tolerance),
        (config.population_dt / 2, config.max_age, config.stationary_mass_tolerance),
        (config.population_dt, config.max_age, config.stationary_mass_tolerance / 100),
        (config.population_dt, config.max_age + 15, config.stationary_mass_tolerance),
    ]
    for dt, age_limit, tolerance in settings:
        solver = AgeGridSolver(model, dt, age_limit)
        result = solver.stationary_renewal(
            solver.steps_for_time(config.baseline_tau), tolerance=tolerance
        )
        phases = []
        for phase in (0, config.baseline_tau / 2, config.baseline_tau):
            full = solver.propagate_full(
                solver.embed_nondegraded(result.distribution), solver.steps_for_time(phase)
            )
            phases.extend(solver.state_ratios(full))
        phases = np.asarray(phases)
        solutions.append(phases)
        population_records.append({
            "population_dt": dt, "max_age": age_limit, "series_tolerance": tolerance,
            "stationary_repair_fraction": result.repair_fraction,
            "absolute_error_vs_refined_survival_series": abs(result.repair_fraction-reference_theta),
            "retained_terms": result.cycle_count_terms,
            "first_omitted_survival_term": result.omitted_survival_mass,
            "omitted_series_tail_bound": result.omitted_series_tail_bound,
            "invariant_residual_tv": result.invariant_residual_tv,
            "overflow_mass": solver.overflow_mass(result.distribution),
            "max_phase_state_ratio_change_vs_baseline": float(np.max(abs(phases-solutions[0]))),
        })
    pd.DataFrame(population_records).to_csv(data / "population_grid_refinement.csv", index=False)

    # Independent calibration is against a survival convolution, rather than
    # the same return-map discretization used for the round trip.
    baseline = pd.read_csv(data / "threshold_to_cycle_curve.csv")
    threshold_records = []
    for dt in (config.population_dt, config.population_dt / 2):
        solver = AgeGridSolver(model, dt, config.max_age)
        for target in config.reported_thresholds:
            if dt == config.population_dt:
                row = baseline.iloc[np.argmin(abs(baseline["threshold"]-target))]
                cycle_time = float(row["threshold_policy_tau"])
                cycles = int(row["threshold_cycles_to_convergence"])
                residual = float(row["threshold_invariance_residual_tv"])
            else:
                _, cycle_time, cycles, residual = _stationary_threshold_policy(
                    solver, target, 4 * model.mean_degradation_time,
                    config.threshold_max_cycles, config.threshold_convergence_tolerance,
                )
            calibrated = refined.calibrate_cycle(target)
            recovered = refined.stationary_repair_fraction(cycle_time)
            threshold_records.append({
                "population_dt": dt, "reference_survival_dt": refined.dt,
                "threshold": target, "iterated_cycle": cycle_time,
                "independent_calibrated_cycle": calibrated,
                "independent_cycle_error": abs(cycle_time-calibrated),
                "independent_calibration_error": abs(recovered-target),
                "iterations": cycles, "successive_iterate_tv": residual,
            })
    pd.DataFrame(threshold_records).to_csv(data / "threshold_grid_refinement.csv", index=False)

    cost_records = []
    truncation_records = []
    for factor in (1, .5, .25):
        dist = DegradationTimeDistribution.build(
            model, config.survival_dt * factor, config.survival_individual_horizon
        )
        for tau in (1., config.baseline_tau, 12., 20.):
            cost_records.append(dict(components(dist, tau), survival_dt=dist.dt))
        for tau in (.1, config.baseline_tau, 100.):
            terms = dist.survival_series_terms(tau, config.survival_tolerance)
            truncation_records.append({
                "survival_dt": dist.dt, "tau": tau,
                "last_retained_index": len(terms)-1,
                "last_retained_time": tau*(len(terms)-1),
                "continuous_omitted_series_bound": dist.continuous_tail_bound(tau,len(terms)-1),
            })
    pd.DataFrame(cost_records).to_csv(data / "cost_point_refinement.csv", index=False)
    pd.DataFrame(truncation_records).to_csv(data / "survival_series_truncation.csv", index=False)

    parameter_tails = [float(model.survival(i, config.survival_individual_horizon))
                       for i in range(model.nondegraded_count)]
    metrics = pd.read_csv(data / "validation_error_metrics.csv")
    ratio_errors = metrics[metrics.window_or_phase == "all observations"]
    snapshot_errors = metrics[metrics.binned_tv_distance.notna()]
    summary = {
        "mixed_initial_distribution": [
            {"state":1,"age":0.,"weight":.4},
            {"state":2,"age":.6,"weight":.35},
            {"state":3,"age":1.,"weight":.25},
        ],
        "component_tail_probabilities_at_horizon": parameter_tails,
        "component_tail_probability_sum": sum(parameter_tails),
        "maximum_state_proportion_simulation_error": float(ratio_errors.maximum_absolute_error.max()),
        "snapshot_projected_tv": dict(zip(snapshot_errors.window_or_phase, snapshot_errors.binned_tv_distance)),
        "maximum_round_trip_error_same_grid": float(baseline.round_trip_absolute_error.max()),
        "maximum_independent_calibration_error": float(baseline.independent_calibration_error.max()),
        "maximum_independent_cycle_error": float(baseline.independent_cycle_error.max()),
        "threshold_max_iterations_observed": int(baseline.threshold_cycles_to_convergence.max()),
        "scope": "Truncation bounds exclude floating-point and discretization error; mesh checks are reported separately.",
    }
    (output / "verification_summary.json").write_text(json.dumps(summary,indent=2)+"\n")


# 9. Figure generation

COLORS = ("#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
LINESTYLES = ("-", ":", "--", "-.", (0, (5, 1, 1, 1)))


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


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("pdf", None), ("png", 320)):
        buffer = io.BytesIO()
        kwargs: dict[str, str | int] = {"format": extension}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(buffer, **kwargs)
        payload = buffer.getvalue()
        temporary = directory / f".{stem}.{extension}.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, directory / f"{stem}.{extension}")
    plt.close(fig)


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
    save_figure(fig, figure_dir, "fig01_state_proportions")


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
    save_figure(fig, figure_dir, "fig02_state_age_snapshots")


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
    save_figure(fig, figure_dir, "fig03_tv_convergence")


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
    axes[0].set_ylabel(r"Corresponding cycle $\tau_\theta$")
    axes[0].set_title("Threshold policy", loc="center", pad=7)

    axes[1].plot(
        fixed_to_threshold["tau"],
        fixed_to_threshold["fixed_policy_threshold"],
        color="#D55E00",
        linewidth=1.25,
    )
    axes[1].set_xlim(0.0, float(fixed_to_threshold["tau"].max()))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel(r"Fixed cycle $\tau$")
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
    save_figure(fig, figure_dir, "fig04_policy_correspondence")


def plot_maintenance_cost(
    scenarios: pd.DataFrame,
    summary: dict,
    figure_dir: Path,
) -> None:
    """Draw the combined cost figure without an embedded overall title."""
    opt = summary["optimal"]
    tstar = opt["tau"]
    # Apply the cost-figure style within a local plotting context.
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        plt.rcParams.update({'font.family':'serif','font.serif':['Nimbus Roman','Times New Roman'],
                             'mathtext.fontset':'stix','font.size':13,'axes.titlesize':15,
                             'axes.labelsize':14,'legend.fontsize':12,'xtick.labelsize':12,
                             'ytick.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
                             'pdf.fonttype':42,'savefig.dpi':320,
                             'savefig.bbox':'tight','savefig.pad_inches':.04})
        # One combined cost figure, based on the cost comparison.
        fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
        for cd,color,style in [(.3,'#0072B2','-.'),(.4,'#009E73','--'),(1.,'#111111','-')]:
            ss = scenarios[scenarios['loss_rate'] == cd]
            ax.plot(ss.tau,ss.total,c=color,ls=style,lw=1.6,label=rf'$c_D={cd:g}$')
            ax.axhline(cd,c=color,ls=':',lw=.85,alpha=.7)
        ax.scatter([tstar],[opt['total']],s=26,marker='o',c='#D55E00',zorder=5)
        ax.annotate(r'$\tau^*\approx '+f'{tstar:.3f}'+r'$',
                    xy=(tstar,opt['total']),xytext=(23,.67),
                    arrowprops=dict(arrowstyle='-',color='#D55E00'),color='#B54700',fontsize=12)
        ax.set(xscale='linear',xlim=(1,100),ylim=(.25,1.4),xlabel=r'Cycle length $\tau$',
               ylabel='Long-run average cost')
        ax.set_xticks([1,20,40,60,80,100])
        ax.legend(loc='upper right');ax.grid(alpha=.22)
        save_figure(fig, figure_dir, 'fig05_long_run_average_maintenance')


# 10. Data and table export

def prepare_output_directories(root: Path) -> tuple[Path, Path, Path]:
    data = root / "data"
    figures = root / "figures"
    tables = root / "tables"
    for directory in (root, data, figures, tables):
        directory.mkdir(parents=True, exist_ok=True)
    return data, figures, tables


def save_config(config: ExperimentConfig, root: Path, quick: bool) -> None:
    payload = config.to_dict()
    payload["quick_mode"] = quick
    with (root / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_runtime_versions(root: Path) -> None:
    content = (
        f"Python {platform.python_version()}\n"
        f"NumPy {np.__version__}\n"
        f"pandas {pd.__version__}\n"
        f"SciPy {scipy.__version__}\n"
        f"Matplotlib {matplotlib.__version__}\n"
    )
    (root / "runtime_versions.txt").write_text(content, encoding="utf-8")


def save_frames(data_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False, float_format="%.12g")


def _write_table(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_parameter_table(parameters: pd.DataFrame, path: Path) -> None:
    rows = []
    for row in parameters.itertuples(index=False):
        rows.append(
            f"{int(row.state)} & {row.classification} & {row.weibull_shape:.2f} & "
            f"{row.weibull_scale:.4f} & {row.sojourn_mean:.2f} \\\\"
        )
    content = r"""
\begin{tabular}{clrrr}
\hline
State & Classification & Shape $\beta_i$ & Scale $\eta_i$ & Mean $m_i$ \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
"""
    _write_table(path, content)


def write_selected_cost_table(costs: pd.DataFrame, optimal_tau: float, path: Path) -> None:
    """Export the five cycle lengths reported in the manuscript cost table."""
    rows = []
    for tau in (1.0, 3.0, optimal_tau, 12.0, 20.0):
        matched = costs[np.isclose(costs["tau"], tau, rtol=1e-8, atol=1e-10)]
        if len(matched) != 1:
            raise ValueError(f"Expected one saved cost row for tau={tau:g}")
        row = matched.iloc[0]
        rows.append(
            " & ".join(f"{row[key]:.4f}" for key in
                       ("tau", "theta", "setup", "repair", "degraded_loss", "total"))
            + r" \\"
        )
    content = r"""
\begin{tabular}{rrrrrr}
\hline
\shortstack{Cycle length\\$\tau$} & \shortstack{Repair fraction\\$\Theta(\tau)$} & Fixed cost & Repair cost & \shortstack{Degradation\\loss} & Total cost \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
"""
    _write_table(path, content)


# 11. Command-line interface and experiment runner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all selective-maintenance numerical experiments."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick",
        action="store_true",
        help="run a reduced smoke-test experiment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: outputs or outputs_quick)",
    )
    modes.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate figures and tables from saved results without running experiments",
    )
    return parser.parse_args()


def render_figures(output: Path) -> None:
    """Use the same plotting path for fresh and previously saved results."""
    data_dir = output / "data"
    figure_dir = output / "figures"
    required = (
        output / "configuration.json",
        output / "cost_summary.json",
        *(data_dir / f"{name}.csv" for name in (
            "fixed_cycle_state_trajectory",
            "state_age_snapshots",
            "different_initial_distributions",
            "threshold_to_cycle_curve",
            "fixed_cycle_to_threshold_curve",
            "existence_boundary_curves",
        )),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Saved results are incomplete. Run without --plot-only first. Missing: "
            + ", ".join(missing)
        )
    config = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "cost_summary.json").read_text(encoding="utf-8"))
    plot_state_proportions(
        pd.read_csv(data_dir / "fixed_cycle_state_trajectory.csv"),
        config["baseline_tau"],
        figure_dir,
    )
    plot_distribution_snapshots(pd.read_csv(data_dir / "state_age_snapshots.csv"), figure_dir)
    plot_convergence(pd.read_csv(data_dir / "different_initial_distributions.csv"), figure_dir)
    plot_threshold_comparison(
        pd.read_csv(data_dir / "threshold_to_cycle_curve.csv"),
        pd.read_csv(data_dir / "fixed_cycle_to_threshold_curve.csv"),
        figure_dir,
    )
    plot_maintenance_cost(
        pd.read_csv(data_dir / "existence_boundary_curves.csv"), summary, figure_dir
    )


def render_tables(output: Path) -> None:
    """Export the two manuscript tables from the saved numerical results."""
    data_dir = output / "data"
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_parameter_table(
        pd.read_csv(data_dir / "model_parameters.csv"),
        table_dir / "table01_parameters.tex",
    )
    summary = json.loads((output / "cost_summary.json").read_text(encoding="utf-8"))
    write_selected_cost_table(
        pd.read_csv(data_dir / "selected_costs.csv"),
        float(summary["optimal"]["tau"]),
        table_dir / "table02_selected_costs.tex",
    )


def main() -> int:
    args = parse_args()
    config = ExperimentConfig()
    if args.quick:
        config = config.quick()
    output = args.output or Path("outputs_quick" if args.quick else config.output_root)
    config = replace(config, output_root=str(output))
    config.validate()

    if args.plot_only:
        render_figures(output)
        render_tables(output)
        print(f"Regenerated figures and tables; figure directory: {(output / 'figures').resolve()}", flush=True)
        return 0

    # Overwrite generated result files without removing the output directory.
    data_dir, figure_dir, table_dir = prepare_output_directories(output)
    save_config(config, output, args.quick)
    save_runtime_versions(output)

    start = time.perf_counter()
    print("[1/5] Building the shared Weibull model and degradation-time distribution...", flush=True)
    model = SequentialWeibullModel.from_config(config)
    solver = AgeGridSolver(model, config.population_dt, config.max_age)
    degradation = DegradationTimeDistribution.build(
        model, config.survival_dt, config.survival_individual_horizon
    )

    print("[2/5] Running individual-unit and population experiments...", flush=True)
    fixed = run_fixed_cycle_experiments(config, model, solver, degradation)

    print("[3/5] Running threshold-to-fixed round-trip experiments and computing epsilon...", flush=True)
    threshold = run_threshold_experiments(config, solver, degradation)
    epsilon_summary = pd.DataFrame(
        [
            {
                "tau0": config.baseline_tau,
                "epsilon": degradation.uniform_epsilon(config.baseline_tau),
            }
        ]
    )

    frames = {
        "model_parameters": model.parameter_frame(),
        "fixed_cycle_state_trajectory": fixed.trajectory,
        "fixed_cycle_tv_convergence": fixed.cycle_convergence,
        "different_initial_distributions": fixed.deterministic_starts,
        "state_age_snapshots": fixed.snapshots,
        "snapshot_absorbing_atoms": fixed.snapshot_atoms,
        "stationary_phase_state_ratios": fixed.stationary_state_ratios,
        "validation_error_metrics": fixed.validation_errors,
        "fixed_cycle_summary": fixed.summary,
        "threshold_to_cycle_curve": threshold.threshold_to_cycle,
        "fixed_cycle_to_threshold_curve": threshold.fixed_to_threshold,
        "threshold_correspondence_summary": threshold.correspondence_summary,
        "epsilon_summary": epsilon_summary,
    }
    save_frames(data_dir, frames)

    print("[4/5] Running cost optimization, grid refinement and renewal validation...", flush=True)
    run_cost_experiments(config, degradation, output)
    print("  Checking independent calibration, truncation and population-grid refinement...", flush=True)
    run_reference_checks(config, output)

    print("[5/5] Rendering all five figures and writing LaTeX tables...", flush=True)
    render_figures(output)
    render_tables(output)

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.1f} seconds. Results: {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
