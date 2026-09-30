SELECTIVE MAINTENANCE: NUMERICAL EXPERIMENTS

Companion code for:
Stationary Regimes under Fixed-Cycle Selective Maintenance of Homogeneous Units


1. FILES
This package contains two files:

  selective_maintenance.py  All calculations, simulations, and figure/table export.
  README.txt                Installation, execution, methods, and output guide.

All experimental inputs are generated from the specified Weibull model. No
external dataset, separate configuration file, or additional local Python module
is required. Numerical results and figures are created when the script runs;
they are not included in this source-only package.


2. INSTALLATION AND MAIN COMMAND
Use Python 3.12. The reference environment is Python 3.12.14 with:

  NumPy       2.3.5
  pandas      2.2.3
  SciPy       1.17.0
  Matplotlib  3.10.8

Open a terminal in the folder containing the two files, then run:

  python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 matplotlib==3.10.8
  python selective_maintenance.py

The second command runs the full baseline experiment and creates outputs/ in
the terminal's current directory. It prints progress and total elapsed time.
The generated table fragments can optionally be included in a LaTeX document.


3. BASELINE MODEL AND SETTINGS

States: 1 -> 2 -> 3 -> 4 -> 5.
Non-degraded states: 1, 2, 3. Degraded states: 4, 5.
State 5 is absorbing between maintenance epochs.

Weibull shapes, states 1--4:       1.5, 2.0, 2.5, 1.5
Mean sojourn times, states 1--4:   2.0, 2.5, 3.0, 2.0
Scale in state i:                 mean_i / Gamma(1 + 1/shape_i)
Mean time to degradation:         7.5
Fixed maintenance cycle:          3
Individual units:                 200,000
Maintenance cycles:               28
NumPy random seed:                20260903
Population time and age step:     0.02
Maximum grid age:                 35
Age-bin width for comparison:     0.2
Sojourn convolution step:         0.002
Component convolution horizon:   50
Stationary population tail bound: 1e-13
Threshold targets:               0.05, 0.10, ..., 0.95
Threshold iteration tolerance:   1e-10
Maximum policy iterations:       1,200
Cost parameters K, c_R, c_D:      1, 2, 1
Additional degraded-loss rates:   0.3 and 0.4

Change ExperimentConfig near the start of the script to adjust model and
numerical settings. Cost constants K, CR, and CD appear in the cost section.

The published plotting layouts and table exporter target the five-state model;
arbitrary changes to the state structure require corresponding plotting edits.


4. FIGURES AND TABLES

Five figures are generated under outputs/figures/, in both PDF and PNG formats.

  Figure 1: State proportions
  Figure 2: State-age snapshots
  Figure 3: Convergence from different initial distributions
  Figure 4: Fixed-cycle/threshold-policy correspondence
  Figure 5: Long-run average maintenance cost

Two LaTeX table fragments are generated under outputs/tables/:

  table01_parameters.tex       Weibull parameters.
  table02_selected_costs.tex   Costs at cycles 1, 3, the optimizer, 12, and 20.

Their numeric sources are model_parameters.csv
and selected_costs.csv; the latter also contains additional cycle lengths.


5. GENERATED DATA AND RECORDS

Everything below is generated automatically.
Numeric blanks mean not applicable.
State numbers in exports start at 1; Python array indices start at 0.
Costs are per unit and per unit of operating time.

Main CSV files in outputs/data/:

  model_parameters.csv :     State classification, Weibull shape, scale, and mean.
  fixed_cycle_state_trajectory.csv :     Time, cycle, phase, computational source, state, and fraction; Figure 1.
  state_age_snapshots.csv :     Phase, source, state, age-bin limits, mass, and density; Figure 2.
  different_initial_distributions.csv :     Initial distribution, cycle, and distance to stationarity; Figure 3.
  threshold_to_cycle_curve.csv :     Threshold iteration, recovered fraction, residuals, iteration counts, and independent calibration for 19 thresholds; Figure 4.
  fixed_cycle_to_threshold_curve.csv :     Cycle length and stationary repair fraction; Figure 4.
  existence_boundary_curves.csv :     Cycle, repair fraction, cost components, total cost, and loss rate; Figure 5.
  selected_costs.csv :     Cycle, repair fraction, and cost components at selected cycles; Table 2.
  cost_curve.csv :     Baseline cost components over a wider cycle-length grid.

Other CSV files in outputs/data/:

  fixed_cycle_summary.csv             Baseline statistics and convergence.
  fixed_cycle_tv_convergence.csv      Simulation/population/stationary distances.
  stationary_phase_state_ratios.csv   Stationary state fractions at three phases.
  snapshot_absorbing_atoms.csv        State-5 mass at each phase and source.
  validation_error_metrics.csv        State-fraction errors and snapshot distances.
  epsilon_summary.csv                Uniform-reachability lower bound.
  threshold_correspondence_summary.csv  Five selected threshold targets.
  grid_refinement.csv                Cost optimizers and numerical search gaps.
  cost_point_refinement.csv           Fixed-cycle costs across convolution grids.
  population_grid_refinement.csv      Step, age-limit, and tolerance sensitivity.
  threshold_grid_refinement.csv       Calibration across population grids.
  survival_series_truncation.csv      Retained series indices and tail bounds.
  renewal_monte_carlo.csv             Renewal cost estimates and sampling errors.

Other records in outputs/:

  configuration.json         Settings used for the run; an output, not an input.
  cost_summary.json          Cost parameters, optimizer, refinement, and checks.
  verification_summary.json  Simulation and independent calibration summaries.
  runtime_versions.txt       Python and dependency versions used for the run.