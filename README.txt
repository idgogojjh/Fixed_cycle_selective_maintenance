SELECTIVE MAINTENANCE: NUMERICAL EXPERIMENTS

Companion code for:
Stationary Regimes for Periodic Selective Maintenance of
Large-scale Systems with Homogeneous Units


1. FILES AND EXECUTION

This source-only package contains two files:

  selective_maintenance.py   Model, experiments, figures, and table export.
  README.txt                 Installation, methods, and output guide.

No external dataset or additional local Python module is required.
Use Python 3.12. The tested environment is Python 3.12.14 with NumPy 2.3.5,
pandas 2.2.3, SciPy 1.17.0, and Matplotlib 3.10.8.

It creates outputs/ in the terminal's current directory
and prints the main results and elapsed time.


2. BASELINE MODEL

States: 1 -> 2 -> 3 -> 4 -> 5.
Repair set: {4, 5}.
State 5 is the absorbing state.

  Weibull shapes, states 1--4:          1.5, 2.0, 2.5, 1.5
  Mean sojourn times, states 1--4:      2.0, 2.5, 3.0, 2.0
  Scale in state i:                    mean_i / Gamma(1 + 1/shape_i)
  Fixed maintenance cycle length:      3
  Number of maintenance cycles:        28
  Number of individual units:          200,000
  Threshold targets:                   0.05, 0.10, ..., 0.95
  Cost parameters K, c_R, c_D:          1, 2, 1
  Additional loss rates c_D:            0.3 and 0.4


3. FIGURES AND TABLES

Five figures are saved in outputs/figures/, each in PDF and PNG formats.

  Figure 1: Comparison of state proportions from the individual-unit simulation
            and the population model.
  Figure 2: Comparison of state-age distributions from the individual-unit
            simulation and the population model.
  Figure 3: Convergence in the projected distance d_h from different initial
            distributions.
  Figure 4: Correspondence between threshold-based and periodic maintenance
            policies.
  Figure 5: Long-run average cost versus maintenance cycle length under the
            baseline Weibull degradation model.


Two LaTeX tabular fragments are saved in outputs/tables/:

  table01_parameters.tex
    Weibull sojourn-time parameters for the baseline model.

  table02_selected_costs.tex
    Stationary repair fractions and costs per unit of operating time at
    selected cycle lengths: 1, 3, the numerical optimizer, 12, and 20.


4. NUMERIC SOURCES AND MAIN RESULTS

Eight CSV files in outputs/data/:

  figure1_state_proportions.csv       Figure 1; all 28 cycles are retained.
  figure2_state_age.csv               Figure 2; bin masses and densities.
  figure3_convergence.csv             Figure 3; distances for three starts.
  figure4_threshold_to_cycle.csv      Figure 4; thresholds, cycle lengths, recovered values, independent calibration.
  figure4_cycle_to_threshold.csv      Figure 4; periodic-policy mapping.
  figure5_cost.csv                    Figure 5; the three loss-rate curves.
  table1_parameters.csv               Table 1; Weibull parameters.
  table2_selected_costs.csv           Table 2; exactly five cycle lengths.
