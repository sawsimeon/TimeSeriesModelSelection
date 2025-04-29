# Time Series Model Selection with REP Metric and Penalty Reduction

## Overview
This repository contains code and documentation for a project analyzing the **Representativeness (REP)** metric and its implicit regularization properties in the context of Exponential Smoothing (ETS) model selection for time series forecasting. The project focuses on datasets from the M1, M3, and M4 competitions, implementing rolling origin cross-validation, optimizing ETS model parameters, and reducing the penalty function in the REP metric by adjusting the `delta` parameter and introducing explicit regularization terms.

Key objectives:
- Explore the theoretical foundation of the REP metric, which measures how well ETS models’ fitted values and forecasts align with the standardized, Box-Cox-transformed time series distribution.
- Investigate implicit regularization in REP via exponential weighting of recent data segments (`delta = 0.5`) and propose penalty reduction strategies (e.g., lowering `delta` to 0.3, adding L2 and complexity penalties).
- Optimize ETS smoothing parameters (`alpha`, `beta`, `gamma`, `phi`) using cross-validation with a penalized objective.
- Evaluate performance on yearly time series data (M1, M3, M4) using high-CPU cloud resources.


## Datasets
- **M1, M3, M4 Yearly Data**: Time series from the M1, M3, and M4 forecasting competitions, accessed via R packages `Mcomp` and `M4comp2018`. Series lengths vary (~20–100 years).
- **Toy Dataset**: A synthetic dataset in  (100 daily samples, Jan 1–Apr 10, 2023) for regression-based forecasting.

## Key Components
### REP Metric
- **Definition**: Measures the discrepancy between standardized ETS model outputs (fitted values and forecasts) and the Box-Cox-transformed time series, with exponential weighting (`(1-delta)^(k-1)`) prioritizing recent segments.
- **Role**: Complements forecast accuracy (CVMAEs) and information criteria for model selection, ensuring distributional alignment.
- **Implicit Regularization**: The weighting scheme (`delta = 0.5`) penalizes models misaligned with recent data, acting as a constraint.

### Penalty Reduction
- **Lower Delta**: Reduced `delta` from 0.5 to 0.3, increasing weights for older segments (e.g., 1, 0.7, 0.49 vs. 1, 0.5, 0.25) to lessen the penalty on models fitting older data.
- **Explicit Penalties**: Added L2 penalty (0.001 * sum(parameters^2)) on ETS parameters and complexity penalty (0.01 * number of parameters) in REP, with small weights to minimize impact.
- **Purpose**: Balances distributional fit and model simplicity, improving generalization.

### ETS Parameter Optimization
- Optimizes smoothing parameters (`alpha`, `beta`, `gamma`, `phi`) via grid search, minimizing CVMAEs with an L2 penalty (0.01 * sum(parameters^2)).
- Implemented in `supplementary_code_modified.R` and `supplementary_code_modified_v2.R`.

### Rolling Origin Cross-Validation
- Evaluates ETS models using expanding training windows (e.g., 8–14 years for a 20-year series) and fixed test horizons (6 years).
- Visualized in `tscv_diagram.png`.

## Requirements
- **R**: Packages `Mcomp`, `M4comp2018`, `forecast`, `foreach`, `doSNOW`.
- **Python**: Packages `numpy`, `pandas`, `sklearn`, `matplotlib`.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sawsimeon/TimeSeriesModelSelection.git
   cd TimeSeriesModelSelection
   ```
