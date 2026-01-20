# CHANGELOG

All notable changes to PRESTOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Gaussian parameterization model**: New `Gaussian` class for direct gradient parameterization with analytically-integrated profiles
  - Models a/Ly as `b + A·G(x; c, w)` where G is a Gaussian centered at c with width w
  - Closed-form profile reconstruction via error function integrals
  - Automatically determines width w from boundary conditions
  - Solves bifurcation issues with `c` parameter near separatrix
- **GaussianDipole parameterization**: Curvature-based model with morphological transition parameter
  - Parameterizes `k(x) = d²y/dx²` directly using composite Gaussian structures
  - Morphology parameter S ∈ [0,1] smoothly transitions between single-lobe (S=0) and dipole (S=1) regimes
  - Fixed relationship `δ = 2.5w` ensures quasi-sinusoidal dipole profiles
  - Double integration with exact boundary condition enforcement
- **IvpSolver (TimeStepperSolver)**: Pseudo-time integration solver for parameter evolution
  - Replaces deprecated TimeStepperSolver with cleaner implementation
  - Integrates ODE system for parameter dynamics using scipy.integrate
- **Constraint support in solvers**: Flexible constraint compilation and enforcement
  - Expression parsing with alias normalization (Ti/Te, Pi/Pe, etc.)
  - Inequality constraints (>=, <=) with hinge loss
  - Ramp enforcement option for gradual constraint activation
  - Backend mapping to state attributes for runtime evaluation
- **Parameter uncertainty propagation**: Comprehensive uncertainty tracking through solver pipeline
  - Parameter standard deviations from `curve_fit` covariance matrices
  - Residual covariance via Jacobian linearization: `C_R = J·Cx·J^T + C_meas`
  - Objective uncertainty via delta method: `var(Z) ≈ ∇Z^T · C_R · ∇Z`
  - Posterior parameter covariance: `Cx_post = (J^T·C_R^{-1}·J)^{-1}`
- **Log-space parameter handling**: Proper uncertainty treatment for log-transformed parameters
  - Parameters like `log_A` stored in log-space with additive uncertainties
  - Automatic conversion between solver space and physical space
  - Fixes previous incorrect multiplicative uncertainty for log parameters
- **Analysis improvements**:
  - Interactive figure display: plt.show() keeps plots open until user closes them
  - Parameter history plotting for both Spline and Gaussian/GaussianDipole models
  - Bifurcation handling for Gaussian `c` parameter in uncertainty bands
  - Feature importance heatmaps for surrogate models
  - Surrogate sensitivity scatter plots with hierarchical importance ranking

### Changed
- **Refactored parameterizations.py**:
  - Renamed `SplineParameterModel` → `Spline`, `MTanhParameterModel` → `Mtanh`
  - Renamed `RbfParameterModel` → `GaussianDipole` with complete rewrite
  - All models now return `(params, params_std)` tuple from `parameterize()`
  - Unified boundary condition format: `{'val': float, 'loc': float}`
  - Added `sigma` attribute to base class for relative uncertainty (default 0.1)
- **Solver improvements**:
  - Increased default `model_iter_to_stall` from 3 to 10 for better convergence robustness
  - Best model restoration now occurs after convergence/stall check rather than during
  - Removed automatic bounds refresh with BC (caused numerical issues)
  - Jacobian caching and adaptive recomputation for efficiency
  - Proper bounds projection in finite-difference Jacobian computation
- **State processing**:
  - Fixed rotation decomposition in `_update_w0_by_decompose()` to match GACODE theory
  - Radial electric field: `Er = -dpi/dr/(Z·e·ni) - R·Bp·w0`
  - ExB shear: `γ_E = r·d/dr(v_E/r)` where `v_E = Er/B`
  - Parallel flow: `vpar = (BT/B)·R·w0`
  - Removed broken poloidal/toroidal decomposition
- **Transport model updates**:
  - Simplified ExB source handling: `ExB_source='state'` uses `gamma_exb_norm` directly
  - Removed separate 'state-pol' and 'state-both' options
- **Surrogate changes**:
  - Reduced default `max_train_samples` from 50 to 10 to prevent overfitting
  - Fixed parameter feature extraction for nested dict keys with underscores
- **Analysis plotting**:
  - Disabled automatic figure closing (commented `plt.close(fig)`) for interactive use
  - Added `plt.show()` call in main() to display all figures
  - Fixed power flow indexing bug (removed `[:-1]` slice on data series)
  - Corrected `base_ix` calculation for Gaussian parameter bifurcation handling

### Fixed
- **Boundary condition handling**: Removed broken `_apply_boundary_condition_bounds()` that caused parameter locking
- **Parameter flattening**: Changed from sorted() to list() to preserve insertion order for dict keys
- **Log-space uncertainties**: Fixed `X_new_std` calculation in RelaxSolver for `log_*` parameters
- **Bifurcation constraints**: Proper handling of Gaussian `c` parameter near x=1 in uncertainty bands
- **Surrogate feature splitting**: Fixed string split to handle parameter names with underscores (changed `split('_')` to `split('_', 1)`)

### Documentation
- Updated README with Gaussian and GaussianDipole parameterization options
- Added mathematical formulation documentation in docstrings for new models
- Updated example `run_config.yaml` to use Gaussian parameterization and IvpSolver
- Clarified log-space parameter handling in code comments

## [0.1.0] - 2025-01-12

### Initial Release
- Modular transport solver framework
- Multiple solver implementations (RelaxSolver, BayesianOptSolver, TimeStepperSolver)
- Surrogate acceleration with Gaussian processes
- Spline-based profile parameterization
- Fingerprints transport model
- Analytic target model with fusion heating and radiation
- GACODE file I/O
- Basic analysis and plotting tools
