# CHANGELOG

All notable changes to PRESTOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Platform infrastructure for distributed execution**:
  - `PlatformSpec`, `PlatformManager`, `CommandExecutor`, `FileManager`, `SLURMJobSubmitter` in `src/tools/io.py`
  - Run transport models locally, on remote SSH machines, or on SLURM HPC clusters
  - Automatic file staging and job submission with configurable scratch directories
  - Support for SSH tunneling (jump hosts) for firewall-behind clusters
  - Comprehensive platform documentation: `PLATFORM_README.md` and `PLATFORM_SUBMISSION_GUIDE.md`
  - Example configurations and working examples in `example/platform_*.py`
  
- **Transport model evaluation caching system** (`src/evaluation_log.py`):
  - SQLite database for storing transport evaluations across runs and users
  - Automatic deduplication via SHA-256 content hashing
  - Queryable by model class, settings, roa range, and timestamp
  - `get_for_surrogate()` method for warm-starting surrogates with historical data
  - Configurable via `run_config.yaml`: `transport.args.evaluation_log`
  - Cross-user knowledge sharing via shared network database paths
  - Useful for parameter scans and surrogate acceleration
  
- **Workflow restart and checkpointing**:
  - `--restart <checkpoint_dir>` option in `workflow.py` for resuming from previous checkpoint
  - Save/restore solver state, transport model, and parameter history
  - Apply modified configurations without re-running earlier iterations
  - Enables efficient parameter scans and iterative refinement
  - New functions: `load_checkpoint()`, `restore_module_from_checkpoint()` in workflow
  
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
  
- **IvpSolver (pseudo-time integration)**: New solver for parameter evolution
  - Integrates ODE system for parameter dynamics using scipy.integrate
  - Smoother convergence paths compared to relaxation methods
  - Replaces deprecated `TimeStepperSolver` with cleaner implementation
  - Supports Jacobian caching and adaptive step selection
  
- **Constraint support in solvers**: Flexible constraint compilation and enforcement
  - Expression parsing with alias normalization (Ti/Te, Pi/Pe, etc.)
  - Inequality constraints (>=, <=) with hinge loss
  - Ramp enforcement option for gradual constraint activation
  - Backend mapping to state attributes for runtime evaluation
  - Constraints specified as list of dict in solver config
  
- **Parameter uncertainty propagation**: Comprehensive uncertainty tracking through solver pipeline
  - Parameter standard deviations from `curve_fit` covariance matrices
  - Residual covariance via Jacobian linearization: `C_R = J·Cx·J^T + C_meas`
  - Objective uncertainty via delta method: `var(Z) ≈ ∇Z^T · C_R · ∇Z`
  - Posterior parameter covariance: `Cx_post = (J^T·C_R^{-1}·J)^{-1}`
  - Full tracking through SolverData container
  
- **Log-space parameter handling**: Proper uncertainty treatment for log-transformed parameters
  - Parameters like `log_A` stored in log-space with additive uncertainties
  - Automatic conversion between solver space and physical space
  - Fixes previous incorrect multiplicative uncertainty for log parameters
  
- **GACODE control file templates** in `src/tools/gacode_controls/`:
  - `input.cgyro.controls`: Template for CGYRO simulations (nonlinear gyrokinetic)
  - `input.neo.controls`: Template for NEO simulations (neoclassical transport)
  - `input.qlgyro.controls`: Template for QLGYRO simulations (quasi-linear)
  - `input.tglf.controls`: Template for TGLF simulations (turbulence transport)
  - Reusable in custom transport models via tools/io.py

- **Analysis improvements**:
  - Interactive figure display: plt.show() keeps plots open until user closes them
  - Parameter history plotting for both Spline and Gaussian/GaussianDipole models
  - Bifurcation handling for Gaussian `c` parameter in uncertainty bands
  - Feature importance heatmaps for surrogate models
  - Surrogate sensitivity scatter plots with hierarchical importance ranking
  - Fixed power flow indexing bug (removed `[:-1]` slice on data series)
  - Corrected `base_ix` calculation for Gaussian parameter bifurcation handling

### Changed
- **Refactored parameterizations.py**:
  - Renamed `SplineParameterModel` → `Spline`, `MTanhParameterModel` → `Mtanh`
  - Renamed `RbfParameterModel` → `GaussianDipole` with complete rewrite
  - All models now return `(params, params_std)` tuple from `parameterize()`
  - Unified boundary condition format: `{'val': float, 'loc': float}`
  - Added `sigma` attribute to base class for relative uncertainty (default 0.1)
  - Parameter uncertainty automatically propagated through solver pipeline
  
- **Solver improvements**:
  - Increased default `model_iter_to_stall` from 3 to 10 for better convergence robustness
  - Best model restoration now occurs after convergence/stall check rather than during
  - Removed automatic bounds refresh with BC (caused numerical issues)
  - Jacobian caching and adaptive recomputation for efficiency
  - Proper bounds projection in finite-difference Jacobian computation
  - Fixed `curve_fit` bounds handling for proper uncertainty estimation
  
- **State processing**:
  - Fixed rotation decomposition in `_update_w0_by_decompose()` to match GACODE theory
  - Radial electric field: `Er = -dpi/dr/(Z·e·ni) - R·Bp·w0`
  - ExB shear: `γ_E = r·d/dr(v_E/r)` where `v_E = Er/B`
  - Parallel flow: `vpar = (BT/B)·R·w0`
  - Removed broken poloidal/toroidal decomposition
  
- **Transport model updates**:
  - Simplified ExB source handling: `ExB_source='state'` uses `gamma_exb_norm` directly
  - Removed separate 'state-pol' and 'state-both' options
  - Added platform support via TransportBase.run_on_platform()
  
- **Surrogate changes**:
  - Reduced default `max_train_samples` from 50 to 10 to prevent overfitting
  - Fixed parameter feature extraction for nested dict keys with underscores
  - Support for warm-starting from evaluation_log database
  
- **Analysis plotting**:
  - Disabled automatic figure closing (commented `plt.close(fig)`) for interactive use
  - Added `plt.show()` call in main() to display all figures
  - Fixed power flow indexing bug (removed `[:-1]` slice on data series)
  - Corrected `base_ix` calculation for Gaussian parameter bifurcation handling
  
- **Dependencies**:
  - Added `paramiko` for SSH/SFTP support (required for platform infrastructure)
  - Updated numpy requirement to `>=1.24,<2.0`

### Fixed
- **Boundary condition handling**: Removed broken `_apply_boundary_condition_bounds()` that caused parameter locking
- **Parameter flattening**: Changed from sorted() to list() to preserve insertion order for dict keys
- **Log-space uncertainties**: Fixed `X_new_std` calculation in RelaxSolver for `log_*` parameters
- **Bifurcation constraints**: Proper handling of Gaussian `c` parameter near x=1 in uncertainty bands
- **Evaluation logging deduplication**: Robust hash computation avoiding floating-point issues
- **SFTP file transfer**: Proper error handling for remote path creation and cleanup
- **Module import paths**: Consistent handling of nested module references in platform infrastructure
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
