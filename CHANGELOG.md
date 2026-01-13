# CHANGELOG

All notable changes to PRESTOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Batch processing support**: TransportBase and TargetModel now support batch evaluation of multiple PlasmaState objects
  - Child classes (FingerprintsModel, AnalyticTargetModel, etc.) unchanged - batching handled in base class
  - Enables efficient parallel evaluation for future batch optimization methods
- **Best model restoration on stall**: Solver now tracks and restores the best model evaluation when stalling is detected
  - Prevents exiting at a worse solution than earlier iterations
  - Particularly useful when gradient-based methods cause oscillations
- **Comprehensive README**: Added detailed configuration guide, examples, troubleshooting, and extension documentation

### Changed
- Code cleanup in solvers.py: Removed unused TODO context managers, simplified docstrings
- Code cleanup in surrogates.py: Removed pragma comments, improved clarity
- Renamed child class `evaluate()` methods to `_evaluate_single()` for batch processing pattern
- Improved docstrings throughout transport and targets modules

### Fixed
- Corrected batch processing detection logic for numpy arrays and tuples

### Documentation
- Added batch processing implementation summary (BATCH_PROCESSING_CHANGES.md)
- Updated README with comprehensive installation, configuration, and usage guides
- Added troubleshooting section
- Added examples of extending PRESTOS with custom models

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
