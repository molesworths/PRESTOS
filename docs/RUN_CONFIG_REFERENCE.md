# PRESTOS Run Configuration Reference

This document describes the modules, classes, and options available for configuring PRESTOS simulations through `run_config.yaml`.

## Table of Contents

1. [Overview](#overview)
2. [Top-Level Configuration](#top-level-configuration)
3. [State Module](#state-module)
4. [Boundary Module](#boundary-module)
5. [Parameters Module](#parameters-module)
6. [Neutrals Module](#neutrals-module)
7. [Transport Module](#transport-module)
8. [Targets Module](#targets-module)
9. [Solver Module](#solver-module)
10. [Surrogate Module](#surrogate-module)
11. [Complete Example](#complete-example)

---

## Overview

The `run_config.yaml` file defines the complete workflow for a PRESTOS simulation, including:
- Initial plasma state
- Boundary condition models
- Profile parameterizations
- Neutral transport
- Turbulent transport models
- Target/heating models
- Numerical solver configuration
- Optional surrogate model acceleration

Each module follows the pattern:
```yaml
module_name:
  class: package.ClassName
  args:
    option1: value1
    option2: value2
```

---

## Top-Level Configuration

### `max_iterations`
**Type:** `int`  
**Description:** Maximum number of solver iterations (rarely reached if solver converges).  
**Example:** `max_iterations: 50`

### `work_dir`
**Type:** `string` (directory path)  
**Description:** Working directory where output files will be saved. The directory will be created if it doesn't exist. Supports `~` expansion and environment variables. If not specified, outputs are saved in the current directory.  
**Example:** `work_dir: ~/git/PRESTOS/example/`

---

## State Module

Defines the initial plasma state from which the simulation begins.

### Configuration Structure
```yaml
state:
  args:
    from_gacode: <path_to_gacode_file>
    boundary: <boundary_class_reference>
```

### Arguments

#### `from_gacode`
**Type:** `string` (file path)  
**Required:** Yes  
**Description:** Path to a GACODE profiles file containing initial plasma state. Supports `~` expansion and environment variables.  
**Example:** `from_gacode: ~/git/scratch/input.gacode`

#### `boundary`
**Type:** `string`  
**Description:** Reference to boundary condition class (used for initialization).  
**Example:** `boundary: boundary.TwoFluidTwoPoint_PeretSSF`

---

## Boundary Module

Handles boundary conditions at the last closed flux surface (LCFS/separatrix).

### Available Classes

#### `boundary.TwoFluidTwoPoint_PeretSSF`
**Description:** Combined two-fluid two-point model with Peret 2025 SSF (Scrape-off Layer Flux) decay length model. Iteratively solves for self-consistent LCFS conditions.

**Arguments:** None (uses default parameters internally)

**Physical Model:**
- Peret SSF model for flux decay lengths (λ_pe, λ_n, λ_T)
- Two-point model for parallel heat transport
- Iterative coupling with momentum loss factors

**Example:**
```yaml
boundary:
  class: boundary.TwoFluidTwoPoint_PeretSSF
  args: {}
```

#### `boundary.TwoPoint_EichManz`
**Description:** Classical two-point model with Eich-Manz SOL width scaling.

**Arguments:**
- `ne_sep` (float): Separatrix electron density [10^19 m^-3]
- `Tratio_sep` (float): Ti/Te ratio at separatrix
- `aLTratio_sep` (float): Ratio of ion to electron temperature gradient scale lengths

**Example:**
```yaml
boundary:
  class: boundary.TwoPoint_EichManz
  args:
    ne_sep: 1.0
    Tratio_sep: 1.0
```

---

## Parameters Module

Defines the parameterization of plasma profiles (density, temperature).

### Available Classes

#### `parameterizations.SplineParameterModel`
**Description:** Spline-based parameterization using control points at user-defined knots.

**Arguments:**

##### `knots`
**Type:** `list[float]`  
**Required:** Yes  
**Description:** Radial locations (ρ = r/a) where parameters control profile gradients.  
**Example:** `knots: [0.88, 0.91, 0.94, 0.97, 1.0]`

##### `spline_type`
**Type:** `string`  
**Options:** `akima`, `pchip`, `cubic`  
**Default:** `akima`  
**Description:** Type of spline interpolation between knots.  
**Example:** `spline_type: pchip`

##### `defined_on`
**Type:** `string`  
**Options:** `aLy` (gradient scale length), `y` (profile values)  
**Default:** `aLy`  
**Description:** Whether parameters represent gradient scale lengths (a/Ly) or direct profile values.  
**Example:** `defined_on: aLy`

##### `include_zero_grad_on_axis`
**Type:** `bool`  
**Default:** `True`  
**Description:** Enforce zero gradient at magnetic axis (ρ=0) for physical profiles.  
**Example:** `include_zero_grad_on_axis: True`

##### `sigma`
**Type:** `float`  
**Default:** `0.05`  
**Description:** Relative uncertainty on parameters (1σ) for uncertainty quantification.  
**Example:** `sigma: 0.05`

**Example:**
```yaml
parameters:
  class: parameterizations.SplineParameterModel
  args:
    knots: [0.88, 0.91, 0.94, 0.97, 1.0]
    spline_type: pchip
    defined_on: aLy
    include_zero_grad_on_axis: True
    sigma: 0.05
```

#### `parameterizations.RbfParameterModel`
**Description:** Two-Gaussian curvature parameterization for profile shapes.

**Arguments:**
- `A` (float): Amplitude parameter
- `roa_center` (float): Center location in normalized radius
- `sigma` (float): Width parameter
- `B` (float): Separation factor between Gaussians

**Example:**
```yaml
parameters:
  class: parameterizations.RbfParameterModel
  args:
    sigma: 0.05
```

---

## Neutrals Module

Models neutral particle transport in the plasma.

### Available Classes

#### `neutrals.DiffusiveNeutralModel`
**Description:** Simplified diffusive neutral transport with 1D diffusion equation.

**Arguments:**

##### `n0_edge`
**Type:** `list[float]`  
**Required:** Yes  
**Description:** Edge neutral density for each species [10^19 m^-3]. Single value applies to all species.  
**Example:** `n0_edge: [1e-4, 1e-7]`

**Example:**
```yaml
neutrals:
  class: neutrals.DiffusiveNeutralModel
  args:
    n0_edge: [1e-4, 1e-7]
```

#### `neutrals.KineticNeutralModel`
**Description:** Kinetic (ballistic) neutral transport with charge exchange.

**Arguments:**
- `n0_edge` (list[float]): Edge neutral density [10^19 m^-3]
- `include_cx_firstgen` (bool): Include first-generation charge exchange neutrals
- `max_cx_iter` (int): Maximum iterations for CX convergence

**Example:**
```yaml
neutrals:
  class: neutrals.KineticNeutralModel
  args:
    n0_edge: [1e-4, 1e-7]
    include_cx_firstgen: True
    max_cx_iter: 20
```

---

## Transport Module

Calculates turbulent and neoclassical transport fluxes.

### Available Classes

#### `transport.FingerprintsModel`
**Description:** Physics-based critical gradient transport model including ITG, ETG, KBM turbulence and neoclassical transport.

**Arguments:**

##### `modes`
**Type:** `string`  
**Options:** `all`, `turb`, `neo`, `ITG`, `ETG`, `KBM`  
**Default:** `all`  
**Description:** Which transport channels to include.  
**Example:** `modes: all`

##### `sigma`
**Type:** `float`  
**Default:** `0.05`  
**Description:** Relative uncertainty on transport model outputs (1σ).  
**Example:** `sigma: 0.05`

##### `ITG_lcorr`
**Type:** `float`  
**Default:** `0.1`  
**Description:** ITG correlation length [m] for non-local closure.  
**Example:** `ITG_lcorr: 0.1`

##### `ExBon`
**Type:** `bool`  
**Default:** `True`  
**Description:** Include E×B shear suppression of turbulence.  
**Example:** `ExBon: True`

##### `non_local`
**Type:** `bool`  
**Default:** `False`  
**Description:** Use non-local closure with correlation length effects.  
**Example:** `non_local: False`

**Example:**
```yaml
transport:
  class: transport.FingerprintsModel
  args:
    modes: all
    sigma: 0.05
    ExBon: True
    non_local: False
```

#### `transport.FixedTransport`
**Description:** Fixed diffusivity/conductivity for testing.

**Arguments:**
- `D` (float): Particle diffusivity [m^2/s]
- `chi` (float): Thermal diffusivity [m^2/s]

---

## Targets Module

Defines target values the solver attempts to match (typically from heating/radiation balance).

### Available Classes

#### `targets.AnalyticTargetModel`
**Description:** Analytical models for fusion heating, radiation, and energy exchange.

**Arguments:**

##### `scale_Pe_beam`
**Type:** `float`  
**Default:** `1.0`  
**Description:** Scaling factor for electron beam heating power.  
**Example:** `scale_Pe_beam: 1.0`

##### `scale_Pi_beam`
**Type:** `float`  
**Default:** `1.0`  
**Description:** Scaling factor for ion beam heating power.  
**Example:** `scale_Pi_beam: 1.0`

##### `sigma`
**Type:** `float`  
**Default:** `0.0`  
**Description:** Relative uncertainty on target values (1σ).  
**Example:** `sigma: 0.1`

**Physical Models:**
- Alpha particle heating (fusion power)
- Bremsstrahlung radiation
- Line radiation (via Aurora atomic data)
- Synchrotron radiation
- Classical electron-ion energy exchange

**Example:**
```yaml
targets:
  class: targets.AnalyticTargetModel
  args:
    scale_Pe_beam: 1.0
    scale_Pi_beam: 1.0
    sigma: 0.0
```

---

## Solver Module

Controls the numerical optimization algorithm.

### Available Classes

#### `solvers.RelaxSolver`
**Description:** Gradient-based relaxation solver with optional Jacobian assistance.

**Arguments:**

##### `predicted_profiles`
**Type:** `list[string]`  
**Required:** Yes  
**Description:** Profile names to be predicted/optimized (must match parameterization).  
**Example:** `predicted_profiles: [ne, te, ti]`

##### `target_vars`
**Type:** `list[string]`  
**Required:** Yes  
**Description:** Target variable names from transport/targets to match.  
**Example:** `target_vars: [Ce, Pe, Pi]`

##### `roa_eval`
**Type:** `list[float]`  
**Required:** Yes  
**Description:** Radial locations (ρ = r/a) where residuals are evaluated.  
**Example:** `roa_eval: [0.88, 0.91, 0.94, 0.97, 1.0]`

##### `domain`
**Type:** `list[float]` (length 2)  
**Required:** Yes  
**Description:** Radial domain [ρ_min, ρ_max] for solver operation.  
**Example:** `domain: [0.88, 1.0]`

##### `tol`
**Type:** `float`  
**Default:** `1e-3`  
**Description:** Convergence tolerance for objective function.  
**Example:** `tol: 1e-3`

##### `max_iter`
**Type:** `int`  
**Default:** `1000`  
**Description:** Maximum solver iterations.  
**Example:** `max_iter: 1000`

##### `use_surrogate`
**Type:** `bool`  
**Default:** `True`  
**Description:** Enable surrogate model acceleration.  
**Example:** `use_surrogate: True`

##### `use_jacobian`
**Type:** `bool`  
**Default:** `True`  
**Description:** Use Jacobian for gradient computation (faster convergence).  
**Example:** `use_jacobian: True`

##### `objective`
**Type:** `string`  
**Options:** `mse` (mean squared error), `sse` (sum of squares), `mae` (mean absolute error)  
**Default:** `mse`  
**Description:** Objective function type.  
**Example:** `objective: mse`

##### `normalize_residual`
**Type:** `bool`  
**Default:** `True`  
**Description:** Normalize residuals by target values (relative errors).  
**Example:** `normalize_residual: True`

##### `scale_objective`
**Type:** `bool`  
**Default:** `True`  
**Description:** Scale objective by number of channels.  
**Example:** `scale_objective: True`

##### `step_size`
**Type:** `float`  
**Default:** `1e-1`  
**Description:** Relaxation parameter α for gradient steps.  
**Example:** `step_size: 1e-1`

##### `bounds`
**Type:** Various formats  
**Description:** Parameter bounds. Multiple formats supported:

**Format 1: Uniform bounds for all parameters**
```yaml
bounds: [0, 100]  # [lower, upper] applied to all
```

**Format 2: Per-profile bounds**
```yaml
bounds:
  ne: [0, 100]
  te: [0, 50]
  ti: [0, 50]
```

**Format 3: Per-parameter bounds (advanced)**
```yaml
bounds: [[0, 10], [5, 20], [10, 30], ...]  # One pair per parameter
```

**Example:**
```yaml
solver:
  class: solvers.RelaxSolver
  args:
    predicted_profiles: [ne, te, ti]
    target_vars: [Ce, Pe, Pi]
    roa_eval: [0.88, 0.91, 0.94, 0.97, 1.0]
    domain: [0.88, 1.0]
    tol: 1e-3
    max_iter: 1000
    use_surrogate: True
    use_jacobian: True
    objective: mse
    normalize_residual: True
    scale_objective: True
    step_size: 1e-1
    bounds: [0, 100]
```

#### `solvers.TimeStepperSolver`
**Description:** Time-evolution solver using pseudo-time integration.

**Arguments:**
- `dt` (float): Time step size
- `method` (string): Integration method (`BDF`, `RK45`, etc.)

---

## Surrogate Module

Optional module for accelerating transport evaluations using machine learning surrogates.

### Available Classes

#### `surrogates.SurrogateManager`
**Description:** Manages training and evaluation of surrogate models for transport outputs.

**Arguments:**

##### `type`
**Type:** `string`  
**Options:** `gp` (Gaussian Process), `neural_network`, `polynomial`  
**Default:** `gp`  
**Description:** Type of surrogate model.  
**Example:** `type: gp`

##### `mode`
**Type:** `string`  
**Options:** `global`, `local`  
**Default:** `global`  
**Description:** Global model across all radii or separate local models per radius.  
**Example:** `mode: global`

##### `kwargs`
**Type:** `dict`  
**Description:** Surrogate-specific configuration (kernel parameters, etc.).

**Gaussian Process kwargs:**
- `length_scale` (float): RBF kernel length scale
- `variance` (float): Signal variance
- `noise` (float): Observation noise level
- `kernel` (string): Kernel type (`RBF`, `Matern52`, etc.)

**Example:**
```yaml
surrogate:
  class: surrogates.SurrogateManager
  args:
    type: gp
    mode: global
    kwargs:
      length_scale: 1.0
      variance: 1.0
      noise: 1e-4
```

---

## Complete Example

Here is a complete `run_config.yaml` demonstrating all modules:

```yaml
# Working directory for outputs
work_dir: ~/git/PRESTOS/example/

max_iterations: 50

# Initial plasma state
state:
  args:
    from_gacode: ~/git/scratch/input.gacode
    boundary: boundary.TwoFluidTwoPoint_PeretSSF

# Boundary conditions at LCFS
boundary:
  class: boundary.TwoFluidTwoPoint_PeretSSF
  args: {}

# Profile parameterization
parameters:
  class: parameterizations.SplineParameterModel
  args:
    knots: [0.88, 0.91, 0.94, 0.97, 1.0]
    spline_type: pchip
    defined_on: aLy
    include_zero_grad_on_axis: True
    sigma: 0.05

# Neutral transport
neutrals:
  class: neutrals.DiffusiveNeutralModel
  args:
    n0_edge: [1e-4, 1e-7]

# Turbulent transport
transport:
  class: transport.FingerprintsModel
  args:
    modes: all
    sigma: 0.05
    ExBon: True
    non_local: False

# Target heating/radiation model
targets:
  class: targets.AnalyticTargetModel
  args:
    scale_Pe_beam: 1.0
    scale_Pi_beam: 1.0
    sigma: 0.0

# Numerical solver
solver:
  class: solvers.RelaxSolver
  args:
    predicted_profiles: [ne, te, ti]
    target_vars: [Ce, Pe, Pi]
    roa_eval: [0.88, 0.91, 0.94, 0.97, 1.0]
    domain: [0.88, 1.0]
    tol: 1e-3
    max_iter: 1000
    use_surrogate: True
    use_jacobian: True
    objective: mse
    normalize_residual: True
    scale_objective: True
    step_size: 1e-1
    bounds: [0, 100]

# Optional surrogate acceleration
surrogate:
  class: surrogates.SurrogateManager
  args:
    type: gp
    mode: global
    kwargs:
      length_scale: 1.0
      variance: 1.0
      noise: 1e-4
```

---

## Notes and Best Practices

### Choosing Knot Locations
- Place more knots in regions of steep gradients (typically near the edge)
- Typical pedestal simulations use 5-10 knots between ρ=0.85-1.0
- Always include a knot at ρ=1.0 (separatrix)

### Convergence Tips
1. Start with `step_size: 1e-1` and reduce if oscillations occur
2. Enable `use_jacobian: True` for faster convergence (if stable)
3. Use `normalize_residual: True` to balance different variable scales
4. Set reasonable `bounds` to prevent unphysical parameter values

### Surrogate Model Usage
- Surrogate models accelerate repeated transport evaluations
- Requires ~10-20 initial training samples (warm-up iterations)
- Most beneficial for expensive transport models (not simple Fingerprints)
- Monitor surrogate accuracy via solver history output

### Uncertainty Quantification
- Set `sigma` values to represent model uncertainties
- Solver propagates uncertainties through calculations
- Check `solver_history.csv` for uncertainty estimates

---

## Advanced Configuration Topics

### Custom Evaluation Grids
Instead of uniform `roa_eval`, you can specify custom points:
```yaml
roa_eval: [0.85, 0.88, 0.90, 0.93, 0.96, 0.99, 1.0]
```

### Multi-Ion Species
Neutral and transport models automatically handle multiple ion species defined in the GACODE input file.

### Diagnostic Outputs
The solver automatically saves:
- `solver_history.csv`: Full iteration history with parameters, residuals, objectives
- Checkpoint files every N iterations (configurable via `iter_between_save`)

---

## Troubleshooting

### Solver Not Converging
1. Reduce `step_size`
2. Increase `max_iter`
3. Check bounds are reasonable
4. Verify initial state from GACODE file is physical
5. Try `use_jacobian: False` if getting NaN values

### Surrogate Training Failures
1. Increase `noise` parameter in surrogate kwargs
2. Ensure sufficient training samples (>10)
3. Try different kernel types
4. Check for NaN values in plasma state

### Boundary Condition Issues
1. Verify target powers are reasonable
2. Check if boundary model converges independently
3. Try simpler `TwoPoint_EichManz` if SSF model fails

---

## Version Compatibility

This reference is for PRESTOS codebase as of November 2025. Check repository for updates.

## References

For implementation details, see:
- `workflow.py`: Main execution logic
- Individual module files: `boundary.py`, `parameterizations.py`, `neutrals.py`, etc.
- `README.md`: Project overview and installation

For questions or contributions, see the PRESTOS GitHub repository.
