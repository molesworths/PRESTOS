# PRESTOS

**Parametric Rapid Extensible Surrogate Transport Optimization Solver**

A modular, parametric tokamak plasma transport solver with integrated surrogate modeling for rapid kinetic profile optimization via flux matching.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

---

## Status

Much of the PRESTOS functionality remains under development. Please report issues, suggest improvements, and ask to become a contributor. 

---

## Overview

PRESTOS is designed for efficient transport analysis and profile optimization in tokamak plasmas. It parameterizes kinetic profiles (density, temperature) and iteratively solves for equilibrium by matching predicted fluxes to target values. Key features include:

- **Modular architecture**: Swap boundary conditions, transport models, neutrals, parameterizations, and solvers via configuration
- **Surrogate acceleration**: Gaussian process surrogates reduce expensive transport evaluations during optimization
- **Flexible parameterization**: Spline-based profile representation with customizable knot placement and bounds
- **Multiple solvers**: Relaxation, finite-difference, Bayesian optimization, and time-stepping methods
- **Built-in analysis**: Automated plotting and reporting tools for convergence, profiles, and surrogate sensitivity

PRESTOS aims to provide rapid iteration capabilities compared to heavier frameworks like MITIM-fusion/PORTALS, with a focus on extensibility and ease of experimentation.

---

## Installation

### Requirements
- Python 3.8+
- [pixi](https://pixi.sh/) (recommended) or conda/mamba

### Using pixi (recommended)

```bash
# Clone the repository
git clone https://github.com/molesworths/PRESTOS.git
cd PRESTOS

# Install dependencies via pixi
pixi install

# Activate environment
pixi shell
```

### Using conda/pip

```bash
# Clone the repository
git clone https://github.com/molesworths/PRESTOS.git
cd PRESTOS

# Create conda environment
conda create -n prestos python=3.11
conda activate prestos

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn scikit-learn scikit-optimize jax jaxlib pyqt6
```

### Verify installation

```bash
python <PRESTOS_ROOT>/src/workflow.py --help
```

---

## Quick Start

### 1. Prepare input data

PRESTOS reads plasma state from GACODE format files. Ensure you have an `input.gacode` file with radial profiles of kinetic quantities (ne, te, ti, etc.).

Example location in `run_config.yaml`:
```yaml
state:
  args:
    from_gacode: ~/path/to/input.gacode
```

### 2. Configure your run

Edit `run_config.yaml` to specify:
- Transport model (Fingerprints, TGLF, Fixed)
- Solver type (RelaxSolver, FiniteDifferenceSolver, BayesianOptSolver, TimeStepperSolver)
- Parameterization scheme (SplineParameterModel)
- Target variables and evaluation domain
- Surrogate settings

See [Configuration Guide](#configuration-guide) below.

### 3. Run the solver

```bash
python <PRESTOS_ROOT>/src/workflow.py run_config.yaml
```

Or using pixi:
```bash
pixi run python <PRESTOS_ROOT>/src/workflow.py run_config.yaml
```

Output:
- `solver_history.csv`: iteration history with parameters, objectives, residuals, model predictions
- `solver_checkpoint.pkl`: serialized state for restarts (optional)

### 4. Generate analysis plots

```bash
python analysis.reporting --config analysis/plot_config.yaml
```

Outputs saved to `analysis_outputs/` by default. See [Analysis & Reporting](#analysis--reporting) for details.

---

## Running the Example

PRESTOS includes a working example in the `example/` directory with pre-configured files.

### 1. Navigate to example directory

```bash
cd example
```

### 2. Run the solver

From the example directory, call the workflow script from the source directory:

```bash
python <PRESTOS_ROOT>/src/workflow.py ./run_config.yaml
```

Where `<PRESTOS_ROOT>` is the path to your PRESTOS installation (e.g., `~/git/PRESTOS`).

Using pixi:
```bash
pixi run python <PRESTOS_ROOT>/src/workflow.py ./run_config.yaml
```

This will:
- Load the plasma state from `input.gacode`
- Run the solver according to `run_config.yaml`
- Generate `solver_history.csv` with iteration data

### 3. Run analysis

Generate visualization plots from the solver output:

```bash
python <PRESTOS_ROOT>/src/analysis.py -c ./analysis_config.yaml -w .
```

Arguments:
- `-c, --config`: Path to analysis configuration file
- `-w, --workdir`: Working directory containing solver output files (use `.` for current directory)

Outputs will be saved to `analysis_outputs/` (configurable in `analysis_config.yaml`).

---

## Configuration Guide

### `run_config.yaml` Structure

The workflow is controlled by a single YAML file specifying all modules and their options.

#### Plasma State

```yaml
state:
  args:
    from_gacode: ~/path/to/input.gacode  # Required: path to GACODE profiles file
```

The state is built from GACODE format and processed to compute derived quantities (e.g., gradient scale lengths, shear, normalizations).

#### Boundary Conditions

```yaml
boundary:
  class: boundary.TwoFluidTwoPoint_PeretSSF  # Module.ClassName
  args: {}  # Optional arguments for boundary model
```

Available models:
- `TwoFluidTwoPoint_PeretSSF`: Simple separatrix BC model

#### Neutrals

```yaml
neutrals:
  class: neutrals.DiffusiveNeutralModel
  args:
    n0_edge: [1e-4, 1e-7]  # Neutral density range at edge [min, max]
```

Available models:
- `DiffusiveNeutralModel`: Simplified neutral transport with diffusive approximation

#### Parameterization

```yaml
parameters:
  class: parameterizations.SplineParameterModel
  args:
    knots: [0.88, 0.91, 0.94, 0.97, 1.0]  # Radial locations (rho or roa)
    spline_type: pchip  # 'pchip' | 'cubic' | 'linear'
    defined_on: aLy  # 'aLy' (gradient scale length) | 'y' (raw profile)
    include_zero_grad_on_axis: true  # Force zero gradient at axis
    sigma: 0.05  # Relative uncertainty on parameters (1-sigma)
```

This converts kinetic profiles into parameter vectors (e.g., spline coefficients) and reconstructs profiles during optimization.

#### Transport Model

```yaml
transport:
  class: transport.FingerprintsModel
  args:
    modes: all  # 'all' | 'ITG' | 'ETG' | 'KBM' | 'neo'
    ExBon: true  # Include ExB shear suppression
    ExB_source: model  # 'model' | 'state-pol' | 'state-both'
    ExB_scale: 1.0  # Scaling factor for ExB shear
    non_local: false  # Use non-local closure
    ITG_lcorr: 0.1  # ITG correlation length [m]
    sigma: 0.05  # Relative model uncertainty (1-sigma)
```

Available models:
- `FingerprintsModel`: Simplified critical-gradient transport model (ITG/ETG/KBM + neoclassical)
- `TGLFModel`: Direct TGLF interface (not yet implemented)
- `FixedTransport`: Fixed diffusivities for testing

#### Targets

```yaml
targets:
  class: targets.AnalyticTargetModel
  args:
    scale_Pe_beam: 1.0  # Scaling for electron beam power
    scale_Pi_beam: 1.0  # Scaling for ion beam power
    sigma: 0.0  # Relative target uncertainty (1-sigma)
```

Available models:
- `AnalyticTargetModel`: Construct targets from auxililary heating, fusion, radiation, etc.

#### Solver

```yaml
solver:
  class: solvers.RelaxSolver
  args:
    predicted_profiles: [ne, te, ti]  # Profiles to optimize
    target_vars: [Ce, Pe, Pi]  # Target quantities (fluxes/powers)
    roa_eval: [0.88, 0.91, 0.94, 0.97, 1.0]  # Radial evaluation grid (normalized minor radius)
    domain: [0.88, 1.0]  # Radial domain for solver
    tol: 1e-3  # Convergence tolerance on objective
    max_iter: 1000  # Maximum iterations
    objective: mse  # 'mse' | 'mae' | 'sse'
    normalize_residual: true  # Normalize residual by |target|
    scale_objective: true  # Divide objective by number of channels
    step_size: 1e-1  # Relaxation step size (alpha)
    use_jacobian: true  # Use Jacobian for gradient-based update (RelaxSolver)
    bounds: [0, 100]  # Parameter bounds (uniform or per-profile dict)
```

**Bounds formats:**
- Uniform: `[lower, upper]` applied to all parameters
- Per-profile dict: `{ne: [0,100], te: [0,100], ti: [0,100]}`
- Per-parameter list: `[[lower, upper], [lower, upper], ...]` (length = n_params_per_profile)

Available solvers:
- `RelaxSolver`: Simple relaxation with optional Jacobian
- `FiniteDifferenceSolver`: Finite-difference gradient descent
- `BayesianOptSolver`: Bayesian optimization with acquisition function
- `TimeStepperSolver`: Pseudo-time integration of parameter evolution

#### Surrogate

```yaml
surrogate:
  class: surrogates.SurrogateManager
  args:
    type: gp  # 'gp' | 'neural_network' | 'polynomial'
    mode: global  # 'global' | 'local'
    kwargs:
      length_scale: 1.0
      variance: 1.0
      noise: 1e-4
```

Surrogates are trained on-the-fly during solver iterations and can replace expensive transport evaluations on scheduled iterations.

---

## Running PRESTOS

### Basic workflow execution

```bash
python <PRESTOS_ROOT>/src/workflow.py run_config.yaml
```

### Using alternative setup files

```bash
python <PRESTOS_ROOT>/src/workflow.py path/to/custom_run_config.yaml
```

### Command-line interface

```bash
python <PRESTOS_ROOT>/src/workflow.py --help
python <PRESTOS_ROOT>/src/workflow.py --setup my_run_config.yaml
```

### Expected outputs

After a successful run:
- `solver_history.csv`: Full iteration history
  - Columns: `iter`, `Z` (objective), `X` (parameters), `R` (residuals), `Y` (model predictions), `Y_target`, `used_surrogate`
- `solver_checkpoint.pkl`: Checkpoint for restart (optional, currently disabled)
- Console output: Convergence status and final objective value

---

## Analysis & Reporting

PRESTOS includes an automated analysis toolkit for visualizing solver performance and surrogate behavior.

### Configuration: `analysis_config.yaml`

```yaml
reporting:
  level: standard  # 'minimal' | 'standard' | 'full'
  
  data:
    solver_history: solver_history.csv
    run_config: run_config.yaml
  
  output:
    directory: analysis_outputs
  
  style:
    dpi: 120
    figsize:
      small: [6, 4]
      medium: [7, 5]
      wide: [10, 5]
  
  sensitivity:
    n_features: 10  # Number of top features to plot in sensitivity analysis
```

### Analysis levels

- **minimal**: Objective convergence, initial vs final profiles, final target comparison
- **standard**: minimal + residual channel evolution, PCA on parameter space, surrogate heatmap
- **full**: standard + detailed surrogate sensitivity scatter plots with hierarchical feature importance

### Generate plots

```bash
python <PRESTOS_ROOT>/src/analysis.py -c analysis_config.yaml -w <work_directory>
```

Arguments:
- `-c, --config`: Path to analysis configuration YAML file
- `-w, --workdir`: Directory containing `solver_history.csv` and `run_config.yaml`

Outputs saved to configured output directory (default: `analysis_outputs/`):

**Minimal level:**
- `objective.png`: Objective vs iteration (model vs surrogate markers)
- `profile_<name>.png`: Parameter comparison (initial vs final) for each profile
- `target_<name>.png`: Model vs target at final iteration

**Standard level (includes minimal plus):**
- `residual_channels.png`: Residual component evolution
- `surrogate_pca.png`: PCA component evolution and feature loadings
- `surrogate_heatmap.png`: Feature importance from GP length scales (inverse RBF length scale 1/l_rbf)
  - Rows: features, Columns: outputs, Color: importance magnitude
  - Global mode: single heatmap; Local mode: grid of heatmaps per evaluation point

**Full level (includes standard plus):**
- `surrogate_sensitivity.png`: Scatter plots of top features vs outputs, colored by evaluation index
  - Feature importance computed hierarchically:
    1. Permutation importance on rebuilt surrogate models
    2. Fallback: Mutual information regression
    3. Final fallback: Pearson correlation
  - Normalized per output variable

---

## Extending PRESTOS

PRESTOS is designed for easy extension. Add new models by subclassing base classes and registering them.

### Adding a new transport model

1. Create a new class in `transport.py` inheriting from `TransportBase`:

```python
class MyTransportModel(TransportBase):
    def __init__(self, options: dict):
        super().__init__(options)
        self.my_param = options.get('my_param', 1.0)
    
    def evaluate(self, state) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # Compute fluxes from state
        Qe = ...  # electron heat flux
        Qi = ...  # ion heat flux
        Ge = ...  # particle flux
        
        # Return predictions and uncertainties
        output_dict = {'Qe': Qe, 'Qi': Qi, 'Ge': Ge, ...}
        std_dict = {k: self.sigma * np.abs(v) for k, v in output_dict.items()}
        return output_dict, std_dict
```

2. Register in `TRANSPORT_MODELS` dict:

```python
TRANSPORT_MODELS = {
    'fingerprints': FingerprintsModel,
    'my_model': MyTransportModel,
}
```

3. Use in `run_config.yaml`:

```yaml
transport:
  class: transport.MyTransportModel
  args:
    my_param: 2.0
```

### Adding a new solver

1. Create a new class in `solvers.py` inheriting from `SolverBase`:

```python
class MySolver(SolverBase):
    def __init__(self, options=None):
        super().__init__(options)
        self.my_option = self.options.get('my_option', 'default')
    
    def propose_parameters(self, state, surrogate=None) -> Dict[str, Dict[str, float]]:
        # Implement parameter update logic
        X_new = self.X.copy()  # Start from current parameters
        # ... modify X_new based on residuals, Jacobian, etc.
        return self._project_bounds(X_new)
```

2. Register in `SOLVER_MODELS` dict:

```python
SOLVER_MODELS = {
    'relax': RelaxSolver,
    'my_solver': MySolver,
}
```

3. Use in `run_config.yaml`:

```yaml
solver:
  class: solvers.MySolver
  args:
    my_option: custom_value
```

### Adding options to existing modules

All module options are passed through the `args` dict in `run_config.yaml` and accessed via `self.options` in the class constructor.

Example: Add a new option to `FingerprintsModel`:

```python
# In transport.py
class FingerprintsModel(TransportBase):
    def __init__(self, options: dict):
        super().__init__(options)
        self.new_feature = options.get('new_feature', False)  # Default: False
```

Then use in config:

```yaml
transport:
  class: transport.FingerprintsModel
  args:
    new_feature: true
```

---

## Workflow Overview

PRESTOS follows a straightforward iteration loop:

```
1. Load plasma state from GACODE file
2. Initialize modules (boundary, neutrals, transport, targets, parameters, solver)
3. Compute initial parameter vector from state
4. Solver loop:
   a. Update state profiles from current parameters
   b. Evaluate transport model (or surrogate) → predicted fluxes
   c. Evaluate targets → target fluxes
   d. Compute residuals: R = Y_model - Y_target
   e. Compute objective: J = f(R)  (e.g., mean squared error)
   f. Check convergence: if J < tol, stop
   g. Propose new parameters (gradient-based or heuristic)
   h. Apply bounds and repeat
5. Save history and checkpoint
```

### Key abstractions

- **PlasmaState**: Container for all plasma quantities (kinetic profiles, geometry, derived quantities)
- **Parameters**: Converts profiles ↔ parameter vectors (e.g., spline coefficients)
- **Transport**: Computes fluxes from state
- **Targets**: Constructs target fluxes from auxiliary heating, fusion power, etc.
- **Solver**: Orchestrates the optimization loop
- **SurrogateManager**: Trains and evaluates surrogate models to accelerate transport calls

### Modularity

Each module is instantiated via a factory pattern from `run_config.yaml`. Classes are dynamically loaded by `workflow.build_module()`, enabling easy swapping of implementations without code changes.

---

## Troubleshooting

### Common issues

**ModuleNotFoundError: No module named 'pandas'**
- Ensure all dependencies are installed: `pixi install` or `pip install -r requirements.txt`

**FileNotFoundError: input.gacode**
- Check the path in `run_config.yaml` under `state.args.from_gacode`
- Use absolute paths or paths relative to execution directory

**Solver not converging**
- Reduce `step_size` in solver args
- Tighten or loosen parameter `bounds`
- Check that `target_vars` match available outputs from transport model
- Increase `max_iter`

**Surrogate accuracy issues**
- Increase surrogate warmup iterations: `surrogate_warmup: 10`
- Adjust GP hyperparameters: `length_scale`, `variance`, `noise`
- Disable surrogate temporarily: set `surrogate_retrain_every` very large or remove surrogate config

---

## Citation

If you use PRESTOS in your research, please cite:

```
@software{prestos2025,
  author = {Molesworth, Steve},
  title = {PRESTOS: Parametric Rapid Extensible Surrogate Transport Optimization Solver},
  year = {2025},
  url = {https://github.com/molesworths/PRESTOS}
}
```

---

## License

Not available at this time.

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

For major changes, please discuss in an issue first to coordinate development.

---

## Contact

- GitHub: [@molesworths](https://github.com/molesworths)
- Repository: [https://github.com/molesworths/PRESTOS](https://github.com/molesworths/PRESTOS)

---

## Acknowledgments

PRESTOS builds on concepts from MITIM-fusion/PORTALS and leverages physics models from the fusion transport community. Special thanks to contributors and early testers.
