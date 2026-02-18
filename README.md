# PRESTOS

**Parametric Rapid Extensible Surrogate Transport Optimization Solver**

A modular, parametric tokamak plasma transport solver with integrated surrogate modeling for rapid kinetic profile optimization via flux matching.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

---

## Status

Much of the PRESTOS functionality remains under development. Please report issues, suggest improvements, and ask to become a contributor. 

---

## Overview

PRESTOS is designed for efficient transport analysis and profile optimization in tokamak plasmas. It parameterizes kinetic profiles (density, temperature) and iteratively solves for equilibrium by matching predicted fluxes to target values. Key features include:

- **Modular architecture**: Swap boundary conditions, transport models, neutrals, parameterizations, and solvers via configuration
- **Surrogate acceleration**: Gaussian process surrogates reduce expensive transport evaluations during optimization
- **Flexible parameterization**: Spline, Gaussian, and curvature-based profile models with customizable knot placement and bounds
- **Multiple solvers**: Relaxation, Bayesian optimization, and pseudo-time integration (IvpSolver) methods
- **Constraint support**: Flexible inequality and equality constraints with automatic enforcement
- **Uncertainty quantification**: Full parameter and residual covariance tracking through the solver pipeline
- **Platform extensibility**: Execute modules (transport, targets) on local, remote, or HPC platforms
- **Evaluation caching**: Shared transport evaluation database for surrogate warm-start and cross-user knowledge sharing
- **Workflow checkpointing**: Save/restore solver state for seamless restarts with modified configurations
- **Built-in analysis**: Automated plotting, surrogate sensitivity, and convergence tracking

PRESTOS aims to provide rapid iteration capabilities compared to heavier frameworks like MITIM-fusion/PORTALS, with a focus on extensibility and ease of experimentation.

---

## Installation

### Requirements
- Python 3.11+
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
pip install numpy scipy pandas matplotlib seaborn scikit-learn scikit-optimize jax jaxlib pyqt6 paramiko
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
- Solver type (RelaxSolver, BayesianOptSolver, TimeStepperSolver)
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
  class: parameterizations.Spline  # Spline | Gaussian | GaussianDipole | Mtanh
  args:
    # Spline options:
    knots: [0.88, 0.91, 0.94, 0.97, 1.0]  # Radial locations (rho or roa)
    spline_type: pchip  # 'pchip' | 'akima' | 'cubic'
    defined_on: aLy  # 'aLy' (gradient scale length) | 'y' (raw profile)
    
    # Gaussian options:
    # (no knots needed - uses continuous Gaussian model)
    # Parameters: log_A (amplitude), b (baseline), c (center), w (width)
    
    # GaussianDipole options:
    # (curvature-based model with morphology parameter)
    # Parameters: log_A, x_c (center), w (width), S (symmetry 0-1)
    
    # Common options:
    include_zero_grad_on_axis: true  # Force zero gradient at axis
    sigma: 0.1  # Relative uncertainty on parameters (1-sigma)
    # lcfs_aLti_in_params: false  # Optional: include LCFS a/L_Ti as free parameter
```

This converts kinetic profiles into parameter vectors and reconstructs profiles during optimization. Each model returns `(params, params_std)` tuples for full uncertainty propagation.

**Available parameterization models:**
- `Spline`: Spline-based a/Ly interpolation at user-defined knots
- `Gaussian`: Direct gradient model using `a/Ly = b + A·G(x; c, w)` with analytical integration
- `GaussianDipole`: Curvature-based model with morphological transition parameter
- `Mtanh`: Modified-tanh pedestal model (stub)

**Gaussian model details:**
- Models gradient scale length directly: `a/Ly(x) = b + A·exp[-(x-c)²/(2w²)]`
- Width `w` automatically determined from boundary conditions
- Closed-form profile reconstruction via error function integrals
- Parameters stored in log-space (`log_A`) for proper uncertainty handling
- Solves bifurcation issues with `c` parameter near separatrix via bounded optimization

**GaussianDipole model details:**
- Models curvature: `k(x) = d²y/dx²` using composite Gaussian structures
- Morphology parameter `S` transitions between single-lobe (0) and dipole (1)
- Fixed separation `δ = 2.5w` ensures quasi-sinusoidal profiles
- Double integration with exact boundary condition enforcement
- Parameters: `log_A` (amplitude), `x_c` (center), `w` (width), `S` (morphology)

#### Transport Model

```yaml
transport:
  class: transport.FingerprintsModel
  args:
    modes: all  # 'all' | 'ITG' | 'ETG' | 'KBM' | 'neo'
    ExBon: true  # Include ExB shear suppression
    ExB_source: state  # 'state' (uses gamma_exb_norm from state) | 'model' (compute from gradients)
    ExB_scale: 1.0  # Scaling factor for ExB shear
    non_local: false  # Use non-local closure
    ITG_lcorr: 0.1  # ITG correlation length [m]
    sigma: 0.1  # Relative model uncertainty (1-sigma)
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
    scale_Qpar_wall: 1.0  # Scaling for parallel wall heat flux
    scale_Qpar_beam: 1.0  # Scaling for parallel beam heat flux
    sigma: 0.1  # Relative target uncertainty (1-sigma)
```

Available models:
- `AnalyticTargetModel`: Construct targets from auxiliary heating, fusion, radiation, etc.

#### Solver

```yaml
solver:
  class: solvers.RelaxSolver  # RelaxSolver | BayesianOptSolver | IvpSolver
  args:
    predicted_profiles: [ne, te, ti]  # Profiles to optimize
    target_vars: [Ce, Pe, Pi]  # Target quantities (fluxes/powers)
    roa_eval: [0.88, 0.91, 0.94, 0.97, 1.0]  # Radial evaluation grid (normalized minor radius)
    domain: [0.88, 1.0]  # Radial domain for solver
    tol: 1e-4  # Convergence tolerance on objective
    max_iter: 1000  # Maximum iterations
    model_iter_to_stall: 10  # Iterations without improvement before stall (increased from 3)
    objective: mse  # 'mse' | 'mae' | 'sse' | 'rmse'
    normalize_residual: true  # Normalize residual by |target|
    scale_objective: true  # Divide objective by number of channels
    step_size: 0.1  # Relaxation step size (RelaxSolver)
    adaptive_step: true  # Adaptive step sizing (RelaxSolver)
    use_jacobian: true  # Use Jacobian for gradient-based update (with Jacobian caching)
    bounds: [[0,10], [0.0,100.0], [0.9,1.2]]  # Parameter bounds (auto-computed from curve_fit if omitted)
    use_surrogate: false  # Use surrogate for evaluation speedup
    surrogate_warmup: 5  # Iterations before first surrogate training
    surrogate_retrain_every: 5  # Retrain surrogate every N iterations (reduced from 50 max_train_samples)
    
    # Optional constraints (flexible expression parsing):
    # constraints:
    #   - location: 0.88  # Radial location (r/a)
    #     expression: "Ti/Te = Pi/Pe"  # Constraint expression (with aliases)
    #     weight: 1.0  # Penalty weight
    #     norm: log  # Optional: 'log' normalization for ratios
    #     enforcement: ramp  # 'exact' | 'ramp' (gradual activation)
    #   - location: 0.95
    #     expression: "ne >= 3.0"  # Inequality constraint
    #     weight: 0.5
    #     enforcement: ramp
```

**Available solvers:**
- `RelaxSolver`: Simple relaxation with optional Jacobian-assisted gradient descent
  - Good for quick iterations and debugging
  - Supports adaptive step sizing for robustness
  - New: Jacobian caching and adaptive recomputation for efficiency
- `BayesianOptSolver`: Bayesian optimization with Monte Carlo acquisition functions
  - For expensive evaluations or when surrogate model is primary method
  - Supports expected improvement (EI) and upper confidence bound (UCB)
- `IvpSolver`: **NEW** - Pseudo-time integration using scipy ODE solvers
  - Integrates parameter dynamics `dp/dt = -∇Z(p)` using RK45 or other methods
  - Suitable for smooth convergence paths and systematic parameter evolution
  - Requires use_jacobian=true for efficiency

**Constraint support (NEW):**
- Expression parsing with built-in aliases (Ti/Te, Pi/Pe, ne, te, ti, etc.)
- Equality (`=`) and inequality (`>=`, `<=`) constraints
- Optional logarithmic normalization for ratio constraints
- Ramp enforcement for gradual constraint activation during iterations
- Hinge loss penalty applied to residual vector

**Bounds formats:**
- Uniform: `[lower, upper]` applied to all parameters
- Per-profile dict: `{ne: [0,100], te: [0,100], ti: [0,100]}`
- Per-parameter list: `[[lower, upper], [lower, upper], ...]` (length = n_params_per_profile)
- Auto-computed from `curve_fit` covariance if not specified

**Surrogate improvements:**
- Reduced default `max_train_samples` from 50 to 10 to prevent overfitting
- Fixed parameter feature extraction for nested dict keys with underscores
- Use `log_*` parameters for proper uncertainty propagation in log-space

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

## Advanced Features

### Workflow Restart & Checkpointing

Restart solver runs from a previous checkpoint, applying changes to the configuration without re-running earlier iterations:

```bash
python <PRESTOS_ROOT>/src/workflow.py run_config_v2.yaml --restart <previous_dir>
```

The `--restart` option:
- Loads the checkpoint from `<previous_dir>/solver_checkpoint.pkl`
- Restores solver state, transport model, and parameter history
- Applies new configuration from `run_config_v2.yaml`
- Continues iterating from the last state

Useful for:
- Parameter scans (modify `scale_*` factors in targets)
- Different solver settings (change `max_iter`, `tol`)
- Transport model adjustments (modify transport settings)

### Platform Infrastructure

Execute PRESTOS modules on local, remote, or HPC platforms without changing code:

```yaml
transport:
  class: transport.FingerprintsModel
  args:
    modes: all
  platform:  # Optional: run on specific platform
    machine: local  # 'local' | hostname
    scratch: ./work
    n_cpu: 16
    scheduler: slurm  # 'none' | 'slurm'
    slurm_partition: gpu  # for SLURM clusters
```

**Platform types:**
- **Local**: Direct execution on your machine
- **Remote SSH**: Automatic file transfer and execution via SSH
- **SLURM Cluster**: Automatic job submission and monitoring

**Example: SLURM cluster execution**

```yaml
transport:
  class: transport.FingerprintsModel
  platform:
    machine: hpc.example.com
    username: user
    scratch: /work/user/prestos
    modules: "module load python/3.9 && module load gcc/11.2.0"
    ssh_identity: ~/.ssh/cluster_key
    scheduler: slurm
    slurm_partition: gpu
```

See [PLATFORM_README.md](docs/PLATFORM_README.md) and [PLATFORM_SUBMISSION_GUIDE.md](docs/PLATFORM_SUBMISSION_GUIDE.md) for comprehensive documentation.

### Transport Evaluation Caching

Cache transport model evaluations in a shared database for:
- Surrogate warm-start (pre-training from previous runs)
- Cross-user knowledge sharing
- Avoiding duplicate expensive evaluations

**Enable in transport configuration:**

```yaml
transport:
  class: transport.FingerprintsModel
  args:
    modes: all
    log_evaluations: true  # Enable evaluation logging
    evaluation_log:
      enabled: true
      path: /shared/prestos/transport_evaluations.db  # Shared database
      # Alternatives:
      # path: ./logs/transport_eval.db  # Local database
      # (uses $PRESTOS_EVAL_LOG env var or default if omitted)
```

**Use cached evaluations for surrogate training:**

```python
from src.surrogates import SurrogateManager
from src.evaluation_log import TransportEvaluationLog

# Query previous evaluations
log = TransportEvaluationLog("/shared/prestos/transport_evaluations.db")
X, Y, roa = log.get_for_surrogate(
    model_class='transport.Fingerprints',
    model_settings={'modes': 'all'},
    target_roa=np.array([0.88, 0.91, 0.94, 0.97]),
    feature_names=['Ti_Te', 'aLTi', 'aLTe', 'aLne'],
    output_names=['Pe_turb', 'Pi_turb']
)

# Build surrogate from warm-start data
surrogate = SurrogateManager('gp')
surrogate.train(X, Y)  # Pre-trained on historical data
```

Database schema:
- `evaluations` table stores: timestamp, model class, settings, roa location, plasma features, model outputs
- Automatic deduplication via content hashing (SHA-256)
- Queryable by model class, settings, roa range, timestamp

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
