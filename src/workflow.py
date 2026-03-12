import importlib
import os
import sys
from typing import Any, Dict, Optional
import argparse
import yaml
import pickle
from pathlib import Path

# Local imports for state construction
from interfaces import gacode
from tools.platforms import resolve_platform_config
from state import PlasmaState

def load_class(path: str, module_prefix: Optional[str] = None):
    """Dynamically import class given 'Module.ClassName' string or just 'ClassName' with module_prefix.
    
    Supports case-insensitive matching and intelligent suffix handling:
    - "anderson" → finds "AndersonMixing" in solvers module
    - "ivp" → finds "IvpSolver" in solvers module
    - "Tglf" → finds "Tglf" in transport module (exact match)
    
    Parameters
    ----------
    path : str
        Either full path like 'boundary.FixedInitial' or just class name like 'anderson'
    module_prefix : str, optional
        Module name to use if path doesn't contain a dot (e.g., 'boundary', 'solvers')
        Typically comes from the config subsection header.
    """
    # Legacy full-path mappings
    legacy_map = {
        "surrogates.GaussianProcessModel": "surrogates.GaussianProcess",
        "surrogates.GaussianProcessSurrogate": "surrogates.GaussianProcess",
    }
    if path in legacy_map:
        path = legacy_map[path]
    
    # Parse module and class names
    if '.' in path:
        module_name, class_name = path.rsplit('.', 1)
    elif module_prefix:
        module_name = module_prefix
        class_name = path
    else:
        raise ValueError(f"Cannot load class '{path}': no module prefix provided and path contains no dot")
    
    # Import the module
    module = importlib.import_module(module_name)
    
    # Strategy 1: Try exact match (case-sensitive)
    if hasattr(module, class_name):
        obj = getattr(module, class_name)
        # Ensure it's actually a class, not a module or function
        if isinstance(obj, type):
            return obj
    
    # Strategy 2: Try case-insensitive exact match
    # Get only actual classes (exclude modules, functions, etc.)
    available_classes = {name: obj for name, obj in vars(module).items() 
                        if isinstance(obj, type) and not name.startswith('_')}
    
    class_name_lower = class_name.lower()
    for name, obj in available_classes.items():
        if name.lower() == class_name_lower:
            return obj
    
    # Strategy 3: Try with common suffixes for each module type
    suffix_patterns = {
        'solvers': ['Solver', 'Mixing', ''],
        'transport': ['', 'Model'],
        'boundary': ['', 'Conditions'],
        'targets': ['', 'Model'],
        'neutrals': ['', 'Model'],
        'parameterizations': ['', 'Base'],
        'surrogates': ['', 'Surrogate'],
    }
    
    # Get module base name (e.g., 'solvers' from 'solvers' or from 'src.solvers')
    module_base = module_name.split('.')[-1]
    patterns = suffix_patterns.get(module_base, [''])
    
    # Try each suffix pattern
    for suffix in patterns:
        candidate = class_name + suffix
        candidate_lower = candidate.lower()
        
        for name, obj in available_classes.items():
            if name.lower() == candidate_lower:
                return obj
    
    # Strategy 4: Try partial match if class_name is a substring (case-insensitive)
    # This handles cases like "anderson" matching "AndersonMixing"
    for name, obj in available_classes.items():
        if class_name_lower in name.lower() and not name.endswith('Base') and not name.endswith('Mixin'):
            # Prefer matches that start with the search term
            if name.lower().startswith(class_name_lower):
                return obj
    
    # If no exact or prefix match, try any substring match
    for name, obj in available_classes.items():
        if class_name_lower in name.lower() and not name.endswith('Base') and not name.endswith('Mixin'):
            return obj
    
    # If all strategies fail, provide helpful error message
    available_names = [name for name in available_classes.keys() if not name.endswith('Base') and not name.endswith('Mixin')]
    raise ValueError(
        f"Cannot find class '{class_name}' in module '{module_name}'. "
        f"Available classes: {', '.join(sorted(available_names))}"
    )

def build_module(config_entry: Dict[str, Any], module_prefix: Optional[str] = None):
    """Instantiate a module given its dict from setup.

    Expects: {"class": "package.Module.Class", "args": {...}}
    or {"class": "ClassName", "args": {...}} with module_prefix provided.
    
    Parameters
    ----------
    config_entry : Dict[str, Any]
        Configuration dictionary with 'class' and optional 'args'
    module_prefix : str, optional
        Module name from config subsection (e.g., 'boundary', 'solvers')
        Used when class path doesn't include module name.
    """
    if not config_entry:
        return None
    class_path = config_entry.get("class")
    if not class_path:
        return None
    cls = load_class(class_path, module_prefix=module_prefix)
    args = config_entry.get("args", {})
    return cls(args)


def _inject_verbose(config_entry: Optional[Dict[str, Any]], verbose: bool) -> Optional[Dict[str, Any]]:
    """Inject verbose flag into module config args."""
    if not config_entry:
        return config_entry
    if not isinstance(config_entry, dict):
        return config_entry
    
    # Make a shallow copy to avoid modifying the original
    cfg = dict(config_entry)
    if "args" not in cfg:
        cfg["args"] = {}
    cfg["args"]["verbose"] = verbose
    return cfg



def build_state(config_entry: Dict[str, Any]):
    """Build and process a PlasmaState from config.

    Supports:
    state:
      args:
        from_gacode: ~/path/to/input.gacode
    """
    args = (config_entry or {}).get("args", {})
    gacode_path = args.get("from_gacode")

    # If a GACODE file path is provided, load it and construct PlasmaState
    if gacode_path:
        file_path = os.path.expanduser(os.path.expandvars(str(gacode_path)))
        gc = gacode(filepath=file_path)
        st = PlasmaState.from_gacode(gc)
        # Populate derived quantities with access to original profiles
        st.process(gc)
        return st

    # Fallback: try to instantiate a class if provided
    if "class" in (config_entry or {}):
        cls = load_class(config_entry["class"], module_prefix="state")
        st = cls(**config_entry.get("args", {}))
        # If it looks like a PlasmaState, process it
        if isinstance(st, PlasmaState) and getattr(st, "metadata", None) is not None:
            st.process()
        return st

    raise ValueError("State configuration must include args.from_gacode or a class to instantiate.")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load solver checkpoint from pickle file.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to solver_checkpoint.pkl file
        
    Returns
    -------
    Dict[str, Any]
        Module specifications from checkpoint
    """
    checkpoint_path = os.path.expanduser(os.path.expandvars(str(checkpoint_path)))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path, "rb") as fh:
        module_specs = pickle.load(fh)
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    if 'timestamp' in module_specs:
        print(f"  Checkpoint timestamp: {module_specs['timestamp']}")
    
    return module_specs


def restore_module_from_checkpoint(module_name: str, module_specs: Dict[str, Any],
                                   new_config: Optional[Dict[str, Any]] = None) -> Any:
    """Restore a module from checkpoint specifications.
    
    Parameters
    ----------
    module_name : str
        Name of module to restore (e.g., 'state', 'transport', 'targets')
    module_specs : Dict[str, Any]
        Full checkpoint dictionary
    new_config : Dict[str, Any], optional
        New configuration to override checkpoint values
        
    Returns
    -------
    Any
        Reconstructed module instance
    """
    if module_name not in module_specs:
        return None
    
    spec = module_specs[module_name]
    class_path = spec.get("class_path")
    if not class_path:
        return None

    # If a new config explicitly overrides the class, honor it.
    if new_config is not None:
        override_class = new_config.get("class")
        if override_class:
            class_path = override_class
    
    # Map module names to module prefixes for loading
    module_prefix_map = {
        "parameters": "parameterizations",
        "neutrals": "neutrals",
        "transport": "transport",
        "targets": "targets",
        "surrogate": "surrogates",
        "solver": "solvers",
        "boundary": "boundary",
    }
    module_prefix = module_prefix_map.get(module_name)
    
    # Reconstruct the module by instantiating its class
    cls = load_class(class_path, module_prefix=module_prefix)
    
    # Use new config if provided, otherwise extract from checkpoint
    if new_config is not None:
        args = new_config.get("args", {})
        module = cls(args)
    else:
        # Try to reconstruct from saved attributes
        # This is a fallback; prefer providing new_config
        module = cls({})
        
        # Restore public attributes from checkpoint
        attrs = spec.get("attributes", {})
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, tuple) and len(attr_value) == 2:
                value, type_name = attr_value
            else:
                value = attr_value
            
            try:
                setattr(module, attr_name, value)
            except Exception:
                pass  # Skip attributes that can't be set
    
    return module


def run_workflow(setup_file="setup.yaml"):
    """Run the solver workflow with optional restart from checkpoint.
    
    Parameters
    ----------
    setup_file : str
        Path to run configuration YAML file
    Restart configuration is read from the run_config file:
        restart: null
        restart: /path/to/restart_dir
    If provided, modules will be loaded from checkpoint and updated
    with new run_config parameters.
    """
    print(f"Using setup file: {setup_file}")
    with open(setup_file) as f:
        config = yaml.safe_load(f)

    # Extract verbose flag from top-level config
    verbose = config.get("verbose", False)

    # Apply platform environment (e.g., GACODE_ROOT)
    platform_cfg = resolve_platform_config(config.get("platform", {})) if isinstance(config, dict) else {}
    if platform_cfg:
        config["platform"] = platform_cfg

    # Set working directory if specified
    work_dir = config.get("work_dir")
    if work_dir:
        work_dir = os.path.expanduser(os.path.expandvars(str(work_dir)))
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        print(f"Working directory set to: {work_dir}")

    # Handle restart from checkpoint
    module_specs = None
    restart_dir = config.get("restart")
    if restart_dir:
        restart_path = os.path.expanduser(os.path.expandvars(str(restart_dir)))
        checkpoint_file = os.path.join(restart_path, "solver_checkpoint.pkl")
        
        if os.path.exists(checkpoint_file):
            print(f"\n{'='*60}")
            print(f"RESTARTING FROM CHECKPOINT")
            print(f"{'='*60}")
            module_specs = load_checkpoint(checkpoint_file)
            print(f"Applying new configuration from: {setup_file}")
            print(f"{'='*60}\n")
        else:
            print(f"Warning: restart_dir provided but checkpoint not found at {checkpoint_file}")
            print(f"Starting fresh run instead.\n")
            restart_dir = None

    # Build modules: use checkpoint if available, otherwise build fresh
    if module_specs and restart_dir:
        # Restore state from checkpoint (state typically doesn't change between runs)
        state = restore_module_from_checkpoint("state", module_specs, config.get("state", {}))
        if state is None:
            # Fallback to building fresh state
            state = build_state(config.get("state", {}))
        
        # Restore other modules with new configurations from run_config
        params_cfg = config.get("parameters") or config.get("parameterization") or config.get("parameterizations")
        parameters = restore_module_from_checkpoint("parameters", module_specs, _inject_verbose(params_cfg, verbose))
        if parameters is None and params_cfg:
            parameters = build_module(_inject_verbose(params_cfg, verbose), module_prefix="parameterizations")
        
        neutrals = restore_module_from_checkpoint("neutrals", module_specs, _inject_verbose(config.get("neutrals", {}), verbose))
        if neutrals is None:
            neutrals = build_module(_inject_verbose(config.get("neutrals", {}), verbose), module_prefix="neutrals")
        
        transport_cfg = config.get("transport", {})
        if transport_cfg and "args" in transport_cfg:
            transport_cfg["args"]["work_dir"] = work_dir or config.get("work_dir", ".")
            # Pass platform config to transport so model execution can run on remote platform
            if "platform" in config:
                transport_cfg["args"]["platform"] = config["platform"]
        transport = restore_module_from_checkpoint("transport", module_specs, _inject_verbose(transport_cfg, verbose))
        if transport is None:
            transport = build_module(_inject_verbose(transport_cfg, verbose), module_prefix="transport")
        if transport is not None:
            print(
                "Transport model initialized: "
                f"{transport.__class__.__module__}.{transport.__class__.__name__}"
            )
        
        targets_cfg = config.get("targets", {})
        targets = restore_module_from_checkpoint("targets", module_specs, _inject_verbose(targets_cfg, verbose))
        if targets is None:
            targets = build_module(_inject_verbose(targets_cfg, verbose), module_prefix="targets")
        
        # Apply scaling factors to state for parameter scans
        # This must happen at initialization and be held constant
        if targets and hasattr(targets, 'targets') and hasattr(targets.targets, 'options'):
            _apply_target_scaling_to_state(state, targets.targets.options)
        
        surrogate = restore_module_from_checkpoint("surrogate", module_specs, _inject_verbose(config.get("surrogate", {}), verbose))
        if surrogate is None:
            surrogate = build_module(_inject_verbose(config.get("surrogate", {}), verbose), module_prefix="surrogates")
    else:
        # Fresh run: build all modules from config
        # Build state first (supports from_gacode path)
        state = build_state(config.get("state", {}))

        # Parameters: accept either 'parameters' or legacy 'parameterization' or 'parameterizations'
        params_cfg = config.get("parameters") or config.get("parameterization") or config.get("parameterizations")
        parameters = build_module(_inject_verbose(params_cfg, verbose), module_prefix="parameterizations") if params_cfg else None
        
        # Other components
        neutrals = build_module(_inject_verbose(config.get("neutrals", {}), verbose), module_prefix="neutrals")
        
        # Inject work_dir and platform config into transport module
        transport_cfg = config.get("transport", {})
        if transport_cfg and "args" in transport_cfg:
            transport_cfg["args"]["work_dir"] = work_dir or config.get("work_dir", ".")
            # Pass platform config to transport so model execution can run on remote platform
            if "platform" in config:
                transport_cfg["args"]["platform"] = config["platform"]
        transport = build_module(_inject_verbose(transport_cfg, verbose), module_prefix="transport")
        if transport is not None:
            print(
                "Transport model initialized: "
                f"{transport.__class__.__module__}.{transport.__class__.__name__}"
            )
        
        targets_cfg = config.get("targets", {})
        targets = build_module(_inject_verbose(targets_cfg, verbose), module_prefix="targets")
        
        # Apply scaling factors to state for parameter scans
        # This must happen at initialization and be held constant
        if targets and hasattr(targets, 'targets') and hasattr(targets.targets, 'options'):
            _apply_target_scaling_to_state(state, targets.targets.options)
        
        surrogate = build_module(_inject_verbose(config.get("surrogate", {}), verbose), module_prefix="surrogates")
    
    # Update solver config with parameter count
    if parameters:
        config["solver"]["args"]["n_params_per_profile"] = parameters.n_params_per_profile
    # Update solver config with parameter count
    if parameters:
        config["solver"]["args"]["n_params_per_profile"] = parameters.n_params_per_profile
    
    solver = build_module(_inject_verbose(config.get("solver", {}), verbose), module_prefix="solvers")
    boundary = build_module(_inject_verbose(config.get("boundary", {}), verbose), module_prefix="boundary")

    # Now execute workflow logic
    solver.run(state, boundary, parameters, neutrals, transport, targets, surrogate=surrogate)


def _apply_target_scaling_to_state(state: PlasmaState, target_options: Dict[str, Any]):
    """Apply target scaling factors to state power and particle sources.
    
    This function applies scaling multipliers to the original input.gacode
    heating and particle flux densities. Scaling occurs once at initialization
    and is held constant throughout the solver iterations.
    
    Parameters
    ----------
    state : PlasmaState
        Plasma state object to modify
    target_options : Dict[str, Any]
        Target configuration options containing scale_* parameters
        
    Supported scaling factors:
    -------------------------
    - scale_Pe_beam : float, default 1.0
        Multiplier for electron beam heating (qbeame)
    - scale_Pi_beam : float, default 1.0
        Multiplier for ion beam heating (qbeami)
    - scale_Qpar_beam : float, default 1.0
        Multiplier for parallel particle flux (qpar_beam)
    - scale_ohmic_e : float, default 1.0
        Multiplier for electron ohmic heating (qohme)
    - scale_ohmic_i : float, default 1.0
        Multiplier for ion ohmic heating (qohmi)
    - scale_rf_e : float, default 1.0
        Multiplier for electron RF heating (qrfe)
    - scale_rf_i : float, default 1.0
        Multiplier for ion RF heating (qrfi)
    
    Notes
    -----
    - Scaling is applied BEFORE integration to powers
    - Original unscaled values are preserved in state.metadata
    - This enables parameter scans by only changing run_config
    """
    import numpy as np
    
    # Store original values if not already stored
    if 'unscaled_heating' not in state.metadata:
        state.metadata['unscaled_heating'] = {}
        
        # Preserve originals for all potentially scaled quantities
        for var in ['qbeame', 'qbeami', 'qpar_beam', 'qohme', 'qohmi', 'qrfe', 'qrfi']:
            if hasattr(state, var):
                state.metadata['unscaled_heating'][var] = np.array(getattr(state, var)).copy()
    
    # Define scaling mappings: scale_factor_name -> state_variable
    scaling_map = {
        'scale_Pe_beam': 'qbeame',
        'scale_Pi_beam': 'qbeami',
        'scale_Qpar_beam': 'qpar_beam',
        'scale_ohmic_e': 'qohme',
        'scale_ohmic_i': 'qohmi',
        'scale_rf_e': 'qrfe',
        'scale_rf_i': 'qrfi',
    }
    
    # Apply scaling factors
    scaling_applied = []
    for scale_key, state_var in scaling_map.items():
        scale_factor = target_options.get(scale_key, 1.0)
        
        # Only apply if scale factor differs from 1.0 and variable exists
        if scale_factor != 1.0 and hasattr(state, state_var):
            original = state.metadata['unscaled_heating'].get(state_var)
            if original is not None:
                scaled_value = original * scale_factor
                setattr(state, state_var, scaled_value)
                scaling_applied.append(f"{scale_key}={scale_factor:.3f}")
    
    # Report applied scaling
    if scaling_applied:
        print(f"\nApplied target scaling to state:")
        for msg in scaling_applied:
            print(f"  {msg}")
        print()
    
    # Store applied scaling in metadata for transparency
    state.metadata['applied_scaling'] = {k: v for k, v in target_options.items() if k.startswith('scale_')}


def main(argv: list[str] | None = None):
    """CLI entrypoint to run the workflow with a specific setup YAML."""
    parser = argparse.ArgumentParser(description="Run MinT workflow")
    parser.add_argument("setup", nargs="?", default="setup.yaml", help="Path to setup YAML file")
    parser.add_argument("--setup", "-s", dest="setup_opt", help="Path to setup YAML file (alias)")
    args = parser.parse_args(argv)

    setup_path = args.setup_opt or args.setup
    setup_path = os.path.expanduser(os.path.expandvars(setup_path))
    if not os.path.isabs(setup_path):
        setup_path = os.path.abspath(setup_path)

    if not os.path.exists(setup_path):
        print(f"Error: setup file not found: {setup_path}")
        sys.exit(2)

    run_workflow(setup_path)
    # try:
    #     run_workflow(setup_path)
    # except Exception as e:
    #     print(f"Workflow failed: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
