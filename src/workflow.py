import importlib
import os
import sys
from typing import Any, Dict
import argparse
import yaml

# Local imports for state construction
from interfaces import gacode
from state import PlasmaState

def load_class(path: str):
    """Dynamically import class given 'Module.ClassName' string."""
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def build_module(config_entry: Dict[str, Any]):
    """Instantiate a module given its dict from setup.

    Expects: {"class": "package.Module.Class", "args": {...}}
    """
    if not config_entry:
        return None
    class_path = config_entry.get("class")
    if not class_path:
        return None
    cls = load_class(class_path)
    args = config_entry.get("args", {})
    return cls(args)


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
        cls = load_class(config_entry["class"])
        st = cls(**config_entry.get("args", {}))
        # If it looks like a PlasmaState, process it
        if isinstance(st, PlasmaState) and getattr(st, "metadata", None) is not None:
            st.process()
        return st

    raise ValueError("State configuration must include args.from_gacode or a class to instantiate.")

def run_workflow(setup_file="setup.yaml"):
    with open(setup_file) as f:
        config = yaml.safe_load(f)

    # Set working directory if specified
    work_dir = config.get("work_dir")
    if work_dir:
        work_dir = os.path.expanduser(os.path.expandvars(str(work_dir)))
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        print(f"Working directory set to: {work_dir}")

    # Build state first (supports from_gacode path)
    state = build_state(config.get("state", {}))

    # Parameters: accept either 'parameters' or legacy 'parameterization'
    params_cfg = config.get("parameters") or config.get("parameterization")
    parameters = build_module(params_cfg) if params_cfg else None
    config["solver"]["args"]["n_params_per_profile"] = parameters.n_params_per_profile  

    # Other components
    neutrals = build_module(config.get("neutrals", {}))
    transport = build_module(config.get("transport", {}))
    targets = build_module(config.get("targets", {}))
    surrogate = build_module(config.get("surrogate", {}))
    solver = build_module(config.get("solver", {}))
    boundary = build_module(config.get("boundary", {}))

    # Now execute workflow logic
    solver.run(state, boundary, parameters, neutrals, transport, targets, surrogate)


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
