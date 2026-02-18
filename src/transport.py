"""
Transport model interface and implementations.

Refactored to follow the project pattern used by other modules (e.g., boundary):
- Base class attaches a simple container object at state.transport to store options and outputs.
- Concrete models compute fluxes on state.roa grid and store them under state.transport.*

Currently includes a Fingerprints-like simplified model and a fixed-transport test model.
"""

import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
import traceback
import os
import subprocess
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from state import PlasmaState
    from tools.io import PlatformManager


def _resolve_gacode_executable(executable: str, options: Dict[str, Any]) -> str:
    """Resolve a GACODE executable from $GACODE_ROOT or PATH."""
    gacode_root = options.get("gacode_root") or os.environ.get("GACODE_ROOT")
    if gacode_root:
        candidate = Path(gacode_root) / "bin" / executable
        if candidate.exists():
            return str(candidate)
    return shutil.which(executable) or executable


def _format_namelist_value(value: Any) -> str:
    """Format a value for Fortran namelist output."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        # Check if already formatted as Fortran boolean
        v_lower = value.lower().strip()
        if v_lower in ('.true.', '.false.', 't', 'f'):
            return 'True' if v_lower in ('.true.', 't') else 'False'
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(_format_namelist_value(v) for v in value)
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _parse_namelist_lines(lines: List[str]) -> Dict[str, str]:
    parsed = {}
    for line in lines:
        clean = line.strip()
        if not clean or clean.startswith("#") or clean.startswith("!"):
            continue
        if "=" not in clean:
            continue
        key, val = clean.split("=", 1)
        parsed[key.strip().upper()] = val.strip()
    return parsed


def _update_namelist_lines(lines: List[str], updates: Dict[str, Any]) -> List[str]:
    updates_upper = {k.upper(): v for k, v in updates.items()}
    updated_keys = set()
    out_lines = []
    for line in lines:
        clean = line.strip()
        if not clean or clean.startswith("#") or clean.startswith("!") or "=" not in clean:
            out_lines.append(line)
            continue
        key, _ = clean.split("=", 1)
        key_upper = key.strip().upper()
        if key_upper in updates_upper:
            value = _format_namelist_value(updates_upper[key_upper])
            out_lines.append(f"{key_upper} = {value}\n")
            updated_keys.add(key_upper)
        else:
            out_lines.append(line)

    for key, value in updates_upper.items():
        if key not in updated_keys:
            out_lines.append(f"{key} = {_format_namelist_value(value)}\n")
    return out_lines


def _with_python_in_path(
    env: Optional[Dict[str, str]] = None,
    gacode_root: Optional[str] = None,
) -> Dict[str, str]:
    """Ensure the current Python executable and GACODE python modules are discoverable."""
    resolved = dict(env or os.environ)
    if gacode_root:
        resolved["GACODE_ROOT"] = gacode_root

    python_dir = str(Path(sys.executable).parent)
    path = resolved.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if python_dir not in parts:
        resolved["PATH"] = python_dir + os.pathsep + path

    gacode_root = resolved.get("GACODE_ROOT")
    if gacode_root:
        extra = [
            Path(gacode_root) / "f2py" / "pygacode",
            Path(gacode_root) / "f2py",
            Path(gacode_root) / "tglf" / "bin",
        ]
        existing = resolved.get("PYTHONPATH", "")
        extra_str = os.pathsep.join(str(p) for p in extra if p.exists())
        if extra_str:
            resolved["PYTHONPATH"] = extra_str + (os.pathsep + existing if existing else "")

    return resolved


def _read_numeric_file(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        text = f.read()
    values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", text)]
    return np.array(values, dtype=float)


def _get_template_path(template_type: str) -> Path:
    """Get path to a GACODE template file from src/tools/gacode_templates/.
    
    Parameters
    ----------
    template_type : str
        One of 'tglf', 'cgyro', 'qlgyro', 'neo'
        
    Returns
    -------
    Path
        Absolute path to template file
    """
    # Find src/tools/gacode_templates relative to this file
    this_file = Path(__file__).resolve()
    template_dir = this_file.parent / "tools" / "gacode_templates"
    
    template_file = template_dir / f"input.{template_type.lower()}"
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    
    return template_file


def _load_template(template_type: str) -> List[str]:
    """Load and parse a GACODE template file.
    
    Parameters
    ----------
    template_type : str
        One of 'tglf', 'cgyro', 'qlgyro', 'neo'
        
    Returns
    -------
    List[str]
        Template lines with keepends=True for safe reconstruction
    """
    template_path = _get_template_path(template_type)
    with open(template_path, "r") as f:
        return f.readlines()


def _resolve_locpargen_executable(options: Dict[str, Any]) -> str:
    """Resolve profiles_gen/locpargen executable from GACODE_ROOT or PATH."""
    gacode_root = options.get("gacode_root") or os.environ.get("GACODE_ROOT")
    explicit = options.get("locpargen_executable")
    if explicit:
        return explicit
    profiles_gen = _resolve_gacode_executable("profiles_gen", options)
    if shutil.which(profiles_gen) or Path(profiles_gen).exists():
        return profiles_gen
    if gacode_root:
        candidate = Path(gacode_root) / "profiles_gen" / "locpargen" / "locpargen"
        if candidate.exists():
            return str(candidate)
    return "profiles_gen"


def _get_controls_path(model: str) -> Path:
    """Get path to a GACODE controls file from src/tools/gacode_controls/.
    
    Parameters
    ----------
    model : str
        One of 'tglf', 'cgyro', 'qlgyro', 'neo'
        
    Returns
    -------
    Path
        Absolute path to controls file
    """
    this_file = Path(__file__).resolve()
    controls_dir = this_file.parent / "tools" / "gacode_controls"
    
    controls_file = controls_dir / f"input.{model.lower()}.controls"
    if not controls_file.exists():
        raise FileNotFoundError(f"Controls file not found: {controls_file}")
    
    return controls_file


def _load_controls(model: str) -> List[str]:
    """Load GACODE controls file, filtering out template header comments.
    
    Parameters
    ----------
    model : str
        One of 'tglf', 'cgyro', 'qlgyro', 'neo'
        
    Returns
    -------
    List[str]
        Control file lines (without template headers)
    """
    controls_path = _get_controls_path(model)
    with open(controls_path, "r") as f:
        lines = f.readlines()
    
    # Filter out template header block
    filtered = []
    in_header = False
    for line in lines:
        stripped = line.strip()
        # Detect header block delimiters (#--- or #===)
        if stripped.startswith('#---') or stripped.startswith('#==='):
            in_header = not in_header
            continue
        # Skip lines within header block
        if in_header:
            continue
        # Skip standalone "Template" comment lines
        if 'Template' in line and line.strip().startswith('#'):
            continue
        filtered.append(line)
    
    return filtered


def _run_locpargen(state, rho: float, rho_dir: Path, input_gacode: Path) -> Dict[str, Path]:
    """Run profiles_gen/locpargen to create state-specific input files.
    
    Returns dict mapping filenames to paths (e.g., {'input.tglf.locpargen': Path(...)})
    """
    if not hasattr(state, "to_gacode"):
        raise RuntimeError("locpargen requires state.to_gacode() to create input.gacode")

    rho_dir.mkdir(parents=True, exist_ok=True)
    input_gacode = Path(input_gacode).resolve()
    if not input_gacode.exists():
        state.to_gacode().write(str(input_gacode))

    # Copy input.gacode to rho_dir so profiles_gen can find it
    rho_input_gacode = rho_dir / "input.gacode"
    shutil.copy2(str(input_gacode), str(rho_input_gacode))

    executable = _resolve_locpargen_executable({})
    # Use relative path (just "input.gacode") since we run with cwd=rho_dir
    cmd = [executable, "-i", "input.gacode", "-loc_rho", str(rho)]
    result = subprocess.run(
        cmd,
        cwd=str(rho_dir),
        capture_output=True,
        text=True,
        env=_with_python_in_path(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"profiles_gen/locpargen failed at rho={rho}: {result.stderr.strip()}\nCommand: {' '.join(cmd)}")

    loc_files = list(rho_dir.glob("*.locpargen"))
    if not loc_files:
        raise RuntimeError(f"No .locpargen files produced in {rho_dir} (locpargen stderr: {result.stderr.strip()})")

    return {p.name: p for p in loc_files}


def _pick_locpargen_file(loc_files: Dict[str, Path], keyword: str) -> Optional[Path]:
    """Find locpargen file matching keyword (e.g., 'tglf' -> 'input.tglf.locpargen')."""
    key = f"input.{keyword}.locpargen"
    if key in loc_files:
        return loc_files[key]
    for name, path in loc_files.items():
        if keyword in name:
            return path
    return None


def _merge_controls_and_locpargen(
    controls_lines: List[str],
    locpargen_path: Path,
    settings: Dict[str, Any],
    multipliers: Dict[str, float],
) -> List[str]:
    """Merge controls, locpargen data, and user settings (removing duplicates).
    
    Parameters
    ----------
    controls_lines : List[str]
        Base control file lines
    locpargen_path : Path
        Path to .locpargen file (has state-specific values)
    settings : dict
        User overrides (e.g., {'SAT_RULE': 2, 'USE_BPER': 0})
    multipliers : dict
        Scaling factors (e.g., {'RLTS_1': 1.5})
        
    Returns
    -------
    lines : List[str]
        Final namelist lines ready to write
    """
    # Step 1: Parse controls to get parameter names
    controls_params = set()
    for line in controls_lines:
        clean = line.strip()
        if clean and not clean.startswith("#") and "=" in clean:
            key = clean.split("=", 1)[0].strip().upper()
            controls_params.add(key)
    
    # Step 2: Read locpargen file and filter out duplicates
    with open(locpargen_path, "r") as f:
        locpargen_lines = f.readlines()
    
    # Keep only locpargen lines that don't duplicate control parameters
    filtered_locpargen = []
    for line in locpargen_lines:
        clean = line.strip()
        if clean and not clean.startswith("#") and "=" in clean:
            key = clean.split("=", 1)[0].strip().upper()
            if key not in controls_params:
                filtered_locpargen.append(line)
        elif not clean or clean.startswith("#"):
            # Keep comments and blank lines from locpargen
            filtered_locpargen.append(line)
    
    # Step 3: Merge controls + filtered locpargen
    merged = controls_lines + filtered_locpargen
    
    # Step 4: Apply user settings
    if settings:
        merged = _update_namelist_lines(merged, settings)
    
    # Step 5: Apply multipliers
    if multipliers:
        parsed = _parse_namelist_lines(merged)
        mult_updates = {}
        for key, mult in multipliers.items():
            key_upper = str(key).upper()
            if key_upper in parsed:
                try:
                    base = float(parsed[key_upper].split()[0])
                    mult_updates[key_upper] = base * float(mult)
                except (ValueError, IndexError):
                    pass
        if mult_updates:
            merged = _update_namelist_lines(merged, mult_updates)
    
    return merged


def _run_neo_single_rho(
    rho: float,
    state,
    work_dir: Path,
    input_gacode: Path,
    neo_exec: str,
    settings: Dict[str, Any],
    multipliers: Dict[str, float],
    n_threads: int,
    keep_files: str,
) -> None:
    """Run NEO for a single radial location.
    
    This function is designed to be called by ProcessPoolExecutor for parallel execution.
    
    Parameters
    ----------
    rho : float
        Normalized radius point [0,1]
    state : PlasmaState
        Plasma state with profiles and geometry
    work_dir : Path
        Working directory for NEO runs
    input_gacode : Path
        Path to input.gacode file
    neo_exec : str
        Path to NEO executable
    settings : dict
        User settings overrides
    multipliers : dict
        Scaling factors for parameters
    n_threads : int
        Number of OpenMP threads
    keep_files : str
        File retention policy ('none', 'minimal', 'all')
    """
    rho_dir = work_dir / f"rho_{rho:.3f}"
    rho_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run locpargen to get state-specific input file
    loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
    locpargen_file = _pick_locpargen_file(loc_files, "neo")
    
    if locpargen_file is None:
        raise RuntimeError(f"locpargen did not produce input.neo.locpargen at rho={rho}")
    
    # Step 2: Load controls and merge with locpargen data
    controls_lines = _load_controls("neo")
    lines = _merge_controls_and_locpargen(
        controls_lines,
        locpargen_file,
        settings,
        multipliers
    )
    
    # Step 3: Write final input file
    input_neo = rho_dir / "input.neo"
    input_neo.write_text("".join(lines))

    # Run NEO subprocess with relative path and cwd
    cmd = [neo_exec, "-e", rho_dir.name, "-nomp", str(n_threads)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        env=_with_python_in_path(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"NEO failed at rho={rho:.3f}: {result.stderr.strip()}")

    if keep_files == "none":
        shutil.rmtree(rho_dir, ignore_errors=True)


def _run_tglf_single_rho(
    rho: float,
    state,
    work_dir: Path,
    input_gacode: Path,
    tglf_exec: str,
    settings: Dict[str, Any],
    multipliers: Dict[str, float],
    n_threads: int,
    keep_files: str,
) -> None:
    """Run TGLF for a single radial location.

    This function is designed to be called by ProcessPoolExecutor for parallel execution.
    """
    rho_dir = work_dir / f"rho_{rho:.3f}"
    rho_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run locpargen to get state-specific input file
    loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
    locpargen_file = _pick_locpargen_file(loc_files, "tglf")

    if locpargen_file is None:
        raise RuntimeError(f"locpargen did not produce input.tglf.locpargen at rho={rho}")

    # Step 2: Load controls and merge with locpargen data
    controls_lines = _load_controls("tglf")
    lines = _merge_controls_and_locpargen(
        controls_lines,
        locpargen_file,
        settings,
        multipliers,
    )

    # Step 3: Write final input file
    input_tglf = rho_dir / "input.tglf"
    input_tglf.write_text("".join(lines))

    # Run TGLF subprocess
    cmd = [tglf_exec, "-p", str(work_dir), "-e", rho_dir.name, "-n", str(n_threads)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_with_python_in_path(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"TGLF failed at rho={rho:.3f}: {result.stderr.strip()}")

    if keep_files == "none":
        shutil.rmtree(rho_dir, ignore_errors=True)


class TransportBase:
    """Base class for transport models using PlasmaState.

    Pattern:
    - call init(state, **kwargs) to attach a container at state.transport
    - call evaluate(state) to populate state.transport.* fields on state.roa grid
    
    Supports batch processing:
    - evaluate(state) handles both single PlasmaState and list of PlasmaState objects
    - Returns results for single state or list of results for batched states
    
    Platform execution (optional):
    - run_on_platform(state, platform) runs model on remote/local platform
    
    Evaluation logging:
    - Auto-logs evaluations to TransportEvaluationLog for surrogate warm-start
    - Configure via options['evaluation_log'] = {'enabled': True, 'path': '...'}
    """

    def __init__(self, options, **kwargs):
        self.options = options
        self.options.update(dict(kwargs) if kwargs else {})
        self.sigma = self.options.get('sigma', 0.1) # relative epistemic uncertainty for transport model outputs
        self.neoclassical_model = self.options.get('neoclassical_model', 'analytic')
    
        # Get workflow work_dir (from options or use default)
        workflow_work_dir = Path(self.options.get('work_dir', '.'))
        self.neo_work_dir = workflow_work_dir / 'neo'
       
        # Support both old neoclassical_options and new NEO_options
        self.neo_opts = self.options.get('NEO_options', self.options.get('neoclassical_options', {}))
        self.roa_eval = self.options.get('roa_eval', None)
        self.output_vars = self.options.get('output_vars', None)
        self.state_vars_extracted = False

        self.n_parallel = int(self.options.get('n_parallel', 1))
        self.n_threads = int(self.options.get('n_threads', 1))
        
        # Evaluation logging
        self.eval_log = None
        self.log_evaluations = self.options.get('log_evaluations', False)
        eval_log_config = self.options.get('evaluation_log', {})
        
        if self.log_evaluations or eval_log_config.get('enabled', False):
            from evaluation_log import TransportEvaluationLog, get_default_log_path
            log_path = eval_log_config.get('path') or get_default_log_path(self.options)
            try:
                self.eval_log = TransportEvaluationLog(str(log_path))
                print(f"Transport evaluation logging enabled: {log_path}")
            except Exception as e:
                print(f"Warning: Could not initialize evaluation log: {e}")
                self.eval_log = None

    def _is_batched(self, state) -> bool:
        """Check if state is a batch (list/array of states) vs single state."""
        if isinstance(state, (list, tuple)):
            return True
        # Check if it's a numpy array of PlasmaState objects
        if isinstance(state, np.ndarray) and state.dtype == object:
            return True
        return False

    def evaluate(self, state) -> Any:
        """Evaluate transport model for single or batch of states.
        
        Parameters
        ----------
        state : PlasmaState or list of PlasmaState
            Single plasma state or list of states for batch processing
            
        Returns
        -------
        Any
            For single state: result from _evaluate_single()
            For batch: list of results from _evaluate_single() for each state
        """
        if self._is_batched(state):
            results = []
            for single_state in state:
                result = self._evaluate_single(single_state)
                results.append(result)
            return results
        else:
            return self._evaluate_single(state)

    def _evaluate_single(self, state) -> None:
        """Evaluate transport model for a single state.
        
        To be overridden by child classes with actual implementation.
        
        Parameters
        ----------
        state : PlasmaState
            Single plasma state
        """
        raise NotImplementedError
    
    def _extract_from_state(self,state) -> None:
        """Extract commonly used variables from state for transport calculations."""

        # Extract quantities from state
        self.x = state.roa
        self.a = state.a
        self.aspect_ratio = state.aspect_ratio
        self.eps = state.eps
        self.Te = state.te
        self.Ti = state.ti
        self.ne = state.ne
        self.ni = state.ni
        self.pe = state.pe
        self.pi = state.pi
        self.aLne = state.aLne
        self.aLni = state.aLni
        self.aLTe = state.aLte
        self.aLTi = state.aLti
        self.kappa = state.kappa
        self.q = state.q
        self.Zeff = state.Zeff
        self.mi_over_mp = state.mi_ref
        self.f_trap = state.f_trap
        self.beta = state.betae * (1 + self.Ti/self.Te) # beta_norm * ne * (Ti + Te)
        self.rhostar = state.rhostar # rhostar_norm * np.sqrt(Ti)
        self.dne_dx = -self.aLne * self.ne # dne_dr * r/a
        self.dTe_dx = -self.aLTe * self.Te
        self.dTi_dx = -self.aLTi * self.Ti
        self.dni_dx = -self.aLni * self.ni
        self.dpe_dx = self.ne*self.dTe_dx + self.Te*self.dne_dx
        self.dpi_dx = self.ni * self.dTi_dx + self.Ti * self.dni_dx
        self.aLpe = - (self.dpe_dx) / self.pe
        self.d2ne_dx2 = state.d2ne * self.a**2 # d2ne/dr2 * a**2
        self.d2ni_dx2 = self.d2ne_dx2 / self.Zeff # assume ni ~ ne / Zeff
        self.d2Te_dx2 = state.d2te * self.a**2 # d2Te/dr2 * a**2
        self.d2Ti_dx2 = state.d2ti * self.a**2 # d2Ti/dr2 * a**2
        self.d2pi_dx2 = self.Ti*self.d2ni_dx2 + 2*self.dni_dx*self.dTi_dx + self.ni*self.d2Ti_dx2
                # Collision frequencies
        self.nuii = state.nuii*state.tau_norm
        self.nuei = state.nuei*state.tau_norm
        self.Qnorm_to_P = state.Qnorm_to_P
        self.g_gb = state.g_gb
        self.q_gb = state.q_gb
        self.state_vars_extracted = True
    
    def _log_evaluation(self, state, outputs: Dict[str, Any], roa_idx: int):
        """Log this evaluation to the transport evaluation database.
        
        Parameters
        ----------
        state : PlasmaState
            Current plasma state
        outputs : Dict[str, Any]
            Transport model outputs at this roa location
        roa_idx : int
            Index in roa_eval array for this evaluation
        """
        if self.eval_log is None:
            return
        
        try:
            # Get current roa value
            roa = self.roa_eval[roa_idx] if self.roa_eval is not None else state.roa[roa_idx]
            
            # Extract dimensionless state features at this location
            state_features = self._extract_state_features(state, roa)
            
            # Get model class name
            model_class = f"{self.__class__.__module__}.{self.__class__.__name__}"
            
            # Get model settings (specific to each transport class)
            model_settings = self._get_model_settings()
            
            # Log to database
            self.eval_log.add_evaluation(
                model_class=model_class,
                model_settings=model_settings,
                roa=float(roa),
                state_features=state_features,
                outputs=outputs,
                skip_duplicates=True
            )
        except Exception as e:
            # Don't let logging errors crash the simulation
            if hasattr(self, '_log_warn_once'):
                pass  # Already warned
            else:
                print(f"Warning: Evaluation logging failed: {e}")
                self._log_warn_once = True
    
    def _extract_state_features(self, state, roa: float) -> Dict[str, float]:
        """Extract dimensionless state features at given roa location.
        
        Uses same features as SurrogateManager for consistency.
        
        Parameters
        ----------
        state : PlasmaState
            Current plasma state
        roa : float
            Radial location
            
        Returns
        -------
        Dict[str, float]
            Dimensionless state features
        """
        # Interpolate state quantities to target roa
        def interp(var_name):
            var = getattr(state, var_name, None)
            if var is None:
                return 0.0
            return float(np.interp(roa, state.roa, var))
        
        # Standard feature set (matches SurrogateManager.state_features)
        features = {
            "aLne": interp("aLne"),
            "aLte": interp("aLte"),
            "aLti": interp("aLti"),
            "tite": interp("ti") / max(interp("te"), 1e-6),
            "betae": interp("betae"),
            "nustar": interp("nustar"),
            "rhostar": interp("rhostar"),
            "gamma_exb": interp("gamma_exb"),
            "gamma_par": interp("gamma_par"),
            "q": interp("q"),
            "Zeff": interp("Zeff"),
            "shear": interp("shear"),
            "delta": interp("delta"),
            "kappa": interp("kappa"),
            "eps": interp("eps"),
            "roa": float(roa),
        }
        
        return features
    
    def _get_model_settings(self) -> Dict[str, Any]:
        """Get model-specific settings for logging.
        
        Override in child classes to log model-specific configuration.
        
        Returns
        -------
        Dict[str, Any]
            Model settings that affect outputs
        """
        # Base implementation returns common options
        return {
            'neoclassical_model': self.neoclassical_model,
            'sigma': self.sigma,
        }

    def _gbflux_to_physical(self, rho: float, Ge: float, Qi: float, Qe: float) -> Tuple[float, float, float]:
        """Convert gyroBohm-normalized fluxes to physical units at rho.

        Ge -> 1e19/m^2/s (using g_gb)
        Qi/Qe -> MW/m^2 (using q_gb)
        """
        g_gb = np.asarray(self.g_gb)
        q_gb = np.asarray(self.q_gb)
        if g_gb.size == 1:
            g_val = float(g_gb)
        else:
            g_val = float(np.interp(rho, self.x, g_gb))
        if q_gb.size == 1:
            q_val = float(q_gb)
        else:
            q_val = float(np.interp(rho, self.x, q_gb))

        # g_gb is in 1e20/(m^2 s); q_gb is in MW/m^2 (both are per-area fluxes).
        # Convert particle flux to 1e19/m^2/s for consistency with state.Ge convention.
        g_area_1e19 = g_val * 10.0  # 1e20 -> 1e19

        return Ge * g_area_1e19, Qi * q_val, Qe * q_val

    def _compute_neoclassical(self,state) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute neoclassical fluxes according to selected model."""
        if self.neoclassical_model == 'analytic':
            self._extract_from_state(state)
            Gamma_neo, Qi_neo, Qe_neo = self.compute_analytic()
            if self.roa_eval is not None:
                roa_eval = np.atleast_1d(self.roa_eval)
                if len(Gamma_neo) != len(roa_eval):
                    Gamma_neo = np.interp(roa_eval, state.roa, Gamma_neo)
                    Qi_neo = np.interp(roa_eval, state.roa, Qi_neo)
                    Qe_neo = np.interp(roa_eval, state.roa, Qe_neo)
            return Gamma_neo, Qi_neo, Qe_neo
        if self.neoclassical_model == 'neo':
            roa_eval = np.array(self.roa_eval) if self.roa_eval is not None else np.array(state.roa)
            Ge_neo, Qi_neo, Qe_neo = self.compute_neo(state, roa_eval)
            # Expand to full grid by interpolation if needed
            if len(Ge_neo) == len(roa_eval):
                Gamma_neo = np.interp(state.roa, roa_eval, Ge_neo)
                Qi_neo = np.interp(state.roa, roa_eval, Qi_neo)
                Qe_neo = np.interp(state.roa, roa_eval, Qe_neo)
                return Gamma_neo, Qi_neo, Qe_neo
            raise RuntimeError("NEO output size does not match evaluation grid")
        raise ValueError(f"Unknown neoclassical_model: {self.neoclassical_model}")

    def compute_analytic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute analytic neoclassical fluxes (GB units)."""

        chii_nc = self.f_trap * (self.Ti * (self.q / np.maximum(self.eps, 1e-9))**2) * self.nuii
        chie_nc = self.f_trap * ((self.Te * (self.q / np.maximum(self.eps, 1e-9))**2) / (1840.0 * self.mi_over_mp)) * self.nuei

        Gamma_neo = chie_nc * (
            -1.53 * (1.0 + self.Ti / self.Te) * self.dne_dx
            + 0.59 * (self.ne / self.Te) * self.dTe_dx
            + 0.26 * (self.ne / self.Te) * self.dTi_dx
        )
        Qi_neo = -self.ne * chii_nc * self.dTi_dx + 1.5 * self.Ti * Gamma_neo
        Qe_neo = -self.ne * chie_nc * self.dTe_dx + 1.5 * self.Te * Gamma_neo
        return Gamma_neo, Qi_neo, Qe_neo

    def compute_neo(self, state, roa_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run or parse NEO to obtain neoclassical fluxes (GB units).
        
        Parameters
        ----------
        state : PlasmaState
            Plasma state with profiles and geometry
        roa_eval : np.ndarray
            Array of normalized radius points [0,1] for evaluation
            
        Returns
        -------
        Ge_neo, Qi_neo, Qe_neo : (np.ndarray, np.ndarray, np.ndarray)
            Particle and heat fluxes on roa_eval grid
        """

        work_dir = self.neo_work_dir
        keep_files = self.neo_opts.get("keep_files", "minimal")
        settings = self.neo_opts.get("settings", {})
        multipliers = self.neo_opts.get("multipliers", {})
        
        work_dir.mkdir(parents=True, exist_ok=True)
        neo_exec = _resolve_gacode_executable("neo", self.neo_opts)

        input_gacode = work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        # Run NEO at each radial point (with parallel execution if n_parallel > 1)
        roa_eval_list = list(roa_eval)
        
        if self.n_parallel > 1:
            # Run in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = {
                    executor.submit(
                        _run_neo_single_rho,
                        rho, state, work_dir, input_gacode, neo_exec, 
                        settings, multipliers, self.n_threads, keep_files
                    ): rho 
                    for rho in roa_eval_list
                }
                
                for future in as_completed(futures):
                    rho = futures[future]
                    try:
                        future.result()  # Raises exception if NEO failed
                    except Exception as e:
                        raise RuntimeError(f"NEO failed at rho={rho:.3f}: {str(e)}")
        else:
            # Run sequentially
            for rho in roa_eval_list:
                _run_neo_single_rho(
                    rho, state, work_dir, input_gacode, neo_exec,
                    settings, multipliers, self.n_threads, keep_files
                )

        # Parse NEO outputs
        rho_dirs = {p.name.replace("rho_", ""): p for p in work_dir.glob("rho_*") if p.is_dir()}
        available_rhos = np.array([float(k) for k in rho_dirs.keys()]) if rho_dirs else np.array([])

        def parse_transport_flux(path: Path) -> Tuple[float, float, float]:
            """Parse NEO out.neo.transport_flux output."""
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            if not lines:
                raise RuntimeError(f"Empty NEO output: {path}")
            
            # Format: Z  Gamma_e  Q_e  [M_e  ...]
            # Parse first 3 columns (Z, Gamma, Q)
            data = []
            for ln in lines:
                try:
                    cols = [float(x) for x in ln.split()[:4]]
                    data.append(cols)
                except (ValueError, IndexError):
                    continue
            
            if not data:
                raise RuntimeError(f"Could not parse NEO output: {path}")
            
            data = np.array(data)
            Z = data[:, 0]
            G = data[:, 1]
            Q = data[:, 2]
            
            # Find electron entry (Z = -1)
            ie = int(np.where(np.isclose(Z, -1.0))[0][0])
            Ge = float(G[ie])
            Qe = float(Q[ie])
            Qi = float(np.sum(np.delete(Q, ie)))
            
            return Ge, Qi, Qe

        # Interpolate to requested roa_eval points
        Ge_list, Qi_list, Qe_list = [], [], []
        for rho in roa_eval:
            rho_key = f"{rho:.3f}"
            if rho_key in rho_dirs:
                rho_dir = rho_dirs[rho_key]
            elif available_rhos.size > 0:
                nearest = available_rhos[np.argmin(np.abs(available_rhos - float(rho)))]
                rho_key_nearest = f"{nearest:.3f}"
                rho_dir = rho_dirs.get(rho_key_nearest)
            else:
                rho_dir = None

            if rho_dir is None:
                raise RuntimeError(f"NEO output folder for rho={rho:.3f} not found in {work_dir}")

            output_path = rho_dir / "out.neo.transport_flux"
            if not output_path.exists():
                raise RuntimeError(f"NEO output file missing: {output_path}")

            Ge, Qi, Qe = parse_transport_flux(output_path)
            Ge_list.append(Ge)
            Qi_list.append(Qi)
            Qe_list.append(Qe)

        return np.array(Ge_list), np.array(Qi_list), np.array(Qe_list)

    def _assemble_fluxes(
        self,
        state,
        *,
        Gamma_turb: np.ndarray,
        Gamma_neo: np.ndarray,
        Qi_turb: np.ndarray,
        Qi_neo: np.ndarray,
        Qe_turb: np.ndarray,
        Qe_neo: np.ndarray,
        model_label: str = "Transport",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Assemble and store flux components and output dictionaries."""

        roa_eval = np.atleast_1d(self.roa_eval)

        def _to_roa_eval(data: np.ndarray) -> np.ndarray:
            if self.roa_eval is not None:
                if len(data) == len(state.roa):
                    return np.interp(roa_eval, state.roa, data)
            return data
        
        Zeff = _to_roa_eval(state.Zeff)
        Qnorm_to_P = _to_roa_eval(state.Qnorm_to_P)

        # Convert particle flux to convective power flow
        Ge_to_Ce = _to_roa_eval(1.5 * self.Te * self.Qnorm_to_P)
        Gi_to_Ci = _to_roa_eval(1.5 * self.Ti * self.Qnorm_to_P)

        # Storage - process all flux components through _to_roa_eval for consistent lengths
        Gamma_turb_eval = _to_roa_eval(Gamma_turb)
        Gamma_neo_eval = _to_roa_eval(Gamma_neo)
        Qi_turb_eval = _to_roa_eval(Qi_turb)
        Qi_neo_eval = _to_roa_eval(Qi_neo)
        Qe_turb_eval = _to_roa_eval(Qe_turb)
        Qe_neo_eval = _to_roa_eval(Qe_neo)
        
        self.Ge_turb = Gamma_turb_eval
        self.Ge_neo = Gamma_neo_eval
        self.Ge = Gamma_turb_eval + Gamma_neo_eval
        self.Gi_turb = self.Ge_turb / Zeff
        self.Gi_neo = self.Ge_neo / Zeff
        self.Gi = self.Gi_turb + self.Gi_neo
        self.Ce_turb = self.Ge_turb * Ge_to_Ce
        self.Ce_neo = self.Ge_neo * Ge_to_Ce
        self.Ce = self.Ce_turb + self.Ce_neo
        self.Ci_turb = self.Gi_turb * Gi_to_Ci
        self.Ci_neo = self.Gi_neo * Gi_to_Ci
        self.Ci = self.Ci_turb + self.Ci_neo
        self.Qi_turb = Qi_turb_eval
        self.Qi_neo = Qi_neo_eval
        self.Qi = self.Qi_turb + self.Qi_neo
        self.Pi_turb = self.Qi_turb * Qnorm_to_P
        self.Pi_neo = self.Qi_neo * Qnorm_to_P
        self.Pi = self.Pi_turb + self.Pi_neo
        self.Qe_turb = Qe_turb_eval
        self.Qe_neo = Qe_neo_eval
        self.Qe = self.Qe_turb + self.Qe_neo
        self.Pe_turb = self.Qe_turb * Qnorm_to_P
        self.Pe_neo = self.Qe_neo * Qnorm_to_P
        self.Pe = self.Pe_turb + self.Pe_neo

        # Defaults for output selection
        if self.roa_eval is None:
            self.roa_eval = list(state.roa)
        if self.output_vars is None:
            if hasattr(self, "labels"):
                self.output_vars = list(self.labels)
            else:
                self.output_vars = [
                    key for key in ["Ge", "Gi", "Ce", "Ci", "Pe", "Pi", "Qe", "Qi"]
                    if hasattr(self, key)
                ]

        # Provide dict for requested outputs
        def _value_at_roa(data: np.ndarray, roa: float) -> float:
            arr = np.asarray(data)
            if arr.ndim == 0:
                return float(arr)
            if arr.ndim == 1:
                if len(arr) == len(state.roa):
                    xp = state.roa
                elif self.roa_eval is not None and len(arr) == len(self.roa_eval):
                    xp = np.asarray(self.roa_eval)
                else:
                    return float(np.nan_to_num(arr, nan=0.0)[0])
                if np.any(np.isclose(xp, roa, atol=1e-3)):
                    idx = int(np.where(np.isclose(xp, roa, atol=1e-3))[0][0])
                    return float(np.nan_to_num(arr, nan=0.0)[idx])
                return float(np.interp(roa, xp, np.nan_to_num(arr, nan=0.0)))
            return float(np.nan_to_num(arr, nan=0.0).flat[0])

        output_dict = {
            key: [
                _value_at_roa(getattr(self, key), roa)
                for roa in self.roa_eval
            ]
            for key in self.output_vars
        }

        std_dict = {
            key: [self.sigma * abs(output_dict[key][i]) for i in range(len(self.roa_eval))]
            for key in self.output_vars
        }

        self.output_dict = output_dict
        self.std_dict = std_dict

        # reset state_vars_extracted flag at end of each evaluation
        self.state_vars_extracted = False

        return output_dict, std_dict

    
    def run_on_platform(
        self,
        state: "PlasmaState",
        platform: Union["PlatformManager", Dict[str, Any]],
        work_dir: Optional[Path] = None,
        model_name: str = "transport_model",
        cleanup: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run transport model evaluation on a platform (local or remote).
        
        This method serializes the model and state, sends them to the platform,
        executes the evaluation, and retrieves results.
        
        Parameters
        ----------
        state : PlasmaState
            Plasma state to evaluate
        platform : PlatformManager or dict
            Platform configuration and manager
        work_dir : Path, optional
            Working directory (default: temporary)
        model_name : str
            Name for this model run
        cleanup : bool
            If True, remove remote scratch directory after completion
        
        Returns
        -------
        output_dict : dict
            Model outputs (e.g., fluxes)
        std_dict : dict
            Model output uncertainties
        """
        from tools.io import PlatformManager
        
        # Import here to avoid circular imports
        if isinstance(platform, dict):
            platform = PlatformManager(platform)
        elif not isinstance(platform, PlatformManager):
            raise TypeError("platform must be PlatformManager or dict")
        
        # Use temporary directory if not specified
        if work_dir is None:
            import tempfile
            work_dir = Path(tempfile.mkdtemp(prefix="transport_"))
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Prepare local files
            model_file = work_dir / "model.pkl"
            state_file = work_dir / "state.pkl"
            config_file = work_dir / "config.json"
            output_file = work_dir / "output.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(self, f)
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            
            config = {
                "model_file": "model.pkl",
                "state_file": "state.pkl",
                "output_file": "output.pkl",
            }
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Stage 2: Transfer to platform
            remote_work_dir = platform.platform.get_scratch_path() / model_name
            platform.stage_inputs(work_dir, remote_work_dir)
            
            # Stage 3: Execute on platform
            script_path = remote_work_dir / "run_transport.py"
            python_script = self._generate_execution_script()
            
            if platform.platform.is_local():
                with open(script_path, 'w') as f:
                    f.write(python_script)
            else:
                import tempfile as tmp
                with tmp.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(python_script)
                    temp_script = f.name
                platform.file_manager.upload_file(Path(temp_script), script_path)
                import os
                os.remove(temp_script)
            
            # Make executable and run
            platform.executor.execute(f"chmod +x {script_path}", check=False)
            returncode, stdout, stderr = platform.executor.execute(
                f"cd {remote_work_dir} && python run_transport.py",
                check=True,
            )
            
            if returncode != 0:
                raise RuntimeError(f"Transport model execution failed:\n{stderr}")
            
            # Stage 4: Retrieve results
            platform.retrieve_outputs(remote_work_dir, work_dir, ["output.pkl", "*.log"])
            
            # Stage 5: Load results
            with open(output_file, 'rb') as f:
                output_dict, std_dict = pickle.load(f)
            
            return output_dict, std_dict
        
        finally:
            if cleanup and not platform.platform.is_local():
                platform.file_manager.remove_directory(remote_work_dir)
            platform.cleanup()
    
    @staticmethod
    def _generate_execution_script() -> str:
        """Generate Python script to execute transport model on platform."""

        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            with open(config['model_file'], 'rb') as f:
                model = pickle.load(f)
            
            with open(config['state_file'], 'rb') as f:
                state = pickle.load(f)
            
            # Run the model
            output_dict, std_dict = model._evaluate_single(state)
            
            # Save results
            with open(config['output_file'], 'wb') as f:
                pickle.dump((output_dict, std_dict), f)
            
            print("Transport model execution completed successfully")
            sys.exit(0)

        except Exception as e:
            with open('error.log', 'w') as f:
                f.write(f"Transport model execution failed:\\n{traceback.format_exc()}")
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

class Fingerprints(TransportBase):
    """
    Critical gradient fingerprints transport model.
    
    Implements simplified physics-based model combining:
    - ITG (Ion Temperature Gradient) turbulence
    - ETG (Electron Temperature Gradient) turbulence  
    - KBM (Kinetic Ballooning Mode) turbulence
    - Neoclassical transport
    
    Ported from TRANSPORTmodels.fingerprints
    """
    
    def __init__(self, options: dict):
        """
        Parameters
        ----------
        output : str
            'all', 'turb', or 'neo'
        ITG_lcorr : float
            ITG correlation length [m]
        ExBon : bool
            Include ExB shear suppression
        non_local : bool
            Use non-local closure for ITG/KBM
        """
        super().__init__(options)
        self.ITG_lcorr = self.options.get('ITG_lcorr', 0.1)
        self.ExBon = self.options.get('ExBon', True)
        self.non_local = self.options.get('non_local', False)
        self.labels = ["Ge","Gi", "Ce", "Ci", "Pe", "Pi"]
        self.modes = self.options.get('modes', 'all') # ['neo','turb','ITG','ETG','KBM']
        self.ExB_source = self.options.get('ExB_source', 'model')  # 'model' | 'state-pol' | 'state-both'
        self.ExB_scale = float(self.options.get('ExB_scale', 1.0))
    
    def _evaluate_single(self, state) -> Dict[str, np.ndarray]:
        """Compute and store turbulent and neoclassical fluxes on state.transport.* and return labeled powers.

        Returns
        -------
        Dict[str, np.ndarray]
            {"Pe": P_e [MW], "Pi": P_i [MW]} based on edge flux times edge surface area.
        """

        if self.state_vars_extracted is False:
            self._extract_from_state(state)

        if self.ExBon:
            exb_src = getattr(self, 'ExB_source', self.options.get('ExB_source', 'model'))
            if exb_src == 'state':
                gamma_ExB = getattr(state, 'gamma_exb_norm', np.zeros_like(self.x))
            else:
                V_ExB = self.rhostar*self.dpi_dx/self.ni
                gamma_ExB = self.rhostar*self.d2pi_dx2/self.ni + self.aLni*V_ExB
            gamma_ExB = self.ExB_scale * gamma_ExB
        else:
            gamma_ExB = 0*self.x

        # Neoclassical transport (analytic or NEO)
        if self.modes == 'neo' or self.modes == 'all':
            Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(state)
        
        # Critical gradients
        RLTi_crit = np.maximum((4.0 / 3.0) * (1.0 + self.Ti / self.Te), 0.8 * self.aLne * self.aspect_ratio)
        RLTe_crit = np.maximum((4.0 / 3.0) * (1.0 + self.Te / self.Ti), 1.4 * self.aLne * self.aspect_ratio)
        
        # Turbulent transport (ITG)
        ky_ITG = 0.3

        if self.non_local:
            # Non-local closure with correlation length
            aLTi_eff = np.maximum(self.aLTi - RLTi_crit / self.aspect_ratio, 0.0)
            gamma_ITG = ky_ITG * (aLTi_eff / self.ITG_lcorr)**0.5
        else:
            gamma_ITG = ky_ITG * np.maximum(self.aLTi - RLTi_crit / self.aspect_ratio, 0.0)**0.5

        gamma_eff = np.maximum(gamma_ITG - abs(gamma_ExB), 0.)
        I_ITG = (gamma_eff / ky_ITG**2)**2
        chi_ITG = I_ITG * (self.Ti**1.5)
        Gamma_ITG = 0.1*self.f_trap * chi_ITG * (-self.dne_dx - 0.25 * self.ne * self.aLTe / self.a)
        Qi_ITG = -self.ne * chi_ITG * self.dTi_dx + 1.5 * self.Ti * Gamma_ITG
        Qe_ITG = -self.ne * self.f_trap * chi_ITG * self.dTe_dx + 1.5 * self.Te * Gamma_ITG
        
        # ETG transport
        z_ETG = np.maximum(self.aLTe - RLTe_crit / self.aspect_ratio, 0.0) / np.maximum(self.aLne, 1e-12)
        chi_ETG = (1.0 / 60.0) * 1.5 * (self.Te**1.5) * self.aLTe * z_ETG
        Qe_ETG = -self.ne * chi_ETG * self.dTe_dx
        
        # KBM transport (simplified)
        ky_KBM = 0.1
        alpha_crit = 2.0
        RLp_crit = alpha_crit / (np.maximum(self.beta, 1e-12) * np.maximum(self.q, 1e-9)**2)

        if self.non_local:
            aLp_eff = np.maximum(self.aLpe - RLp_crit / self.aspect_ratio, 0.0)
            gamma_KBM = ky_KBM * (aLp_eff / self.ITG_lcorr)**0.5
        else:
            gamma_KBM = ky_KBM * np.maximum(self.aLpe - RLp_crit / self.aspect_ratio, 0.0)**0.5
        
        I_KBM = (gamma_KBM / ky_KBM**2)**2
        chi_KBM = I_KBM * (self.Ti**1.5)
        Gamma_KBM = 0.1*-chi_KBM * self.dne_dx
        Qi_KBM = -self.ne * chi_KBM * self.dTi_dx + 1.5 * self.Ti * Gamma_KBM
        Qe_KBM = -self.ne * chi_KBM * self.dTe_dx + 1.5 * self.Te * Gamma_KBM
        
        # Total turbulent
        if self.modes=='all':
            Gamma_turb = Gamma_ITG + Gamma_KBM
            Qi_turb = Qi_ITG + Qi_KBM
            Qe_turb = Qe_ITG + Qe_ETG + Qe_KBM
        if self.modes=='ITG':
            Gamma_turb = Gamma_ITG
            Qi_turb = Qi_ITG
            Qe_turb = Qe_ITG
        if self.modes=='ETG':
            Gamma_turb = 0*self.x
            Qi_turb = 0*self.x
            Qe_turb = Qe_ETG
        if self.modes=='KBM':
            Gamma_turb = Gamma_KBM
            Qi_turb = Qi_KBM
            Qe_turb = Qe_KBM
        if self.modes=='neo':
            Gamma_turb = 0*self.x
            Qi_turb = 0*self.x
            Qe_turb = 0*self.x

        return self._assemble_fluxes(
            state,
            Gamma_turb=Gamma_turb,
            Gamma_neo=Gamma_neo,
            Qi_turb=Qi_turb,
            Qi_neo=Qi_neo,
            Qe_turb=Qe_turb,
            Qe_neo=Qe_neo,
            model_label="Fingerprints",
        )


class Tglf(TransportBase):
    """Direct TGLF transport model using $GACODE_ROOT (no mitim-fusion dependency).

    Loads template from src/tools/gacode_templates/input.tglf and updates with
    user-provided options from TGLF_options dict.
    
    Configuration
    --------------
    transport:
      class: transport.Tglf
      args:
        TGLF_options:
          settings:         # Override any input.tglf parameter
            SAT_RULE: 2
            USE_BPER: 0
          multipliers:      # Scale parameters by factor
            EPSB: 0.5
          keep_files: minimal    # 'minimal'|'all'|'none'
          cores: 4
        roa_eval: [0.88, 0.94, 1.0]
        output_vars: ['Ge', 'Qi', 'Qe']
    """

    def __init__(self, options: dict):
        super().__init__(options)
        
        # Extract TGLF-specific options from unified TGLF_options dict
        tglf_opts = self.options.get("TGLF_options", {})
        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.work_dir = workflow_work_dir / "tglf"
        self.settings = tglf_opts.get("settings", {})
        self.multipliers = tglf_opts.get("multipliers", {})
        self.keep_files = tglf_opts.get("keep_files", "minimal")

    def _write_tglf_inputs(self, state, rho: float, rho_dir: Path, input_gacode: Path) -> Path:
        """Create input.tglf from controls + locpargen + user settings.
        
        Uses GEOMETRY_FLAG=1 (Miller) so s-alpha parameters are ignored.
        """
        rho_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Run locpargen to get state-specific input file
        loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
        locpargen_file = _pick_locpargen_file(loc_files, "tglf")
        
        if locpargen_file is None:
            raise RuntimeError(f"locpargen did not produce input.tglf.locpargen at rho={rho}")
        
        # Step 2: Load controls and merge with locpargen data
        controls_lines = _load_controls("tglf")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            self.settings,
            self.multipliers
        )
        
        # Step 3: Write final input file
        input_tglf = rho_dir / "input.tglf"
        input_tglf.write_text("".join(lines))
        
        return input_tglf

    def _read_tglf_gbflux(self, path: Path, rho: float) -> Tuple[float, float, float]:
        """Read TGLF out.tglf.gbflux file.
        
        File format: [Gamma_e, Q_e, Pi_e, S_e, Gamma_i1, Q_i1, Pi_i1, S_i1, Gamma_i2, ...]
        Reshape to (4, n_species) where:
        - Row 0: Particle fluxes [Gamma_e, Gamma_i1, Gamma_i2, ...]
        - Row 1: Heat fluxes [Q_e, Q_i1, Q_i2, ...]
        - Row 2: Momentum fluxes [Pi_e, Pi_i1, Pi_i2, ...]
        - Row 3: Exchange terms [S_e, S_i1, S_i2, ...]
        """
        if not path.exists():
            run_log = path.parent / "out.tglf.run"
            details = ""
            if run_log.exists():
                details = f"\nTGLF run log:\n{run_log.read_text().strip()}"
            raise RuntimeError(f"TGLF output missing: {path}{details}")
        
        # Read flux data and reshape to (4, n_species)
        data = _read_numeric_file(path)
        if data.size % 4 != 0:
            raise RuntimeError(f"TGLF gbflux size {data.size} not divisible by 4")
        n_species = data.size // 4
        data = data.reshape((4, n_species))
        
        # Extract electron quantities (column 0)
        Ge = float(data[0, 0])  # Gamma_electron
        Qe = float(data[1, 0])  # Q_electron
        
        # Sum heat flux over all ion species (columns 1:)
        Qi = float(np.sum(data[1, 1:]))
        
        # Convert gyroBohm units to physical units
        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge, Qi, Qe)
        
        return Ge, Qi, Qe

    def _evaluate_single(self, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._extract_from_state(state)

        if self.roa_eval is None:
            self.roa_eval = state.roa
        roa_eval = np.atleast_1d(self.roa_eval)

        tglf_exec = _resolve_gacode_executable("tglf", self.options)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        Gamma_turb = np.zeros_like(roa_eval, dtype=float)
        Qi_turb = np.zeros_like(roa_eval, dtype=float)
        Qe_turb = np.zeros_like(roa_eval, dtype=float)

        input_gacode = self.work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        roa_eval_list = list(roa_eval)
        max_workers = min(len(roa_eval_list), max(1, int(self.n_parallel)))

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_tglf_single_rho,
                        rho,
                        state,
                        self.work_dir,
                        input_gacode,
                        tglf_exec,
                        self.settings,
                        self.multipliers,
                        self.n_threads,
                        self.keep_files,
                    ): rho
                    for rho in roa_eval_list
                }

                for future in as_completed(futures):
                    rho = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        raise RuntimeError(f"TGLF failed at rho={rho:.3f}: {str(e)}")
        else:
            for rho in roa_eval_list:
                _run_tglf_single_rho(
                    rho,
                    state,
                    self.work_dir,
                    input_gacode,
                    tglf_exec,
                    self.settings,
                    self.multipliers,
                    self.n_threads,
                    self.keep_files,
                )

        for i, rho in enumerate(roa_eval_list):
            rho_dir = self.work_dir / f"rho_{rho:.3f}"
            out_path = rho_dir / "out.tglf.gbflux"
            Ge, Qi, Qe = self._read_tglf_gbflux(out_path, rho)
            Gamma_turb[i] = Ge
            Qi_turb[i] = Qi
            Qe_turb[i] = Qe

            if self.keep_files == "none":
                shutil.rmtree(rho_dir, ignore_errors=True)

        # Compute neoclassical fluxes if requested
        Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(state)

        return self._assemble_fluxes(
            state,
            Gamma_turb=Gamma_turb,
            Gamma_neo=Gamma_neo,
            Qi_turb=Qi_turb,
            Qi_neo=Qi_neo,
            Qe_turb=Qe_turb,
            Qe_neo=Qe_neo,
            model_label="TGLF",
        )


class Cgyro(TransportBase):
    """Direct CGYRO transport model using $GACODE_ROOT (no mitim-fusion dependency).
    
    Loads template from src/tools/gacode_templates/input.cgyro and updates with
    user-provided options from CGYRO_options dict.
    
    Configuration
    --------------
    transport:
      class: transport.Cgyro
      args:
        CGYRO_options:
          settings:         # Override any input.cgyro parameter
            N_TOROIDAL: 4
            NONLINEAR_FLAG: 1
          multipliers:      # Scale parameters by factor
            TRATE: 1.5
          auto:             # Auto-scale based on physics
            enabled: true
            n_toroidal: 4
          keep_files: minimal     # 'minimal'|'all'|'none'
          cores: 8
        roa_eval: [0.88, 0.94, 1.0]
        output_vars: ['Ge', 'Qi', 'Qe']
    """

    def __init__(self, options: dict):
        super().__init__(options)
        
        # Extract CGYRO-specific options from unified CGYRO_options dict
        cgyro_opts = self.options.get("CGYRO_options", {})
        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.work_dir = workflow_work_dir / "cgyro"
        self.settings = cgyro_opts.get("settings", {})
        self.multipliers = cgyro_opts.get("multipliers", {})
        self.keep_files = cgyro_opts.get("keep_files", "minimal")
        self.auto = cgyro_opts.get("auto", {})
        self.read_mode = cgyro_opts.get("read_mode", "pygacode")

    def _auto_cgyro_settings(self, state, updates: Dict[str, Any]) -> Dict[str, Any]:
        if not self.auto or not self.auto.get("enabled", True):
            return updates

        n_species = int(np.asarray(state.ni).shape[1]) if np.asarray(state.ni).ndim == 2 else 1

        def _as_scalar(value, default):
            if value is None:
                return float(default)
            arr = np.asarray(value)
            if arr.ndim == 0:
                return float(arr)
            return float(arr.ravel()[-1])

        rhostar = _as_scalar(getattr(state, "rhostar", None), 1e-3)
        betae = _as_scalar(getattr(state, "betae", None), 0.0)
        nuei = _as_scalar(getattr(state, "nuei", None), 0.0)

        def set_if_absent(key, value):
            if key not in updates:
                updates[key] = value

        # Radial resolution
        n_radial = int(np.clip(8 + 4 * np.log10(1.0 / max(rhostar, 1e-4)), 8, 24))
        set_if_absent("N_RADIAL", n_radial)

        # Poloidal resolution
        n_theta = int(np.clip(24 + 8 * n_species, 24, 96))
        set_if_absent("N_THETA", n_theta)

        # Parallel velocity grid
        n_xi = 24 if nuei > 0.5 else 16
        set_if_absent("N_XI", n_xi)

        # Toroidal modes
        set_if_absent("N_TOROIDAL", int(self.auto.get("n_toroidal", 1)))

        # Field count
        if self.auto.get("electromagnetic", None) is True or betae > 0.01:
            set_if_absent("N_FIELD", 3)
        elif self.auto.get("electromagnetic", None) is False:
            set_if_absent("N_FIELD", 1)

        # Box size and ky
        if "box_size" in self.auto:
            set_if_absent("BOX_SIZE", self.auto.get("box_size", 1))
        else:
            set_if_absent("BOX_SIZE", 2 if rhostar < 5e-4 else 1)

        if "ky" in self.auto:
            set_if_absent("KY", self.auto.get("ky", 0.3))
        else:
            if betae > 0.02:
                set_if_absent("KY", 0.2)
            elif betae < 0.005:
                set_if_absent("KY", 0.35)
            else:
                set_if_absent("KY", 0.3)

        return updates

    def _write_cgyro_inputs(self, state, rho: float, rho_dir: Path, input_gacode: Path) -> Path:
        """Create input.cgyro from controls + locpargen + user settings."""
        rho_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Run locpargen to get state-specific input file
        loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
        locpargen_file = _pick_locpargen_file(loc_files, "cgyro")
        
        if locpargen_file is None:
            raise RuntimeError(f"locpargen did not produce input.cgyro.locpargen at rho={rho}")
        
        # Step 2: Apply auto-scaling to user settings
        settings = dict(self.settings) if self.settings else {}
        settings = self._auto_cgyro_settings(state, settings)
        
        # Step 3: Load controls and merge with locpargen data
        controls_lines = _load_controls("cgyro")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            settings,
            self.multipliers
        )
        
        # Step 4: Write final input file
        input_cgyro = rho_dir / "input.cgyro"
        input_cgyro.write_text("".join(lines))
        
        return input_cgyro

    def _read_cgyro_fluxes_json(self, path: Path, rho: float) -> Tuple[float, float, float]:
        """Read CGYRO flux JSON file.
        
        JSON contains pre-summed Ge, Qi, Qe from CGYRO's post-processing.
        Returns fluxes in physical units (1e19/m^3/s for Ge, MW/m^3 for Qi/Qe).
        """
        if not path.exists():
            raise RuntimeError(f"CGYRO flux JSON not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        rho_list = data.get("rho", [])
        if rho_list:
            idx = int(np.argmin(np.abs(np.asarray(rho_list, dtype=float) - float(rho))))
        else:
            idx = 0
        Qe = data.get("Qe", data.get("Qe_mean", []))
        Qi = data.get("Qi", data.get("Qi_mean", []))
        Ge = data.get("Ge", data.get("Ge_mean", []))
        
        Ge_gb = float(np.asarray(Ge)[idx])
        Qi_gb = float(np.asarray(Qi)[idx])
        Qe_gb = float(np.asarray(Qe)[idx])
        
        # Convert gyroBohm units to physical units
        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge_gb, Qi_gb, Qe_gb)
        
        return Ge, Qi, Qe

    def _read_cgyro_fluxes_pygacode(self, rho_dir: Path, rho: float) -> Tuple[float, float, float]:
        """Read CGYRO fluxes using pygacode.
        
        Uses pygacode's cgyrodata_plot to extract and sum fluxes by species.
        Returns fluxes in physical units (1e19/m^3/s for Ge, MW/m^3 for Qi/Qe).
        """
        try:
            from pygacode.cgyro.data_plot import cgyrodata_plot
        except ModuleNotFoundError:
            gacode_root = os.environ.get("GACODE_ROOT")
            if gacode_root:
                candidate = Path(gacode_root) / "f2py"
                if candidate.exists():
                    sys.path.append(str(candidate))
            try:
                from pygacode.cgyro.data_plot import cgyrodata_plot
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "CGYRO reading requires pygacode. "
                    "Install pygacode, set GACODE_ROOT, or set cgyro_read_mode='json'."
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                "CGYRO reading requires pygacode. "
                "Install pygacode, set GACODE_ROOT, or set cgyro_read_mode='json'."
            ) from exc

        cgyrodata = cgyrodata_plot(f"{rho_dir.resolve()}{os.sep}")
        cgyrodata.getflux(cflux="auto")
        cgyrodata.getnorm("elec")
        cgyrodata.getgeo()
        cgyrodata.getxflux()

        z = np.asarray(getattr(cgyrodata, "z", []))
        if z.size == 0:
            raise RuntimeError("CGYRO output missing species charges (z)")
        electron_idx = int(np.where(z == -1)[0][0]) if np.any(z == -1) else -1
        ion_idxs = [i for i in range(len(z)) if i != electron_idx]

        ky_flux = cgyrodata.ky_flux
        fields = ["phi", "apar", "bpar"][: cgyrodata.n_field]

        def sum_fields(species_idx, moment_idx):
            total = None
            for i_field, field in enumerate(fields):
                block = ky_flux[species_idx, moment_idx, i_field, :, :]
                total = block if total is None else total + block
            return total

        def time_avg(arr):
            if arr.ndim >= 2:
                arr = arr.sum(axis=-2)
            if arr.ndim == 1:
                n = max(1, int(0.5 * arr.size))
                return float(np.nanmean(arr[-n:]))
            return float(np.nanmean(arr))

        Qe_gb = time_avg(sum_fields(electron_idx, 1))
        Ge_gb = time_avg(sum_fields(electron_idx, 0))
        Qi_gb = 0.0
        for i in ion_idxs:
            Qi_gb += time_avg(sum_fields(i, 1))
        
        # Convert gyroBohm units to physical units
        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge_gb, Qi_gb, Qe_gb)

        return Ge, Qi, Qe

    def _evaluate_single(self, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._extract_from_state(state)

        if self.roa_eval is None:
            self.roa_eval = state.roa
        roa_eval = np.atleast_1d(self.roa_eval)

        cgyro_exec = _resolve_gacode_executable("cgyro", self.options)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        Gamma_turb = np.zeros_like(roa_eval, dtype=float)
        Qi_turb = np.zeros_like(roa_eval, dtype=float)
        Qe_turb = np.zeros_like(roa_eval, dtype=float)

        input_gacode = self.work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        for i, rho in enumerate(roa_eval):
            rho_dir = self.work_dir / f"rho_{rho:.3f}"
            self._write_cgyro_inputs(state, rho, rho_dir, input_gacode)

            sim_name = rho_dir.name
            cmd = [cgyro_exec, "-p", str(self.work_dir), "-e", sim_name, "-n", str(self.n_parallel), "-nomp", str(self.n_threads)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=_with_python_in_path(),
            )
            if result.returncode != 0:
                raise RuntimeError(f"CGYRO failed at rho={rho}: {result.stderr.strip()}")

            if self.read_mode == "json":
                json_path = rho_dir / "fluxes_turb.json"
                Ge, Qi, Qe = self._read_cgyro_fluxes_json(json_path, rho)
            else:
                Ge, Qi, Qe = self._read_cgyro_fluxes_pygacode(rho_dir, rho)

            Gamma_turb[i] = Ge
            Qi_turb[i] = Qi
            Qe_turb[i] = Qe

            if self.keep_files == "none":
                shutil.rmtree(rho_dir, ignore_errors=True)

        Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(state)

        return self._assemble_fluxes(
            state,
            Gamma_turb=Gamma_turb,
            Gamma_neo=Gamma_neo,
            Qi_turb=Qi_turb,
            Qi_neo=Qi_neo,
            Qe_turb=Qe_turb,
            Qe_neo=Qe_neo,
            model_label="CGYRO",
        )


class Qlgyro(TransportBase):
    """Direct QLGYRO transport model using $GACODE_ROOT (no mitim-fusion dependency).
    
    QLGYRO is a quasi-linear wrapper around CGYRO. Loads input.cgyro and input.qlgyro
    templates from src/tools/gacode_templates/ and updates with QLGYRO_options dict.
    
    Configuration
    --------------
    transport:
      class: transport.Qlgyro
      args:
        QLGYRO_options:
          settings:         # Override any input.qlgyro parameter
            N_RADIAL: 3
          keep_files: minimal     # 'minimal'|'all'|'none'
          cores: 1
        CGYRO_options:     # Used for CGYRO input file within QLGYRO
          settings:
            N_TOROIDAL: 4
        roa_eval: [0.88, 0.94, 1.0]
        output_vars: ['Ge', 'Qi', 'Qe']
    """

    def __init__(self, options: dict):
        super().__init__(options)
        
        # Extract QLGYRO-specific options
        qlgyro_opts = self.options.get("QLGYRO_options", {})
        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.work_dir = workflow_work_dir / "qlgyro"
        self.settings = qlgyro_opts.get("settings", {})
        self.keep_files = qlgyro_opts.get("keep_files", "minimal")
        self.wait_timeout = float(qlgyro_opts.get("wait_timeout", 3600.0))
        self.wait_poll = float(qlgyro_opts.get("wait_poll", 5.0))

        # CGYRO options are also used for input.cgyro generation within QLGYRO
        cgyro_opts = self.options.get("CGYRO_options", {})
        self.cgyro_settings = cgyro_opts.get("settings", {})
        self.cgyro_auto = cgyro_opts.get("auto", {})

    def _auto_cgyro_settings(self, state, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-scale CGYRO settings based on physics (same as Cgyro class)."""
        if not self.cgyro_auto or not self.cgyro_auto.get("enabled", True):
            return updates

        n_species = int(np.asarray(state.ni).shape[1]) if np.asarray(state.ni).ndim == 2 else 1

        def _as_scalar(value, default):
            if value is None:
                return float(default)
            arr = np.asarray(value)
            if arr.ndim == 0:
                return float(arr)
            return float(arr.ravel()[-1])

        rhostar = _as_scalar(getattr(state, "rhostar", None), 1e-3)
        betae = _as_scalar(getattr(state, "betae", None), 0.0)
        nuei = _as_scalar(getattr(state, "nuei", None), 0.0)

        def set_if_absent(key, value):
            if key not in updates:
                updates[key] = value

        n_radial = int(np.clip(8 + 4 * np.log10(1.0 / max(rhostar, 1e-4)), 8, 24))
        set_if_absent("N_RADIAL", n_radial)

        n_theta = int(np.clip(24 + 8 * n_species, 24, 96))
        set_if_absent("N_THETA", n_theta)

        n_xi = 24 if nuei > 0.5 else 16
        set_if_absent("N_XI", n_xi)

        set_if_absent("N_TOROIDAL", int(self.cgyro_auto.get("n_toroidal", 1)))

        if self.cgyro_auto.get("electromagnetic", None) is True or betae > 0.01:
            set_if_absent("N_FIELD", 3)
        elif self.cgyro_auto.get("electromagnetic", None) is False:
            set_if_absent("N_FIELD", 1)

        if "box_size" in self.cgyro_auto:
            set_if_absent("BOX_SIZE", self.cgyro_auto.get("box_size", 1))
        else:
            set_if_absent("BOX_SIZE", 2 if rhostar < 5e-4 else 1)

        if "ky" in self.cgyro_auto:
            set_if_absent("KY", self.cgyro_auto.get("ky", 0.3))
        else:
            if betae > 0.02:
                set_if_absent("KY", 0.2)
            elif betae < 0.005:
                set_if_absent("KY", 0.35)
            else:
                set_if_absent("KY", 0.3)

        return updates

    def _write_cgyro_inputs(self, state, rho: float, rho_dir: Path, input_gacode: Path) -> Path:
        """Create input.cgyro for QLGYRO from controls + locpargen + user settings."""
        rho_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Run locpargen to get state-specific input file
        loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
        locpargen_file = _pick_locpargen_file(loc_files, "cgyro")
        
        if locpargen_file is None:
            raise RuntimeError(f"locpargen did not produce input.cgyro.locpargen at rho={rho}")
        
        # Step 2: Apply auto-scaling to user settings
        settings = dict(self.cgyro_settings) if self.cgyro_settings else {}
        settings = self._auto_cgyro_settings(state, settings)
        
        # Step 3: Load controls and merge with locpargen data
        controls_lines = _load_controls("cgyro")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            settings,
            {}  # No multipliers for QLGYRO's CGYRO
        )
        
        # Step 4: Write final input file
        input_cgyro = rho_dir / "input.cgyro"
        input_cgyro.write_text("".join(lines))
        
        return input_cgyro

    def _write_qlgyro_input(self, rho_dir: Path) -> Path:
        """Create input.qlgyro from controls + user settings (no locpargen)."""
        input_qlgyro = rho_dir / "input.qlgyro"

        # Load controls and apply user settings
        lines = _load_controls("qlgyro")
        
        if self.settings:
            lines = _update_namelist_lines(lines, self.settings)

        # Ensure n_parallel reflects current configuration
        lines = _update_namelist_lines(lines, {"n_parallel": self.n_parallel})

        input_qlgyro.write_text("".join(lines))
        return input_qlgyro

    def _read_qlgyro_gbflux(self, path: Path, rho: float) -> Tuple[float, float, float]:
        """Read QLGYRO out.qlgyro.gbflux file.
        
        File format: [Gamma_e, Q_e, Pi_e, S_e, Gamma_i1, Q_i1, Pi_i1, S_i1, Gamma_i2, ...]
        Reshape to (4, n_species) where:
        - Row 0: Particle fluxes [Gamma_e, Gamma_i1, Gamma_i2, ...]
        - Row 1: Heat fluxes [Q_e, Q_i1, Q_i2, ...]
        - Row 2: Momentum fluxes [Pi_e, Pi_i1, Pi_i2, ...]
        - Row 3: Exchange terms [S_e, S_i1, S_i2, ...]
        """
        if not path.exists():
            raise RuntimeError(f"QLGYRO output missing: {path}")
        
        # Read flux data and reshape to (4, n_species)
        data = _read_numeric_file(path)
        if data.size % 4 != 0:
            raise RuntimeError(f"QLGYRO gbflux size {data.size} not divisible by 4")
        n_species = data.size // 4
        data = data.reshape((4, n_species))
        
        # Extract electron quantities (column 0)
        Ge = float(data[0, 0])  # Gamma_electron
        Qe = float(data[1, 0])  # Q_electron
        
        # Sum heat flux over all ion species (columns 1:)
        Qi = float(np.sum(data[1, 1:]))
        
        # Convert gyroBohm units to physical units
        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge, Qi, Qe)
        
        return Ge, Qi, Qe

    def _wait_for_qlgyro_gbflux(self, rho_dir: Path) -> None:
        gbflux_path = rho_dir / "out.qlgyro.gbflux"
        status_path = rho_dir / "out.qlgyro.status"
        run_path = rho_dir / "out.qlgyro.run"

        start = time.time()
        last_status = ""
        last_run = ""
        while time.time() - start < self.wait_timeout:
            if gbflux_path.exists():
                return
            if status_path.exists():
                last_status = status_path.read_text().strip()
            if run_path.exists():
                last_run = run_path.read_text().strip()
            time.sleep(self.wait_poll)

        details = []
        if last_status:
            details.append(f"Last out.qlgyro.status:\n{last_status}")
        if last_run:
            details.append(f"Last out.qlgyro.run:\n{last_run}")
        detail_text = "\n\n".join(details) if details else "(no status or run log found)"
        raise RuntimeError(
            f"QLGYRO timed out waiting for out.qlgyro.gbflux in {rho_dir}\n{detail_text}"
        )

    def _evaluate_single(self, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._extract_from_state(state)

        if self.roa_eval is None:
            self.roa_eval = state.roa
        roa_eval = np.atleast_1d(self.roa_eval)

        qlgyro_exec = _resolve_gacode_executable("qlgyro", self.options)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        input_gacode = self.work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        Gamma_turb = np.zeros_like(roa_eval, dtype=float)
        Qi_turb = np.zeros_like(roa_eval, dtype=float)
        Qe_turb = np.zeros_like(roa_eval, dtype=float)

        for i, rho in enumerate(roa_eval):
            rho_dir = self.work_dir / f"rho_{rho:.3f}"
            rho_dir.mkdir(parents=True, exist_ok=True)
            
            # Write input.cgyro using locpargen approach (like Tglf/Cgyro)
            self._write_cgyro_inputs(state, rho, rho_dir, input_gacode)
            
            # Write input.qlgyro from controls + user settings
            self._write_qlgyro_input(rho_dir)

            # QLGYRO init mode to parse both input.cgyro and input.qlgyro
            # Use relative path with cwd=self.work_dir to avoid qlgyro path concatenation bugs
            init_cmd = [qlgyro_exec, "-i", rho_dir.name]
            init_res = subprocess.run(
                init_cmd,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                env=_with_python_in_path(),
            )
            if init_res.returncode != 0:
                raise RuntimeError(f"QLGYRO init failed at rho={rho}: {init_res.stderr.strip()}")

            # Run QLGYRO
            run_cmd = [qlgyro_exec, "-e", rho_dir.name, "-n", str(self.n_parallel), "-nomp", str(self.n_threads)]
            run_res = subprocess.run(
                run_cmd,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                env=_with_python_in_path(),
            )
            if run_res.returncode != 0:
                raise RuntimeError(f"QLGYRO failed at rho={rho}: {run_res.stderr.strip()}")

            self._wait_for_qlgyro_gbflux(rho_dir)

            out_path = rho_dir / "out.qlgyro.gbflux"
            Ge, Qi, Qe = self._read_qlgyro_gbflux(out_path, rho)
            Gamma_turb[i] = Ge
            Qi_turb[i] = Qi
            Qe_turb[i] = Qe

            if self.keep_files == "none":
                shutil.rmtree(rho_dir, ignore_errors=True)

        Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(state)

        return self._assemble_fluxes(
            state,
            Gamma_turb=Gamma_turb,
            Gamma_neo=Gamma_neo,
            Qi_turb=Qi_turb,
            Qi_neo=Qi_neo,
            Qe_turb=Qe_turb,
            Qe_neo=Qe_neo,
            model_label="QLGYRO",
        )


class Fixed(TransportBase):
    """
    Fixed diffusivity/conductivity model for testing.
    
    Useful for debugging solver logic without transport model overhead.
    """
    
    def __init__(self, D: float = 1.0, chi: float = 1.0):
        """
        Parameters
        ----------
        D : float
            Particle diffusivity [m^2/s]
        chi : float
            Thermal diffusivity [m^2/s]
        """
        self.D = D
        self.chi = chi
    
    def _evaluate_single(self, state) -> Dict[str, np.ndarray]:
        """Compute fluxes using fixed diffusivities and store on state.transport.

        Returns
        -------
        Dict[str, np.ndarray]
            {"Pe": P_e [MW], "Pi": P_i [MW]} computed from edge flux times edge area.
        """
        x = getattr(state, 'roa', state.r / state.a)
        n_roa = len(x)
        n_species = np.asarray(state.ni).shape[1] if np.asarray(state.ni).ndim == 2 else 1
        
        # Compute gradients
        dne_dr = np.gradient(state.ne, state.r)
        dte_dr = np.gradient(state.te, state.r)
        dti_dr = np.gradient(state.ti, state.r)
        
        # Particle fluxes
        Ge = -self.D * dne_dr
        Gi = np.zeros((n_roa, n_species))
        for i in range(n_species):
            dni_dr = np.gradient(state.ni[:, i] if np.asarray(state.ni).ndim == 2 else state.ni, state.r)
            Gi[:, i] = -self.D * dni_dr
        
        # Energy fluxes (convert to MW/m^2)
        # Q [MW/m^2] = -χ [m^2/s] * n [1e19 m^-3] * dT/dr [keV/m] * 1.6e-16 [J/keV]
        Qe = -self.chi * state.ne * dte_dr * 1.6e-3  # 1e19 * 1.6e-16 = 1.6e-3
        
        Qi = np.zeros((n_roa, n_species))
        for i in range(n_species):
            ni_i = state.ni[:, i] if np.asarray(state.ni).ndim == 2 else state.ni
            Qi[:, i] = -self.chi * ni_i * dti_dr * 1.6e-3

        # Store
        if not hasattr(state, 'transport'):
            class TransportContainer:
                pass
            state.transport = TransportContainer()
        tr = state.transport
        self.model = 'Fixed'
        self.Ge = Ge
        self.Gi = Gi
        self.Qe = Qe
        self.Qi = Qi

        # Edge surface area (approximate using same form as Fingerprints)
        R0 = float(getattr(state, 'R0', getattr(state, 'Rmaj', 1.0)))
        a = float(getattr(state, 'a', 1.0))
        kappa = np.asarray(getattr(state, 'kappa', np.ones_like(x)))
        aspect_ratio = R0 / max(a, 1e-9)
        dVdx = (2 * np.pi * aspect_ratio) * (2 * np.pi * x * np.sqrt((1 + kappa**2) / 2))
        surfArea = dVdx * a**2
        A_edge = float(surfArea[-1])

        # Total powers at edge
        P_e = float(np.asarray(Qe)[-1]) * A_edge
        Qi_sum = np.asarray(Qi)
        if Qi_sum.ndim == 2:
            Qi_edge = float(np.sum(Qi_sum[-1, :]))
        else:
            Qi_edge = float(Qi_sum[-1])
        P_i = Qi_edge * A_edge

        self.labels = ["Pe", "Pi"]
        return {"Pe": np.atleast_1d(P_e), "Pi": np.atleast_1d(P_i)}


class Analytic(TransportBase):
    """
    Analytic transport model for edge plasma. Effectively Fingerprints v2.

    Compute steady-state turbulent particle and heat fluxes in the closed-flux
    edge region (0.85 < r/a < 1.0) using an analytic, mode-resolved transport model.

    Physics included:
    -----------------
    - Ion-scale turbulence: ITG, TEM, KBM
    - Electron-scale turbulence: ETG (Hatch 2023 saturation)
    - Critical-gradient stiffness with geometric modification
    - ExB shear suppression and nonlinear decorrelation
    - Impurity dilution effects (trace limit)
    - Regime-dependent scaling between quasilinear and strong turbulence
    via a Kubo-number proxy

    Parameters:
    -----------

    """
    
    def __init__(self, **kwargs):
        """
        Edge Turbulent Transport Model — Parameter Definitions and Defaults
        ===================================================================

        This parameter table defines all tunable coefficients used in the
        steady-state analytic turbulent transport model for the closed-flux
        edge region (0.85 < r/a < 1.0).

        Design philosophy:
        ------------------
        - Minimize the number of free parameters
        - Each parameter corresponds to a known physical effect
        - Parameters are weakly correlated to improve optimizer robustness
        - Defaults are chosen to be machine-agnostic and order-unity

        Users are encouraged to tune *threshold* parameters before *amplitude*
        parameters when matching experimental profiles.

        --------------------------------------------------------------------
        GLOBAL / STIFFNESS PARAMETERS
        --------------------------------------------------------------------

        p_stiff : float, default = 2.0
            Stiffness exponent applied once a critical gradient is exceeded.
            Controls how strongly profiles are pinned near marginal stability.
            Typical range: 1.5 – 3.0

        --------------------------------------------------------------------
        WAVENUMBER / SPECTRAL PARAMETERS
        --------------------------------------------------------------------

        ky_rhos_ITG : float, default = 0.30
            Effective binormal wavenumber (k_y * rho_s) representing the
            ion-scale turbulence spectrum (ITG / TEM).
            Represents the spectral peak after implicit k-integration.

        ky_rhoe_ETG : float, default = 0.25
            Effective binormal wavenumber (k_y * rho_e) for electron-scale ETG
            turbulence, consistent with Hatch (2023).

        sigma_k : float, default = 0.7
            Logarithmic width of the assumed turbulence k-spectrum.
            Absorbed into transport prefactors; retained for future extensions.

        --------------------------------------------------------------------
        GEOMETRY / SHAPING MODIFIERS
        --------------------------------------------------------------------

        geom_kappa_coeff : float, default = 0.6
            Stabilization coefficient for elongation (kappa > 1).
            Captures reduced curvature drive and increased connection length.

        geom_delta_coeff : float, default = 0.3
            Stabilization coefficient for triangularity.
            Represents favorable average curvature effects near the edge.

        geom_shear_coeff : float, default = 0.4
            Destabilization coefficient for magnetic shear (s_hat).
            Higher shear enhances ballooning-type mode drive.

        --------------------------------------------------------------------
        CRITICAL GRADIENT THRESHOLDS (BASE VALUES)
        --------------------------------------------------------------------

        eta_i_crit_0 : float, default = 1.0
            Base critical ion temperature gradient (ITG threshold),
            before geometric and impurity corrections.

        eta_e_crit_0 : float, default = 1.2
            Base trapped-electron-mode (TEM) threshold for eta_e.

        RLT_e_crit_0 : float, default = 3.0
            Base critical electron temperature gradient for ETG onset,
            consistent with pedestal-ETG studies.

        alpha_crit_0 : float, default = 0.7
            Base critical normalized pressure gradient for KBM onset.

        --------------------------------------------------------------------
        SHEAR AND NONLINEAR DECORRELATION
        --------------------------------------------------------------------

        gammaE_coeff : float, default = 1.0
            Converts ExB shear rate into an effective turbulence
            decorrelation rate.

        gammaNL_coeff : float, default = 0.5
            Strength of nonlinear self-decorrelation relative to
            linear growth rate. Controls saturation strength.

        --------------------------------------------------------------------
        TURBULENCE REGIME (KUBO-BASED SCALING)
        --------------------------------------------------------------------

        kubo_alpha_min : float, default = 1.0
            Lower bound for transport scaling exponent.
            Corresponds to strong-turbulence (non-quasilinear) regime.

        kubo_alpha_max : float, default = 2.0
            Upper bound for transport scaling exponent.
            Corresponds to quasilinear transport regime.

        --------------------------------------------------------------------
        ETG MULTISCALE INTERACTION
        --------------------------------------------------------------------

        ETG_ITG_supp_coeff : float, default = 1.0
            Strength of ETG suppression by ion-scale turbulence.
            Models multiscale interaction observed in nonlinear simulations.

        ETG_prefactor : float, default = 1.0
            Overall amplitude multiplier for ETG transport.
            Should remain O(1) if thresholds are tuned correctly.

        --------------------------------------------------------------------
        IMPURITY (TRACE LIMIT) EFFECTS
        --------------------------------------------------------------------

        impurity_dilution_coeff : float, default = 0.8
            Reduces ion-scale turbulence drive due to main-ion dilution
            by trace impurities.

        impurity_threshold_coeff : float, default = 0.5
            Increases effective critical gradients in the presence of
            impurities, consistent with recent pedestal studies.

        --------------------------------------------------------------------
        NOTES
        --------------------------------------------------------------------

        - All parameters are dimensionless unless stated otherwise.
        - Defaults are intended for predictive profile modeling, not
        channel-by-channel validation.
        - Radiation, ionization, and charge-state physics must be handled
        externally (e.g., via Aurora/ADAS).
        - This table is stable across machines and should not require
        re-tuning for modest extrapolation in size or field strength.
        """

        super().__init__(kwargs)
        
        # model parameters
        self.p_stiff = self.options.get('p_stiff', 2.0)
        self.ky_rhos_ITG = self.options.get('ky_rhos_ITG', 0.30)
        self.ky_rhoe_ETG = self.options.get('ky_rhoe_ETG', 0.25)
        self.sigma_k = self.options.get('sigma_k', 0.7)
        self.geom_kappa_coeff = self.options.get('geom_kappa_coeff', 0.6)
        self.geom_delta_coeff = self.options.get('geom_delta_coeff', 0.3)
        self.geom_shear_coeff = self.options.get('geom_shear_coeff', 0.4)
        self.eta_i_crit_0 = self.options.get('eta_i_crit_0', 1.0)
        self.eta_e_crit_0 = self.options.get('eta_e_crit_0', 1.2)
        self.RLT_e_crit_0 = self.options.get('RLT_e_crit_0', 3.0)
        self.alpha_crit_0 = self.options.get('alpha_crit_0', 0.7)
        self.gammaE_coeff = self.options.get('gammaE_coeff', 1.0)
        self.gammaNL_coeff = self.options.get('gammaNL_coeff', 0.5)
        self.kubo_alpha_min = self.options.get('kubo_alpha_min', 1.0)
        self.kubo_alpha_max = self.options.get('kubo_alpha_max', 2.0)
        self.ETG_ITG_supp_coeff = self.options.get('ETG_ITG_supp_coeff', 1.0)
        self.ETG_prefactor = self.options.get('ETG_prefactor', 1.0)
        self.impurity_dilution_coeff = self.options.get('impurity_dilution_coeff', 0.8)
        self.impurity_threshold_coeff = self.options.get('impurity_threshold_coeff', 0.5)

    def _get_critical_gradients(self, state) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """
        -----------------------------------------------------------------------
        GENERAL FORM
        -----------------------------------------------------------------------

        All critical gradients follow the form:

            G_crit = G_0 * (1 + sum_i C_i * X_i)

        where:
            - G_0 is the base (machine-independent) threshold
            - X_i are dimensionless geometric or plasma parameters
            - C_i are order-unity tuning coefficients

        ExB shear, rho*, and source-dependent quantities are deliberately
        excluded, as they influence saturation rather than linear onset.

        -----------------------------------------------------------------------
        ION TEMPERATURE GRADIENT (ITG) CRITICAL THRESHOLD
        -----------------------------------------------------------------------

        eta_i_crit =
            eta_i_crit_0 * (
                1
                + C_kappa_ITG * (kappa - 1)
                + C_delta_ITG * abs(delta)
                + C_s_ITG * s_hat
                + C_beta_ITG * beta_e
                + C_Z_ITG * fZ * Zeff
            )

        Definitions:
            eta_i      = (R / L_Ti) / (R / L_n)
            kappa      = plasma elongation
            delta      = plasma triangularity (absolute value used)
            s_hat      = magnetic shear
            beta_e     = local electron beta
            fZ         = trace impurity density fraction
            Zeff       = effective charge

        Physical interpretation:
            - Elongation and triangularity reduce bad curvature drive
            - Magnetic shear increases ballooning localization
            - Electromagnetic effects weakly stabilize ITG
            - Impurities dilute main-ion drive

        -----------------------------------------------------------------------
        TRAPPED ELECTRON MODE (TEM) CRITICAL THRESHOLD
        -----------------------------------------------------------------------

        eta_e_crit =
            eta_e_crit_0 * (
                1
                + C_kappa_TEM * (kappa - 1)
                + C_delta_TEM * abs(delta)
                + C_s_TEM * s_hat
                + C_nu_TEM * nu_star
            )

        Definitions:
            eta_e   = (R / L_Te) / (R / L_n)
            nu_star = normalized electron collisionality

        Physical interpretation:
            - TEMs are sensitive to collisional detrapping
            - Geometry modifies effective trapped particle fraction
            - Impurity effects enter indirectly and are neglected here

        -----------------------------------------------------------------------
        ELECTRON TEMPERATURE GRADIENT (ETG) CRITICAL THRESHOLD
        -----------------------------------------------------------------------

        (R / L_Te)_crit =
            RLT_e_crit_0 * (
                1
                + C_kappa_ETG * (kappa - 1)
                + C_delta_ETG * abs(delta)
                + C_beta_ETG * beta_e
            )

        Definitions:
            R / L_Te = major-radius-normalized electron temperature gradient

        Physical interpretation:
            - ETG onset is stiff and geometry-dependent in the pedestal
            - Electromagnetic stabilization becomes important near the edge
            - Collisions affect saturation rather than linear onset

        -----------------------------------------------------------------------
        KINETIC BALLOONING MODE (KBM) CRITICAL THRESHOLD
        -----------------------------------------------------------------------

        alpha_crit =
            alpha_crit_0 * (
                1
                + C_s_KBM * s_hat
                + C_q_KBM * q**2
            )

        Definitions:
            alpha = -(R q^2 / B^2) * dp/dr  (normalized pressure gradient)
            q     = safety factor

        Physical interpretation:
            - KBM onset is governed primarily by electromagnetic drive
            - Magnetic shear and safety factor strongly affect stability
            - Geometry enters weakly and is omitted for robustness

        -----------------------------------------------------------------------
        NOTES AND LIMITATIONS
        -----------------------------------------------------------------------

        - Critical gradients are local and steady-state.
        - Thresholds are intended to capture *ordering and sensitivity*, not
        exact gyrokinetic stability boundaries.
        - Threshold coefficients (C_*) should be tuned before transport
        amplitude parameters.
        - Negative triangularity is stabilizing for microturbulence via abs(delta),
        while KBM remains pressure-gradient limited.

        -----------------------------------------------------------------------
        REFERENCES (GUIDING)
        -----------------------------------------------------------------------

        - Snyder et al., Nucl. Fusion (EPED model)
        - Hatch et al., NF (2023), ETG pedestal transport
        - Ashourvan et al., PRL (2024), strong turbulence regimes
        - Romanelli et al., NF, critical-gradient transport models
        """

        eta_i_crit = self.eta_i_crit_0 * (
                1
                + self.C_kappa_ITG * (state.kappa - 1)
                + self.C_delta_ITG * abs(state.delta)
                + self.C_s_ITG * state.s_hat
                + self.C_beta_ITG * state.beta_e
                + self.C_Z_ITG * state.fZ * state.Zeff
            )
        
        eta_e_crit = self.eta_e_crit_0 * (
                1
                + self.C_kappa_TEM * (state.kappa - 1)
                + self.C_delta_TEM * abs(state.delta)
                + self.C_s_TEM * state.s_hat
                + self.C_nu_TEM * state.nu_star_e
            )
        
        RLT_e_crit = self.RLT_e_crit_0 * (
                1
                + self.C_kappa_ETG * (state.kappa - 1)
                + self.C_delta_ETG * abs(state.delta)
                + self.C_beta_ETG * state.beta_e
            )
        
        alpha_crit = self.alpha_crit_0 * (
                1
                + self.C_s_KBM * state.s_hat
                + self.C_q_KBM * state.q**2
            )
        
        RLp_crit = alpha_crit / (np.maximum(state.beta_e, 1e-12) * np.maximum(state.q, 1e-9)**2)

        return eta_i_crit, eta_e_crit, RLT_e_crit, RLp_crit
    
    def _evaluate_single(self, state) -> None:
        """Compute analytic fluxes and store on state.transport.


        Formulations:

        Ion Temperature Gradient (ITG) Turbulent Transport
        --------------------------------------------------

        Ion heat diffusivity due to ion-scale ITG turbulence is modeled as:

            chi_i_ITG =
                chi_gB_i
                * f_geom_micro
                * (Δ_ITG) ** p_stiff
                * (gamma_l_ITG / gamma_decorr_ITG) ** alpha_ITG
                * f_impurity

        where:

            Δ_ITG = max(0, eta_i - eta_i_crit)

            chi_gB_i = rho_s^2 * c_s / a

            f_geom_micro =
                [1
                + a_kappa * (kappa - 1)
                + a_delta * |delta|
                + a_s * s_hat]^{-1}

            gamma_l_ITG ~ (c_s / R) * Δ_ITG

            gamma_decorr_ITG =
                gamma_E
                + gammaNL_coeff * gamma_l_ITG

            alpha_ITG =
                2 - K_ITG / (1 + K_ITG),
            with
                K_ITG = gamma_l_ITG / gamma_decorr_ITG

            f_impurity = (1 + b_Z * fZ * Zeff)^{-1}

        Physical interpretation:
        ------------------------
        - Profiles are pinned near marginal stability via p_stiff
        - ExB shear suppresses transport
        - Strong turbulence transitions smoothly from quasilinear (α≈2)
        to strong-turbulence scaling (α≈1)
        - Negative triangularity reduces ITG drive through |delta|

        
        Trapped Electron Mode (TEM) Transport
        -------------------------------------

        Electron heat diffusivity due to TEM turbulence:

            chi_e_TEM =
                chi_gB_e
                * f_geom_micro
                * (Δ_TEM) ** p_stiff
                * (gamma_l_TEM / gamma_decorr_TEM) ** alpha_TEM

        Particle diffusivity:
            D_e_TEM = C_n * chi_e_TEM

        where:

            Δ_TEM = max(0, eta_e - eta_e_crit)

            gamma_l_TEM ~ (c_s / R) * Δ_TEM

            gamma_decorr_TEM =
                gamma_E
                + gammaNL_coeff * gamma_l_TEM

            alpha_TEM defined via Kubo-number proxy (same as ITG)

        Notes:
        ------
        - TEM dominates edge particle transport
        - Negative triangularity stabilizes TEM via geometry modifier
        - Convective fluxes may be added separately if desired

        
        Electron Temperature Gradient (ETG) Transport
        ---------------------------------------------

        Electron-scale ETG heat diffusivity follows a Hatch-style saturation model:

            chi_e_ETG =
                C_ETG
                * rho_e^2
                * gamma_l_ETG / ky_e^2
                * f_ITG_supp
                * (gamma_l_ETG / gamma_decorr_ETG) ** (alpha_ETG - 1)

        where:

            gamma_l_ETG ~ (v_te / R) * Δ_ETG

            Δ_ETG = max(0, R/LTe - (R/LTe)_crit)

            f_ITG_supp =
                gamma_l_ETG / (gamma_l_ETG + gamma_l_ITG)

            gamma_decorr_ETG =
                gamma_E + gamma_l_ITG

            alpha_ETG determined via Kubo-number proxy

        Physical interpretation:
        ------------------------
        - Recovers quasilinear ETG when weakly driven
        - Allows strong turbulence enhancement when decorrelation is weak
        - Naturally suppresses ETG in strong ITG regimes
        - Consistent with nonlinear multiscale simulations (Hatch 2023)

        
        Kinetic Ballooning Mode (KBM) Transport
        ---------------------------------------

        KBM transport activates when the normalized pressure gradient exceeds
        a critical threshold:

            chi_KBM =
                chi_gB_i
                * (Δ_KBM) ** p_stiff

        where:

            Δ_KBM = max(0, alpha - alpha_crit)

        Notes:
        ------
        - No explicit geometry modifier is applied
        - KBM provides the dominant transport channel when
        microturbulence is suppressed (e.g. negative triangularity)
        - Heat flux is typically split evenly between ions and electrons
        - Acts as the steady-state edge-limiting mechanism


        Total Turbulent Flux Assembly
        -----------------------------

        Ion heat flux:
            Qi = - n * chi_i * dTi/dr

        Electron heat flux:
            Qe = - n * chi_e * dTe/dr

        Electron particle flux:
            Gamma_e = - D_e * dn/dr

        with:
            chi_i = chi_i_ITG + 0.5 * chi_KBM
            chi_e = chi_e_TEM + chi_e_ETG + 0.5 * chi_KBM
            D_e   = D_e_TEM + D_e_ETG
        """

        # Extract quantities from state
        x = state.roa
        a = state.a
        eps = state.eps
        Te = state.te
        Ti = state.ti
        ne = state.ne
        ni = state.ni
        pe = state.pe
        pi = state.pi
        aLne = state.aLne
        aLni = state.aLni
        aLTe = state.aLte
        aLTi = state.aLti
        kappa = state.kappa
        q = state.q
        Zeff = state.Zeff
        mi_over_mp = state.mi_ref
        f_trap = state.f_trap
        beta = state.betae * (1 + Ti/Te) # beta_norm * ne * (Ti + Te)
        rhostar = state.rhostar # rhostar_norm * np.sqrt(Ti)

        dne_dx = -aLne * ne # dne_dr * r/a
        dTe_dx = -aLTe * Te
        dTi_dx = -aLTi * Ti
        dni_dx = -aLni * ni
        dpe_dx = ne*dTe_dx + Te*dne_dx
        dpi_dx = ni * dTi_dx + Ti * dne_dx
        aLpe = - (dpe_dx) / pe
        d2ne_dx2 = state.d2ne * a**2 # d2ne/dr2 * a**2
        d2ni_dx2 = np.gradient(dni_dx, x) 
        d2Te_dx2 = state.d2te * a**2 # d2Te/dr2 * a**2
        d2Ti_dx2 = state.d2ti * a**2 # d2Ti/dr2 * a**2
        d2pi_dx2 = Ti*d2ni_dx2 + 2*dni_dx*dTi_dx + ni*d2Ti_dx2
        
        # Collision frequencies
        nuii = state.nuii*state.tau_norm
        nuei = state.nuei*state.tau_norm

        if self.ExBon:
            pass
        else:
            pass


        # Neoclassical transport (analytic or NEO)
        Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(
            state,
            eps=eps,
            q=q,
            Ti=Ti,
            Te=Te,
            ne=ne,
            f_trap=f_trap,
            mi_over_mp=mi_over_mp,
            nuii=nuii,
            nuei=nuei,
            dne_dx=dne_dx,
            dTe_dx=dTe_dx,
            dTi_dx=dTi_dx,
        )

        # Critical gradients

        
        # Total turbulent
        if self.modes=='all':
            Gamma_turb = Gamma_ITG + Gamma_KBM
            Qi_turb = Qi_ITG + Qi_KBM
            Qe_turb = Qe_ITG + Qe_ETG + Qe_KBM
        if self.modes=='ITG':
            Gamma_turb = Gamma_ITG
            Qi_turb = Qi_ITG
            Qe_turb = Qe_ITG
        if self.modes=='ETG':
            Gamma_turb = 0*x
            Qi_turb = 0*x
            Qe_turb = Qe_ETG
        if self.modes=='KBM':
            Gamma_turb = Gamma_KBM
            Qi_turb = Qi_KBM
            Qe_turb = Qe_KBM
        if self.modes=='neo':
            Gamma_turb = 0*x
            Qi_turb = 0*x
            Qe_turb = 0*x

        return self._assemble_fluxes(
            state,
            Gamma_turb=Gamma_turb,
            Gamma_neo=Gamma_neo,
            Qi_turb=Qi_turb,
            Qi_neo=Qi_neo,
            Qe_turb=Qe_turb,
            Qe_neo=Qe_neo,
            model_label="Analytic",
        )

TRANSPORT_MODELS = {
    'fingerprints': Fingerprints,
    'tglf': Tglf,
    'cgyro': Cgyro,
    'qlgyro': Qlgyro,
    'fixed': Fixed,
    'analytic': Analytic,
}


def create_transport_model(config: Dict[str, Any]) -> TransportBase:
    """Factory to create a transport model instance using a config dict.

    Expected config format:
    {"type": "fingerprints"|"tglf"|"cgyro"|"qlgyro"|"fixed", "kwargs": { ... }}
    """
    if isinstance(config, str):
        model_type = config
        kwargs = {}
    else:
        model_type = (config or {}).get('type', 'fingerprints')
        kwargs = (config or {}).get('kwargs', {})

    cls = TRANSPORT_MODELS.get(model_type.lower())
    if cls is None:
        raise ValueError(f"Unknown transport model: {model_type}")
    return cls(**kwargs)
