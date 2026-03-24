"""Shared utilities for transport models."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.constants import e


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
        v_lower = value.lower().strip()
        if v_lower in (".true.", ".false.", "t", "f"):
            return "True" if v_lower in (".true.", "t") else "False"
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


def _enforce_cgyro_thread_multiple(settings: Dict[str, Any], n_threads: int) -> int:
    if not settings or n_threads is None:
        return n_threads

    key_map = {k.upper(): k for k in settings.keys()}

    n_key = key_map.get("N_TOROIDAL")
    if n_key is None:
        return n_threads

    try:
        n_toroidal = int(settings[n_key])
    except (TypeError, ValueError):
        return n_threads

    if n_toroidal <= 0:
        return n_threads

    tpp_key = key_map.get("TOROIDALS_PER_PROC") or key_map.get("TOROIDALS_PER_PROCESS")
    if tpp_key is None:
        return n_threads

    try:
        tpp = int(settings[tpp_key])
    except (TypeError, ValueError):
        return n_threads

    if tpp <= 0:
        return n_threads

    if n_toroidal % tpp != 0:
        for candidate in range(min(tpp, n_toroidal), 0, -1):
            if n_toroidal % candidate == 0:
                tpp = candidate
                break
        else:
            tpp = 1

    settings["TOROIDALS_PER_PROC"] = tpp
    settings.pop("TOROIDALS_PER_PROCESS", None)

    groups = max(1, n_toroidal // tpp)
    if n_threads % groups != 0:
        n_threads = int(((n_threads + groups - 1) // groups) * groups)

    return n_threads


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
    values = [
        float(x)
        for x in re.findall(
            r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
            text,
        )
    ]
    return np.array(values, dtype=float)


def _get_template_path(template_type: str) -> Path:
    """Get path to a GACODE template file from src/tools/gacode_templates/."""
    this_file = Path(__file__).resolve()
    template_dir = this_file.parent.parent / "tools" / "gacode_templates"

    template_file = template_dir / f"input.{template_type.lower()}"
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    return template_file


def _load_template(template_type: str) -> List[str]:
    """Load and parse a GACODE template file."""
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
    """Get path to a GACODE controls file from src/tools/gacode_controls/."""
    this_file = Path(__file__).resolve()
    controls_dir = this_file.parent.parent / "tools" / "gacode_controls"

    controls_file = controls_dir / f"input.{model.lower()}.controls"
    if not controls_file.exists():
        raise FileNotFoundError(f"Controls file not found: {controls_file}")

    return controls_file


def _load_controls(model: str) -> List[str]:
    """Load GACODE controls file, filtering out template header comments."""
    controls_path = _get_controls_path(model)
    with open(controls_path, "r") as f:
        lines = f.readlines()

    filtered = []
    in_header = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#---") or stripped.startswith("#==="):
            in_header = not in_header
            continue
        if in_header:
            continue
        if "Template" in line and line.strip().startswith("#"):
            continue
        filtered.append(line)

    return filtered


def _run_locpargen(state, roa: float, rho_dir: Path, input_gacode: Path) -> Dict[str, Path]:
    """Run profiles_gen/locpargen to create state-specific input files."""
    if not hasattr(state, "to_gacode"):
        raise RuntimeError("locpargen requires state.to_gacode() to create input.gacode")

    rho_dir.mkdir(parents=True, exist_ok=True)
    input_gacode = Path(input_gacode).resolve()
    if not input_gacode.exists():
        state.to_gacode().write(str(input_gacode))

    rho_input_gacode = rho_dir / "input.gacode"
    shutil.copy2(str(input_gacode), str(rho_input_gacode))

    executable = _resolve_locpargen_executable({})
    cmd = [executable, "-i", "input.gacode", "-loc_rad", str(roa)]
    result = subprocess.run(
        cmd,
        cwd=str(rho_dir),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_with_python_in_path(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "profiles_gen/locpargen failed at roa={roa}: {stderr}\nCommand: {cmd}".format(
                roa=roa,
                stderr=result.stderr.strip(),
                cmd=" ".join(cmd),
            )
        )

    loc_files = list(rho_dir.glob("*.locpargen"))
    if not loc_files:
        raise RuntimeError(
            "No .locpargen files produced in {rho_dir} (locpargen stderr: {stderr})".format(
                rho_dir=rho_dir,
                stderr=result.stderr.strip(),
            )
        )

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
    """Merge controls, locpargen data, and user settings (removing duplicates)."""
    controls_params = set()
    for line in controls_lines:
        clean = line.strip()
        if clean and not clean.startswith("#") and "=" in clean:
            key = clean.split("=", 1)[0].strip().upper()
            controls_params.add(key)

    with open(locpargen_path, "r") as f:
        locpargen_lines = f.readlines()

    filtered_locpargen = []
    for line in locpargen_lines:
        clean = line.strip()
        if clean and not clean.startswith("#") and "=" in clean:
            key = clean.split("=", 1)[0].strip().upper()
            if key not in controls_params:
                filtered_locpargen.append(line)
        elif not clean or clean.startswith("#"):
            filtered_locpargen.append(line)

    merged = controls_lines + filtered_locpargen

    if settings:
        merged = _update_namelist_lines(merged, settings)

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
    """Run NEO for a single radial location."""
    rho_dir = work_dir / f"rho_{rho:.3f}"
    rho_dir.mkdir(parents=True, exist_ok=True)

    loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
    locpargen_file = _pick_locpargen_file(loc_files, "neo")

    if locpargen_file is None:
        raise RuntimeError(f"locpargen did not produce input.neo.locpargen at rho={rho}")

    controls_lines = _load_controls("neo")

    settings_run = dict(settings or {})
    dphi_setting = settings_run.get("DPHI0DR")
    dphi_requested = False
    if isinstance(dphi_setting, str):
        dphi_requested = dphi_setting.strip().lower() in {"1", "1.0", "true", ".true.", "t", "yes", "on"}
    elif isinstance(dphi_setting, bool):
        dphi_requested = dphi_setting
    elif dphi_setting is not None:
        try:
            dphi_requested = float(dphi_setting) == 1.0
        except (TypeError, ValueError):
            dphi_requested = False

    if dphi_requested:
        if not all(hasattr(state, name) for name in ("Er", "te", "roa", "a")):
            raise RuntimeError("DPHI0DR=1 requested, but state is missing one of Er/te/roa/a")

        roa = np.asarray(state.roa, dtype=float)
        er = np.asarray(state.Er, dtype=float)
        te = np.asarray(state.te, dtype=float)

        er_eval = float(np.interp(float(rho), roa, er))
        te_eval = float(np.interp(float(rho), roa, te))
        te_safe = te_eval if abs(te_eval) > 1e-30 else np.copysign(1e-30, te_eval if te_eval != 0.0 else 1.0)

        # User-requested normalization: DPHI0DR = -Er * (a * e / te)
        settings_run["DPHI0DR"] = -er_eval * (float(state.a) * e / te_safe)

    lines = _merge_controls_and_locpargen(
        controls_lines,
        locpargen_file,
        settings_run,
        multipliers,
    )

    input_neo = rho_dir / "input.neo"
    input_neo.write_text("".join(lines))

    cmd = [neo_exec, "-e", rho_dir.name, "-nomp", str(n_threads)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
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
    """Run TGLF for a single radial location."""
    rho_dir = work_dir / f"rho_{rho:.3f}"
    rho_dir.mkdir(parents=True, exist_ok=True)

    loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
    locpargen_file = _pick_locpargen_file(loc_files, "tglf")

    if locpargen_file is None:
        raise RuntimeError(f"locpargen did not produce input.tglf.locpargen at rho={rho}")

    controls_lines = _load_controls("tglf")
    lines = _merge_controls_and_locpargen(
        controls_lines,
        locpargen_file,
        settings,
        multipliers,
    )

    input_tglf = rho_dir / "input.tglf"
    input_tglf.write_text("".join(lines))

    cmd = [tglf_exec, "-p", str(work_dir), "-e", rho_dir.name, "-n", str(n_threads)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_with_python_in_path(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"TGLF failed at rho={rho:.3f}: {result.stderr.strip()}")

    if keep_files == "none":
        shutil.rmtree(rho_dir, ignore_errors=True)
