"""TGLF transport model."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import time

import numpy as np

from .TransportBase import TransportBase
from .utils import (
    _load_controls,
    _merge_controls_and_locpargen,
    _pick_locpargen_file,
    _read_numeric_file,
    _run_locpargen,
)

if TYPE_CHECKING:
    from state import PlasmaState


class Tglf(TransportBase):
    """Direct TGLF transport model using $GACODE_ROOT."""

    def __init__(self, options: dict):
        super().__init__(options)
        self.external = True  # External dependencies
        
        tglf_opts = self.options.get("TGLF_options", {})
        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.work_dir = workflow_work_dir / "tglf"
        self.settings = tglf_opts.get("settings", {})
        self.multipliers = tglf_opts.get("multipliers", {})
        self.keep_files = tglf_opts.get("keep_files", "minimal")

    def _write_tglf_inputs(self, state, rho: float, rho_dir: Path, input_gacode: Path) -> Path:
        """Create input.tglf from controls + locpargen + user settings."""
        rho_dir.mkdir(parents=True, exist_ok=True)

        loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
        locpargen_file = _pick_locpargen_file(loc_files, "tglf")

        if locpargen_file is None:
            raise RuntimeError(f"locpargen did not produce input.tglf.locpargen at rho={rho}")

        controls_lines = _load_controls("tglf")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            self.settings,
            self.multipliers,
        )

        input_tglf = rho_dir / "input.tglf"
        input_tglf.write_text("".join(lines))

        return input_tglf

    def _read_tglf_gbflux(self, path: Path, rho: float) -> Tuple[float, float, float]:
        """Read TGLF out.tglf.gbflux file."""
        max_wait = 120.0
        poll_interval = 1.0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if path.exists() and path.stat().st_size > 50:
                break
            time.sleep(poll_interval)
        else:
            run_log = path.parent / "out.tglf.run"
            details = ""
            if run_log.exists():
                details = f"\nTGLF run log:\n{run_log.read_text().strip()}"
            raise FileNotFoundError(
                f"TGLF output not generated after {max_wait}s: {path}{details}"
            )

        data = _read_numeric_file(path)
        if data.size % 4 != 0:
            raise RuntimeError(f"TGLF gbflux size {data.size} not divisible by 4")
        n_species = data.size // 4
        data = data.reshape((4, n_species))

        Ge = float(data[0, 0])
        Qe = float(data[1, 0])

        Qi = float(np.sum(data[1, 1:]))

        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge, Qi, Qe)

        return Ge, Qi, Qe

    def _evaluate_single(self, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        platform = self._get_platform_manager(self.work_dir)
        output_dict, std_dict = self.run_on_platform(
            state,
            platform,
            work_dir=self.work_dir,
            model_name="tglf",
        )
        if self.keep_files == "none":
            self._cleanup_run_dirs(self.work_dir)
        return output_dict, std_dict

    def _prepare_platform_inputs(self, state: "PlasmaState", work_dir: Path) -> None:
        """Prepare TGLF input files for platform execution."""
        self._extract_from_state(state)
        if self.roa_eval is None:
            self.roa_eval = state.roa
        roa_eval = np.atleast_1d(self.roa_eval)

        input_gacode = work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        print(f"  Radial locations: {list(roa_eval)}")

        for rho in roa_eval:
            rho_dir = work_dir / f"rho_{rho:.3f}"
            self._write_tglf_inputs(state, rho, rho_dir, input_gacode)
            print(f"  ✓ Created input.tglf for rho={rho:.3f}")

    def _get_platform_file_patterns(self) -> Dict[str, List[str]]:
        """Return file patterns for staging inputs and retrieving outputs."""
        return {
            "input_files": ["rho_*/input.tglf"],
            "input_folders": ["rho_*"],
            "output_files": ["*.gbflux", "out.tglf.*"],
        }

    def _get_platform_commands(self, remote_work_dir: Path, state: "PlasmaState", platform) -> List[str]:
        """Generate TGLF execution commands for the platform."""
        tglf_exec = "tglf"
        roa_eval = np.atleast_1d(self.roa_eval)
        commands = []

        for rho in roa_eval:
            sim_name = f"rho_{rho:.3f}"
            cmd = (
                f"cd {remote_work_dir} && {tglf_exec} "
                f"-p {remote_work_dir} -e {sim_name} -n {self.threads_per_task}"
            )
            commands.append(cmd)

        return commands

    def _read_platform_results(self, work_dir: Path, state: "PlasmaState") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read TGLF output files and build result dictionaries."""
        roa_eval = np.atleast_1d(self.roa_eval)

        Gamma_turb = np.zeros_like(roa_eval, dtype=float)
        Qi_turb = np.zeros_like(roa_eval, dtype=float)
        Qe_turb = np.zeros_like(roa_eval, dtype=float)

        for i, rho in enumerate(roa_eval):
            rho_dir = work_dir / f"rho_{rho:.3f}"
            out_path = rho_dir / "out.tglf.gbflux"
            Ge, Qi, Qe = self._read_tglf_gbflux(out_path, rho)
            Gamma_turb[i] = Ge
            Qi_turb[i] = Qi
            Qe_turb[i] = Qe
            print(f"  ✓ Read fluxes for rho={rho:.3f}: Ge={Ge:.3e}, Qi={Qi:.3e}, Qe={Qe:.3e}")

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
