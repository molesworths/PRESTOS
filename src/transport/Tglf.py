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

        f_trap_at_rho = float(np.interp(rho, state.roa, state.f_trap))
        settings = {**self.settings, "THETA_TRAPPED": f_trap_at_rho}

        controls_lines = _load_controls("tglf")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            settings,
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

        data_flat = _read_numeric_file(path)
        if data_flat.size % 4 != 0:
            raise RuntimeError(f"TGLF gbflux size {data_flat.size} not divisible by 4")

        # [quantity=G,Q,P,S; species]
        data = np.reshape(data_flat, (4, data_flat.size // 4))

        Ge_gB = float(data[0, 0])
        Qe_gB = float(data[1, 0])
        Me_gB = float(data[2, 0])
        Se_gB = float(data[3, 0])

        Gi_gB = data[0, 1:]
        Qi_i_gB = data[1, 1:]
        Mi_gB = data[2, 1:]
        Si_gB = data[3, 1:]

        # Keep API compatible: return total ion heat flux (summed over ion species).
        Qi_gB = float(np.sum(Qi_i_gB))

        return Ge_gB, Qi_gB, Qe_gB

    def _read_tglf_exchange_spectrum(self, path: Path, rho: float) -> float:
        """Read turbulent exchange from out.tglf.sum_flux_spectrum.

        Returns the electron-species exchange channel in Sgb-normalized units.
        """
        if not path.exists() or path.stat().st_size == 0:
            return 0.0

        exchange_by_species: Dict[int, float] = {}
        current_species = None

        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue

                if s.lower().startswith("species"):
                    parts = s.replace("=", " ").split()
                    try:
                        i_species = parts.index("species") + 1
                        current_species = int(parts[i_species])
                    except Exception:
                        current_species = None
                    continue

                if current_species is None:
                    continue

                vals = []
                for token in s.split():
                    try:
                        vals.append(float(token))
                    except ValueError:
                        pass
                if len(vals) >= 5:
                    exchange_by_species[current_species] = exchange_by_species.get(current_species, 0.0) + vals[4]

        if not exchange_by_species:
            return 0.0

        electron_species = 1 if 1 in exchange_by_species else min(exchange_by_species.keys())
        return float(exchange_by_species[electron_species])

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

        Ge_turb_gB = np.zeros_like(roa_eval, dtype=float)
        Qi_turb_gB = np.zeros_like(roa_eval, dtype=float)
        Qe_turb_gB = np.zeros_like(roa_eval, dtype=float)
        Qie_turb_gB = np.zeros_like(roa_eval, dtype=float)

        for i, rho in enumerate(roa_eval):
            rho_dir = work_dir / f"rho_{rho:.3f}"
            out_path = rho_dir / "out.tglf.gbflux"
            exch_path = rho_dir / "out.tglf.sum_flux_spectrum"
            ge_gB, qi_gB, qe_gB = self._read_tglf_gbflux(out_path, rho)
            qie_gB = self._read_tglf_exchange_spectrum(exch_path, rho)
            Ge_turb_gB[i] = ge_gB
            Qi_turb_gB[i] = qi_gB
            Qe_turb_gB[i] = qe_gB
            Qie_turb_gB[i] = qie_gB
            print(
                f"  ✓ Read gB fluxes for rho={rho:.3f}: "
                f"Ge={ge_gB:.3e}, Qi={qi_gB:.3e}, Qe={qe_gB:.3e}, Qie={qie_gB:.3e}"
            )

        Ge_neo_gB, Qi_neo_gB, Qe_neo_gB, Qie_neo_gB = self._compute_neoclassical(state)

        return self._assemble_fluxes(
            state,
            Ge_turb_gB=Ge_turb_gB,
            Ge_neo_gB=Ge_neo_gB,
            Qi_turb_gB=Qi_turb_gB,
            Qi_neo_gB=Qi_neo_gB,
            Qe_turb_gB=Qe_turb_gB,
            Qe_neo_gB=Qe_neo_gB,
            Qie_turb_gB=Qie_turb_gB,
            Qie_neo_gB=Qie_neo_gB,
            model_label="TGLF",
        )
