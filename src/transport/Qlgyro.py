"""QLGYRO transport model."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from .TransportBase import TransportBase
from .utils import (
    _enforce_cgyro_thread_multiple,
    _load_controls,
    _merge_controls_and_locpargen,
    _pick_locpargen_file,
    _read_numeric_file,
    _resolve_gacode_executable,
    _run_locpargen,
    _update_namelist_lines,
    _with_python_in_path,
)

if TYPE_CHECKING:
    from state import PlasmaState


class Qlgyro(TransportBase):
    """Direct QLGYRO transport model using $GACODE_ROOT."""

    def __init__(self, options: dict):
        super().__init__(options)
        self.external = True  # External dependencies

        qlgyro_opts = self.options.get("QLGYRO_options", {})
        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.work_dir = workflow_work_dir / "qlgyro"
        self.settings = qlgyro_opts.get("settings", {})
        self.keep_files = qlgyro_opts.get("keep_files", "minimal")
        self.wait_timeout = float(qlgyro_opts.get("wait_timeout", 3600.0))
        self.wait_poll = float(qlgyro_opts.get("wait_poll", 5.0))

        cgyro_opts = self.options.get("CGYRO_options", {})
        self.cgyro_settings = cgyro_opts.get("settings", {})
        self.cgyro_auto = cgyro_opts.get("auto", {})

    def _auto_cgyro_settings(self, state, updates: Dict[str, Any], rho: float = None) -> Dict[str, Any]:
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

        if rho is not None and hasattr(state, "q") and hasattr(state, "shear"):
            q_local = float(np.interp(rho, state.roa, state.q))
            s_local = float(np.interp(rho, state.roa, state.shear))
        else:
            q_local = 2.0
            s_local = 1.0

        def set_if_absent(key, value):
            if key not in updates:
                updates[key] = value

        if "box_size" in self.cgyro_auto:
            box_size = int(self.cgyro_auto.get("box_size", 1))
        else:
            box_size = 1
        set_if_absent("BOX_SIZE", box_size)

        nonlinear = updates.get("NONLINEAR_FLAG", self.cgyro_settings.get("NONLINEAR_FLAG", 0)) == 1

        if nonlinear:
            n_radial_min = box_size / max(abs(q_local) * abs(s_local) * rhostar, 1e-6)
            n_radial = int(np.ceil(max(n_radial_min * 1.2, 8)))
            n_radial = int(np.clip(n_radial, 8, 48))
        else:
            n_radial = 4

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

        loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
        locpargen_file = _pick_locpargen_file(loc_files, "cgyro")

        if locpargen_file is None:
            raise RuntimeError(f"locpargen did not produce input.cgyro.locpargen at rho={rho}")

        settings = dict(self.cgyro_settings) if self.cgyro_settings else {}
        settings = self._auto_cgyro_settings(state, settings, rho=rho)
        self.threads_per_task = _enforce_cgyro_thread_multiple(settings, self.threads_per_task)

        controls_lines = _load_controls("cgyro")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            settings,
            {},
        )

        input_cgyro = rho_dir / "input.cgyro"
        input_cgyro.write_text("".join(lines))

        return input_cgyro

    def _write_qlgyro_input(self, rho_dir: Path) -> Path:
        """Create input.qlgyro from controls + user settings (no locpargen)."""
        input_qlgyro = rho_dir / "input.qlgyro"

        lines = _load_controls("qlgyro")

        if self.settings:
            lines = _update_namelist_lines(lines, self.settings)

        lines = _update_namelist_lines(lines, {"n_parallel": self.mpi_tasks})

        input_qlgyro.write_text("".join(lines))
        return input_qlgyro

    def _read_qlgyro_gbflux(self, path: Path) -> Tuple[float, float, float]:
        """Read QLGYRO out.qlgyro.gbflux file."""
        if not path.exists():
            raise FileNotFoundError(f"QLGYRO gbflux not found: {path}")

        data = _read_numeric_file(path)
        if data.size == 0:
            raise RuntimeError(f"QLGYRO gbflux is empty: {path}")
        if data.size % 4 != 0:
            raise RuntimeError(f"QLGYRO gbflux size {data.size} not divisible by 4")

        n_species = data.size // 4
        data = data.reshape((4, n_species))

        Ge_gB = float(data[0, 0])
        Qe_gB = float(data[1, 0])
        Qi_gB = float(np.sum(data[1, 1:]))

        return Ge_gB, Qi_gB, Qe_gB

    def _evaluate_single(self, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        platform = self._get_platform_manager(self.work_dir)
        output_dict, std_dict = self.run_on_platform(
            state,
            platform,
            work_dir=self.work_dir,
            model_name="qlgyro",
        )
        if self.keep_files == "none":
            self._cleanup_run_dirs(self.work_dir)
        return output_dict, std_dict

    def _prepare_platform_inputs(self, state: "PlasmaState", work_dir: Path) -> None:
        """Prepare QLGYRO input files for platform execution."""
        self._extract_from_state(state)
        if self.roa_eval is None:
            self.roa_eval = state.roa
        roa_eval = np.atleast_1d(self.roa_eval)

        input_gacode = work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        print(f"  Radial locations: {list(roa_eval)}")

        qlgyro_exec = _resolve_gacode_executable("qlgyro", self.options)

        for rho in roa_eval:
            rho_dir = work_dir / f"rho_{rho:.3f}"
            rho_dir.mkdir(parents=True, exist_ok=True)

            self._write_cgyro_inputs(state, rho, rho_dir, input_gacode)
            self._write_qlgyro_input(rho_dir)

            init_cmd = [qlgyro_exec, "-i", rho_dir.name]
            init_res = subprocess.run(
                init_cmd,
                cwd=str(work_dir),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env=_with_python_in_path(),
            )
            if init_res.returncode != 0:
                raise RuntimeError(f"QLGYRO init failed at rho={rho}: {init_res.stderr.strip()}")

            print(f"  ✓ Created and initialized QLGYRO inputs for rho={rho:.3f}")

    def _get_platform_file_patterns(self) -> Dict[str, List[str]]:
        """Return file patterns for staging inputs and retrieving outputs."""
        return {
            "input_files": ["rho_*/input.cgyro", "rho_*/input.qlgyro"],
            "input_folders": ["rho_*"],
            "output_files": ["*.gbflux", "out.qlgyro.*"],
        }

    def _get_platform_commands(self, remote_work_dir: Path, state: "PlasmaState", platform) -> List[str]:
        """Generate QLGYRO execution commands for the platform.

        Each command launches qlgyro then polls out.qlgyro.status until all ky
        entries have left the 'fresh'/'running' state.  This blocks the shell
        until computation is genuinely complete, guarding against platforms where
        qlgyro spawns background MPI workers and returns before finishing.
        """
        qlgyro_exec = "qlgyro"
        roa_eval = np.atleast_1d(self.roa_eval)
        poll = max(1, int(self.wait_poll))
        timeout = max(poll, int(self.wait_timeout))
        commands = []

        for rho in roa_eval:
            sim_name = f"rho_{rho:.3f}"
            status_file = f"{sim_name}/out.qlgyro.status"
            cmd = (
                f"cd {remote_work_dir} && "
                f"{qlgyro_exec} -e {sim_name} -n {self.mpi_tasks} -nomp {self.threads_per_task}; "
                f"_t=0; "
                f"while [ $_t -lt {timeout} ]; do "
                f"  if [ -f {status_file} ] && ! grep -qwE 'fresh|running' {status_file}; then break; fi; "
                f"  sleep {poll}; _t=$((_t + {poll})); "
                f"done"
            )
            commands.append(cmd)

        return commands

    def _read_platform_results(self, work_dir: Path, state: "PlasmaState") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read QLGYRO output files and build result dictionaries."""
        roa_eval = np.atleast_1d(self.roa_eval)

        Ge_turb_gB = np.zeros_like(roa_eval, dtype=float)
        Qi_turb_gB = np.zeros_like(roa_eval, dtype=float)
        Qe_turb_gB = np.zeros_like(roa_eval, dtype=float)

        for i, rho in enumerate(roa_eval):
            rho_dir = work_dir / f"rho_{rho:.3f}"
            out_path = rho_dir / "out.qlgyro.gbflux"
            ge_gB, qi_gB, qe_gB = self._read_qlgyro_gbflux(out_path)
            Ge_turb_gB[i] = ge_gB
            Qi_turb_gB[i] = qi_gB
            Qe_turb_gB[i] = qe_gB
            print(f"  ✓ Read gB fluxes for rho={rho:.3f}: Ge={ge_gB:.3e}, Qi={qi_gB:.3e}, Qe={qe_gB:.3e}")

        Ge_neo_gB, Qi_neo_gB, Qe_neo_gB = self._compute_neoclassical(state)

        return self._assemble_fluxes(
            state,
            Ge_turb_gB=Ge_turb_gB,
            Ge_neo_gB=Ge_neo_gB,
            Qi_turb_gB=Qi_turb_gB,
            Qi_neo_gB=Qi_neo_gB,
            Qe_turb_gB=Qe_turb_gB,
            Qe_neo_gB=Qe_neo_gB,
            model_label="QLGYRO",
        )
