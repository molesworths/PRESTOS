"""CGYRO transport model."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .TransportBase import TransportBase
from .utils import (
    _enforce_cgyro_thread_multiple,
    _load_controls,
    _merge_controls_and_locpargen,
    _parse_namelist_lines,
    _pick_locpargen_file,
    _run_locpargen,
)

if TYPE_CHECKING:
    from state import PlasmaState


class Cgyro(TransportBase):
    """Direct CGYRO transport model using $GACODE_ROOT."""

    def __init__(self, options: dict):
        super().__init__(options)
        self.external = True  # External dependencies

        cgyro_opts = self.options.get("CGYRO_options", {})
        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.work_dir = workflow_work_dir / "cgyro"
        self.settings = cgyro_opts.get("settings", {})
        self.multipliers = cgyro_opts.get("multipliers", {})
        self.keep_files = cgyro_opts.get("keep_files", "minimal")
        self.auto = cgyro_opts.get("auto", {})
        self.read_mode = cgyro_opts.get("read_mode", "pygacode")
        self.linear_ky = cgyro_opts.get("linear_ky")
        self.linear_ky_scan = cgyro_opts.get("linear_ky_scan")

    def _prepare_platform_inputs(self, state: "PlasmaState", work_dir: Path) -> None:
        """Prepare CGYRO input files for platform execution."""
        self._extract_from_state(state)
        if self.roa_eval is None:
            self.roa_eval = state.roa
        roa_eval = np.atleast_1d(self.roa_eval)

        input_gacode = work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        nonlinear = int(self.settings.get("NONLINEAR_FLAG", 0)) == 1

        print(f"  Mode: {'Nonlinear' if nonlinear else 'Linear'}")
        print(f"  Radial locations: {list(roa_eval)}")

        for rho in roa_eval:
            if nonlinear:
                rho_dir = work_dir / f"rho_{rho:.3f}"
                self._write_cgyro_inputs(state, rho, rho_dir, input_gacode)
                print(f"  ✓ Created input.cgyro for rho={rho:.3f}")
            else:
                ky_list = self._resolve_linear_ky_list(self.settings)
                if not ky_list:
                    ky_list = [0.3]

                multiple_ky = len(ky_list) > 1
                for ky in ky_list:
                    if multiple_ky:
                        sim_dir = work_dir / f"rho_{rho:.3f}_ky_{float(ky):.3f}"
                    else:
                        sim_dir = work_dir / f"rho_{rho:.3f}"

                    self._write_cgyro_inputs(
                        state,
                        rho,
                        sim_dir,
                        input_gacode,
                        settings_override={"KY": float(ky), "NONLINEAR_FLAG": 0},
                    )
                    print(f"  ✓ Created input.cgyro for rho={rho:.3f}, ky={ky:.3f}")

    def _get_platform_file_patterns(self) -> Dict[str, List[str]]:
        """Return file patterns for staging inputs and retrieving outputs."""
        return {
            "input_files": ["rho_*/input.cgyro"],
            "input_folders": ["rho_*"],
            "output_files": ["*.json", "out.cgyro.*", "bin.cgyro.*"],
        }

    def _get_platform_commands(self, remote_work_dir: Path, state: "PlasmaState", platform) -> List[str]:
        """Generate CGYRO execution commands for the platform."""
        cgyro_exec = "cgyro"

        roa_eval = np.atleast_1d(self.roa_eval)
        nonlinear = int(self.settings.get("NONLINEAR_FLAG", 0)) == 1
        commands = []

        for rho in roa_eval:
            if nonlinear:
                sim_name = f"rho_{rho:.3f}"
                cmd = (
                    f"cd {remote_work_dir} && {cgyro_exec} -e {sim_name} "
                    f"-n {self.mpi_tasks} -nomp {self.threads_per_task} -p {remote_work_dir}"
                )
                commands.append(cmd)
            else:
                ky_list = self._resolve_linear_ky_list(self.settings)
                if not ky_list:
                    ky_list = [0.3]

                multiple_ky = len(ky_list) > 1
                for ky in ky_list:
                    if multiple_ky:
                        sim_name = f"rho_{rho:.3f}_ky_{float(ky):.3f}"
                    else:
                        sim_name = f"rho_{rho:.3f}"
                    cmd = (
                        f"cd {remote_work_dir} && {cgyro_exec} -e {sim_name} "
                        f"-n {self.mpi_tasks} -nomp {self.threads_per_task} -p {remote_work_dir}"
                    )
                    commands.append(cmd)

        return commands

    def _read_platform_results(self, work_dir: Path, state: "PlasmaState") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read CGYRO output files and build result dictionaries."""
        roa_eval = np.atleast_1d(self.roa_eval)
        nonlinear = int(self.settings.get("NONLINEAR_FLAG", 0)) == 1

        Gamma_turb = np.zeros_like(roa_eval, dtype=float)
        Qi_turb = np.zeros_like(roa_eval, dtype=float)
        Qe_turb = np.zeros_like(roa_eval, dtype=float)

        for i, rho in enumerate(roa_eval):
            if nonlinear:
                rho_dir = work_dir / f"rho_{rho:.3f}"
                if self.read_mode == "json":
                    json_path = rho_dir / "fluxes_turb.json"
                    Ge, Qi, Qe = self._read_cgyro_fluxes_json(json_path, rho)
                else:
                    Ge, Qi, Qe = self._read_cgyro_fluxes_pygacode(rho_dir, rho)

                Gamma_turb[i] = Ge
                Qi_turb[i] = Qi
                Qe_turb[i] = Qe
                print(f"  ✓ Read fluxes for rho={rho:.3f}: Ge={Ge:.3e}, Qi={Qi:.3e}, Qe={Qe:.3e}")
            else:
                ky_list = self._resolve_linear_ky_list(self.settings)
                if not ky_list:
                    ky_list = [0.3]

                Ge_sum = 0.0
                Qi_sum = 0.0
                Qe_sum = 0.0
                multiple_ky = len(ky_list) > 1

                for ky in ky_list:
                    if multiple_ky:
                        sim_dir = work_dir / f"rho_{rho:.3f}_ky_{float(ky):.3f}"
                    else:
                        sim_dir = work_dir / f"rho_{rho:.3f}"

                    if self.read_mode == "json":
                        json_path = sim_dir / "fluxes_turb.json"
                        Ge, Qi, Qe = self._read_cgyro_fluxes_json(json_path, rho)
                    else:
                        Ge, Qi, Qe = self._read_cgyro_fluxes_pygacode(sim_dir, rho)

                    Ge_sum += Ge
                    Qi_sum += Qi
                    Qe_sum += Qe

                Gamma_turb[i] = Ge_sum
                Qi_turb[i] = Qi_sum
                Qe_turb[i] = Qe_sum
                print(
                    f"  ✓ Read fluxes for rho={rho:.3f}: Ge={Ge_sum:.3e}, Qi={Qi_sum:.3e}, Qe={Qe_sum:.3e}"
                )

        self._extract_from_state(state)
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

    def _auto_cgyro_settings(self, state, updates: Dict[str, Any], rho: float = None) -> Dict[str, Any]:
        """Auto-scale CGYRO settings based on plasma physics."""
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

        if rho is not None and hasattr(state, "q") and hasattr(state, "shear"):
            q_local = float(np.interp(rho, state.roa, state.q))
            s_local = float(np.interp(rho, state.roa, state.shear))
        else:
            q_local = 2.0
            s_local = 1.0

        def set_if_absent(key, value):
            if key not in updates:
                updates[key] = value

        if "box_size" in self.auto:
            box_size = int(self.auto.get("box_size", 1))
        else:
            box_size = 1
        set_if_absent("BOX_SIZE", box_size)

        nonlinear = updates.get("NONLINEAR_FLAG", self.settings.get("NONLINEAR_FLAG", 0)) == 1

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

        set_if_absent("N_TOROIDAL", int(self.auto.get("n_toroidal", 1)))

        if self.auto.get("electromagnetic", None) is True or betae > 0.01:
            set_if_absent("N_FIELD", 3)
        elif self.auto.get("electromagnetic", None) is False:
            set_if_absent("N_FIELD", 1)

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

    def _resolve_linear_ky_list(self, settings: Dict[str, Any]) -> List[float]:
        def _as_list(value: Any) -> List[float]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, np.ndarray)):
                return [float(v) for v in value]
            return [float(value)]

        def _from_scan(scan: Any) -> List[float]:
            if isinstance(scan, dict):
                ky_min = float(scan.get("min", scan.get("ky_min", 0.1)))
                ky_max = float(scan.get("max", scan.get("ky_max", ky_min)))
                n_ky = int(scan.get("n", scan.get("n_ky", 1)))
                if n_ky <= 1:
                    return [ky_min]
                return [float(v) for v in np.linspace(ky_min, ky_max, n_ky)]
            return _as_list(scan)

        if self.linear_ky is not None:
            return _as_list(self.linear_ky)

        if self.linear_ky_scan is not None:
            return _from_scan(self.linear_ky_scan)

        if isinstance(self.auto, dict):
            if "linear_ky" in self.auto:
                return _as_list(self.auto.get("linear_ky"))
            if "linear_ky_scan" in self.auto:
                return _from_scan(self.auto.get("linear_ky_scan"))

        for key in ("LINEAR_KY", "LINEAR_KY_LIST", "KY_LIST"):
            if key in settings:
                return _as_list(settings[key])

        if all(key in settings for key in ("KY_MIN", "KY_MAX", "NKY")):
            try:
                ky_min = float(settings["KY_MIN"])
                ky_max = float(settings["KY_MAX"])
                n_ky = int(settings["NKY"])
            except (TypeError, ValueError):
                return []
            if n_ky <= 1:
                return [ky_min]
            return [float(v) for v in np.linspace(ky_min, ky_max, n_ky)]

        ky_val = settings.get("KY")
        if ky_val is None and isinstance(self.auto, dict):
            ky_val = self.auto.get("ky")
        if ky_val is None:
            try:
                controls = _load_controls("cgyro")
                parsed = _parse_namelist_lines(controls)
                if "KY" in parsed:
                    ky_val = float(parsed["KY"])
            except Exception:
                ky_val = None
        return [float(ky_val)] if ky_val is not None else []

    def _write_cgyro_inputs(
        self,
        state,
        rho: float,
        rho_dir: Path,
        input_gacode: Path,
        settings_override: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create input.cgyro from controls + locpargen + user settings."""
        rho_dir.mkdir(parents=True, exist_ok=True)

        loc_files = _run_locpargen(state, rho, rho_dir, input_gacode)
        locpargen_file = _pick_locpargen_file(loc_files, "cgyro")

        if locpargen_file is None:
            raise RuntimeError(f"locpargen did not produce input.cgyro.locpargen at rho={rho}")

        settings = dict(self.settings) if self.settings else {}
        settings = self._auto_cgyro_settings(state, settings, rho=rho)
        if settings_override:
            settings.update(settings_override)
        self.threads_per_task = _enforce_cgyro_thread_multiple(settings, self.threads_per_task)

        controls_lines = _load_controls("cgyro")
        lines = _merge_controls_and_locpargen(
            controls_lines,
            locpargen_file,
            settings,
            self.multipliers,
        )

        input_cgyro = rho_dir / "input.cgyro"
        input_cgyro.write_text("".join(lines))

        return input_cgyro

    def _read_cgyro_fluxes_json(self, path: Path, rho: float) -> Tuple[float, float, float]:
        """Read CGYRO flux JSON file."""
        max_wait = 120.0
        poll_interval = 1.0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if path.exists() and path.stat().st_size > 50:
                break
            time.sleep(poll_interval)
        else:
            raise FileNotFoundError(
                f"CGYRO flux JSON not generated after {max_wait}s: {path}"
            )

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

        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge_gb, Qi_gb, Qe_gb)

        return Ge, Qi, Qe

    def _read_cgyro_fluxes_pygacode(self, rho_dir: Path, rho: float) -> Tuple[float, float, float]:
        """Read CGYRO fluxes using pygacode."""
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

        grids_file = rho_dir / "out.cgyro.grids"
        phi_file = rho_dir / "bin.cgyro.phi_t"

        max_wait = 3600.0
        poll_interval = 1.0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            grids_ok = grids_file.exists() and grids_file.stat().st_size > 100
            phi_ok = phi_file.exists() and phi_file.stat().st_size > 1000
            if grids_ok and phi_ok:
                break
            time.sleep(poll_interval)
        else:
            missing = []
            if not (grids_file.exists() and grids_file.stat().st_size > 100):
                missing.append(f"{grids_file}")
            if not (phi_file.exists() and phi_file.stat().st_size > 1000):
                missing.append(f"{phi_file}")
            raise FileNotFoundError(
                "CGYRO output files not fully generated after "
                f"{max_wait}s: {', '.join(missing)}"
            )

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

        Ge, Qi, Qe = self._gbflux_to_physical(rho, Ge_gb, Qi_gb, Qe_gb)

        return Ge, Qi, Qe


    def check_cgyro_saturation(
        self,
        rho_dir: Path,
        window_frac: float = 0.3,
        flux_tol: float = 0.05,
        energy_tol: float = 0.05,
        max_wait: float = 3600.0,
        poll_interval: float = 10.0,
        min_time: float = 50.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if CGYRO nonlinear run has reached turbulence saturation."""
        rho_dir = Path(rho_dir)
        ky_flux_file = rho_dir / "bin.cgyro.ky_flux"
        info_file = rho_dir / "out.cgyro.info"
        time_file = rho_dir / "out.cgyro.time"

        start_time = time.time()
        last_info_size = 0
        last_flux_size = 0
        stall_count = 0
        max_stall_checks = 3

        while time.time() - start_time < max_wait:
            files_exist = ky_flux_file.exists() and info_file.exists()

            if files_exist:
                current_flux_size = ky_flux_file.stat().st_size
                current_info_size = info_file.stat().st_size

                if current_info_size > 100:
                    try:
                        with open(info_file, "r") as f:
                            info_content = f.read()
                            if (
                                "ERROR" in info_content
                                or "ABORT" in info_content
                                or "FATAL" in info_content
                            ):
                                return False, {
                                    "converged": False,
                                    "error": "CGYRO reported ERROR/ABORT/FATAL in out.cgyro.info",
                                }
                    except Exception:
                        pass

                if current_flux_size > 1000:
                    break

                if current_flux_size == last_flux_size and current_info_size == last_info_size:
                    stall_count += 1
                    if stall_count >= max_stall_checks:
                        stall_info = ""
                        if time_file.exists():
                            try:
                                with open(time_file, "r") as f:
                                    lines = f.readlines()
                                    if lines:
                                        stall_info = (
                                            " Last line in out.cgyro.time: "
                                            f"{lines[-1].strip()}"
                                        )
                            except Exception:
                                pass
                        return False, {
                            "converged": False,
                            "error": (
                                "CGYRO process appears stalled (files not growing for "
                                f"{stall_count * poll_interval}s).{stall_info}"
                            ),
                        }
                else:
                    stall_count = 0

                last_flux_size = current_flux_size
                last_info_size = current_info_size

            time.sleep(poll_interval)
        else:
            missing = []
            if not info_file.exists():
                missing.append("out.cgyro.info")
            if not ky_flux_file.exists():
                missing.append("bin.cgyro.ky_flux")

            error_msg = f"Output files not generated after {max_wait} seconds"
            if missing:
                error_msg += f" (missing: {', '.join(missing)})"

            return False, {
                "converged": False,
                "error": error_msg,
            }

        try:
            n_field, n_species, n_ky = self._parse_cgyro_info_dimensions(info_file)
        except Exception as e:
            return False, {
                "converged": False,
                "error": f"Could not parse CGYRO dimensions: {e}",
            }

        diagnostics = {
            "flux_converged": False,
            "energy_converged": False,
            "time": None,
            "messages": [],
        }

        consecutive_read_failures = 0
        max_read_failures = 5
        last_data_size = 0
        data_stall_count = 0

        while time.time() - start_time < max_wait:
            try:
                try:
                    with open(info_file, "r") as f:
                        info_lines = f.readlines()
                        for line in info_lines[-10:]:
                            if "ERROR" in line or "ABORT" in line or "FATAL" in line:
                                return False, {
                                    "converged": False,
                                    "error": f"CGYRO error detected: {line.strip()}",
                                }
                except Exception:
                    pass

                time_val, fluxes = self._read_cgyro_ky_flux(
                    ky_flux_file, n_field, n_species, n_ky
                )

                consecutive_read_failures = 0

                if time_val is None or len(time_val) < 10:
                    diagnostics["messages"].append(
                        f"Insufficient time points: {len(time_val) if time_val is not None else 0}"
                    )
                    time.sleep(poll_interval)
                    continue

                current_data_size = len(time_val)

                if current_data_size == last_data_size and current_data_size > 0:
                    data_stall_count += 1
                    if data_stall_count >= max_stall_checks:
                        return False, {
                            "converged": False,
                            "error": (
                                "CGYRO data not growing (stalled at "
                                f"{current_data_size} time points for "
                                f"{data_stall_count * poll_interval}s)"
                            ),
                        }
                else:
                    data_stall_count = 0

                last_data_size = current_data_size

                diagnostics["time"] = float(time_val[-1])

                if time_val[-1] < min_time:
                    msg = f"Sim time {time_val[-1]:.1f} < min {min_time}"
                    diagnostics["messages"].append(msg)
                    time.sleep(poll_interval)
                    continue

                n_window = max(int(len(time_val) * window_frac), 10)
                flux_window = fluxes[-n_window:, :, :2]
                flux_mean = flux_window.mean(axis=0)
                flux_std = flux_window.std(axis=0)
                rel_std_flux = flux_std / np.maximum(np.abs(flux_mean), 1e-10)

                flux_converged = np.all(rel_std_flux < flux_tol)
                diagnostics["flux_converged"] = flux_converged
                diagnostics["messages"].append(
                    f"Flux std/mean: {rel_std_flux.max():.3f} (tol={flux_tol})"
                )

                energy_mean = np.abs(fluxes[-n_window:]).mean()
                energy_std = np.abs(fluxes[-n_window:]).std()
                rel_std_energy = energy_std / max(energy_mean, 1e-10)

                energy_converged = rel_std_energy < energy_tol
                diagnostics["energy_converged"] = energy_converged
                diagnostics["messages"].append(
                    f"Energy std/mean: {rel_std_energy:.3f} (tol={energy_tol})"
                )

                if flux_converged and energy_converged:
                    return True, diagnostics

                time.sleep(poll_interval)

            except Exception as e:
                consecutive_read_failures += 1
                diagnostics["messages"].append(f"Error reading fluxes: {e}")

                if consecutive_read_failures >= max_read_failures:
                    return False, {
                        "converged": False,
                        "error": (
                            "CGYRO appears to have failed "
                            f"({consecutive_read_failures} consecutive read errors). "
                            f"Last error: {e}"
                        ),
                        "messages": diagnostics["messages"],
                    }

                time.sleep(poll_interval)

        diagnostics["error"] = f"Saturation not detected after {max_wait} seconds"
        return False, diagnostics

    def _parse_cgyro_info_dimensions(self, info_file: Path) -> Tuple[int, int, int]:
        """Parse out.cgyro.info to extract simulation dimensions."""
        n_field = 1
        n_species = 1
        n_ky = 1

        with open(info_file, "r") as f:
            for line in f:
                if "N_FIELD" in line:
                    n_field = int(line.split("=")[1].strip())
                elif "n_species" in line:
                    n_species = int(line.split("|")[0].strip().split()[-1])
                elif "ky*rho:" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "n" and i + 1 < len(parts):
                            try:
                                n_ky = int(parts[i + 1])
                                break
                            except ValueError:
                                pass

        return n_field, n_species, n_ky

    def _read_cgyro_ky_flux(
        self,
        ky_flux_file: Path,
        n_field: int,
        n_species: int,
        n_ky: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Read time history from bin.cgyro.ky_flux."""
        try:
            if not ky_flux_file.exists():
                return None, None

            data = np.fromfile(str(ky_flux_file), dtype=np.float64)

            vals_per_time = n_field * n_species * 4 * n_ky
            n_time = len(data) // vals_per_time

            if n_time < 5:
                return None, None

            fluxes = data[: n_time * vals_per_time].reshape(
                (n_time, n_field, n_species, 4, n_ky)
            )

            time_array = np.arange(n_time, dtype=float)

            return time_array, fluxes

        except Exception:
            return None, None

    def _evaluate_single(self, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        platform = self._get_platform_manager(self.work_dir)
        output_dict, std_dict = self.run_on_platform(
            state,
            platform,
            work_dir=self.work_dir,
            model_name="cgyro",
        )
        if self.keep_files == "none":
            self._cleanup_run_dirs(self.work_dir)
        return output_dict, std_dict
