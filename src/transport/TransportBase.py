"""Base transport model and shared evaluation logic."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import shutil
import time

import numpy as np

from .utils import _resolve_gacode_executable, _run_neo_single_rho

if TYPE_CHECKING:
    from state import PlasmaState
    from tools.platforms import PlatformManager

logger = logging.getLogger(__name__)


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
        self.verbose = options.get("verbose", False)
        self.options = options
        self.options.update(dict(kwargs) if kwargs else {})
        self.sigma = self.options.get("sigma", 0.1)
        self.neoclassical_model = self.options.get("neoclassical_model", "analytic")

        # ExB settings
        self.exb_on = self.options.get("exb_on", True)
        self.exb_scale = self.options.get("exb_scale", 1.0)
        self.exb_src = self.options.get("exb_source", "model")

        workflow_work_dir = Path(self.options.get("work_dir", "."))
        self.neo_work_dir = workflow_work_dir / "neo"

        self.neo_opts = self.options.get("NEO_options", self.options.get("neoclassical_options", {}))
        self.roa_eval = self.options.get("roa_eval", None)
        self.output_vars = self.options.get("output_vars", None)
        self.state_vars_extracted = False

        self.external = False # if True, model is responsible for its own platform execution and evaluate() only orchestrates inputs/outputs
        self.platform_config = self.options.get("platform", None)
        self.platform = None
        _pcfg = self.platform_config if isinstance(self.platform_config, dict) else {}
        self.mpi_tasks = int(_pcfg.get("mpi_tasks", _pcfg.get("n_parallel", self.options.get("mpi_tasks", self.options.get("n_parallel", 1)))))
        self.threads_per_task = int(_pcfg.get("threads_per_task", _pcfg.get("n_threads", self.options.get("threads_per_task", self.options.get("n_threads", 1)))))
        self.n_gpus = int(_pcfg.get("n_gpus", self.options.get("n_gpus", 0)))
        self.n_ram_gb = int(_pcfg.get("n_ram_gb", self.options.get("n_ram_gb", 16)))
        if self.platform_config and isinstance(self.platform_config, dict):
            platform_name = self.platform_config.get("name", "").lower()
            machine = str(self.platform_config.get("machine", "")).lower()
            is_remote = (
                (platform_name and platform_name not in ("local", "localhost"))
                or (machine and machine not in ("local", "localhost"))
                or self.platform_config.get("ssh_tunnel")
            )
            if is_remote:
                try:
                    from tools.platforms import PlatformManager

                    self.platform = PlatformManager(self.platform_config)
                    print(
                        "Transport model will run on platform: "
                        f"{self.platform_config.get('machine', 'configured platform')}"
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize platform manager: {e}")
                    self.platform = None

        self.eval_log = None
        self.log_evaluations = self.options.get("log_evaluations", False)
        eval_log_config = self.options.get("evaluation_log", {})

        if self.log_evaluations or eval_log_config.get("enabled", False):
            from evaluation_log import TransportEvaluationLog, get_default_log_path

            log_path = eval_log_config.get("path") or get_default_log_path(self.options)
            try:
                self.eval_log = TransportEvaluationLog(str(log_path))
                print(f"Transport evaluation logging enabled: {log_path}")
            except Exception as e:
                print(f"Warning: Could not initialize evaluation log: {e}")
                self.eval_log = None

    def __del__(self):
        """Destructor - cleanup platform connections if they exist."""
        if hasattr(self, 'platform') and self.platform is not None:
            try:
                self.platform.cleanup()
            except Exception:
                pass  # Silently fail in destructor to avoid issues during shutdown

    def _is_batched(self, state) -> bool:
        """Check if state is a batch (list/array of states) vs single state."""
        if isinstance(state, (list, tuple)):
            return True
        if isinstance(state, np.ndarray) and state.dtype == object:
            return True
        return False

    def evaluate(self, state) -> Any:
        """Evaluate transport model for single or batch of states.

        If a remote platform is configured, _evaluate_single() runs on that platform.
        All orchestration and batch processing remains local.
        """
        if self._is_batched(state):
            results = []
            for single_state in state:
                if self.platform is not None and self.external:
                    try:
                        result = self.run_on_platform(single_state, self.platform)
                    except Exception as e:
                        print(f"Warning: Platform execution failed, falling back to local: {e}")
                        result = self._evaluate_single(single_state)
                else:
                    result = self._evaluate_single(single_state)
                results.append(result)
            return results

        if self.platform is not None and self.external:
            try:
                return self.run_on_platform(state, self.platform)
            except Exception as e:
                print(f"Warning: Platform execution failed, falling back to local: {e}")
                return self._evaluate_single(state)
        return self._evaluate_single(state)

    def _get_platform_manager(self, work_dir: Optional[Path] = None) -> "PlatformManager":
        """Return configured platform manager or a local default."""
        if self.platform is not None:
            return self.platform

        if getattr(self, "_local_platform", None) is not None:
            return self._local_platform

        from tools.platforms import PlatformManager

        platform_cfg = dict(self.platform_config) if isinstance(self.platform_config, dict) else {}
        if not platform_cfg:
            platform_cfg = {"name": "local", "machine": "local"}

        platform_cfg.setdefault("name", "local")
        platform_cfg.setdefault("machine", "local")
        if work_dir is not None and "scratch" not in platform_cfg:
            platform_cfg["scratch"] = str(work_dir)

        self._local_platform = PlatformManager(platform_cfg)
        return self._local_platform

    def _cleanup_run_dirs(self, work_dir: Path, patterns: Optional[List[str]] = None) -> None:
        """Remove per-rho run directories when keep_files='none'."""
        for pattern in (patterns or ["rho_*"]):
            for path in work_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    def _evaluate_single(self, state) -> None:
        """Evaluate transport model for a single state."""
        raise NotImplementedError

    def _prepare_platform_inputs(self, state, work_dir: Path) -> None:
        """Prepare input files for platform execution."""
        pass

    def _get_platform_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for staging and retrieval."""
        return {
            "input_files": ["input.gacode"],
            "input_folders": ["roa_*"],
            "output_files": ["*.json", "out.*", "bin.*"],
        }

    def _get_platform_commands(self, remote_work_dir: Path, state, platform) -> List[str]:
        """Generate commands to execute on remote platform."""
        return []

    def _read_platform_results(self, work_dir: Path, state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read results from retrieved output files."""
        raise NotImplementedError("Model must implement _read_platform_results")

    def _extract_from_state(self, state) -> None:
        """Extract commonly used variables from state for transport calculations."""
        self.x = state.roa
        self.a = state.a
        self.R = state.R
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
        self.aLpe = self.aLne + self.aLTe
        self.aLpi = self.aLni + self.aLTi
        self.tite = state.tite
        self.kappa = state.kappa
        self.delta = state.delta
        self.shear = state.shear # magnetic shear
        self.B = state.B
        self.q = state.q
        self.Zeff = state.Zeff
        self.mi_over_mp = state.mi_ref
        self.f_trap = state.f_trap
        self.beta = state.betae * (1 + self.Ti / self.Te)
        self.alpha = state.alpha
        self.rhostar = state.rhostar
        self.c_s = state.c_s
        self.rho_s = state.rho_s
        self.dne_dx = -self.aLne * self.ne
        self.dTe_dx = -self.aLTe * self.Te
        self.dTi_dx = -self.aLTi * self.Ti
        self.dni_dx = -self.aLni * self.ni
        self.dpe_dx = -self.aLpe * self.pe
        self.dpi_dx = -self.aLpi * self.pi
        self.d2ne_dx2 = state.d2ne * self.a**2
        self.d2ni_dx2 = self.d2ne_dx2 / self.Zeff
        self.d2Te_dx2 = state.d2te * self.a**2
        self.d2Ti_dx2 = state.d2ti * self.a**2
        self.d2pi_dx2 = self.Ti * self.d2ni_dx2 + 2 * self.dni_dx * self.dTi_dx + self.ni * self.d2Ti_dx2
        self.nuii = state.nuii * state.tau_norm
        self.nuei = state.nuei * state.tau_norm
        self.nustar = state.nustar
        self.nu_eff = state.nu_eff
        self.Qnorm_to_P = state.Qnorm_to_P
        self.g_gb = state.g_gb
        self.q_gb = state.q_gb
        self.tau_norm = state.tau_norm
        self.mi_over_me = state.mi_over_me
        self.gamma_exb_state = getattr(state, "gamma_exb_norm", np.zeros_like(self.x))
        self.f_imp = state.f_imp if hasattr(state, "f_imp") else np.ones_like(self.x) * 0.01
        self.state_vars_extracted = True
        

    def _log_evaluation(self, state, outputs: Dict[str, Any], roa_idx: int):
        """Log this evaluation to the transport evaluation database."""
        if self.eval_log is None:
            return

        try:
            roa = self.roa_eval[roa_idx] if self.roa_eval is not None else state.roa[roa_idx]
            state_features = self._extract_state_features(state, roa)
            model_class = f"{self.__class__.__module__}.{self.__class__.__name__}"
            model_settings = self._get_model_settings()
            self.eval_log.add_evaluation(
                model_class=model_class,
                model_settings=model_settings,
                roa=float(roa),
                state_features=state_features,
                outputs=outputs,
                skip_duplicates=True,
            )
        except Exception as e:
            if hasattr(self, "_log_warn_once"):
                pass
            else:
                print(f"Warning: Evaluation logging failed: {e}")
                self._log_warn_once = True

    def _extract_state_features(self, state, roa: float) -> Dict[str, float]:
        """Extract dimensionless state features at given roa location."""
        def interp(var_name):
            var = getattr(state, var_name, None)
            if var is None:
                return 0.0
            return float(np.interp(roa, state.roa, var))

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
        """Get model-specific settings for logging."""
        return {
            "neoclassical_model": self.neoclassical_model,
            "sigma": self.sigma,
        }

    def _gbflux_to_physical(self, rho: float, Ge: float, Qi: float, Qe: float) -> Tuple[float, float, float]:
        """Convert gyroBohm-normalized fluxes to physical units at rho."""
        g_gb = np.asarray(self.g_gb)
        q_gb = np.asarray(self.q_gb)
        if g_gb.size == 1:
            g_val = float(g_gb)
        else:
            g_val = np.interp(rho, self.x, g_gb)
        if q_gb.size == 1:
            q_val = float(q_gb)
        else:
            q_val = np.interp(rho, self.x, q_gb)

        g_area_1e19 = g_val * 10.0

        return Ge * g_area_1e19, Qi * q_val, Qe * q_val

    def _compute_neoclassical(self, state) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute neoclassical fluxes according to selected model."""
        if self.neoclassical_model == "analytic":
            self._extract_from_state(state)
            Gamma_neo, Qi_neo, Qe_neo = self.compute_analytic()
            if self.roa_eval is not None:
                roa_eval = np.atleast_1d(self.roa_eval)
                if len(Gamma_neo) != len(roa_eval):
                    Gamma_neo = np.interp(roa_eval, state.roa, Gamma_neo)
                    Qi_neo = np.interp(roa_eval, state.roa, Qi_neo)
                    Qe_neo = np.interp(roa_eval, state.roa, Qe_neo)
            return Gamma_neo, Qi_neo, Qe_neo
        if self.neoclassical_model == "neo":
            roa_eval = np.array(self.roa_eval) if self.roa_eval is not None else np.array(state.roa)
            Ge_neo_gb, Qi_neo_gb, Qe_neo_gb = self.compute_neo(state, roa_eval)
            Ge_neo, Qi_neo, Qe_neo = self._gbflux_to_physical(roa_eval, Ge_neo_gb, Qi_neo_gb, Qe_neo_gb)
            if len(Ge_neo) == len(roa_eval):
                Gamma_neo = np.interp(state.roa, roa_eval, Ge_neo)
                Qi_neo = np.interp(state.roa, roa_eval, Qi_neo)
                Qe_neo = np.interp(state.roa, roa_eval, Qe_neo)
                return Gamma_neo, Qi_neo, Qe_neo
            raise RuntimeError("NEO output size does not match evaluation grid")
        raise ValueError(f"Unknown neoclassical_model: {self.neoclassical_model}")

    def compute_analytic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute analytic neoclassical fluxes."""
        chii_nc = self.f_trap * (self.Ti * (self.q / np.maximum(self.eps, 1e-9)) ** 2) * self.nuii
        chie_nc = (
            self.f_trap
            * ((self.Te * (self.q / np.maximum(self.eps, 1e-9)) ** 2) / (1840.0 * self.mi_over_mp))
            * self.nuei
        )

        Gamma_neo = chie_nc * (
            -1.53 * (1.0 + self.Ti / self.Te) * self.dne_dx
            + 0.59 * (self.ne / self.Te) * self.dTe_dx
            + 0.26 * (self.ne / self.Te) * self.dTi_dx
        )
        Qi_neo = -self.ne * chii_nc * self.dTi_dx + 1.5 * self.Ti * Gamma_neo
        Qe_neo = -self.ne * chie_nc * self.dTe_dx + 1.5 * self.Te * Gamma_neo
        
        return Gamma_neo, Qi_neo, Qe_neo

    # def compute_analytic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """Compute analytic neoclassical fluxes across all collisionality regimes.
        
    #     Uses comprehensive neoclassical theory including:
    #     - Banana regime (nu_star << 1): Trapped particle dominance
    #     - Plateau regime (nu_star ~ 1): Transition regime
    #     - Pfirsch-Schlüter regime (nu_star >> 1): Collisional regime
        
    #     References:
    #     - Hirschman & Sigmar, Nucl. Fusion 21, 1079 (1981)
    #     - Hinton & Hazeltine, Rev. Mod. Phys. 48, 239 (1976)
    #     - Chang & Hinton, Phys. Fluids 25, 1493 (1982)
    #     """
    #     # Geometric factors
    #     eps_safe = np.maximum(self.eps, 1e-9)
    #     q_safe = np.maximum(np.abs(self.q), 1e-9)
    #     f_trap = self.f_trap
        
    #     # Normalized collisionality
    #     nu_star = self.nustar  # Already available from state
        
    #     # Define regime interpolation functions
    #     def K_banana(nu):
    #         """Banana regime interpolation factor"""
    #         return 1.0 / (1.0 + nu)
        
    #     def K_plateau(nu, eps):
    #         """Plateau regime interpolation factor"""
    #         return np.sqrt(nu) / (1.0 + np.sqrt(nu))
        
    #     def K_PS(nu):
    #         """Pfirsch-Schlüter regime factor"""
    #         return nu / (1.0 + nu)
        
    #     # === ION HEAT DIFFUSIVITY ===
    #     # Banana regime contribution
    #     chi_i_banana = f_trap * (self.Ti * (q_safe / eps_safe) ** 2) * self.nuii
        
    #     # Plateau regime contribution (Hinton-Hazeltine eq. 6.8)
    #     chi_i_plateau = 0.66 * self.Ti * q_safe**2 * np.sqrt(self.nuii / eps_safe)
        
    #     # Pfirsch-Schlüter regime contribution
    #     chi_i_PS = 4.0 * self.Ti * q_safe**2 * self.nuii / eps_safe**1.5
        
    #     # Combine regimes with smooth interpolation
    #     w_banana = K_banana(nu_star / eps_safe**1.5)
    #     w_plateau = K_plateau(nu_star, eps_safe) * (1.0 - w_banana)
    #     w_PS = K_PS(nu_star * eps_safe**1.5) * (1.0 - w_banana - w_plateau)
        
    #     chi_i_nc = (w_banana * chi_i_banana + 
    #                 w_plateau * chi_i_plateau + 
    #                 w_PS * chi_i_PS)
        
    #     # === ELECTRON HEAT DIFFUSIVITY ===
    #     # Mass ratio factor
    #     mass_ratio = 1.0 / (1840.0 * self.mi_over_mp)
        
    #     # Banana regime
    #     chi_e_banana = f_trap * (self.Te * (q_safe / eps_safe) ** 2) * self.nuei * mass_ratio
        
    #     # Plateau regime  
    #     chi_e_plateau = 0.66 * self.Te * q_safe**2 * np.sqrt(self.nuei / eps_safe) * mass_ratio
        
    #     # Pfirsch-Schlüter regime (enhanced by Zeff)
    #     chi_e_PS = 4.0 * self.Te * q_safe**2 * self.nuei * self.Zeff / eps_safe**1.5
        
    #     chi_e_nc = (w_banana * chi_e_banana + 
    #                 w_plateau * chi_e_plateau + 
    #                 w_PS * chi_e_PS)
        
    #     # === PARTICLE TRANSPORT (Ware pinch + diffusion) ===
    #     # Main ion particle flux coefficients from Chang-Hinton
        
    #     # Banana regime coefficients
    #     D11_banana = 0.67 * f_trap * self.Te * (q_safe / eps_safe)**2 * self.nuii
    #     D12_banana = -1.17 * f_trap * (q_safe / eps_safe)**2 * self.nuii
    #     D13_banana = 0.74 * f_trap * (q_safe / eps_safe)**2 * self.nuii
        
    #     # Plateau regime coefficients
    #     D11_plateau = 0.35 * self.Te * q_safe**2 * np.sqrt(self.nuii / eps_safe)
    #     D12_plateau = -0.61 * q_safe**2 * np.sqrt(self.nuii / eps_safe)
    #     D13_plateau = 0.39 * q_safe**2 * np.sqrt(self.nuii / eps_safe)
        
    #     # PS regime coefficients (classical)
    #     D11_PS = 2.0 * self.Te * q_safe**2 * self.nuii / eps_safe**1.5
    #     D12_PS = -1.5 * q_safe**2 * self.nuii / eps_safe**1.5
    #     D13_PS = 0.5 * q_safe**2 * self.nuii / eps_safe**1.5
        
    #     # Interpolate coefficients
    #     D11 = w_banana * D11_banana + w_plateau * D11_plateau + w_PS * D11_PS
    #     D12 = w_banana * D12_banana + w_plateau * D12_plateau + w_PS * D12_PS
    #     D13 = w_banana * D13_banana + w_plateau * D13_plateau + w_PS * D13_PS
        
    #     # Include impurity effects on electron diffusion (Hirschman-Sigmar eq. 4.25)
    #     Te_safe = np.maximum(self.Te, 1e-6)
    #     alpha_e = D11 * (self.Zeff - 1.0) / self.Zeff  # Impurity-modified electron diffusion
        
    #     # Particle flux (Chang-Hinton form with all three regimes)
    #     Gamma_neo_gb = (
    #         -D11 * self.dne_dx / self.ne +  # Density gradient drive
    #         D12 * self.dTe_dx / Te_safe +    # Electron temperature gradient (Ware pinch)
    #         D13 * self.dTi_dx / self.Ti -    # Ion temperature gradient
    #         alpha_e * self.dne_dx / self.ne  # Impurity correction
    #     ) * self.ne
        
    #     # === ION HEAT FLUX ===
    #     # Conductive + convective contributions
    #     Qi_conduct = -self.ne * chi_i_nc * self.dTi_dx
    #     Qi_convect = 1.5 * self.Ti * Gamma_neo_gb  # 3/2 convective factor
        
    #     # Add bootstrap-related correction in banana regime (Hirschman-Sigmar eq. 5.12)
    #     bootstrap_correction = w_banana * 0.5 * f_trap * self.Ti * self.ne * q_safe**2 * (
    #         self.dTi_dx / self.Ti - self.dne_dx / self.ne
    #     )
        
    #     Qi_neo_gb = Qi_conduct + Qi_convect + bootstrap_correction
        
    #     # === ELECTRON HEAT FLUX ===
    #     Qe_conduct = -self.ne * chi_e_nc * self.dTe_dx
    #     Qe_convect = 1.5 * self.Te * Gamma_neo_gb  # 3/2 convective factor
        
    #     # Electron-ion collisional energy exchange in PS regime
    #     collisional_exchange = w_PS * 3.0 * mass_ratio * self.ne * self.nuei * (self.Te - self.Ti)
        
    #     Qe_neo_gb = Qe_conduct + Qe_convect - collisional_exchange

    #     Ge_neo, Qi_neo, Qe_neo = Gamma_neo_gb, Qi_neo_gb, Qe_neo_gb #self._gbflux_to_physical(self.x, Gamma_neo_gb, Qi_neo_gb, Qe_neo_gb)
        
    #     return Ge_neo, Qi_neo, Qe_neo

    def compute_neo(self, state, roa_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run or parse NEO to obtain neoclassical fluxes (GB units)."""
        work_dir = self.neo_work_dir
        keep_files = self.neo_opts.get("keep_files", "minimal")
        settings = self.neo_opts.get("settings", {})
        multipliers = self.neo_opts.get("multipliers", {})

        work_dir.mkdir(parents=True, exist_ok=True)
        neo_exec = _resolve_gacode_executable("neo", self.neo_opts)

        input_gacode = work_dir / "input.gacode"
        state.to_gacode().write(str(input_gacode))

        roa_eval_list = list(roa_eval)

        if self.mpi_tasks > 1:
            with ProcessPoolExecutor(max_workers=self.mpi_tasks) as executor:
                futures = {
                    executor.submit(
                        _run_neo_single_rho,
                        rho,
                        state,
                        work_dir,
                        input_gacode,
                        neo_exec,
                        settings,
                        multipliers,
                        self.threads_per_task,
                        keep_files,
                    ): rho
                    for rho in roa_eval_list
                }

                for future in as_completed(futures):
                    rho = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        raise RuntimeError(f"NEO failed at rho={rho:.3f}: {str(e)}")
        else:
            for rho in roa_eval_list:
                _run_neo_single_rho(
                    rho,
                    state,
                    work_dir,
                    input_gacode,
                    neo_exec,
                    settings,
                    multipliers,
                    self.threads_per_task,
                    keep_files,
                )

        rho_dirs = {p.name.replace("rho_", ""): p for p in work_dir.glob("rho_*") if p.is_dir()}
        available_rhos = np.array([float(k) for k in rho_dirs.keys()]) if rho_dirs else np.array([])

        def parse_transport_flux(path: Path) -> Tuple[float, float, float]:
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            if not lines:
                raise RuntimeError(f"Empty NEO output: {path}")

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

            ie = int(np.where(np.isclose(Z, -1.0))[0][0])
            Ge = float(G[ie])
            Qe = float(Q[ie])
            Qi = float(np.sum(np.delete(Q, ie)))

            return Ge, Qi, Qe

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

            max_wait = 120.0
            poll_interval = 1.0
            start_time = time.time()

            while time.time() - start_time < max_wait:
                if output_path.exists() and output_path.stat().st_size > 50:
                    break
                time.sleep(poll_interval)
            else:
                raise FileNotFoundError(
                    f"NEO output not generated after {max_wait}s: {output_path}"
                )

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

        Ge_to_Ce = _to_roa_eval(1.5 * self.Te * self.Qnorm_to_P)
        Gi_to_Ci = _to_roa_eval(1.5 * self.Ti * self.Qnorm_to_P)

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
            key: [_value_at_roa(getattr(self, key), roa) for roa in self.roa_eval]
            for key in self.output_vars
        }

        std_dict = {
            key: [self.sigma * abs(output_dict[key][i]) for i in range(len(self.roa_eval))]
            for key in self.output_vars
        }

        self.output_dict = output_dict
        self.std_dict = std_dict

        self.state_vars_extracted = False

        return output_dict, std_dict

    def run_on_platform(
        self,
        state: "PlasmaState",
        platform: Union["PlatformManager", Dict[str, Any]],
        work_dir: Optional[Path] = None,
        model_name: str = "transport_model",
        cleanup: bool = True,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Simplified platform execution: stage inputs, run commands, retrieve outputs."""
        from tools.platforms import PlatformManager

        # Track whether we created the platform manager (so we know to clean it up)
        platform_created_here = False
        if isinstance(platform, dict):
            platform = PlatformManager(platform)
            platform_created_here = True
        elif not isinstance(platform, PlatformManager):
            raise TypeError("platform must be PlatformManager or dict")

        if model_name == "transport_model":
            model_name = self.__class__.__name__.lower()

        if work_dir is None:
            import tempfile

            work_dir = Path(tempfile.mkdtemp(prefix=f"{model_name}_"))
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        remote_work_dir = None

        try:
            if verbose: print(f"\n[{model_name.upper()} Platform Execution]")

            if verbose: print("  Preparing input files...")
            self._prepare_platform_inputs(state, work_dir)
            if verbose: print("  ✓ Input files ready")

            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_dir_name = f"prestos_{model_name}_{timestamp}"

            remote_scratch = platform.platform.get_scratch_path()
            remote_work_dir = remote_scratch / job_dir_name

            if not platform.platform.is_local():
                platform.executor.execute(f"mkdir -p {remote_work_dir}", check=False)
                if verbose: print(f"  ✓ Remote directory: {remote_work_dir}")
            else:
                remote_work_dir.mkdir(parents=True, exist_ok=True)

            patterns = self._get_platform_file_patterns()
            if verbose: print("  Transferring files...")
            platform.stage_inputs(
                work_dir,
                remote_work_dir,
                file_patterns=patterns.get("input_files", []),
                folder_patterns=patterns.get("input_folders", []),
            )
            if verbose: print("  ✓ Files transferred")

            commands = self._get_platform_commands(remote_work_dir, state, platform)
            if not commands:
                raise RuntimeError(f"{model_name} generated no execution commands")

            if verbose: print(f"  Executing {len(commands)} command(s)...")

            if platform.platform.scheduler == "slurm" and not platform.platform.is_local():
                script_commands = "\n".join(commands)

                script_content = platform.slurm_submitter.generate_batch_script(
                    command=script_commands,
                    job_name=f"prestos_{model_name}",
                    n_tasks=max(1, self.mpi_tasks),
                    cpus_per_task=max(1, self.threads_per_task),
                    walltime_minutes=int(self.options.get("platform_walltime_minutes", 30)),
                    memory_gb=int(self.options.get("platform_memory_gb")),
                    n_gpus=self.n_gpus,
                )

                script_path = remote_work_dir / "prestos_run.sbatch"
                job_id = platform.slurm_submitter.submit_job(script_content, script_path)
                if verbose: print(f"  ✓ SLURM job submitted: {job_id}")

                success = platform.wait_for_job(
                    job_id,
                    check_interval=10,
                    timeout=int(self.options.get("platform_wait_timeout", 3600)),
                )

                if not success:
                    raise RuntimeError(f"SLURM job {job_id} failed or timed out")
                if verbose: print("  ✓ Job completed")
            else:
                for i, cmd in enumerate(commands, 1):
                    if verbose: print(f"  [{i}/{len(commands)}] {cmd[:60]}...")
                    returncode, stdout, stderr = platform.executor.execute(
                        cmd,
                        cwd=remote_work_dir,
                        check=False,
                        timeout=3600,
                    )
                    if returncode != 0:
                        raise RuntimeError(
                            f"Command {i} failed:\n  {cmd}\n  Error: {stderr[:200]}"
                        )
                if verbose: print("  ✓ All commands completed")

            if verbose: print("  Retrieving results...")
            output_patterns = patterns.get("output_files", [])
            if output_patterns:
                platform.retrieve_outputs(remote_work_dir, work_dir, output_patterns)
            if verbose: print("  ✓ Results retrieved")

            if verbose: print("  Parsing output files...")
            output_dict, std_dict = self._read_platform_results(work_dir, state)
            if verbose: print("  ✓ Parsing complete")

            return output_dict, std_dict

        finally:
            if remote_work_dir and cleanup:
                try:
                    platform.file_manager.remove_directory(remote_work_dir)
                    if verbose:
                        scope = "Local" if platform.platform.is_local() else "Remote"
                        print(f"  ✓ {scope} directory cleaned up")
                except Exception as e:
                    if verbose: print(f"  ⚠ Cleanup failed: {e}")
            
            # Always cleanup SSH/SFTP connections if we created the platform
            if platform_created_here:
                try:
                    platform.cleanup()
                except Exception as e:
                    logger.warning("Error closing platform connections: %s", e)
