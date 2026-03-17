"""Base transport model and shared evaluation logic."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import shutil
import time
from tools import calc, plasma

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

        self.fluxes: Dict = {'gB': {}, 'real': {}}   # real: Ge/Gi [1e19/m²/s], Qe/Qi [MW/m²]
        self.flows: Dict = {'gB': {}, 'real': {}}    # real: Pe/Pi [MW]
        self.transport_dict: Dict = {}

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
        self.surfArea = state.surfArea
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

    def _compute_neoclassical(self, state) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute neoclassical channels (Ge_gB, Qi_gB, Qe_gB, Qie_gB)."""
        if self.neoclassical_model == "analytic":
            self._extract_from_state(state)
            Ge_gB, Qi_gB, Qe_gB = self.compute_analytic()
            Qie_gB = np.zeros_like(np.asarray(Qe_gB, dtype=float))
            if self.roa_eval is not None:
                roa_eval = np.atleast_1d(self.roa_eval)
                if len(Ge_gB) != len(roa_eval):
                    Ge_gB = np.interp(roa_eval, state.roa, Ge_gB)
                    Qi_gB = np.interp(roa_eval, state.roa, Qi_gB)
                    Qe_gB = np.interp(roa_eval, state.roa, Qe_gB)
                    Qie_gB = np.interp(roa_eval, state.roa, Qie_gB)
            return Ge_gB, Qi_gB, Qe_gB, Qie_gB
        if self.neoclassical_model == "neo":
            roa_eval = np.array(self.roa_eval) if self.roa_eval is not None else np.array(state.roa)
            Ge_gB, Qi_gB, Qe_gB, Qie_gB = self.compute_neo(state, roa_eval)
            return Ge_gB, Qi_gB, Qe_gB, Qie_gB
        raise ValueError(f"Unknown neoclassical_model: {self.neoclassical_model}")

    def compute_analytic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute analytic neoclassical fluxes (gB-normalised, dimensionless).

        Returns Ge_gB, Qi_gB, Qe_gB — all dimensionless gyroBohm units.

        Uses the banana-regime neoclassical transport in gyroBohm-normalised form:
            χ̃_i^nc  = f_trap · (q/ε)² / √ε · ν̃_ii
            χ̃_e^nc  = χ̃_i^nc / (1840 · mi_over_mp)   (mass-ratio factor)
        where ν̃ = ν_phys · τ_norm = ν_phys · (a / c_s)  [dimensionless].

        Gradient drives use the gyroBohm-normalised inverse scale lengths
        aLne, aLTe, aLTi (all dimensionless and positive for peaked profiles)
        with sign conventions consistent with TGLF/CGYRO gbflux output.
        """
        eps_safe = np.maximum(np.abs(self.eps), 1e-9)
        q_fac    = (np.abs(self.q) / eps_safe) ** 2 / np.sqrt(eps_safe)  # (q/ε)²/√ε

        # Normalised neoclassical diffusivities [dimensionless gB]
        # self.nuii and self.nuei are already nuii_phys * tau_norm (set in _extract_from_state)
        chii_nc = self.f_trap * q_fac * self.nuii
        chie_nc = self.f_trap * q_fac * self.nuei / (1840.0 * self.mi_over_mp)

        # Neoclassical particle flux [gB units]
        # Positive → outward (same sign convention as TGLF gbflux row 0)
        Ge_gB = chie_nc * (
            1.53 * (1.0 + self.tite) * self.aLne   # density-gradient pinch
            - 0.59 * self.aLTe                       # Te-gradient drive
            - 0.26 * self.tite * self.aLTi           # Ti-gradient drive
        )

        # Neoclassical heat fluxes [gB units]: conduction + convection
        Qi_gB = chii_nc * self.aLTi + 1.5 * self.tite * Ge_gB
        Qe_gB = chie_nc * self.aLTe + 1.5 * Ge_gB

        return Ge_gB, Qi_gB, Qe_gB

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

    def compute_neo(self, state, roa_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run or parse NEO to obtain neoclassical channels (GB units)."""
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

        def parse_transport_flux(path: Path) -> Tuple[float, float, float, float]:
            with open(path, "r") as f:
                lines = f.readlines()

            # Read only the dke block:
            #   # Z pflux_dke eflux_dke mflux_dke
            in_dke_block = False
            data = []
            for raw in lines:
                s = raw.strip()
                if not s:
                    continue

                if s.startswith("#"):
                    s_lower = s.lower()
                    if "pflux_dke" in s_lower and "eflux_dke" in s_lower:
                        in_dke_block = True
                    elif in_dke_block and "pflux_" in s_lower and "_dke" not in s_lower:
                        # Reached the next section header (e.g. gv/tgyro).
                        break
                    continue

                if not in_dke_block:
                    continue

                try:
                    cols = [float(x) for x in s.split()]
                except (ValueError, IndexError):
                    continue

                if len(cols) < 3:
                    continue

                # Keep (Z, pflux_dke, eflux_dke)
                data.append(cols[:3])

            if not data:
                raise RuntimeError(f"Could not parse dke flux block from NEO output: {path}")

            data = np.asarray(data, dtype=float)
            Z = data[:, 0]
            G_dke = data[:, 1]
            Q_dke = data[:, 2]

            e_idx = np.where(np.isclose(Z, -1.0))[0]
            if e_idx.size == 0:
                raise RuntimeError(f"Could not find electron row (Z=-1) in NEO output: {path}")

            ie = int(e_idx[0])
            ion_mask = np.ones_like(Z, dtype=bool)
            ion_mask[ie] = False

            Ge = float(G_dke[ie])
            Qe = float(Q_dke[ie])
            Qi = float(np.sum(Q_dke[ion_mask]))
            Qie = 0.0

            return Ge, Qi, Qe, Qie

        Ge_list, Qi_list, Qe_list, Qie_list = [], [], [], []
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

            Ge, Qi, Qe, Qie = parse_transport_flux(output_path)
            Ge_list.append(Ge)
            Qi_list.append(Qi)
            Qe_list.append(Qe)
            Qie_list.append(Qie)

        return np.array(Ge_list), np.array(Qi_list), np.array(Qe_list), np.array(Qie_list)

    def _exchange_density_gB_to_flux(
        self,
        state,
        qie_gB: np.ndarray,
        roa_eval: np.ndarray,
        q_gb_eval: np.ndarray,
    ) -> np.ndarray:
        """Convert exchange density in gB units to heat flux [MW/m^2] on roa_eval."""
        qie_gB = np.asarray(qie_gB, dtype=float)
        if qie_gB.size == 0:
            return np.zeros_like(roa_eval, dtype=float)
        if qie_gB.size != len(roa_eval):
            qie_gB = np.interp(roa_eval, state.roa, qie_gB)

        a_minor = float(getattr(state, "a", self.a if hasattr(self, "a") else 1.0))
        a_minor = max(a_minor, 1e-30)
        s_gb = q_gb_eval / a_minor  # [MW/m^3]
        qie_mw_m3_eval = qie_gB * s_gb

        if np.all(np.abs(qie_mw_m3_eval) < 1e-30):
            return np.zeros_like(roa_eval, dtype=float)

        try:
            qie_mw_m3_state = np.interp(state.roa, roa_eval, qie_mw_m3_eval)
            qie_flux_state = calc.integrated_flux(
                qie_mw_m3_state,
                np.asarray(state.r, dtype=float),
                np.asarray(state.dVdr, dtype=float),
                np.asarray(state.surfArea, dtype=float),
            )
            return np.interp(roa_eval, state.roa, np.asarray(qie_flux_state, dtype=float))
        except Exception:
            # Fallback local-length estimate when full geometry arrays are unavailable.
            return qie_mw_m3_eval * a_minor

    def _assemble_fluxes(
        self,
        state,
        *,
        Ge_turb_gB: Optional[np.ndarray] = None,
        Ge_neo_gB: Optional[np.ndarray] = None,
        Qi_turb_gB: Optional[np.ndarray] = None,
        Qi_neo_gB: Optional[np.ndarray] = None,
        Qe_turb_gB: Optional[np.ndarray] = None,
        Qe_neo_gB: Optional[np.ndarray] = None,
        Qie_turb_gB: Optional[np.ndarray] = None,
        Qie_neo_gB: Optional[np.ndarray] = None,
        model_label: str = "Transport",
        **legacy_kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Assemble flux/flow dicts and output from gB-normalised fluxes.

        Inputs
        ------
        Ge_turb_gB / Ge_neo_gB  : electron particle flux (gB-normalised), at roa_eval
        Qi_turb_gB / Qi_neo_gB  : ion heat flux (gB-normalised), same grid
        Qe_turb_gB / Qe_neo_gB  : electron heat flux (gB-normalised), same grid
        Qie_turb_gB / Qie_neo_gB: exchange density (Sgb-normalised), same grid

        Populates
        ---------
        self.fluxes         – nested dict matching flux_flow_ref.txt
        self.flows          – nested dict matching flux_flow_ref.txt
        self.transport_dict – {'fluxes': self.fluxes, 'flows': self.flows}

        Returns
        -------
        (output_dict, std_dict) – flat dicts of physical values at roa_eval
        """
        Ge_turb_gB = Ge_turb_gB if Ge_turb_gB is not None else legacy_kwargs.pop("Gamma_turb", None)
        Ge_neo_gB = Ge_neo_gB if Ge_neo_gB is not None else legacy_kwargs.pop("Gamma_neo", None)
        Qi_turb_gB = Qi_turb_gB if Qi_turb_gB is not None else legacy_kwargs.pop("Qi_turb", None)
        Qi_neo_gB = Qi_neo_gB if Qi_neo_gB is not None else legacy_kwargs.pop("Qi_neo", None)
        Qe_turb_gB = Qe_turb_gB if Qe_turb_gB is not None else legacy_kwargs.pop("Qe_turb", None)
        Qe_neo_gB = Qe_neo_gB if Qe_neo_gB is not None else legacy_kwargs.pop("Qe_neo", None)

        required = {
            "Ge_turb_gB": Ge_turb_gB,
            "Ge_neo_gB": Ge_neo_gB,
            "Qi_turb_gB": Qi_turb_gB,
            "Qi_neo_gB": Qi_neo_gB,
            "Qe_turb_gB": Qe_turb_gB,
            "Qe_neo_gB": Qe_neo_gB,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Missing required transport channels for _assemble_fluxes: {missing}")

        if self.roa_eval is None:
            self.roa_eval = list(state.roa)
        roa_eval = np.atleast_1d(self.roa_eval)

        def _to_roa_eval(data: np.ndarray) -> np.ndarray:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1 and len(arr) == len(state.roa):
                return np.interp(roa_eval, state.roa, arr)
            return arr

        if not self.state_vars_extracted:
            self._extract_from_state(state)

        # ------------------------------------------------------------------
        # 1. gB normalization factors (at roa_eval)
        # ------------------------------------------------------------------
        Zeff     = _to_roa_eval(self.Zeff)
        surfArea = _to_roa_eval(self.surfArea)
        g_gb     = _to_roa_eval(self.g_gb)    # [1e20/m²/s]
        q_gb     = _to_roa_eval(self.q_gb)    # [MW/m²]
        Te_ev    = _to_roa_eval(self.Te)       # [keV]
        Ti_ev    = _to_roa_eval(self.Ti)       # [keV]

        G_gB = g_gb * 10.0       # [1e19/m²/s]  – particle-flux gB norm
        Q_gB = q_gb               # [MW/m²]       – heat-flux gB norm
        P_gB = q_gb * surfArea    # [MW]           – heat-flow gB norm
        _safe = lambda x: np.where(np.abs(x) < 1e-30, 1e-30, x)

        # ------------------------------------------------------------------
        # 2. Convert gB inputs to real units
        # ------------------------------------------------------------------
        Ge_turb_gB = _to_roa_eval(Ge_turb_gB)
        Ge_neo_gB  = _to_roa_eval(Ge_neo_gB)
        Qe_turb_gB = _to_roa_eval(Qe_turb_gB)
        Qe_neo_gB  = _to_roa_eval(Qe_neo_gB)
        Qi_turb_gB = _to_roa_eval(Qi_turb_gB)
        Qi_neo_gB  = _to_roa_eval(Qi_neo_gB)
        Qie_turb_gB = np.zeros_like(Qe_turb_gB) if Qie_turb_gB is None else _to_roa_eval(Qie_turb_gB)
        Qie_neo_gB = np.zeros_like(Qe_turb_gB) if Qie_neo_gB is None else _to_roa_eval(Qie_neo_gB)

        Ge_turb = Ge_turb_gB * G_gB   # [1e19/m²/s]
        Ge_neo  = Ge_neo_gB  * G_gB
        Qe_t    = Qe_turb_gB * Q_gB   # [MW/m²]
        Qe_n    = Qe_neo_gB  * Q_gB
        Qi_t    = Qi_turb_gB * Q_gB
        Qi_n    = Qi_neo_gB  * Q_gB

        # Exchange source density [MW/m^3] integrated to equivalent heat flux [MW/m^2].
        Qie_t = self._exchange_density_gB_to_flux(state, Qie_turb_gB, roa_eval, Q_gB)
        Qie_n = self._exchange_density_gB_to_flux(state, Qie_neo_gB, roa_eval, Q_gB)

        # PORTALS convention: exchange term is additive in electron channel and
        # subtractive in ion channel for final transport totals.
        Qe_x_t = Qie_t
        Qe_x_n = Qie_n
        Qi_x_t = -Qie_t
        Qi_x_n = -Qie_n

        # ------------------------------------------------------------------
        # 3. Totals
        # ------------------------------------------------------------------
        Ge       = Ge_turb + Ge_neo
        Qi_total = Qi_t + Qi_n + Qi_x_t + Qi_x_n
        Qe_total = Qe_t + Qe_n + Qe_x_t + Qe_x_n

        # ion particle fluxes (ambipolarity: Gi ≈ Ge / Zeff)
        Gi_turb = Ge_turb / np.maximum(Zeff, 1e-12)
        Gi_neo  = Ge_neo  / np.maximum(Zeff, 1e-12)
        Gi      = Gi_turb + Gi_neo

        # ------------------------------------------------------------------
        # 4. Convective heat flows  Ce = 1.5 T Ge A   [MW]
        #    Using plasma.get_convective_flow which takes T[keV], Ge[1e19/m²/s], A[m²]
        # ------------------------------------------------------------------
        Ce_turb = plasma.get_convective_flow(Te_ev, Ge_turb, surfArea)
        Ce_neo  = plasma.get_convective_flow(Te_ev, Ge_neo,  surfArea)
        Ce      = Ce_turb + Ce_neo

        Ci_turb = plasma.get_convective_flow(Ti_ev, Gi_turb, surfArea)
        Ci_neo  = plasma.get_convective_flow(Ti_ev, Gi_neo,  surfArea)
        Ci      = Ci_turb + Ci_neo

        # ------------------------------------------------------------------
        # 5. Heat flows  Pe/Pi = Q * surfArea   [MW]
        # ------------------------------------------------------------------
        Pe_turb = Qe_t * surfArea
        Pe_neo  = Qe_n * surfArea
        Pe_exch_turb = Qe_x_t * surfArea
        Pe_exch_neo  = Qe_x_n * surfArea
        Pe      = Pe_turb + Pe_neo
        Pe      = Pe + Pe_exch_turb + Pe_exch_neo

        Pi_turb = Qi_t * surfArea
        Pi_neo  = Qi_n * surfArea
        Pi_exch_turb = Qi_x_t * surfArea
        Pi_exch_neo  = Qi_x_n * surfArea
        Pi      = Pi_turb + Pi_neo
        Pi      = Pi + Pi_exch_turb + Pi_exch_neo

        # Conductive = total - convective
        De_turb = Pe_turb - Ce_turb
        De_neo  = Pe_neo  - Ce_neo
        De_exch_turb = Pe_exch_turb
        De_exch_neo = Pe_exch_neo
        De      = Pe - Ce

        Di_turb = Pi_turb - Ci_turb
        Di_neo  = Pi_neo  - Ci_neo
        Di_exch_turb = Pi_exch_turb
        Di_exch_neo = Pi_exch_neo
        Di      = Pi - Ci

        # ------------------------------------------------------------------
        # 6. Nested fluxes dict  (matches flux_flow_ref.txt)
        #    real: Ge/Gi [1e19/m²/s], Qe/Qi [MW/m²]
        #    gB: dimensionless; turb/neo inputs are already normalised
        # ------------------------------------------------------------------
        Gi_turb_gB = Ge_turb_gB / np.maximum(Zeff, 1e-12)
        Gi_neo_gB  = Ge_neo_gB  / np.maximum(Zeff, 1e-12)
        self.fluxes = {
            'real': {
                'Ge': {'turb': Ge_turb, 'neo': Ge_neo, 'total': Ge},
                'Gi': {'turb': Gi_turb, 'neo': Gi_neo, 'total': Gi},
                'Qe': {
                    'turb': Qe_t,
                    'neo': Qe_n,
                    'turb_exch': Qe_x_t,
                    'neo_exch': Qe_x_n,
                    'total': Qe_total,
                },
                'Qi': {
                    'turb': Qi_t,
                    'neo': Qi_n,
                    'turb_exch': Qi_x_t,
                    'neo_exch': Qi_x_n,
                    'total': Qi_total,
                },
            },
            'gB': {
                'Ge': {'turb': Ge_turb_gB, 'neo': Ge_neo_gB, 'total': Ge / _safe(G_gB)},
                'Gi': {'turb': Gi_turb_gB, 'neo': Gi_neo_gB, 'total': Gi / _safe(G_gB)},
                'Qe': {
                    'turb': Qe_turb_gB,
                    'neo': Qe_neo_gB,
                    'turb_exch': Qe_x_t / _safe(Q_gB),
                    'neo_exch': Qe_x_n / _safe(Q_gB),
                    'total': Qe_total / _safe(Q_gB),
                },
                'Qi': {
                    'turb': Qi_turb_gB,
                    'neo': Qi_neo_gB,
                    'turb_exch': Qi_x_t / _safe(Q_gB),
                    'neo_exch': Qi_x_n / _safe(Q_gB),
                    'total': Qi_total / _safe(Q_gB),
                },
            },
        }

        # ------------------------------------------------------------------
        # 7. Nested flows dict  (matches flux_flow_ref.txt)
        #    real: [MW]   gB: dimensionless (÷ P_gB = Q_gB × surfArea)
        #    3rd key: 'conv', 'cond', 'total'
        #    each contains:  'turb', 'neo', 'total'
        # ------------------------------------------------------------------
        zeros = np.zeros_like(Pe_turb)

        def _flow_entry(turb, neo, total, turb_exch=None, neo_exch=None):
            entry = {'turb': turb, 'neo': neo, 'total': total}
            if turb_exch is not None:
                entry['turb_exch'] = turb_exch
            if neo_exch is not None:
                entry['neo_exch'] = neo_exch
            return entry

        flows_real = {
            'Pe': {
                'conv':  _flow_entry(Ce_turb, Ce_neo, Ce, zeros, zeros),
                'cond':  _flow_entry(De_turb, De_neo, De, De_exch_turb, De_exch_neo),
                'total': _flow_entry(Pe_turb, Pe_neo, Pe, Pe_exch_turb, Pe_exch_neo),
            },
            'Pi': {
                'conv':  _flow_entry(Ci_turb, Ci_neo, Ci, zeros, zeros),
                'cond':  _flow_entry(Di_turb, Di_neo, Di, Di_exch_turb, Di_exch_neo),
                'total': _flow_entry(Pi_turb, Pi_neo, Pi, Pi_exch_turb, Pi_exch_neo),
            },
        }
        _P = _safe(P_gB)
        flows_gB = {
            P_key: {
                level: {comp: flows_real[P_key][level][comp] / _P for comp in flows_real[P_key][level].keys()}
                for level in ('conv', 'cond', 'total')
            }
            for P_key in ('Pe', 'Pi')
        }
        self.flows = {'real': flows_real, 'gB': flows_gB}
        self.transport_dict = {'fluxes': self.fluxes, 'flows': self.flows}

        # ------------------------------------------------------------------
        # 8. Flat output dict for solver
        # ------------------------------------------------------------------
        all_channels = {
            "Ge": Ge, "Gi": Gi,
            "Qe": Qe_total, "Qi": Qi_total,
            "Pe": Pe, "Pi": Pi,
            "Ce": Ce, "Ci": Ci,
            "De": De, "Di": Di,
        }

        if self.output_vars is None:
            if hasattr(self, "labels"):
                self.output_vars = list(self.labels)
            else:
                self.output_vars = [
                    key for key in ["Ge", "Gi", "Ce", "Ci", "Pe", "Pi", "Qe", "Qi"]
                    if key in all_channels
                ]

        def _value_at_roa(data: np.ndarray, roa: float) -> float:
            arr = np.asarray(data)
            if arr.ndim == 0:
                return float(arr)
            if arr.ndim == 1:
                if len(arr) == len(state.roa):
                    xp = state.roa
                elif len(arr) == len(roa_eval):
                    xp = roa_eval
                else:
                    return float(np.nan_to_num(arr, nan=0.0)[0])
                if np.any(np.isclose(xp, roa, atol=1e-3)):
                    idx = int(np.where(np.isclose(xp, roa, atol=1e-3))[0][0])
                    return float(np.nan_to_num(arr, nan=0.0)[idx])
                return float(np.interp(roa, xp, np.nan_to_num(arr, nan=0.0)))
            return float(np.nan_to_num(arr, nan=0.0).flat[0])

        output_dict = {}
        for key in self.output_vars:
            if key in all_channels:
                output_dict[key] = [_value_at_roa(all_channels[key], roa) for roa in roa_eval]

        std_dict = {
            key: [self.sigma * abs(output_dict[key][i]) for i in range(len(roa_eval))]
            for key in output_dict
        }

        self.output_dict = output_dict
        self.std_dict = std_dict
        self.state_vars_extracted = False

        # transport_dict (self.fluxes / self.flows) is the canonical source for the
        # surrogate — all turb/neo/total components at all normalizations are there.
        # No flat Y_gB needed; solver_base passes transport_dict directly to add_sample.

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
