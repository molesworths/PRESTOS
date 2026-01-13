"""
Transport model interface and implementations.

Refactored to follow the project pattern used by other modules (e.g., boundary):
- Base class attaches a simple container object at state.transport to store options and outputs.
- Concrete models compute fluxes on state.roa grid and store them under state.transport.*

Currently includes a Fingerprints-like simplified model and a fixed-transport test model.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


class TransportBase:
    """Base class for transport models using PlasmaState.

    Pattern:
    - call init(state, **kwargs) to attach a container at state.transport
    - call evaluate(state) to populate state.transport.* fields on state.roa grid
    
    Supports batch processing:
    - evaluate(state) handles both single PlasmaState and list of PlasmaState objects
    - Returns results for single state or list of results for batched states
    """

    def __init__(self, options, **kwargs):
        self.options = options
        self.options.update(dict(kwargs) if kwargs else {})
        self.sigma = self.options.get('sigma', 0.1) # relative epistemic uncertainty for transport model outputs

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
    
    def get_jacobian(self, state, X) -> np.ndarray:
        """Optional method to return Jacobian of transport fluxes w.r.t. parameters.

        Returns
        -------
        J : np.ndarray
            Jacobian matrix of shape (n_fluxes, n_parameters)
        """

        # TODO: implement Jacobian calculation with autodiff if possible

        raise NotImplementedError

class FingerprintsModel(TransportBase):
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
            exb_src = getattr(self, 'ExB_source', self.options.get('ExB_source', 'model'))
            if exb_src == 'state-pol':
                gamma_ExB = np.abs(getattr(state, 'gamma_exb_pol_hat', state.tau_norm * np.gradient(getattr(state, 'vpol', 0*x), state.r)))
            elif exb_src == 'state-both':
                gpol = np.abs(getattr(state, 'gamma_exb_pol_hat', state.tau_norm * np.gradient(getattr(state, 'vpol', 0*x), state.r)))
                gtor = np.abs(getattr(state, 'gamma_exb_tor_hat', state.tau_norm * getattr(state, 'gamma_exb_tor', 0*x)))
                gamma_ExB = gpol + gtor
            else:
                V_ExB = rhostar*dpi_dx/ni
                gamma_ExB = rhostar*d2pi_dx2/ni + aLni*V_ExB
            gamma_ExB = self.ExB_scale * gamma_ExB
        else:
            gamma_ExB = 0*x

        # Neoclassical transport
        chii_nc = f_trap * (Ti * (q / np.maximum(eps, 1e-9))**2) * nuii
        chie_nc = f_trap * ((Te * (q / np.maximum(eps, 1e-9))**2) / (1840.0 * mi_over_mp)) * nuei
        
        Gamma_neo = chie_nc * (-1.53 * (1.0 + Ti / Te) * dne_dx + 
                               0.59 * (ne / Te) * dTe_dx + 
                               0.26 * (ne / Te) * dTi_dx)
        Qi_neo = -ne * chii_nc * dTi_dx + 1.5 * Ti * Gamma_neo
        Qe_neo = -ne * chie_nc * dTe_dx + 1.5 * Te * Gamma_neo
        
        # Critical gradients
        RLTi_crit = np.maximum((4.0 / 3.0) * (1.0 + Ti / Te), 0.8 * aLne * state.aspect_ratio)
        RLTe_crit = np.maximum((4.0 / 3.0) * (1.0 + Te / Ti), 1.4 * aLne * state.aspect_ratio)
        
        # Turbulent transport (ITG)
        ky_ITG = 0.3

        if self.non_local:
            # Non-local closure with correlation length
            aLTi_eff = np.maximum(aLTi - RLTi_crit / state.aspect_ratio, 0.0)
            gamma_ITG = ky_ITG * (aLTi_eff / self.ITG_lcorr)**0.5
        else:
            gamma_ITG = ky_ITG * np.maximum(aLTi - RLTi_crit / state.aspect_ratio, 0.0)**0.5

        gamma_eff = np.maximum(gamma_ITG - abs(gamma_ExB), 0.)
        I_ITG = (gamma_eff / ky_ITG**2)**2
        chi_ITG = I_ITG * (Ti**1.5)
        Gamma_ITG = 0.1*f_trap * chi_ITG * (-dne_dx - 0.25 * ne * aLTe / a)
        Qi_ITG = -ne * chi_ITG * dTi_dx + 1.5 * Ti * Gamma_ITG
        Qe_ITG = -ne * f_trap * chi_ITG * dTe_dx + 1.5 * Te * Gamma_ITG
        
        # ETG transport
        z_ETG = np.maximum(aLTe - RLTe_crit / state.aspect_ratio, 0.0) / np.maximum(aLne, 1e-12)
        chi_ETG = (1.0 / 60.0) * 1.5 * (Te**1.5) * aLTe * z_ETG
        Qe_ETG = -ne * chi_ETG * dTe_dx
        
        # KBM transport (simplified)
        ky_KBM = 0.1
        alpha_crit = 2.0
        RLp_crit = alpha_crit / (np.maximum(beta, 1e-12) * np.maximum(q, 1e-9)**2)

        if self.non_local:
            aLp_eff = np.maximum(aLpe - RLp_crit / state.aspect_ratio, 0.0)
            gamma_KBM = ky_KBM * (aLp_eff / self.ITG_lcorr)**0.5
        else:
            gamma_KBM = ky_KBM * np.maximum(aLpe - RLp_crit / state.aspect_ratio, 0.0)**0.5
        
        I_KBM = (gamma_KBM / ky_KBM**2)**2
        chi_KBM = I_KBM * (Ti**1.5)
        Gamma_KBM = 0.1*-chi_KBM * dne_dx
        Qi_KBM = -ne * chi_KBM * dTi_dx + 1.5 * Ti * Gamma_KBM
        Qe_KBM = -ne * chi_KBM * dTe_dx + 1.5 * Te * Gamma_KBM
        
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

        # Convert particle flux to convective power flow
        Ge_to_Ce = 1.5 * Te * state.Qnorm_to_P
        Gi_to_Ci = 1.5 * Ti * state.Qnorm_to_P

        # Storage

        self.model = 'Fingerprints'
        self.Ge_turb = Gamma_turb
        self.Ge_neo = Gamma_neo
        self.Ge = Gamma_turb + Gamma_neo
        self.Gi_turb = self.Ge_turb / state.Zeff
        self.Gi_neo = self.Ge_neo / state.Zeff
        self.Gi = self.Gi_turb + self.Gi_neo
        self.Ce_turb = self.Ge_turb * Ge_to_Ce
        self.Ce_neo = self.Ge_neo * Ge_to_Ce
        self.Ce = self.Ce_turb + self.Ce_neo
        self.Gi_turb = self.Ge_turb / state.Zeff
        self.Gi_neo = self.Ge_neo / state.Zeff
        self.Gi = self.Gi_turb + self.Gi_neo
        self.Ci_turb = self.Gi_turb * Gi_to_Ci
        self.Ci_neo = self.Gi_neo * Gi_to_Ci
        self.Ci = self.Ci_turb + self.Ci_neo
        self.Qi_turb = Qi_turb
        self.Qi_neo = Qi_neo
        self.Qi = self.Qi_turb + self.Qi_neo
        self.Pi_turb = self.Qi_turb * state.Qnorm_to_P
        self.Pi_neo = self.Qi_neo * state.Qnorm_to_P
        self.Pi = self.Pi_turb + self.Pi_neo
        self.Qe_turb = Qe_turb
        self.Qe_neo = Qe_neo
        self.Qe = self.Qe_turb + self.Qe_neo
        self.Pe_turb = self.Qe_turb * state.Qnorm_to_P
        self.Pe_neo = self.Qe_neo * state.Qnorm_to_P
        self.Pe = self.Pe_turb + self.Pe_neo

        # Provide dict for requested outputs
        output_dict = {
            key: [
            np.nan_to_num(getattr(self, key)[np.where(np.isclose(state.roa, roa, atol=1e-3))[0][0]], nan=0)
            if np.any(np.isclose(state.roa, roa, atol=1e-3))
            else np.interp(roa, state.roa, np.nan_to_num(getattr(self, key), nan=0))
            for roa in self.roa_eval
            ]
            for key in self.output_vars
        }

        std_dict = {
            key: [
                self.sigma * abs(output_dict[key][i])
                for i in range(len(self.roa_eval))
            ]
            for key in self.output_vars
        }

        return output_dict, std_dict


class TGLFModel(TransportBase):
    """
    Direct TGLF SAT-1 transport model.
    
    Runs TGLF executable for each radial location. Slower than
    Fingerprints but can handle edge/non-standard conditions.
    """
    
    def __init__(self, tglf_path: Optional[str] = None,
                 n_parallel: int = 4):
        """
        Parameters
        ----------
        tglf_path : str, optional
            Path to TGLF executable
        n_parallel : int
            Number of parallel TGLF runs
        """
        self.tglf_path = tglf_path
        self.n_parallel = n_parallel
        
        raise NotImplementedError("Direct TGLF interface not yet implemented.")
    
    def _evaluate_single(self, state) -> None:
        """Run TGLF for all radial locations (not yet implemented)."""
        raise NotImplementedError


class FixedTransport(TransportBase):
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


class AnalyticTransport(TransportBase):
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


        # Neoclassical transport
        chii_nc = f_trap * (Ti * (q / np.maximum(eps, 1e-9))**2) * nuii
        chie_nc = f_trap * ((Te * (q / np.maximum(eps, 1e-9))**2) / (1840.0 * mi_over_mp)) * nuei
        
        Gamma_neo = chie_nc * (-1.53 * (1.0 + Ti / Te) * dne_dx + 
                               0.59 * (ne / Te) * dTe_dx + 
                               0.26 * (ne / Te) * dTi_dx)
        Qi_neo = -ne * chii_nc * dTi_dx + 1.5 * Ti * Gamma_neo
        Qe_neo = -ne * chie_nc * dTe_dx + 1.5 * Te * Gamma_neo

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

        # Convert particle flux to convective power flow
        Ge_to_Ce = 1.5 * Te * state.Qnorm_to_P
        Gi_to_Ci = 1.5 * Ti * state.Qnorm_to_P

        # Storage

        self.model = 'Fingerprints'
        self.Ge_turb = Gamma_turb
        self.Ge_neo = Gamma_neo
        self.Ge = Gamma_turb + Gamma_neo
        self.Gi_turb = self.Ge_turb / state.Zeff
        self.Gi_neo = self.Ge_neo / state.Zeff
        self.Gi = self.Gi_turb + self.Gi_neo
        self.Ce_turb = self.Ge_turb * Ge_to_Ce
        self.Ce_neo = self.Ge_neo * Ge_to_Ce
        self.Ce = self.Ce_turb + self.Ce_neo
        self.Gi_turb = self.Ge_turb / state.Zeff
        self.Gi_neo = self.Ge_neo / state.Zeff
        self.Gi = self.Gi_turb + self.Gi_neo
        self.Ci_turb = self.Gi_turb * Gi_to_Ci
        self.Ci_neo = self.Gi_neo * Gi_to_Ci
        self.Ci = self.Ci_turb + self.Ci_neo
        self.Qi_turb = Qi_turb
        self.Qi_neo = Qi_neo
        self.Qi = self.Qi_turb + self.Qi_neo
        self.Pi_turb = self.Qi_turb * state.Qnorm_to_P
        self.Pi_neo = self.Qi_neo * state.Qnorm_to_P
        self.Pi = self.Pi_turb + self.Pi_neo
        self.Qe_turb = Qe_turb
        self.Qe_neo = Qe_neo
        self.Qe = self.Qe_turb + self.Qe_neo
        self.Pe_turb = self.Qe_turb * state.Qnorm_to_P
        self.Pe_neo = self.Qe_neo * state.Qnorm_to_P
        self.Pe = self.Pe_turb + self.Pe_neo

        # Provide dict for requested outputs
        output_dict = {
            key: [
            np.nan_to_num(getattr(self, key)[np.where(np.isclose(state.roa, roa, atol=1e-3))[0][0]], nan=0)
            if np.any(np.isclose(state.roa, roa, atol=1e-3))
            else np.interp(roa, state.roa, np.nan_to_num(getattr(self, key), nan=0))
            for roa in self.roa_eval
            ]
            for key in self.output_vars
        }

        std_dict = {
            key: [
                self.sigma * abs(output_dict[key][i])
                for i in range(len(self.roa_eval))
            ]
            for key in self.output_vars
        }

        return output_dict, std_dict

TRANSPORT_MODELS = {
    'fingerprints': FingerprintsModel,
    'tglf': TGLFModel,
    'fixed': FixedTransport,
    'analytic': AnalyticTransport,
}


def create_transport_model(config: Dict[str, Any]) -> TransportBase:
    """Factory to create a transport model instance using a config dict.

    Expected config format:
    {"type": "fingerprints"|"tglf"|"fixed", "kwargs": { ... }}
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
