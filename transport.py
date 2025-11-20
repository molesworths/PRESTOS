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
    """

    def __init__(self, options, **kwargs):
        self.options = options
        self.options.update(dict(kwargs) if kwargs else {})
        self.sigma = self.options.get('sigma', 0.1) # relative epistemic uncertainty for transport model outputs

    def evaluate(self, state) -> None:
        raise NotImplementedError
    
    def get_jacobian(self, state, X) -> np.ndarray:
        """Optional method to return Jacobian of transport fluxes w.r.t. parameters.

        Returns
        -------
        J : np.ndarray
            Jacobian matrix of shape (n_fluxes, n_parameters)
        """

        # TODO: implement Jacobian calculation with JAX/autodiff if possible

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
    
    def evaluate(self, state) -> Dict[str, np.ndarray]:
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
    
    def evaluate(self, state) -> None:
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
    
    def evaluate(self, state) -> Dict[str, np.ndarray]:
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
            self.init(state)
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


TRANSPORT_MODELS = {
    'fingerprints': FingerprintsModel,
    'tglf': TGLFModel,
    'fixed': FixedTransport,
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
