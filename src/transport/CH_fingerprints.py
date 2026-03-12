
from typing import Dict

import numpy as np

from .TransportBase import TransportBase


class CH_fingerprints(TransportBase):
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
        aLpe = aLne + aLTe
        aLpi = aLni + aLTi
        RLTi = aLTi * state.aspect_ratio
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
        d2ne_dx2 = state.d2ne * a**2 # d2ne/dr2 * a**2
        d2ni_dx2 = np.gradient(dni_dx, x) 
        d2Te_dx2 = state.d2te * a**2 # d2Te/dr2 * a**2
        d2Ti_dx2 = state.d2ti * a**2 # d2Ti/dr2 * a**2
        d2pi_dx2 = Ti*d2ni_dx2 + 2*dni_dx*dTi_dx + ni*d2Ti_dx2
        
        # Collision frequencies
        nuii = state.nuii*state.tau_norm
        nuei = state.nuei*state.tau_norm

        if self.exb_on:
            if self.exb_src == "state":
                gamma_ExB = getattr(state, "gamma_exb_norm", np.zeros_like(x))
            else:
                V_ExB = rhostar * dpi_dx / ni
                gamma_ExB = rhostar * d2pi_dx2 / ni + aLni * V_ExB
            gamma_ExB = self.exb_scale * gamma_ExB
        else:
            gamma_ExB = 0 * state.x
        gamma_ExB = np.maximum(np.absolute(gamma_ExB), 0.0)

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
        ky_ITG = 0.3 #k_y * rho_s
        omega_r_ITG = ky_ITG / state.aspect_ratio # a * omega_r / v_norm = (kyrhos * (cs/vnorm) * a/R)
        #omega_r_ITG *= np.sqrt(Te)
        gamma_lin_ITG = omega_r_ITG * np.sqrt( np.maximum(RLTi - RLTi_crit, 0))
        gamma_eff = np.maximum(gamma_lin_ITG - abs(gamma_ExB), 0.)
        I_ITG = (gamma_eff / ky_ITG**2)**2  #|e delta phi / Te|^2 / rhostar_norm^2
        chi_ITG = ((gamma_eff)/(omega_r_ITG**2 + gamma_eff**2))*I_ITG
        chi_ITG = I_ITG * (Ti**1.5)
        #chi_ITG = [gamma_eff/(omega_r^2 + gamma_eff^2)]*(kyrhos)^2 * cs^2 * |e dphi/Te|^2
        #chi_ITG = (gamma_eff / (omega_r_ITG**2 + gamma_eff**2)) * Te * (ky_ITG**2) * I_ITG
        #chi_ITG = ((Te * gamma_eff)/(omega_r_ITG**2 + gamma_eff**2))*I_ITG
        
        Gamma_ITG = f_trap * chi_ITG * (-dne_dx - 0.25 * ne * aLTe / a)
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

        gamma_KBM = ky_KBM * np.maximum(aLpe - RLp_crit / state.aspect_ratio, 0.0)**0.5
        
        I_KBM = (gamma_KBM / ky_KBM**2)**2
        chi_KBM = I_KBM * (Ti**1.5)
        Gamma_KBM = -chi_KBM * dne_dx
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

        return self._assemble_fluxes(
            state,
            Ge_turb_gB=Gamma_turb,
            Ge_neo_gB=Gamma_neo,
            Qi_turb_gB=Qi_turb,
            Qi_neo_gB=Qi_neo,
            Qe_turb_gB=Qe_turb,
            Qe_neo_gB=Qe_neo,
            model_label="CH_fingerprints",
        )