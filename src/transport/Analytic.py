"""Analytic turbulent transport model for tokamak edge plasma.

Based on reduced critical-gradient model including ITG, TEM, ETG, KBM, RBM modes
with slab/toroidal branch competition, nonlinear suppression, and strong turbulence effects.

Reference: AnalyticTransportModel.md (v1.1, Feb 2026)
"""

from typing import Dict, Tuple

import numpy as np

from src import state

from .TransportBase import TransportBase


class Analytic(TransportBase):
    """Reduced analytic turbulent transport model for edge plasma (0.85 < r/a < 1.0)."""

    def __init__(self, options: dict):
        """Initialize Analytic transport model with all tunable parameters."""
        super().__init__(options)
        self.external = False  # Fully analytic, no external dependencies
        self.modes = self.options.get("modes", "all")  # "all", "ITG", "ETG", "KBM", "RBM", "neo"
        self.mode_list = ["itg", "tem", "etg", "balloon"] if self.modes == "all" else \
            [[i.lower()] for i in self.modes if i.upper() in ["ITG", "TEM", "ETG", "balloon"]]

        # ===== Section 11.1: Geometric & Shaping =====
        self.C_geo = self.options.get("C_geo", 0.2)
        self.C_kappa = self.options.get("C_kappa", 0.2)
        self.C_delta = self.options.get("C_delta", 0.2)
        self.C_shaf = self.options.get("C_shaf", 0.2) # alpha dependence of effective shear

        # ===== Section 11.2-11.4: Critical Gradients =====
        self.a_ITG = self.options.get("a_ITG", 1.5)
        self.b_ITG = self.options.get("b_ITG", 1.5)
        self.s_ITG = self.options.get("s_ITG", 0.15) # shear dependence for ITG critical gradient
        self.c_ITG = self.options.get("c_ITG", 0.3)

        self.a_TEM = self.options.get("a_TEM", 2)
        self.b_TEM = self.options.get("b_TEM", 1.3) # collisional damping coefficient for TEM critical gradient
        self.s_TEM = self.options.get("s_TEM", 0.3) # shear dependence for TEM critical gradient
        self.c_TEM = self.options.get("c_TEM", 2.1)

        self.a_ETG = self.options.get("a_ETG", 1.5)
        self.b_ETG = self.options.get("b_ETG", 0.8)
        self.s_ETG = self.options.get("s_ETG", 0.2) # shear dependence for ETG critical gradient
        self.c_ETG = self.options.get("c_ETG", 1.1)

        self.a_KBM = self.options.get("a_KBM", 2.)
        self.s_KBM = self.options.get("s_KBM", 0.2)
        #self.b_KBM = self.options.get("b_KBM", 0.0)

        #self.a_RBM = self.options.get("a_RBM", 2.0)
        #self.s_RBM = self.options.get("s_RBM", 0.7)
        self.b_RBM = self.options.get("b_RBM", -1.5)

        # self.crit_grads_coefs = {'a':{"itg": self.a_ITG, "tem": self.a_TEM, "etg": self.a_ETG, "kbm": self.a_KBM, "rbm": self.a_RBM}, 
        #                        'b':{"itg": self.b_ITG, "tem": self.b_TEM, "etg": self.b_ETG, "kbm": self.b_KBM, "rbm": self.b_RBM},
        #                        'c':{"itg": self.c_ITG, "tem": self.c_TEM, "etg": self.c_ETG, "kbm": self.c_KBM, "rbm": self.c_RBM}}

        # ===== Section 11.5: Wavenumbers (kyρs) =====
        self.k_hat_ITG = self.options.get("k_hat_ITG", 0.1)
        self.k_hat_TEM = self.options.get("k_hat_TEM", 0.2)
        self.k_hat_ETG = self.options.get("k_hat_ETG", 1.0)
        self.k_hat_balloon = self.options.get("k_hat_balloon", 0.1)
        self.k_hats = {"itg": self.k_hat_ITG, "tem": self.k_hat_TEM, "etg": self.k_hat_ETG, "balloon": self.k_hat_balloon}

        # ===== Section 11.6: Growth Rates =====
        self.p_drive = self.options.get("p_drive", 0.5)
        #self.c_nu_damp = self.options.get("c_nu_damp", 0.85)

        # ===== Section 11.7: Nonlinear Suppression =====
        self.p_ExB = self.options.get("p_ExB", 2)
        self.C_nl = self.options.get("C_nl", 1.)
        self.R_c = self.options.get("R_c", 1.0)
        self.m_exp = self.options.get("m_exp", 1.0)
        self.C_Z = self.options.get("C_Z", 0.0)
        self.C_EM = self.options.get("C_EM", 0.5)

        # Cross-scale coupling coefficients (Section 11.7.4)

        self.C_cross = self.options.get("C_cross", 0.8) # general cross-scale coupling strength, 0 -> OFF
        self.cross_sat = self.options.get("cross_sat", 0.5) # saturation strength for cross-scale coupling effects
        self.C_ion_itg = self.options.get("C_ion_itg", 0.5)
        self.C_ion_etg = self.options.get("C_ion_etg", 0.5)
        self.C_ion_tem = self.options.get("C_ion_tem", 0.5)
        self.C_ion_balloon = self.options.get("C_ion_balloon", 0.5)

        self.cross_coefs = {"itg": self.C_ion_itg, "etg": self.C_ion_etg, "tem": self.C_ion_tem, "balloon": self.C_ion_balloon}

        # ===== Section 11.8: Mixing Length & Transport Ratios =====
        self.C_chi = self.options.get("C_chi", 1.0) # general mixing length scalar

        # ITG transport fractions acting on effective diffusivities
        self.f_i_ITG = self.options.get("f_i_ITG", 1.0)
        self.f_e_ITG = self.options.get("f_e_ITG", 0.6)
        self.f_D_ITG = self.options.get("f_D_ITG", 0.5)

        # TEM transport fractions acting on effective diffusivities
        self.nustar_crit = self.options.get("nustar_crit", 0.2) # for TEM transition
        self.f_i_TEM = self.options.get("f_i_TEM", 1.0)
        self.f_e_TEM = self.options.get("f_e_TEM", 0.7)
        self.f_D_TEM = self.options.get("f_D_TEM", 0.6)

        # ETG transport fractions acting on effective diffusivities
        self.f_i_ETG = self.options.get("f_i_ETG", 0.0)
        self.f_e_ETG = self.options.get("f_e_ETG", 1.0)
        self.f_D_ETG = self.options.get("f_D_ETG", 0.0)

        # Ballooning transport fractions acting on effective diffusivities
        self.f_i_balloon = self.options.get("f_i_balloon", 1.0)
        self.f_e_balloon = self.options.get("f_e_balloon", 1.0)
        self.f_D_balloon = self.options.get("f_D_balloon", 0.6)

        self.chi_ratios = {"f_i": {'itg': self.f_i_ITG, 'tem': self.f_i_TEM, 'etg': self.f_i_ETG, 'balloon': self.f_i_balloon},
                                 "f_e": {'itg': self.f_e_ITG, 'tem': self.f_e_TEM, 'etg': self.f_e_ETG, 'balloon': self.f_e_balloon},
                                 "f_D": {'itg': self.f_D_ITG, 'tem': self.f_D_TEM, 'etg': self.f_D_ETG, 'balloon': self.f_D_balloon}}

        # ===== Section 11.9: Pinch Velocity =====
        self.C_pinch_T = self.options.get("C_pinch_T", 0.1)
        self.C_pinch_n = self.options.get("C_pinch_n", 0.2)
        self.C_pinch_beta = self.options.get("C_pinch_beta", 0.1)
        self.C_pinch_nu = self.options.get("C_pinch_nu", 0.1)
        self.C_pinch_shear = self.options.get("C_pinch_shear", 0.1)
        self.q_ref = self.options.get("q_ref", 1.5) # reference q for pinch scaling

        self.pinch_coeffs = {"T": self.C_pinch_T, "n": self.C_pinch_n, "beta": self.C_pinch_beta, \
                             "nu": self.C_pinch_nu, "shear": self.C_pinch_shear}


    def _compute_geometric_factors(self) -> Dict[str, np.ndarray]:
        """Compute geometric factors for mode growth rates (Section 4).
        
        Returns:
            G_geo: geometric factor from shear, elongation, triangularity
            G_curv: curvature drive factor from edge topology
            G_total: combined geometric factor
        """
        # Access geometric parameters (via state if available)
        s_hat = self.shear - self.C_shaf * self.alpha  # effective shear
        kappa = self.kappa
        delta = self.delta  # Triangularity
        
        # Geometric factor: (1 + C_geo*|s|)(1 + C_geo*kappa^2)(1 - C_geo*delta^2)
        G_geo = (1.0 + self.C_geo * np.abs(s_hat)) * (1.0 + self.C_kappa * kappa**2) * (1.0 - self.C_delta * delta**2)
        
        # Curvature drive factor (edge is unfavorable by default at outboard midplane)
        D_B = 1.0  # Unfavorable curvature
        G_curv = (1.0 + D_B) / 2.0  # = 1.0 for unfavorable
        
        G_total = G_geo * G_curv
        
        return {"geo": G_geo, "curv": G_curv, "total": G_total}

    def _compute_critical_gradients(self) -> Dict[str, np.ndarray]:

        s_hat = self.shear - self.C_shaf * self.alpha
        tau = self.tite / self.Zeff

        # ===== ITG: threshold in R/L_Ti =====
        RLTi_tor  = self.a_ITG * (1 + self.b_ITG * tau) * (1 + self.s_ITG * s_hat ) #2*s_hat/self.q)
        RLTi_etai = self.c_ITG * self.aLne * self.aspect_ratio   # eta_i branch in R/L_Ti space
        RLTi_crit = np.maximum(RLTi_tor, RLTi_etai)

        # ===== TEM: threshold in R/L_ne =====
        # Density-gradient driven: trapped electrons destabilize, collisions restabilize
        # Dominant in high-R/L_n pedestal regime (Roach 2009, Dannert & Jenko 2005)
        RLne_tor  = (self.a_TEM 
                    * (1 + self.s_TEM * s_hat) 
                    * self.f_trap                        # trapped fraction drives
                    / (1 + self.b_TEM * self.nustar))    # collisional detrapping stabilizes
        RLne_etae = self.aLTe * self.aspect_ratio / self.c_TEM  # eta_e branch: R/L_ne ~ R/L_Te/eta_e_crit
        RLne_crit = np.maximum(RLne_tor, RLne_etae)       # same max logic as ITG

        # ===== ETG: threshold in R/L_Te (owns electron temp gradient drive) =====
        # Fix tite direction: ETG scales like ITG with tite (not 1/tite)  
        RLTe_tor  = self.a_ETG * (1 + self.b_ETG / tau) * (1 + self.s_ETG * s_hat) * \
                    (1-1.5*self.eps) # finite aspect ratio effects
        RLTe_etae = self.c_ETG * self.aLne * self.aspect_ratio
        RLTe_crit = np.maximum(RLTe_tor, RLTe_etae)

        # ===== MTM: threshold .... ===== #
        # TODO

        # ===== KBM/RBM: threshold in alpha (unchanged) =====
        alpha_crit_kbm = self.a_KBM * (1 + self.s_KBM * self.shear)
        alpha_crit_rbm = alpha_crit_kbm * self.nustar**self.b_RBM
        alpha_crit     = np.minimum(alpha_crit_kbm, alpha_crit_rbm)

        return {
            "itg":     RLTi_crit,   # R/L_Ti threshold
            "tem":     RLne_crit,   # R/L_ne threshold
            "etg":     RLTe_crit,   # R/L_Te threshold
            "balloon": alpha_crit
        }

    def _compute_drives(self) -> Dict[str, np.ndarray]:
        """Compute mode drives (how far above critical gradient).
        
        Returns:
            Delta_ITG, Delta_TEM, Delta_ETG, Delta_BALLOON
        """
        
        Delta_ITG = np.maximum(self.aLTi - self.crit_grads['itg'] / self.aspect_ratio, 0.0)
        Delta_TEM = np.maximum(self.aLne - self.crit_grads['tem'] / self.aspect_ratio, 0.0)
        Delta_ETG = np.maximum(self.aLTe - self.crit_grads['etg'] / self.aspect_ratio, 0.0)
        Delta_balloon = np.maximum((self.alpha - self.crit_grads['balloon']), 0.0)
        
        return {"itg": Delta_ITG, "tem": Delta_TEM, "etg": Delta_ETG, "balloon": Delta_balloon}

    def _compute_growth_rates(self) -> Dict[str, np.ndarray]:
        """Compute linear growth rates for all modes (Section 6).
        
        Returns:
            gamma_ITG, gamma_TEM, gamma_ETG, gamma_balloon (all unitless)
        """
        
        # ITG growth rate
        gamma_ITG = self.k_hat_ITG * self.geo_factors['total'] * np.maximum(self.mode_drives['itg'], 1e-6) ** self.p_drive
        
        # TEM growth rate with trapped electron fraction dependence
        gamma_TEM = self.k_hat_TEM * self.geo_factors['total'] * self.f_trap**0.5 * np.maximum(self.mode_drives['tem'], 1e-6) ** self.p_drive
        
        # ETG growth rate with (mi/me)^0.5 factor in chi_gB scaling
        gamma_ETG = self.k_hat_ETG * self.geo_factors['total'] * np.maximum(self.mode_drives['etg'], 1e-6) ** self.p_drive

        gamma_balloon = self.k_hat_balloon * self.geo_factors['total'] * np.maximum(self.mode_drives['balloon'], 1e-6) ** self.p_drive
            #np.sqrt(gamma_KBM**2 + gamma_RBM**2) 
            # #np.maximum(gamma_KBM, gamma_RBM)
        
        return {"itg": gamma_ITG, "tem": gamma_TEM, "etg": gamma_ETG, "balloon": gamma_balloon}

    def _compute_suppression_factors(self) -> Dict[str, np.ndarray]:
        """Compute all nonlinear suppression factors (Section 7).
        
        """
        gamma_ITG = self.growth_rates["itg"]
        gamma_TEM = self.growth_rates["tem"]
        gamma_ETG = self.growth_rates["etg"]
        gamma_balloon = self.growth_rates["balloon"]

        # E×B shear suppression: gamma_ExB = d/dr(Er/rB)
        # Use dpi_dx as proxy if Er not available
        if self.exb_on:
            if self.exb_src == "state":
                gamma_ExB = self.gamma_exb_state
            else:
                V_ExB = self.rhostar * self.dpi_dx / self.ni
                gamma_ExB = self.rhostar * self.d2pi_dx2 / self.ni + self.aLni * V_ExB
            gamma_ExB = self.exb_scale * gamma_ExB
        else:
            gamma_ExB = 0 * self.x
        gamma_ExB = np.maximum(np.absolute(gamma_ExB), 0.0)
        self.growth_rates['exb'] = gamma_ExB
        
        # F_E,m = 1 / [1 + (gamma_ExB / gamma_m)^p_ExB]
        F_ExB_ITG = 1.0 / (1.0 + (gamma_ExB / np.maximum(gamma_ITG, 1e-6)) ** self.p_ExB)
        F_ExB_TEM = 1.0 / (1.0 + (gamma_ExB / np.maximum(gamma_TEM, 1e-6)) ** self.p_ExB)
        F_ExB_ETG = 1.0 / (1.0 + (gamma_ExB / np.maximum(gamma_ETG, 1e-6)) ** self.p_ExB)
        F_ExB_balloon = 1.0 / (1.0 + (gamma_ExB / np.maximum(gamma_balloon, 1e-6)) ** self.p_ExB)

        # Regime proxy: R_m = gamma_m / (C_nl * gamma_dominant + gamma_ExB + eps_reg)
        gamma_dominant = np.maximum.reduce([gamma_ITG, gamma_TEM, gamma_balloon])
        denom = self.C_nl * gamma_dominant + gamma_ExB
        
        R_ITG = gamma_ITG / np.maximum(denom, 1e-9)
        R_TEM = gamma_TEM / np.maximum(denom, 1e-9)
        R_ETG = gamma_ETG / np.maximum(denom, 1e-9)
        R_balloon = gamma_balloon / np.maximum(denom, 1e-9)
        
        # Effective exponent: alpha_eff = 1 + 1/[1 + (R/R_c)^m_exp]
        # Ranges from 1 (QL, R<<1) to 2 (SMT, R>>1)
        alpha_eff_ITG = 1.0 + (R_ITG/self.R_c)**self.m_exp / (1 + (R_ITG/self.R_c)**self.m_exp) 
        alpha_eff_TEM = 1.0 + (R_TEM/self.R_c)**self.m_exp / (1 + (R_TEM/self.R_c)**self.m_exp) 
        alpha_eff_ETG = 1.0 + (R_ETG/self.R_c)**self.m_exp / (1 + (R_ETG/self.R_c)**self.m_exp)
        alpha_eff_balloon = 1.0 + (R_balloon/self.R_c)**self.m_exp / (1 + (R_balloon/self.R_c)**self.m_exp)
        
        # Cross-scale interactions
        # ITG, TEM, and balloon suppresses ETG (ion scale zonal flows shear ETG streamers)
        F_cross_ETG = 1.0 / (1.0 + self.C_cross * (self.cross_coefs['etg'] * (gamma_ITG + gamma_TEM + gamma_balloon) / \
                                                   np.maximum(gamma_ETG, 1e-6))**self.cross_sat)

        # ITG suppressed by competition with TEM and ballooning for same free energy
        F_cross_ITG = 1.0 / (1.0 + self.C_cross * (self.cross_coefs['itg'] * (gamma_TEM + gamma_balloon) / \
                                                   np.maximum(gamma_ITG, 1e-6))**self.cross_sat)

        # TEM suppressed by ITG and ballooning competition
        F_cross_TEM = 1.0 / (1.0 + self.C_cross * (self.cross_coefs['tem'] * (gamma_ITG + gamma_balloon) / \
                                                   np.maximum(gamma_TEM, 1e-6))**self.cross_sat)

        # Balloon is ion-scale; primarily suppressed by ITG & TEM competition
        F_cross_balloon = 1.0 / (1.0 + self.C_cross * (self.cross_coefs['balloon']*(gamma_ITG + gamma_TEM) / \
                                                       np.maximum(gamma_balloon, 1e-6))**self.cross_sat)

        # EM stabilization on ITG/TEM
        F_EM = 1.0 / (1.0 + self.C_EM * self.alpha / np.maximum(self.crit_grads['balloon'], 1e-6))
        
        # Impurity dilution
        F_Z = 1.0 / (1.0 + self.C_Z * self.Zeff * self.f_imp)
        
        factors = {
            "exb": {"itg": F_ExB_ITG, "tem": F_ExB_TEM, "etg": F_ExB_ETG,
                    "balloon": F_ExB_balloon},
            "cross": {"itg": F_cross_ITG, "tem": F_cross_TEM, "etg": F_cross_ETG,
                      "balloon": F_cross_balloon},
            "em": {"itg": F_EM, "tem": F_EM},
            "z": {"itg": F_Z, "tem": F_Z},
            "alpha_eff": {"itg": alpha_eff_ITG, "tem": alpha_eff_TEM,
                          "etg": alpha_eff_ETG, "balloon": alpha_eff_balloon},
        }
        
        return factors

    def _compute_diffusivities(self) -> Dict[str, np.ndarray]:
        """Compute saturated diffusivities for all modes (Section 8).
        
        Returns dict with keys: chi_ITG, chi_TEM, chi_ETG, chi_KBM, chi_RBM
        """

        gamma_ITG = self.growth_rates["itg"]
        gamma_TEM = self.growth_rates["tem"]
        gamma_ETG = self.growth_rates["etg"]
        gamma_balloon = self.growth_rates["balloon"]
        factors = self.suppression_factors

        # ky in normalized units: k_y = k_hat / rhos
        # So k_y^2 = (k_hat)^2 / rhos^2
        
        # Base QL estimate: chi_QL = C_chi * (gamma/k_y^2)
        # With k_y = k_hat/rhos: chi_QL = C_chi * (gamma * rhos^2 / k_hat^2)
        
        chi_ITG_QL = self.C_chi * (gamma_ITG / self.k_hat_ITG**2)
        chi_TEM_QL = self.C_chi * (gamma_TEM / self.k_hat_TEM**2)
        chi_ETG_QL = self.C_chi * (gamma_ETG / self.k_hat_ETG**2)
        chi_balloon_QL = self.C_chi * (gamma_balloon / self.k_hat_balloon**2)
        
        # include chi_gB factor ~ rho_s^2 * c_s /R ~ T^(3/2)
        chi_gB = self.rho_s**2 * self.c_s / self.R
        chi_gB_e = chi_gB / self.mi_over_me**0.5

        # Regime-corrected: chi = C_chi * (gamma/k_y^2)^(alpha_eff) * suppression factors
        chi_ITG = chi_gB * chi_ITG_QL ** (factors["alpha_eff"]["itg"]) * \
                  factors["exb"]["itg"] * factors["cross"]["itg"] * \
                  factors["em"]["itg"] * factors["z"]["itg"]
        
        chi_TEM = chi_gB * chi_TEM_QL ** (factors["alpha_eff"]["tem"]) * \
                  factors["exb"]["tem"] * factors["cross"]["tem"] * \
                  factors["em"]["tem"] * factors["z"]["tem"]
        
        chi_ETG = chi_gB_e * chi_ETG_QL ** (factors["alpha_eff"]["etg"]) * \
                  factors["cross"]["etg"] * factors["exb"]["etg"]
        
        chi_balloon = chi_gB * chi_balloon_QL ** (factors["alpha_eff"]["balloon"]) * \
                      factors["exb"]["balloon"] * factors["cross"]["balloon"]
        
        diffusivities = {
            "itg": chi_ITG,
            "tem": chi_TEM,
            "etg": chi_ETG,
            "balloon": chi_balloon,
        }
        
        return diffusivities

    def _compute_effective_diffusivites(self, ratio_D_TEM) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute effective diffusivities for ions and electrons, combining contributions from all modes (Section 9.1-9.3)."""
        
        chi_i = self.chi['itg'] * self.f_i_ITG + self.chi['tem'] * self.f_i_TEM + \
            self.chi['etg'] * self.f_i_ETG + self.chi['balloon'] * self.f_i_balloon
        chi_e = self.chi['itg'] * self.f_e_ITG + self.chi['tem'] * self.f_e_TEM + \
            self.chi['etg'] * self.f_e_ETG + self.chi['balloon'] * self.f_e_balloon
        D_e = self.chi['itg'] * self.f_D_ITG + self.chi['tem'] * ratio_D_TEM + \
            self.chi['etg'] * self.f_D_ETG + self.chi['balloon'] * self.f_D_balloon
        
        return chi_i, chi_e, D_e

    def _compute_pinch(self, De) -> np.ndarray:
        """Compute turbulent pinch velocity (Section 9.4)."""
        # Normalized beta drive
        alpha_norm = self.alpha / np.maximum(self.crit_grads['balloon'], 1e-6)

        # Multiplicative modifiers on curvature pinch
        F_shear   = 1.0 + self.pinch_coeffs['shear'] * self.shear          # signed shear
        F_nu      = 1.0 / (1.0 + self.pinch_coeffs['nu'] * np.sqrt(self.nustar))  # collisional suppression
        F_KBM     = 1.0 - self.pinch_coeffs['beta'] * np.minimum(alpha_norm, 0.99)  # KBM suppression

        # Curvature-driven pinch (ITG: use aLTi; if TEM-only: use aLTe)
        V_curv = -De * self.geo_factors['curv'] \
                    * (self.q / self.q_ref) \
                    * (self.pinch_coeffs['T'] * self.aLTi 
                    - self.pinch_coeffs['n'] * self.aLne) \
                    * F_shear * F_nu * F_KBM

        # Electromagnetic pinch: inward, grows with beta (separate from KBM suppression)
        V_EM = -De * self.pinch_coeffs['beta'] * alpha_norm

        V_pinch = V_curv + V_EM

        return V_pinch
    
    def _evaluate_single(self, state) -> Dict[str, np.ndarray]:
        """Compute and store turbulent and neoclassical fluxes (Section 9-10)."""
        if self.state_vars_extracted is False:
            self._extract_from_state(state)
        
        # Compute all diffusivities
        self.geo_factors = self._compute_geometric_factors()
        self.crit_grads = self._compute_critical_gradients()
        self.mode_drives = self._compute_drives()
        self.growth_rates = self._compute_growth_rates()
        self.suppression_factors = self._compute_suppression_factors()
        self.chi = self._compute_diffusivities()
        
        # collisionality-dependent TEM diffusivity ratios
        # nu_star << 1 → D_TEM/chi_e_TEM ~ 0.5
        # nu_star >> 1 → ratio_D_TEM ~ 1.0
        ratio_D_TEM = self.f_D_TEM * (1.0 + np.tanh(np.log10(np.maximum(self.nustar - self.nustar_crit, 1e-6))))
        
        # ===== Section 9: Total Transport Coefficients =====

        chi_i, chi_e, D_e = self._compute_effective_diffusivites(ratio_D_TEM)
        V_pinch = self._compute_pinch(D_e)
        
        # ===== Section 10: Total Turbulent Fluxes =====
        # Particle flux: Gamma_turb = -D_e*grad(n_e) + n_e*V_p,e
        Gamma_turb = -D_e * self.dne_dx + self.ne * V_pinch
        
        # Ion heat flux: Q_i = -n_i*chi_i*grad(T_i)
        Qi_turb = -self.ni * chi_i * self.dTi_dx + 1.5 * self.Ti * Gamma_turb * (self.ni/self.ne)
        
        # Electron heat flux: Q_e = -n_e*chi_e*grad(T_e) + (3/2)*T_e*Gamma_turb
        Qe_turb = -self.ne * chi_e * self.dTe_dx + 1.5 * self.Te * Gamma_turb
        
        # Neoclassical transport (if modes include "neo" or "all")
        if self.modes == "neo" or self.modes == "all":
            Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(state)
        else:
            Gamma_neo = np.zeros_like(self.x)
            Qi_neo = np.zeros_like(self.x)
            Qe_neo = np.zeros_like(self.x)
        
        # For mode selection
        if self.modes == "neo":
            Gamma_turb = 0 * self.x
            Qi_turb = 0 * self.x
            Qe_turb = 0 * self.x
        
        return self._assemble_fluxes(
            state,
            Gamma_turb=Gamma_turb,
            Gamma_neo=Gamma_neo,
            Qi_turb=Qi_turb,
            Qi_neo=Qi_neo,
            Qe_turb=Qe_turb,
            Qe_neo=Qe_neo,
            model_label="Analytic",
        )
