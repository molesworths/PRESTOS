"""Fingerprints transport model."""

from typing import Dict

import numpy as np

from .TransportBase import TransportBase


class Fingerprints(TransportBase):
    """Critical gradient fingerprints transport model."""

    def __init__(self, options: dict):
        super().__init__(options)
        self.external = False  # No external dependencies; fully analytic
        
        self.ITG_lcorr = self.options.get("ITG_lcorr", 0.1)
        self.non_local = self.options.get("non_local", False)
        self.labels = ["Ge", "Gi", "Ce", "Ci", "Pe", "Pi"]
        self.modes = self.options.get("modes", "all")

    def _evaluate_single(self, state) -> Dict[str, np.ndarray]:
        """Compute and store turbulent and neoclassical fluxes."""
        if self.state_vars_extracted is False:
            self._extract_from_state(state)

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

        if self.modes == "neo" or self.modes == "all":
            Gamma_neo, Qi_neo, Qe_neo = self._compute_neoclassical(state)

        RLTi_crit = np.maximum(
            (4.0 / 3.0) * (1.0 + self.Ti / self.Te),
            0.8 * self.aLne * self.aspect_ratio,
        )
        RLTe_crit = np.maximum(
            (4.0 / 3.0) * (1.0 + self.Te / self.Ti),
            1.4 * self.aLne * self.aspect_ratio,
        )

        ky_ITG = 0.3

        if self.non_local:
            aLTi_eff = np.maximum(self.aLTi - RLTi_crit / self.aspect_ratio, 0.0)
            gamma_ITG = ky_ITG * (aLTi_eff / self.ITG_lcorr) ** 0.5
        else:
            gamma_ITG = ky_ITG * np.maximum(self.aLTi - RLTi_crit / self.aspect_ratio, 0.0) ** 0.5

        gamma_eff = np.maximum(gamma_ITG - abs(gamma_ExB), 0.0)
        I_ITG = (gamma_eff / ky_ITG**2) ** 2
        chi_ITG = I_ITG * (self.Ti**1.5)
        Gamma_ITG = 0.1 * self.f_trap * chi_ITG * (-self.dne_dx - 0.25 * self.ne * self.aLTe / self.a)
        Qi_ITG = -self.ne * chi_ITG * self.dTi_dx + 1.5 * self.Ti * Gamma_ITG
        Qe_ITG = -self.ne * self.f_trap * chi_ITG * self.dTe_dx + 1.5 * self.Te * Gamma_ITG

        z_ETG = np.maximum(self.aLTe - RLTe_crit / self.aspect_ratio, 0.0) / np.maximum(self.aLne, 1e-12)
        chi_ETG = (1.0 / 60.0) * 1.5 * (self.Te**1.5) * self.aLTe * z_ETG
        Qe_ETG = -self.ne * chi_ETG * self.dTe_dx

        ky_KBM = 0.1
        alpha_crit = 2.0
        RLp_crit = alpha_crit / (np.maximum(self.beta, 1e-12) * np.maximum(self.q, 1e-9) ** 2)

        if self.non_local:
            aLp_eff = np.maximum(self.aLpe - RLp_crit / self.aspect_ratio, 0.0)
            gamma_KBM = ky_KBM * (aLp_eff / self.ITG_lcorr) ** 0.5
        else:
            gamma_KBM = ky_KBM * np.maximum(self.aLpe - RLp_crit / self.aspect_ratio, 0.0) ** 0.5

        I_KBM = (gamma_KBM / ky_KBM**2) ** 2
        chi_KBM = I_KBM * (self.Ti**1.5)
        Gamma_KBM = 0.1 * -chi_KBM * self.dne_dx
        Qi_KBM = -self.ne * chi_KBM * self.dTi_dx + 1.5 * self.Ti * Gamma_KBM
        Qe_KBM = -self.ne * chi_KBM * self.dTe_dx + 1.5 * self.Te * Gamma_KBM

        if self.modes == "all":
            Gamma_turb = Gamma_ITG + Gamma_KBM
            Qi_turb = Qi_ITG + Qi_KBM
            Qe_turb = Qe_ITG + Qe_ETG + Qe_KBM
        if self.modes == "ITG":
            Gamma_turb = Gamma_ITG
            Qi_turb = Qi_ITG
            Qe_turb = Qe_ITG
        if self.modes == "ETG":
            Gamma_turb = 0 * self.x
            Qi_turb = 0 * self.x
            Qe_turb = Qe_ETG
        if self.modes == "KBM":
            Gamma_turb = Gamma_KBM
            Qi_turb = Qi_KBM
            Qe_turb = Qe_KBM
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
            model_label="Fingerprints",
        )
