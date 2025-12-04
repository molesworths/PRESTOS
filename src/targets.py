"""
Target models for defining the objective function in transport solves.

This module provides classes to calculate the "target" values that the
transport solver aims to match. For example, in a power-balance-driven
simulation, the target fluxes are derived from heating and radiation sources.
"""

import numpy as np
from typing import Dict
from scipy.integrate import cumulative_trapezoid
from state import PlasmaState
from tools import plasma,calc
from scipy.constants import pi, e, m_u, u, epsilon_0, k
try:
    import aurora
    AURORA_AVAILABLE = True
except ImportError:
    # Optional: print a warning
    import warnings
    warnings.warn("Aurora is not available. Atomic rates will be determined via polynomial interpolation.")
    import tools.atomics as aurora
    AURORA_AVAILABLE = False
class TargetModel:
    """
    Base class for target models.
    """
    def __init__(self, **kwargs):

        class TargetsObj:
            pass
        self.targets = TargetsObj()
        self.targets.options = dict(kwargs) if kwargs else {}
        self.sigma = self.targets.options.get('sigma', 0.1) # relative epistemic uncertainty for target model outputs

    def evaluate(self, state: PlasmaState) -> Dict[str, np.ndarray]:
        """
        Calculate the target quantities for a given plasma state.

        Parameters
        ----------
        state : PlasmaState
            The current plasma state.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of target quantities, e.g., {'Qe': ..., 'Qi': ...},
            on the evaluation grid.
        """
        raise NotImplementedError
    
    def get_jacobian(self, state, X) -> np.ndarray:
        """Optional method to return Jacobian of transport fluxes w.r.t. parameters.

        Returns
        -------
        J : np.ndarray
            Jacobian matrix of shape (n_fluxes, n_parameters)
        """
        raise NotImplementedError
    

class AnalyticTargetModel(TargetModel):
    """
    Calculates targets based on analytical models for heating and radiation,
    inspired by `MINTtargets.py`.
    """
    def __init__(self, config: Dict):
        super().__init__(**config)

    def evaluate(self, state: PlasmaState) -> Dict[str, np.ndarray]:
        """
        Calculate target heat fluxes (Qe, Qi) based on analytical models for
        fusion heating, radiation, and energy exchange.

        - Computes and stores intermediate power densities on state:
          state.qie, state.qfuse, state.qfusi, state.qrad  [MW/m^3]
        - Computes target fluxes on full grid and stores on state:
          state.Qe_target, state.Qi_target  [MW/m^2]
        - If state.eval_indices exists, returns values sliced to that grid.
        """
    # Power densities [MW/m^3]
        self._evaluate_energy_exchange(state)
        self._evaluate_alpha_heating(state)
        self._evaluate_radiation(state)

        # Electron and ion heat density to powers
        self.Paux_e = state.Paux_e
        self.Paux_i = state.Paux_i
        self.Pfus_e = calc.volume_integrate(state.r, state.qfuse, state.dVdr)
        self.Pfus_i = calc.volume_integrate(state.r, state.qfusi, state.dVdr)
        self.Prad_e = calc.volume_integrate(state.r, state.qrade, state.dVdr)
        self.Prad_i = calc.volume_integrate(state.r, state.qradi, state.dVdr)
        self.P_ie = calc.volume_integrate(state.r, state.qie, state.dVdr)*0.01

        # Store to state for power-balance matching (total powers) [MW]
        self.Pe = self.Paux_e - self.Prad_e + self.Pfus_e - self.P_ie
        self.Pi = self.Paux_i - self.Prad_i + self.Pfus_i + self.P_ie
        self.Ge = -state.Gamma0*state.Zeff + state.Gbeam_e # 1e19/m^2/s
        self.Gi = self.Ge / state.Zeff # 1e19/m^2/s
        self.Ce = self.Ge * 1.5 * state.te * state.Qnorm_to_P  # MW
        self.Ci = self.Gi * 1.5 * state.ti * state.Qnorm_to_P  # MW

        # Provide dict for requested outputs
        output_dict = {
            key: [
            getattr(self, key)[np.where(np.isclose(state.roa, roa, atol=1e-3))[0][0]]
            if np.any(np.isclose(state.roa, roa, atol=1e-3))
            else np.interp(roa, state.roa, getattr(self, key))
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

    def _evaluate_energy_exchange(self, state: PlasmaState) -> np.ndarray:
        """Classical electron-ion energy exchange power density (MW/m^3)."""

        q_ie = plasma.energy_exchange(state.ne,state.te,state.ti,state.nuexch)  # MW/m^3
        state.qie = q_ie


    def _evaluate_alpha_heating(self, state: PlasmaState) -> tuple[np.ndarray, np.ndarray]:
        """Alpha heating power densities for electrons and ions [MW/m^3]."""
        # Identify D and T species indices if available
        d_idx = t_idx = None
        if hasattr(state, 'species') and state.species:
            for j, sp in enumerate(state.species):
                name = (sp.get('N') or sp.get('name') or '').upper()
                if name == 'D' or (sp.get('A') == 2 and sp.get('Z') == 1):
                    d_idx = j
                if name == 'T' or (sp.get('A') == 3 and sp.get('Z') == 1):
                    t_idx = j
        # If not found, assume 50/50 using ne
        ne_19 = np.asarray(state.ne)
        if d_idx is None or t_idx is None or state.ni is None:
            n_d_m3 = 0.5 * ne_19 * 1e19
            n_t_m3 = 0.5 * ne_19 * 1e19
            Ti_ref = np.asarray(state.ti)
            if Ti_ref.ndim == 2:
                Ti_ref = Ti_ref[:, 0]
        else:
            ni = np.asarray(state.ni)
            n_d_m3 = ni[:, d_idx] * 1e19
            n_t_m3 = ni[:, t_idx] * 1e19
            Ti_ref = np.asarray(state.ti)

        ti_keV = np.asarray(Ti_ref)

        # Bosch-Hale-like approximation for <σv> in m^3/s (rough)
        bg = 34.3827
        c = [1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5]
        num = (c[1] + ti_keV * (c[3] + ti_keV * c[5]))
        den = (1 + ti_keV * (c[2] + ti_keV * (c[4] + ti_keV * c[6])))
        theta = ti_keV / (1 - (ti_keV * num) / den)
        zeta = bg / np.sqrt(np.maximum(theta, 1e-6))
        # Convert from cm^3/s to m^3/s with 1e-6 factor in denominator
        sigv = c[0] * theta**2 * np.exp(-zeta) / (1e6 * np.maximum(ti_keV, 1e-8)**(2/3))

        E_alpha_MeV = 3.5
        E_alpha_J = E_alpha_MeV * 1.602e-13
        p_alpha_Wm3 = (n_d_m3 * n_t_m3 * sigv) * E_alpha_J
        p_alpha_MWm3 = p_alpha_Wm3 / 1e6

        Ae = 9.1094e-28 / u  # Electron mass in atomic units # 9.1094E-28/u
        Aalpha = 2 * (3.34358e-24) / u  # Alpha mass in atomic units
        c_a = state.te * 0.0
        for i in range(state.ni_full.shape[1]):
            c_a += (state.ni_full[..., i] / state.ne) * \
            state.ions_set_Zi[i] ** 2 * \
            (Aalpha / state.ions_set_mi[i])
        W_crit = (state.te * 1e3) * (4 * (Ae / Aalpha) ** 0.5 / (3 * pi**0.5 * c_a)) ** (
            -2.0 / 3.0)  # in eV

        #frac_e = float(self.options.get('alpha_heat_frac_e', 0.8))
        frac_ai = sivukhin(E_alpha_MeV*1e3 / W_crit)  # This solves Eq 17 of Stix
        state.qfuse = p_alpha_MWm3 * frac_ai
        state.qfusi = p_alpha_MWm3 * (1 - frac_ai)


    def _evaluate_radiation(self, state: PlasmaState) -> np.ndarray:
        """Total radiated power density q_rad [MW/m^3] using Aurora when available.

        Components stored on state when available: qrad_bremms, qrad_line, qrad_sync.
        """

        te_eV = state.te * 1e3
        ti_eV = state.ti_full * 1e3
        ne_m3 = state.ne * 1e19

        # Attempt Aurora radiation model first (returns W/cm^3); convert to MW/m^3 (numerically same)
        q_rad_e = q_rad_i = np.zeros_like(te_eV)

        for sidx, sp in enumerate(state.species):
            
            ne_cm3 = ne_m3 * 1e-6
            ni_cm3 = state.ni_full[:,sidx] * 1e-6
            n0_cm3 = state.n0[:,sidx] * 1e-6
            sp_name = sp['name']
            sp_Z = sp['Z']

            try:
                atom_data = aurora.get_atom_data(sp_name, ['acd', 'scd', 'ccd'])
                _,fZ = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV=te_eV, Ti_eV=ti_eV[:,sidx],
                                                        n0_by_ne=n0_cm3/ne_cm3,plot=False)
                nZ_cm3, _ = self._build_nZ_from_fZ(fZ, n0_cm3, ni_cm3, smooth_weight=True)

                rad = aurora.radiation.compute_rad(sp_name,nZ_cm3,ne_cm3,te_eV,
                                                    n0=n0_cm3,Ti=ti_eV,prad_flag=True,
                                                    thermal_cx_rad_flag=True)

                q_rad += rad['tot'].sum(axis=1)
                q_brem += rad['brems'].sum(axis=1)
                q_cont += rad['cont_rad'].sum(axis=1)
                q_line += rad['line_rad'].sum(axis=1)
                q_cx += rad['thermal_cx_cont_rad'].sum(axis=1)
                q_sync += plasma.synchrotron(state.te, state.ne*0.1, state.B_unit, 
                                            state.aspect_ratio, state.r)
                
                # Electron energy losses [MW/m^3]
                qe_rad_local = (
                    rad.get('line_rad', 0.0)
                    + rad.get('cont_rad', 0.0)
                    + rad.get('synchrotron', 0.0)
                )
                if 'brems' in rad and 'cont_rad' not in rad:
                    qe_rad_local += rad['brems']

                q_rad_e += qe_rad_local

                # Ion energy losses [MW/m^3]
                qi_rad_local = rad.get('thermal_cx_cont_rad', 0.0)
                q_rad_i += qi_rad_local

            except Exception:
                pass

        # Store components
        state.qrade = q_rad_e
        state.qradi = q_rad_i

    def _build_nZ_from_fZ(self, fZ, n0_cm3, ni_cm3, smooth_weight=True):
        """
        Construct nZ_cm3 = fZ * n_total using neutral and fully ionized densities.

        Parameters
        ----------
        fZ : (n_space, n_charge_states)
            Fractional abundances (sum along axis=1 = 1)
        n0_cm3 : (n_space,)
            Neutral density (charge state 0)
        ni_cm3 : (n_space,)
            Fully ionized density (charge state Zmax)
        smooth_weight : bool
            If True, blend n_total estimates to avoid spikes when fZ[:,0] or fZ[:,-1] are small.

        Returns
        -------
        nZ_cm3 : ndarray, shape (n_space, n_charge_states)
            Density in cm^-3 for each charge state
        n_total : ndarray, shape (n_space,)
            Total impurity density
        """
        fZ0 = fZ[:, 0]
        fZmax = fZ[:, -1]

        # avoid divide-by-zero
        eps = 1e-30
        fZ0 = np.clip(fZ0, eps, None)
        fZmax = np.clip(fZmax, eps, None)

        n_total_0 = n0_cm3 / fZ0
        n_total_i = ni_cm3 / fZmax

        if smooth_weight:
            # Weight the two estimates by fractional reliability
            w0 = fZ0 / (fZ0 + fZmax)
            wi = fZmax / (fZ0 + fZmax)
            n_total = w0 * n_total_0 + wi * n_total_i
        else:
            # Simple average if both nonzero
            n_total = 0.5 * (n_total_0 + n_total_i)

        # Prevent unphysical oscillations / negative
        n_total = np.clip(n_total, 0, np.nanmax(n_total))

        # Reconstruct full charge-state density profile
        nZ_cm3 = fZ * n_total[:, None]

        return nZ_cm3, n_total



def create_target_model(config: Dict) -> TargetModel:
    """
    Factory function to create a target model based on configuration.
    """
    model_type = config.get('type', 'analytical')
    if model_type == 'analytical':
        return AnalyticTargetModel(config.get('kwargs', {}))
    else:
        raise ValueError(f"Unknown target model type: {model_type}")

def sigv_fun(ti):
    """
    This script calculates the DT fusion reaction rate coefficient (cm^3/s) from ti (keV), following
    [H.-S. Bosch and G.M. Hale, Nucl. Fusion 32 (1992) 611]

    This method follows the same methodology as in TGYRO [Candy et al. PoP 2009] and all the credits
    are due to the authors of TGYRO. From the source code, this function follows the same procedures
    as in tgyro_auxiliary_routines.f90.
    """

    # For Bosh XS
    c1, c2, c3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
    c4, c5, c6, c7 = 4.60643e-3, 1.3500e-2, -1.06750e-4, 1.36600e-5
    bg, er = 34.3827, 1.124656e6

    r0 = ti * (c2 + ti * (c4 + ti * c6)) / (1.0 + ti * (c3 + ti * (c5 + ti * c7)))
    theta = ti / (1.0 - r0)
    xi = (bg**2 / (4.0 * theta)) ** (1.0 / 3.0)

    sigv = c1 * theta * (xi / (er * ti**3)) ** 0.5 * np.exp(-3.0 * xi)

    return sigv


def sivukhin(x, n=12):
    """
    This script implements the TGYRO's sivukhin algorithm.
    This method follows the same methodology as in TGYRO [Candy et al. PoP 2009] and all the credits
    are due to the authors of TGYRO.

    Improvements have been made to make it faster, by taking into account
    array operations within pytorch rather than loops
    """

    # --------------
    # Asymptotes
    # --------------

    v = 0.866025  # sin(2*pi/3)
    f = (2 * pi / 3) / v - 2.0 / x**0.5 + 0.5 / (x * x)
    sivukhin1 = f / x

    sivukhin3 = 1.0 - 0.4 * x**1.5

    # --------------
    # Numerical (middle)
    # --------------

    dy = x / (n - 1)
    f = 0.0
    for i in range(n):
        yi = i * dy
        if i == 0 or i == n - 1:
            f = f + 0.5 / (1.0 + yi**1.5)
        else:
            f = f + 1.0 / (1.0 + yi**1.5)
    f = f * dy

    sivukhin2 = f / x

    # --------------
    # Construct
    # --------------

    sivukhin = (
        (x > 4.0) * sivukhin1
        + (x < 4.0) * (x > 0.1) * sivukhin2
        + (x < 0.1) * sivukhin3
    )

    return sivukhin

TARGET_MODELS = {
    'analytic': AnalyticTargetModel,
}