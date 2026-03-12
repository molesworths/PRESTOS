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

    Output structure follows the nested flux/flow dict framework:

      targets_dict = {
          'fluxes': {
              'gB'  : {'Ge': {component: arr, ...}, 'Gi': {...}, 'Qe': {...}, 'Qi': {...}},
              'real': {same, physical units},
          },
          'flows': {
              'gB'  : {'Pe': {component: arr, ...}, 'Pi': {...}},
              'real': {same, [MW]},
          },
      }

    Component keys for target models (replacing the transport turb/neo split):
      heat flows  : 'aux', 'alpha', 'rad', 'exch', 'total', 'conv', 'cond'
      particle fluxes : 'beam', 'wall', 'total'

    The flat dict returned by evaluate() contains the channels requested via
    ``output_vars`` in the natural (real-unit) representation.  gB-normalised
    values of the same channels are stored on ``self.Y_gB`` for surrogate use.

    Supports batch processing:
    - evaluate(state) handles both single PlasmaState and list of PlasmaState objects
    """

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)

        class TargetsObj:
            pass
        self.targets = TargetsObj()
        self.targets.options = dict(kwargs) if kwargs else {}
        self.sigma = self.targets.options.get('sigma', 0.1)

        # Containers populated by _evaluate_single in subclasses
        self.fluxes: Dict[str, Dict] = {'gB': {}, 'real': {}}   # per-m² quantities
        self.flows:  Dict[str, Dict] = {'gB': {}, 'real': {}}   # surface-integrated [MW]
        self.targets_dict: Dict = {}   # full nested dict (fluxes + flows)
        self.Y_gB:    Dict[str, list] = {}  # gB values at roa_eval (for surrogate)
        self.Y_target: Dict[str, list] = {}
        self.Y_std:   Dict[str, list] = {}

        # Set by solver before evaluation
        self.roa_eval = None
        self.output_vars = []

    def _is_batched(self, state) -> bool:
        """Check if state is a batch (list/array of states) vs single state."""
        if isinstance(state, (list, tuple)):
            return True
        if isinstance(state, np.ndarray) and state.dtype == object:
            return True
        return False

    def evaluate(self, state: PlasmaState) -> Dict[str, np.ndarray]:
        """Calculate the target quantities for single or batch of plasma states.

        Parameters
        ----------
        state : PlasmaState or list of PlasmaState
            Single plasma state or list of states for batch processing

        Returns
        -------
        Dict[str, np.ndarray] or list of Dict[str, np.ndarray]
            For single state: dictionary of target quantities, e.g., {'Qe': ..., 'Qi': ...}
            For batch: list of dictionaries, one per state
        """
        if self._is_batched(state):
            results = []
            for single_state in state:
                result = self._evaluate_single(single_state)
                results.append(result)
            return results
        else:
            return self._evaluate_single(state)

    def _evaluate_single(self, state: PlasmaState) -> Dict[str, np.ndarray]:
        """Calculate the target quantities for a given plasma state.

        To be overridden by child classes with actual implementation.

        Parameters
        ----------
        state : PlasmaState
            Single plasma state.

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
    

class Analytic(TargetModel):
    """
    Calculates targets based on analytical models for heating and radiation,
    inspired by `MINTtargets.py`.
    """
    def __init__(self, config: Dict):
        super().__init__(**config)

    def _evaluate_single(self, state: PlasmaState) -> Dict[str, np.ndarray]:
        """Build power-balance target fluxes and flows.

        All physical source terms are computed and stored on *state* as
        volumetric power densities [MW/m³].  They are then cumulatively
        integrated to radial flow profiles [MW] and normalized to gyroBohm
        units.

        The full nested dict is stored on ``self.targets_dict`` / ``self.fluxes``
        / ``self.flows`` for diagnostics and surrogate training (``self.Y_gB``).

        Channels available for ``output_vars`` / ``target_vars``:
          'Pe', 'Pi'  – total net heat flow [MW]
          'Ce', 'Ci'  – convective heat flow [MW]
          'De', 'Di'  – conductive heat flow [MW]  (= Pe/Pi - Ce/Ci)
          'Ge', 'Gi'  – total particle flux [1e19/m²/s]
          'Qe', 'Qi'  – total heat flux [MW/m²]  (= Pe/Pi / surfArea)

        Returns
        -------
        tuple(output_dict, std_dict)
            Flat dicts keyed by ``self.output_vars``, values are lists of
            floats at ``self.roa_eval``.
        """
        roa_eval = getattr(self, 'roa_eval', state.roa)
        output_vars = getattr(self, 'output_vars', ['Pe', 'Pi'])

        # ------------------------------------------------------------------
        # 1. Volumetric source/sink densities [MW/m³]
        # ------------------------------------------------------------------
        self._evaluate_energy_exchange(state)   # → state.qie
        self._evaluate_alpha_heating(state)     # → state.qfuse, state.qfusi
        self._evaluate_radiation(state)         # → state.qrade, state.qradi

        # ------------------------------------------------------------------
        # 2. Cumulative component heat flows [MW]  (integrated from r=0 outward)
        # ------------------------------------------------------------------
        Pfus_e = calc.volume_integrate(state.r, state.qfuse, state.dVdr)
        Pfus_i = calc.volume_integrate(state.r, state.qfusi, state.dVdr)
        Prad_e = calc.volume_integrate(state.r, state.qrade, state.dVdr)
        Prad_i = calc.volume_integrate(state.r, state.qradi, state.dVdr)
        # Electron-ion exchange [MW]; *0.01 preserves historical scaling
        Pexch  = calc.volume_integrate(state.r, state.qie,   state.dVdr) * 0.01
        Paux_e = getattr(state, 'Paux_e', np.zeros_like(state.r))
        Paux_i = getattr(state, 'Paux_i', np.zeros_like(state.r))

        # ------------------------------------------------------------------
        # 3. Net total heat flows [MW]
        #    sign convention: Prad / Pexch are losses for electrons, gains for ions
        # ------------------------------------------------------------------
        Pe = Paux_e + Pfus_e - Prad_e - Pexch
        Pi = Paux_i + Pfus_i - Prad_i + Pexch

        # ------------------------------------------------------------------
        # 4. Particle fluxes [1e19/m²/s]
        # ------------------------------------------------------------------
        Gbeam_e = getattr(state, 'Gbeam_e', np.zeros_like(state.r))
        Gwall_e = getattr(state, 'Gwall_e', np.zeros_like(state.r))
        if hasattr(state, 'Gamma0'):
            Ge = -state.Gamma0 * state.Zeff + Gbeam_e
        else:
            Ge = Gbeam_e + Gwall_e
        Zeff_safe = np.maximum(state.Zeff, 1e-12)
        Gi = Ge / Zeff_safe
        Gbeam_i = Gbeam_e / Zeff_safe
        Gwall_i = Gwall_e / Zeff_safe

        # ------------------------------------------------------------------
        # 5. Convective and conductive heat flows [MW]
        # ------------------------------------------------------------------
        Ce = plasma.get_convective_flow(state.te, Ge, state.surfArea)
        Ci = plasma.get_convective_flow(state.ti, Gi, state.surfArea)
        De = Pe - Ce   # conductive electron heat flow [MW]
        Di = Pi - Ci   # conductive ion heat flow [MW]

        # ------------------------------------------------------------------
        # 6. Heat fluxes [MW/m²] = flows / surfArea
        # ------------------------------------------------------------------
        Qe = plasma.heat_flow_to_flux(Pe, state.surfArea)
        Qi = plasma.heat_flow_to_flux(Pi, state.surfArea)

        # ------------------------------------------------------------------
        # 7. gyroBohm normalization factors
        # ------------------------------------------------------------------
        q_gb    = state.q_gb                   # [MW/m²]
        g_gb    = state.g_gb                   # [1e20/m²/s]
        P_gB    = q_gb * state.surfArea        # [MW]  flow-level gB norm
        G_gB    = g_gb * 10.0                  # [1e19/m²/s]  particle-flux gB norm
        _safe   = lambda x: np.maximum(x, 1e-30)

        # ------------------------------------------------------------------
        # 8. Build nested flows dict  (component breakdown replaces turb/neo)
        #    Component keys:  'aux', 'alpha', 'rad', 'exch', 'total', 'conv', 'cond'
        # ------------------------------------------------------------------
        Pe_comps_real = {
            'aux': Paux_e, 'alpha': Pfus_e, 'rad': Prad_e, 'exch': Pexch,
            'total': Pe, 'conv': Ce, 'cond': De,
        }
        Pi_comps_real = {
            'aux': Paux_i, 'alpha': Pfus_i, 'rad': Prad_i, 'exch': Pexch,
            'total': Pi, 'conv': Ci, 'cond': Di,
        }
        self.flows = {
            'real': {'Pe': Pe_comps_real, 'Pi': Pi_comps_real},
            'gB': {
                'Pe': {k: v / _safe(P_gB) for k, v in Pe_comps_real.items()},
                'Pi': {k: v / _safe(P_gB) for k, v in Pi_comps_real.items()},
            },
        }

        # ------------------------------------------------------------------
        # 9. Build nested fluxes dict
        #    Component keys for particle: 'beam', 'wall', 'total'
        #    Component keys for heat: 'total'  (component flows are in flows dict)
        # ------------------------------------------------------------------
        Ge_comps_real = {'beam': Gbeam_e, 'wall': Gwall_e, 'total': Ge}
        Gi_comps_real = {'beam': Gbeam_i, 'wall': Gwall_i, 'total': Gi}
        self.fluxes = {
            'real': {
                'Ge': Ge_comps_real,
                'Gi': Gi_comps_real,
                'Qe': {'total': Qe},
                'Qi': {'total': Qi},
            },
            'gB': {
                'Ge': {k: v / _safe(G_gB) for k, v in Ge_comps_real.items()},
                'Gi': {k: v / _safe(G_gB) for k, v in Gi_comps_real.items()},
                'Qe': {'total': Qe / _safe(q_gb)},
                'Qi': {'total': Qi / _safe(q_gb)},
            },
        }

        self.targets_dict = {'fluxes': self.fluxes, 'flows': self.flows}

        # ------------------------------------------------------------------
        # 10. All available channels mapped to radial profiles for flat output
        # ------------------------------------------------------------------
        all_channels = {
            'Pe': Pe, 'Pi': Pi,
            'Ce': Ce, 'Ci': Ci,
            'De': De, 'Di': Di,
            'Ge': Ge, 'Gi': Gi,
            'Qe': Qe, 'Qi': Qi,
        }
        all_channels_gB = {
            'Pe': Pe / _safe(P_gB), 'Pi': Pi / _safe(P_gB),
            'Ce': Ce / _safe(P_gB), 'Ci': Ci / _safe(P_gB),
            'De': De / _safe(P_gB), 'Di': Di / _safe(P_gB),
            'Ge': Ge / _safe(G_gB), 'Gi': Gi / _safe(G_gB),
            'Qe': Qe / _safe(q_gb), 'Qi': Qi / _safe(q_gb),
        }

        def _extract(arr):
            arr = np.atleast_1d(arr)
            return [float(np.interp(roa, state.roa, arr)) for roa in roa_eval]

        output_dict = {
            key: _extract(all_channels[key])
            for key in output_vars if key in all_channels
        }
        std_dict = {
            key: [self.sigma * abs(v) for v in output_dict[key]]
            for key in output_dict
        }
        self.Y_gB = {
            key: _extract(all_channels_gB[key])
            for key in output_vars if key in all_channels_gB
        }
        self.Y_target = output_dict
        self.Y_std = std_dict

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
        return Analytic(config.get('kwargs', {}))
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
    'analytic': Analytic,
}