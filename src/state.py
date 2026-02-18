"""
Plasma state representation for the standalone solver.

This version centers the state around a gacode profile container.
It reads a gacode file (or pre-parsed profiles), builds a clean
PlasmaState with category dictionaries, and provides round-trip
conversion back to a gacode-like object.
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Union, Optional
import copy
from tools import plasma, calc, geometry, io
from interfaces import gacode 
from scipy.constants import m_p, m_e, e, k, c, mu_0, u, pi, epsilon_0
from scipy.interpolate import Akima1DInterpolator as akima
try:
    import aurora
    AURORA_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("Aurora is not available. Atomic rates will be determined via polynomial interpolation.")
    import tools.atomics as aurora
    AURORA_AVAILABLE = False

@dataclass
class PlasmaState:
    """
    Flat container for derived and raw plasma quantities.
    Each quantity is an attribute (e.g., ne, te, ti, q, rho, etc.).
    
    Supports parameter scan workflows:
    - Heating/particle source scaling via metadata['unscaled_heating']
    - Checkpoint/restart with apply_scaling() method
    """

    # Arbitrary metadata and derived values
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------- conversion hooks ----------

    @classmethod
    def from_gacode(cls, gacode_obj: "gacode") -> "PlasmaState":
        """
        Build a PlasmaState instance from a gacode object.
        Profiles are flattened into direct attributes on the PlasmaState.
        """
        state = cls()  # Initialize an empty PlasmaState

        # Copy profiles into attributes
        for gacode_key, flat_name in gacode_obj.profiles_mapping.items():
            if gacode_key in gacode_obj.profiles:
                setattr(state, flat_name, copy.deepcopy(gacode_obj.profiles[gacode_key]))

        # Store source and mapping metadata
        state.metadata["source_file"] = gacode_obj.file
        state.metadata["source_profiles"] = copy.deepcopy(gacode_obj.profiles)
        state.metadata["gacode_mapping"] = copy.deepcopy(gacode_obj.profiles_mapping)

        return state


    def to_gacode(self) -> "gacode":
        """
        Convert this PlasmaState back into a minimal gacode-like object.
        Only exports variables that exist in metadata['gacode_mapping'].
        """

        profiles = {}

        add_if_not_in_map = {"wtor(rad/s)": "wtor", "n0(10^19/m^3)": "n0"}
        mapping = self.metadata.get("gacode_mapping", {})

        # Normalize mapping to {flat_name: gacode_key}
        if mapping and any("(" in key or ")" in key for key in mapping.keys()):
            export_mapping = {flat_name: gacode_key for gacode_key, flat_name in mapping.items()}
        else:
            export_mapping = dict(mapping)

        for gacode_key, flat_name in add_if_not_in_map.items():
            if flat_name not in export_mapping:
                export_mapping[flat_name] = gacode_key

        for flat_name, gacode_key in export_mapping.items():
            # Only export if attribute exists and is array-like
            if hasattr(self, flat_name):
                val = getattr(self, flat_name)
                if isinstance(val, (list, np.ndarray)):
                    profiles[gacode_key] = np.array(val)

        gacode_obj = gacode(profiles=profiles)
        return gacode_obj

    def process(self, gacode_obj: "gacode" = None, neutrals=None):
        """
        Populate PlasmaState with derived quantities.
        Optionally pass a gacode object for any missing profiles.
        
        Note: Heating/particle source scaling (for parameter scans) should be
        applied AFTER process() is called, using workflow._apply_target_scaling_to_state().
        This ensures scaling happens once at initialization, not repeatedly.
        """
        # --- Helper alias: look up arrays as attributes or from gacode_obj.profiles ---
        def get(key, default=None):
            if hasattr(self, key):
                return getattr(self, key)
            if gacode_obj is not None and key in gacode_obj.profiles:
                return gacode_obj.profiles[key]
            return default

        # --- Ensure essential quantities exist ---
        if gacode_obj is not None:
            r = get("rmin(m)")
            if r is None:
                raise ValueError("Missing radial coordinate 'rmin(m)' in PlasmaState or gacode")
            
            if self.te is None or self.ne is None:
                raise ValueError("Missing core kinetic profiles (te, ne)")
            
            self.ti_full = self.ti
            self.ni_full = self.ni
            self.ti = self.ti[:,0].flatten() if self.ti.ndim == 2 else self.ti.flatten()
            self.ni = self.ni[:,0].flatten() if self.ni.ndim == 2 else self.ni.flatten()
            self.n0 = get("n0(10^19/m^3)", np.ones_like(self.ni_full)*0.01)
            self.metadata['gacode_mapping']['ti(keV)'] = 'ti_full'
            self.metadata['gacode_mapping']['ni(10^19/m^3)'] = 'ni_full'

            # Charge-state distributions (from Aurora atomic physics)
            self.nZ = None   # (n_radial, n_species, max(n_charge)+1) densities [1e19/m^3]
            self.fZ = None  # (n_radial, n_species, n_charge+1) fractional abundances
            self.Kn = None  # (n_radial, n_species) Knudsen numbers [dimensionless]

            self.r = self.rmin
            self.a = self.r[-1]
            self.R0 = float(get("rcentr(m)", [np.nan])[-1])
            self.B0 = float(get("bcentr(T)", [np.nan])[-1])
            self.roa = self.r / self.a
            self.aspect_ratio = self.R0 / self.a
            self.R = self.R0 + self.r

            # --- Magnetic and geometric quantities ---
            self.torflux = self.torfluxa * 2 * np.pi * self.rho ** 2
            self.B_unit = plasma.Bunit(self.torflux, self.r)
            self.B_T = self.bcentr*(self.R0/self.R)
            self.B_p = self.R**(-1)*np.gradient(self.polflux,self.r) #(self.r * self.B_T) / (self.q * self.R) # T
            self.B = np.sqrt(self.B_T**2 + self.B_p**2) # T
            self.kappa_hat = np.sqrt((1 + self.kappa**2)/2)
            self.q_cyl = self.kappa_hat*self.B_T/np.maximum(self.aspect_ratio*np.maximum(self.B_p, 1e-30), 1e-30)

            # --- Compute species and shaping info ---
            self._read_species(gacode_obj=gacode_obj)
            self._produce_shape_lists(gacode_obj=gacode_obj)

            # Gradient scale lengths
            self.aLte = calc.aLy(self.r, self.te)
            self.aLne = calc.aLy(self.r, self.ne)
            self.aLti = calc.aLy(self.r, self.ti)
            self.aLni = calc.aLy(self.r, self.ni)

            # curvatures
            self.d2ne = np.gradient(np.gradient(self.ne,self.r),self.r)
            self.d2te = np.gradient(np.gradient(self.te,self.r),self.r)
            self.d2ti = np.gradient(np.gradient(self.ti,self.r),self.r)

            rates = self.get_atomic_rates()
            self.nu_ion = rates['ion']  # [1/s], ionization
            self.nu_rec = rates['recom']  # [1/s], recombination
            self.nu_cx = rates['cx']  # [1/s], charge exchange

            # --- Geometry first (needed for surface area and related norms) ---
            (
                self.volp_miller,
                self.surf_miller,
                self.gradr_miller,
                self.bp2_miller,
                self.bt2_miller,
                self.geo_bt,

            ) = geometry.calculateGeometricFactors(
                self,
            )
            self.surfArea = self.surf_miller
            self.dVdr = self.volp_miller              # dV/dr
            self.V = calc.volume_integrate(self.r,np.ones_like(self.dVdr),self.dVdr)

                    # Main ion mass/charge from first species entry (fallback to D, Z=1)
            try:
                self.mi_ref = float(self.species[0]["A"]) if self.species else 2.0
                self.Z_ref = float(self.species[0]["Z"]) if self.species else 1.0
            except Exception:
                self.mi_ref = 2.0
                self.Z_ref = 1.0

        # Follow PORTALS approach to ion densities
        # Update thermal ion species to scale proportionally to ne
        # self._ne_old = getattr(self, "_ne_old", self.ne.copy())
        # scale_factor = self.ne / np.maximum(self._ne_old, 1e-30)
        # self.ni_full *= scale_factor[:, np.newaxis]
        # self.ni = self.ni_full[:,0].flatten() if self.ni_full.ndim == 2 else self.ni_full.flatten()
        # self._ne_old = self.ne.copy()
        # self.aLni = calc.aLy(self.r, self.ni)
        # self.f_imp = 1.0 - self.ni / np.maximum(self.ne, 1e-30)

        neutrals_active = neutrals is not None and getattr(neutrals, "n0_edge", None) is not None
        if neutrals_active:
            neutrals.solve(self)

        self.Zeff = plasma.get_Zeff(self.ne, self.ni_full, self.species)

        # --- Basic physical constants and derived scalings ---
        self.c_s = plasma.c_s(self.te, self.mi_ref)
        self.rho_s = plasma.rho_s(self.te, getattr(self, "mi_ref", 2.0), self.B)
        self.vth = plasma.vthermal(self.ti, self.mi_ref)
        self.q_gb, self.g_gb, *_ = plasma.gyrobohm_units(
            self.te,
            self.ne * 1e-1,  # convert 10^19/m^3 -> 10^20/m^3
            self.mi_ref,
            self.B_unit,
            self.a,
        )
        self.LogLam = plasma.loglam(self.te,self.ne)
        self.omega_ci = plasma.omega_c(self.mi_ref,self.Z_ref,self.B_unit)

        # --- Compute normalizations (uses geometry-derived surfaces) ---
        self._get_norms()

        # --- Misc derived ---
        self.shear = plasma.magnetic_shear(self.q, self.r)
        self.rhostar = plasma.rho_star(self.te,self.mi_ref,self.a,self.B)
        self.nustar = plasma.nu_star(self.te,self.ne,self.a,self.mi_ref)
        self.eps = self.r/self.aspect_ratio
        self.q95 = np.interp(0.95, self.rho, self.q) if self.q is not None else np.nan
        self.Lpar = pi * abs(self.q95) * self.R[-1] * self.kappa_hat[-1] / 2  # m
        self.kappa95 = np.interp(0.95, self.rho, self.kappa) if self.kappa is not None else np.nan
        self.delta95 = np.interp(0.95, self.rho, self.delta) if self.delta is not None else np.nan
        self.alphat = (1. + self.ti / self.te) * self.alphat_norm * self.ne / self.te**2
        # Collision frequencies, nuee ~ nuei >> nuii >> nuie
        self.nuii = 4.8e-8 * self.Zeff**4 * self.ni*1e13 * self.LogLam / (np.sqrt(self.mi_over_mp) * (self.ti*1e3)**1.5)
        self.nuei = 2.91e-6 * self.ne*(self.n_norm*1e-6) * self.LogLam / ( (self.te*1e3)**1.5 ) # ~ nu_ee
        self.nuexch = (self.Z_ref/self.mi_over_me)*self.nuei # ~nu_ie
        # Trapping fraction
        self.f_trap = 1.45 * np.sqrt(self.eps)

        (
            self.ptot_manual,
            self.pe,
            self.pi,
        ) = plasma.calculate_pressure(
            self.te,
            self.ti_full,
            self.ne * 0.1,
            self.ni_full * 0.1,
        )

        # --- Compute mean ion mass, background, etc. ---
        self._calculate_mass(gacode_obj=gacode_obj)
        self._update_w0_by_decompose()
        self._get_power_flows()

        # TGLF stuff

        self.tite = self.ti / self.te
        self.tite_full = self.ti_full / self.te[:,np.newaxis]
        self.nine_full = self.ni_full / self.ne[:,np.newaxis]
        self.betae = plasma.betae(self.te, self.ne, self.B)
        self.xnue = plasma.xnue(self.te, self.ne*0.1, self.a, mref_u=self.mi_ref)
        self.debye = plasma.debye(self.te, self.ne*0.1,self.mi_ref,self.B)
        
        self.alpha = -self.R * self.q**2 * np.gradient(self.betae,self.r)   
        self.pprime = plasma.p_prime(self.te, self.ne, self.aLte, self.aLne, 
                                     self.ti, self.ni_full, self.aLti, self.aLni, self.a, self.B_unit, self.q, self.r)
        self.drmindr = np.gradient(self.r,self.r)
        self.dRmajdr = np.gradient(self.rmaj,self.r)
        self.dZmajdr = np.gradient(self.zmag,self.r)

        # assemble gacode species info like 
        # ZS_2 =1
        # MASS_2 =+1.00000E+00 # normalized to m_Deuterium
        # AS_2 =+5.72467E-01 # ns/ne
        # TAUS_2 =+1.09160E+00 # Ts/Te
        # RLNS_2 =+1.22154E+01 # aLns
        # RLTS_2 =+1.25803E+01 # aLTs
        # VPAR_2 =+1.44804E-01 # sign(Ip)*R0*Vtor/(R*c_s)
        # VPAR_SHEAR_2 =+1.69199E+00 # -sign(Ip)*R0*d(VPar)/dr * a

        self.gacode_species = []
        for i, spec in enumerate(self.species):
            gacode_spec = {
                "ZS": spec["Z"],
                "MASS": spec["A"] / 2.0,  # normalized to deuterium
                "AS": self.nine_full[:,i],
                "TAUS": self.tite_full[:,i],
                "RLNS": calc.aLy(self.r, self.ni_full[:,i]),
                "RLTS": calc.aLy(self.r, self.ti_full[:,i]),
                "VPAR": self.vpar,
                "VPAR_SHEAR": self.vpar_shear,
            }
            self.gacode_species.append(gacode_spec)

        return

    # -----------------------------
    # Helper methods (formerly local defs)
    # -----------------------------

    def _calculate_mass(self, gacode_obj: "gacode" = None):
        """Compute average and main ion masses and fractions."""
        masses = getattr(self, "mass", None)
        if masses is None and gacode_obj is not None:
            masses = gacode_obj.profiles.get("mass")

        fi_vol = getattr(self, "fi_vol", None)
        if fi_vol is None:
            self.mbg = np.nan
            return

        self.mbg = np.sum(np.array(masses) * np.array(fi_vol)) if masses is not None else np.nan

        # For DT plasmas
        if getattr(self, "DTplasmaBool", False):
            D = getattr(self, "Dion", 0)
            T = getattr(self, "Tion", 1)
            denom = fi_vol[D] + fi_vol[T]
            self.mbg_main = (
                (masses[D] * fi_vol[D] + masses[T] * fi_vol[T]) / denom if denom > 0 else masses[D]
            )
            self.fmain = fi_vol[D] + fi_vol[T]
        else:
            idx = getattr(self, "Mion", 0)
            self.mbg_main = masses[idx] if masses is not None else np.nan
            self.fmain = fi_vol[idx] if fi_vol is not None else 0.0

    def _produce_shape_lists(self, gacode_obj: "gacode" = None):
        """Extract shape Fourier coefficients."""
        def get_local(key):
            if hasattr(self, key):
                return getattr(self, key)
            if gacode_obj is not None and key in gacode_obj.profiles:
                return gacode_obj.profiles[key]
            return np.zeros_like(getattr(self, "r", np.array([0.0])))

        self.shape_cos = [get_local(f"shape_cos{i}(-)") for i in range(7)]
        self.shape_sin = [get_local(f"shape_sin{i}(-)") for i in range(7)]

    def _read_species(self, gacode_obj: "gacode" = None):
        """Rebuild species info into a list of dicts.

        Sets self.species to a list of dicts, each with name, Z, A, and type.
        """
        def get_local(key):
            val = getattr(self, key, None)
            if val is not None:
                return val
            if gacode_obj is not None and key in gacode_obj.profiles:
                return gacode_obj.profiles[key]
            return []

        names = get_local("name")
        charges = get_local("z")
        masses = get_local("mass")
        types = get_local("type")

        num_species = len(names)
        self.species = []
        for i in range(num_species):
            self.species.append({
                "name": str(names[i]),
                "Z": charges[i] if i < len(charges) else 0,
                "A": masses[i] if i < len(masses) else 0,
                "type": str(types[i]) if i < len(types) else "",
            })
        self.ions_set_mi = masses
        self.ions_set_Zi = charges
        self.ions_set_names = names

    def _get_norms(self, n_norm=1e19, T_norm=1):
        """Compute normalization constants and derived scales."""
        self.keV_to_J = 1e3 * e
        self.J_to_keV = 1.0 / self.keV_to_J
        self.n_norm = n_norm
        self.T_norm = T_norm
        self.tau_norm = self.a / self.c_s
        self.mi_over_mp = self.mi_ref
        self.mi_over_me = self.mi_over_mp * m_p / m_e
        self.v_norm = np.sqrt((self.T_norm*self.keV_to_J)/((self.mi_over_mp)*m_p))
        self.v_norm_e = 4.2e5*self.T_norm*1e3
        self.rhostar_norm = self.v_norm/np.maximum(self.omega_ci, 1e-30)/np.maximum(self.a, 1e-30)
        self.beta_norm = 2*(4e-7 * np.pi)*(self.n_norm)*(self.T_norm*self.keV_to_J)/np.maximum(self.B_T, 1e-30)**2
        self.chiGB_norm = self.a*self.v_norm*(self.rhostar_norm**2)
        self.G_norm = self.n_norm*self.v_norm*(self.rhostar_norm**2)
        self.Q_norm = self.n_norm*(self.T_norm*self.keV_to_J)*self.v_norm*(self.rhostar_norm**2)
        self.Qnorm_to_P = self.Q_norm*self.surfArea*1e-6 # W/m^2 to MW
        self.nuii_norm = 4.8e-8 * self.Zeff * (self.n_norm*1e-6)*self.LogLam/(np.sqrt(self.mi_over_mp) * ( (self.T_norm*1e3)**1.5 ) )*self.tau_norm
        self.nuee_norm = 2.9e-6 * (self.n_norm*1e-6)*self.LogLam/((self.T_norm*1e3)**1.5)*self.tau_norm
        self.nuexch_norm = (self.Zeff/(1840.*self.mi_over_mp))*self.nuee_norm
        self.alphat_norm = (3.13e-18)*self.n_norm*self.R0*(self.q_cyl**2)*self.Zeff/(1e6*self.T_norm**2)


    def _update_w0_by_decompose(self):
        """
        Update Er, ExB shear, and parallel flow diagnostics consistently with
        GACODE rotation theory (https://gacode.io/rotation.html).

        Physics:
            Er = -R*B_p*w0   (total radial electric field) [V/m]
            w0 = w_tor + w1  (total angular frequency) [rad/s]
            w1 = dpdr / (Z*e*ni*R*B_p)  (diamagnetic rotation, SI units) [rad/s]
            
            The input w0 contains total rotation. As dpdr evolves:
            w0_new = w0_input + (w1_new - w1_ref)
            This keeps w_tor constant while w1 evolves with pressure gradient.

        Output quantities:
            SI (physical) units:
                w0, w1 [rad/s]
                Er [V/m]
                vtor [m/s]
                vexb (ExB velocity) [m/s]
                
            Normalized for downstream solvers (NEO, CGYRO, TGLF):
                omega_rot = w0 * tau_norm [-]
                omega_rot_deriv = (dw0/dr) * a [-]
                vexb_shear = (gamma_exb) * tau_norm [-]
                vpar_shear = (gamma_par) * tau_norm [-]
                gamma_e, gamma_p [normalized shearing rates]
        """

        # -------------------------------------------------
        # 1. Ion pressure gradient (evolving)
        # -------------------------------------------------
        spl_pi = akima(self.r, self.pi)
        dpidr = spl_pi.derivative()(self.r)  # [MPa/m]
        ni_SI = self.ni * 1e19  # [1/m^3] SI

        # -------------------------------------------------
        # 2. Diamagnetic rotation frequency (GACODE convention, SI units)
        #    w1 = dp/dr / (Z*e*ni*R*B_p)
        #    Convert dpidr from MPa/m to Pa/m
        # -------------------------------------------------
        dpidr_SI = dpidr * 1e6  # Convert MPa/m -> Pa/m
        w1 = dpidr_SI / (self.Z_ref * e * ni_SI * 
                         self.R * self.B_p)  # [rad/s]

        # -------------------------------------------------
        # 3. Evolve total rotation w0 as diamagnetic part changes
        #    w0 = w_tor + w1  where w_tor = constant (set by NBI, etc.)
        # -------------------------------------------------
        if not hasattr(self, "_w1_ref"):
            # First call: decompose input w0 into w_tor and w1
            self._w1_ref = w1.copy()
            self._w0_input = np.copy(getattr(self, "w0", np.zeros_like(self.r)))
            w0 = self._w0_input
        else:
            # Update: w0 = w_tor + w1_new = (w0_input - w1_ref) + w1_new
            w0 = self._w0_input + (w1 - self._w1_ref)

        self.w0 = w0  # Angular frequency [rad/s]
        self.w1 = w1  # Diamagnetic angular frequency [rad/s]
        
        # Compute derivatives
        spl_w0 = akima(self.r, w0)
        dw0dr = spl_w0.derivative()(self.r)  # [rad/s/m]
        
        # Normalized for downstream solvers (NEO/CGYRO)
        self.omega_rot = w0 * self.tau_norm  # Normalized angular frequency [-]
        self.omega_rot_deriv = dw0dr * self.a  # Normalized angular shear [-]

        # -------------------------------------------------
        # 4. Radial electric field (GACODE: Er = -R*B_p*w0)
        # -------------------------------------------------
        self.Er = -self.R * self.B_p * w0

        # -------------------------------------------------
        # 5. ExB velocity and shearing rate (GACODE/TGLF convention)
        #    VEXB = Er / B [m/s]
        #    GAMMA_E = r*d(VEXB/r)/dr [1/s]
        #    vexb_shear = GAMMA_E * tau_norm [-] for downstream solvers
        # -------------------------------------------------
        self.vexb = self.Er / self.B  # ExB velocity [m/s]
        
        # GAMMA_E = r * d/dr(v_exb/r) [1/s]
        spl_vexb_r = akima(self.r[1:], self.vexb[1:] / self.r[1:], extrapolate=True)
        self.gamma_exb = self.r * spl_vexb_r.derivative()(self.r)  # [1/s]
        
        # Normalized for downstream solvers
        self.vexb_shear = self.gamma_exb * self.tau_norm  # [-]
        self.gamma_e = self.vexb_shear  # CGYRO naming

        # -------------------------------------------------
        # 6. Parallel velocity and shear (TGLF convention)
        #    VPAR = R0 * Vtor / (R * c_s) [-]
        #    where Vtor = R * w0
        # -------------------------------------------------
        vtor = self.R * w0  # Toroidal velocity [m/s]
        
        # VPAR: normalized parallel velocity (sign convention in w0)
        self.vpar = self.R0 * vtor / (self.R * self.c_s)  # [-]
        
        # GAMMA_PAR = R0 * d(Vtor*R)/dr [1/s]
        # d(Vtor*R)/dr = d(R^2*w0)/dr
        spl_vtorR = akima(self.r, vtor * self.R)
        dvtorR_dr = spl_vtorR.derivative()(self.r)
        self.gamma_par = -self.R0 * dvtorR_dr / self.a  # [1/s]
        
        # Normalized for downstream solvers
        self.vpar_shear = self.gamma_par * self.tau_norm  # [-]
        self.gamma_p = self.vpar_shear  # CGYRO naming

        # -------------------------------------------------
        # 7. Toroidal velocity & Mach number
        # -------------------------------------------------
        self.vtor = vtor
        self.Mach = vtor / self.c_s

        # -------------------------------------------------
        # 8. Log-gradient of rotation (for transport models)
        # -------------------------------------------------
        self.aLw0 = calc.aLy(self.r, w0)

        # -------------------------------------------------
        # 9. Normalized results (for transport solvers)
        # -------------------------------------------------     
        self.gamma_exb_norm = self.vexb_shear
        self.gamma_par_norm = self.vpar_shear
        
        return

    def _get_power_flows(self):

        self.Gbeam_e = calc.integrated_flux(getattr(self,'qpar_beam',np.zeros_like(self.roa)), self.r, self.dVdr, self.surfArea) / self.n_norm
        self.Gwall_e = calc.integrated_flux(getattr(self,'qpar_wall',np.zeros_like(self.roa)), self.r, self.dVdr, self.surfArea) / self.n_norm

        self.Pohm_e = calc.volume_integrate(self.r, getattr(self,'qohme',np.zeros_like(self.roa)), self.dVdr)
        self.Pbeam_e = calc.volume_integrate(self.r, getattr(self,'qbeame',np.zeros_like(self.roa)), self.dVdr)
        self.Prf_e = calc.volume_integrate(self.r, getattr(self,'qrfe',np.zeros_like(self.roa)), self.dVdr)

        self.Pohm_i = calc.volume_integrate(self.r, getattr(self,'qohmi',np.zeros_like(self.roa)), self.dVdr)
        self.Pbeam_i = calc.volume_integrate(self.r, getattr(self,'qbeami',np.zeros_like(self.roa)), self.dVdr)
        self.Prf_i = calc.volume_integrate(self.r, getattr(self,'qrfi',np.zeros_like(self.roa)), self.dVdr)
            
        # check for integrated_vars to add integration constant on each flow after trimming
        if hasattr(self, 'integrated_vars'):
            for var in self.integrated_vars:
                var_0 = getattr(self, f"{var}_0", 0.0)
                var_val = getattr(self, var, 0.0)
                setattr(self, var, var_val + var_0)

        self.Gaux_e = self.Gbeam_e + self.Gwall_e
        self.Gaux_i = self.Gaux_e/self.Zeff
        self.Paux_e = self.Pohm_e + self.Pbeam_e + self.Prf_e
        self.Paux_i = self.Pohm_i + self.Pbeam_i + self.Prf_i

        self.integrated_vars = ['Gbeam_e','Gwall_e',
                                'Pohm_e','Pbeam_e','Prf_e',
                                'Pohm_i','Pbeam_i','Prf_i']

    def _trim(self):
        """Trim all array attributes to domain. """
        x0, x1 = self.domain[0], self.domain[1]
        ix0,ix1 = np.searchsorted(self.roa, [x0, x1])
        ix1 += 1
        r_len = ix1 - ix0
        state_keys = list(vars(self).keys())

        # Collect new attributes to add after iteration
        new_attrs = {}
        
        for attr in state_keys:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray) and len(val) > r_len:
                if val.ndim == 1:
                    setattr(self, attr, val[ix0:ix1])
                elif val.ndim == 2:
                    setattr(self, attr, val[ix0:ix1, :])
                if attr in self.integrated_vars:
                    new_attr = f"{attr}_0"
                    new_attrs[new_attr] = val[ix0]  # store initial value for constant added to integration later
        
        # Add new attributes after iteration completes
        for new_attr, new_val in new_attrs.items():
            setattr(self, new_attr, new_val)

        self.nexp = np.array([len(self.roa)], dtype='<U2')

    def get_atomic_rates(self) -> dict:
        """
        Get atomic physics rates from Aurora.

        Parameters
        ----------
        state : object
            The simulation state containing species and their properties.

        Returns
        -------
        rates : dict
            {'ion': nu_ion, 'recom': nu_rec, 'cx': nu_cx}
            All in [1/s] including ne factor
        """

        # Conversions
        te_eV = self.te * 1e3
        ti_eV = self.ti * 1e3
        ne_cm3 = self.ne * 1e13  # [cm^-3]

        Rion = np.ones_like(self.ti_full)*1e-6
        Rrecom = np.ones_like(self.ti_full)*1e-6
        Rcx = np.ones_like(self.ti_full)*1e-6

        try:

            for sidx, sp in enumerate(self.species):


                    atom_data = aurora.get_atom_data(sp['name'], ['acd', 'scd', 'ccd'])
            
                    # Get reaction rates [1/s]
                    _, R_ion, R_recom, R_cx = aurora.atomic.get_cs_balance_terms(
                    atom_data, ne_cm3=ne_cm3, Te_eV=te_eV, Ti_eV=ti_eV, include_cx=True
                    )[:4]

                    Rion[:,sidx] = R_ion[:,0]
                    Rrecom[:,sidx] = R_recom[:,0]
                    Rcx[:,sidx] = R_cx[:,0]

        except:
            pass

        return {
            'ion': Rion,  # [1/s]
            'recom': Rrecom,  # [1/s]
            'cx': Rcx,  # [1/s]
        }

    def update_profile(self, name_or_dict, value=None, mark_dirty: bool = True):
        """
        Update one or multiple PlasmaState attributes.

        Parameters
        ----------
        name_or_dict : str | dict
            Either a single attribute name (str) or a dictionary of {name: value}.
        value : any, optional
            New value if updating a single attribute.
        mark_dirty : bool, default True
            Flag that derived quantities need recomputation.

        Examples
        --------
        state.update_profile("aLne", new_aLne)
        state.update_profile({"aLne": new_aLne, "aLTe": new_aLTe})
        """
        if isinstance(name_or_dict, dict):
            for k, v in name_or_dict.items():
                setattr(self, k, v)
        elif isinstance(name_or_dict, str):
            setattr(self, name_or_dict, value)
        else:
            raise TypeError("update_profile() expects a str or dict as first argument")

        if mark_dirty:
            self._dirty = True
        return self


    def refresh(self):
        """
        Recompute derived quantities after a profile change.
        """

        if not hasattr(self, "_dirty") or not self._dirty:
            return self

        self.process()
        self._dirty = False

        return self


    def update(self, X, parameters, process: bool = True, neutrals=None):
        """
        Update PlasmaState profiles from a Parameters object.

        Parameters
        ----------
        parameters : object
            Must contain dict attributes `y` and `aLy` with keys for each predicted profile.
        process : bool, default=True
            If True, recompute all derived quantities via self.process().
        """

        new_y = parameters.get_y(X,self.roa)
        new_aLy = parameters.get_aLy(X,self.roa)
        new_curv = parameters.get_curvature(X,self.roa)

        # --- Transfer profile values ---
        for prof in X.keys():
            setattr(self, prof, new_y[prof])
            setattr(self, f"aL{prof}", new_aLy[prof])
            setattr(self, f"d2{prof}", np.asarray(new_curv[prof]))

        # --- Trigger downstream processing ---
        if process and hasattr(self, "process"):
            self.process(neutrals=neutrals)

    def apply_scaling(self, scale_factors: Dict[str, float], verbose: bool = True):
        """Apply or re-apply scaling factors to heating/particle sources.
        
        This method allows scaling to be applied or modified after state initialization,
        useful for parameter scans or restart scenarios.
        
        Parameters
        ----------
        scale_factors : Dict[str, float]
            Dictionary of scaling factors with keys like:
            - 'scale_Pe_beam', 'scale_Pi_beam', 'scale_Qpar_beam'
            - 'scale_ohmic_e', 'scale_ohmic_i'
            - 'scale_rf_e', 'scale_rf_i'
        verbose : bool, default True
            If True, print applied scaling factors
            
        Notes
        -----
        - Requires metadata['unscaled_heating'] to be populated (set by workflow)
        - Scaling is applied to original unscaled values, not cumulative
        - Powers (Paux_e, Paux_i) must be recomputed after scaling via _get_power_flows()
        
        Examples
        --------
        >>> state.apply_scaling({'scale_Pe_beam': 1.2, 'scale_Pi_beam': 0.8})
        >>> state._get_power_flows()  # Recompute integrated powers
        """
        if 'unscaled_heating' not in self.metadata:
            if verbose:
                print("Warning: No unscaled_heating in metadata. Cannot apply scaling.")
            return
        
        # Define scaling mappings
        scaling_map = {
            'scale_Pe_beam': 'qbeame',
            'scale_Pi_beam': 'qbeami',
            'scale_Qpar_beam': 'qpar_beam',
            'scale_Qpar_wall': 'qpar_wall',
            'scale_ohmic_e': 'qohme',
            'scale_ohmic_i': 'qohmi',
            'scale_rf_e': 'qrfe',
            'scale_rf_i': 'qrfi',
        }
        
        scaling_applied = []
        for scale_key, state_var in scaling_map.items():
            if scale_key in scale_factors:
                scale_factor = scale_factors[scale_key]
                original = self.metadata['unscaled_heating'].get(state_var)
                
                if original is not None and hasattr(self, state_var):
                    scaled_value = original * scale_factor
                    setattr(self, state_var, scaled_value)
                    scaling_applied.append(f"{scale_key}={scale_factor:.3f}")
        
        if verbose and scaling_applied:
            print(f"Applied scaling: {', '.join(scaling_applied)}")
        
        # Update metadata with current scaling
        self.metadata['applied_scaling'] = {k: v for k, v in scale_factors.items() 
                                           if k.startswith('scale_')}
