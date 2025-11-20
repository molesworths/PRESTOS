"""
Plasma state representation for the standalone solver.

This version centers the state around a gacode profile container.
It reads a gacode file (or pre-parsed profiles), builds a clean
PlasmaState with category dictionaries, and provides round-trip
conversion back to a gacode-like object.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Union
import copy
from tools import plasma, calc, geometry, io
from interfaces import gacode 
from scipy.constants import m_p, m_e, e, k, c, mu_0, u, pi, epsilon_0


@dataclass
class PlasmaState:
    """
    Flat container for derived and raw plasma quantities.
    Each quantity is an attribute (e.g., ne, te, ti, q, rho, etc.).
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

        add_if_not_in_map = {"wtor(rad/s)":"wtor","n0(10^19/m^3)":"n0"}
        mapping = self.metadata.get("gacode_mapping", {})
        for gacode_key, flat_name in add_if_not_in_map.items():
            if flat_name not in mapping.values():
                mapping[flat_name] = gacode_key
        self.metadata["gacode_mapping"] = mapping

        for flat_name, gacode_key in mapping.items():
            # Only export if attribute exists and is array-like
            if hasattr(self, flat_name):
                val = getattr(self, flat_name)
                if isinstance(val, (list, np.ndarray)):
                    profiles[gacode_key] = np.array(val)

        #TODO: Add translation of Pfus_e/i, Pbrem_e/i, Pline_e/i, Pei, Pcx, to qfuse/i, qbreme/i, qohme/i, etc.

        gacode_obj = gacode(profiles=profiles)
        return gacode_obj


    def process(self, gacode_obj: "gacode" = None):
        """
        Populate PlasmaState with derived quantities.
        Optionally pass a gacode object for any missing profiles.
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


        self.Zeff = plasma.get_Zeff(self.ne, self.ni_full, self.species)

        # --- Basic physical constants and derived scalings ---
        self.c_s = plasma.c_s(self.te, self.mi_ref)
        self.rho_s = plasma.rho_s(self.te, getattr(self, "mi_ref", 2.0), self.B)
        self.vth = plasma.vthermal(self.ti, self.mi_ref)
        self.q_gb, self.g_gb, *_ = plasma.gyrobohm_units(
            self.te,
            self.ne * 1e-1,  # convert 10^19/m^3 -> 10^20/m^3
            self.mi_ref,
            np.abs(self.B),
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
        self.betae = plasma.betae(self.te, self.ne, self.B)
        self.xnue = plasma.xnue(self.te, self.ne*0.1, self.a, mref_u=self.mi_ref)
        self.debye = plasma.debye(self.te, self.ne*0.1,self.mi_ref,self.B)

        # self.pprime = 1E-7 * self.q*self.a**2/self.r/self.B**2*np.gradient(self.ptot,self.r)
        # self.drmindr = np.gradient(self.r,self.r)
        # self.dRmajdr = np.gradient(self.rmaj,self.r)
        # self.dZmajdr = np.gradient(self.zmag,self.r)

        # self.s_kappa  = self.r / self.kappa * np.gradient(self.kappa,self.r)
        # self.s_delta  = self.r / self.delta * np.gradient(self.delta,self.r)
        # self.s_zeta   = self.r / self.zeta * np.gradient(self.zeta,self.r)

        s = self.r / self.q*np.gradient(self.q,self.r)
        self.s_q =  np.concatenate([np.array([0.0]),(self.q[1:] / self.roa[1:])**2 * s[1:]]) # infinite in first location

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
        self.Qnorm_to_P = self.Q_norm*self.surfArea*1e-6 # MW/m^2 to MW
        self.nuii_norm = 4.8e-8 * self.Zeff * (self.n_norm*1e-6)*self.LogLam/(np.sqrt(self.mi_over_mp) * ( (self.T_norm*1e3)**1.5 ) )*self.tau_norm
        self.nuee_norm = 2.9e-6 * (self.n_norm*1e-6)*self.LogLam/((self.T_norm*1e3)**1.5)*self.tau_norm
        self.nuexch_norm = (self.Zeff/(1840.*self.mi_over_mp))*self.nuee_norm
        self.alphat_norm = (3.13e-18)*self.n_norm*self.R0*(self.q_cyl**2)*self.Zeff/(1e6*self.T_norm**2)


    def _update_w0_by_decompose(self, store_key: str = 'wtor'):
        """Decompose and update toroidal + diamagnetic rotation components using available vtor, vpol fields.
        
        This needs corrections.
        
        """
        eps = 1e-12

        first_iter = not hasattr(self, store_key)
        if first_iter:
            # if hasattr(self, "vtor"):
            #     wtor = np.copy(self.vtor / np.maximum(self.R, eps))
            # else:
            #     wtor = np.copy(getattr(self, 'w0', np.zeros_like(self.r)))
            wtor = np.copy(getattr(self, 'w0', np.zeros_like(self.r)))
            setattr(self, store_key, wtor)
        else:
            wtor = getattr(self, store_key)

        dpdr = np.gradient(self.pi,self.r)

        ni_SI = self.ni * 1e19
        Er_dia = dpdr / np.maximum(self.Z_ref * e * ni_SI, eps)
        # E×B poloidal velocity component (poloidal), vpol ≈ (Er × B)·e_pol / B^2 ≈ Er * B_T / B^2
        vpol = Er_dia * self.B_T / np.maximum(self.B**2, eps)

        wpol = np.nan_to_num(Er_dia / np.maximum(self.R * self.B_p, eps), 0.0)

        if first_iter:
            wtor -= wpol
            setattr(self, store_key, wtor)

        w0 = wtor + wpol

        dwtordr = np.gradient(wtor, self.r)
        dw0dr = np.gradient(w0, self.r)
        dvpoldr = np.gradient(vpol, self.r)

        aLw0 = calc.aLy(self.r, w0)

        gamma_par = -self.R * dwtordr
        vpar_shear = gamma_par * self.a / np.maximum(self.c_s, 1e-30)
        vpar = self.rmaj * wtor / np.maximum(self.c_s, 1e-30)

        gamma_exb_tor = -(self.r / np.maximum(np.abs(self.q), eps)) * dwtordr
        gamma_exb_pol = dvpoldr
        vexb_shear = (np.abs(gamma_exb_tor) + np.abs(gamma_exb_pol)) * self.a / np.maximum(self.c_s, 1e-30)

        self.w0 = w0
        self.aLw0 = aLw0
        self.vpar = vpar
        self.vpar_shear = vpar_shear
        self.vexb_shear = vexb_shear
        self.gamma_par = gamma_par
        self.gamma_exb_tor = gamma_exb_tor
        self.gamma_exb_pol = gamma_exb_pol
        self.gamma_exb = abs(gamma_exb_tor) + abs(gamma_exb_pol) # TODO FIX
        self.vpol = vpol
        self.vtor = self.R * wtor

    def _get_power_flows(self):
        self.Gbeam_e = calc.integrated_flux(getattr(self,'qpar_beam',np.zeros_like(self.roa)), self.r, self.dVdr, self.surfArea) / self.n_norm
        self.Gwall_e = calc.integrated_flux(getattr(self,'qpar_wall',np.zeros_like(self.roa)), self.r, self.dVdr, self.surfArea) / self.n_norm
        self.Gaux_e = self.Gbeam_e + self.Gwall_e
        self.Gaux_i = self.Gaux_e/self.Zeff

        self.Pohm_e = calc.volume_integrate(self.r, getattr(self,'qohme',np.zeros_like(self.roa)), self.dVdr)
        self.Pbeam_e = calc.volume_integrate(self.r, getattr(self,'qbeame',np.zeros_like(self.roa)), self.dVdr)
        self.Prf_e = calc.volume_integrate(self.r, getattr(self,'qrfe',np.zeros_like(self.roa)), self.dVdr)
        self.Paux_e = self.Pohm_e + self.Pbeam_e + self.Prf_e

        self.Pohm_i = calc.volume_integrate(self.r, getattr(self,'qohmi',np.zeros_like(self.roa)), self.dVdr)
        self.Pbeam_i = calc.volume_integrate(self.r, getattr(self,'qbeami',np.zeros_like(self.roa)), self.dVdr)
        self.Prf_i = calc.volume_integrate(self.r, getattr(self,'qrfi',np.zeros_like(self.roa)), self.dVdr)
        self.Paux_i = self.Pohm_i + self.Pbeam_i + self.Prf_i

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


    def update(self, X, parameters, process: bool = True):
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
            self.process()