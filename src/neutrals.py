"""
Neutral transport models: kinetic (ballistic) and diffusive.

Implements:
- Kinetic model from NEUCG (Burrell 1973) with Aurora atomic physics
- Diffusive model with 1D transport equation
"""

import numpy as np
import scipy as sp
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.integrate import cumulative_trapezoid, odeint
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.constants import k, m_p, e, pi
import aurora
from tools import plasma, calc, io


class NeutralModel:
    """Base class for neutral models."""
    
    def __init__(self, options: dict):
        self.n0_edge = np.asarray(options.get("n0_edge", [1e-6]),dtype='float')  # [1e19 m^-3]

    def solve(self, state, **kwargs) -> None:
        """Solve for neutral profiles."""

        N_species = len(state.species)
        if self.n0_edge.size != N_species:
            self.n0_edge = np.full(N_species, self.n0_edge[0])

        # Get atomic rates
        rates = get_atomic_rates(state)
        state.nu_ion = rates['ionization']  # [1/s]
        state.nu_rec = rates['recombination']  # [1/s]
        state.nu_cx = rates['charge_exchange']  # [1/s]
        
        # unit conversions
        self.ne_SI = state.ne * 1e19  # m^-3
        self.Te_eV = state.te * 1e3  # eV
        self.Ti_eV = state.ti * 1e3  # eV

        state.n0 = np.zeros_like(state.ni_full)
        state.Gamma0 = np.zeros_like(state.ni_full)
        state.Kn = np.zeros_like(state.ni_full)

        return
    
    def get_Knudsen_number(self,state) -> np.ndarray:
        """Compute Knudsen number profile for each species."""
        Kn = np.zeros_like(state.ni)
        for sidx, sp in enumerate(state.species):
            species_name = sp['N']
            if species_name== 'LUMPED':
                continue
            mi_amu = sp.get('A', 2.0)
            mi_kg = mi_amu * m_p
            
            ni_SI = state.ni[:,sidx] * 1e19  # m^-3

            nu_ion = state.nu_ion[:,sidx]  # [1/s]
            nu_cx = state.nu_cx[:,sidx]  # [1/s]
            lambda_mfp = state.vth[:,sidx] / (nu_ion + nu_cx + 1e-30)  # m
            L_scale = calc.scale_length(ni_SI, state.r)  # m
            Kn[:,sidx] = lambda_mfp / (L_scale + 1e-30)
        return Kn
    
    def get_neutral_flux(self,state) -> np.ndarray:
        """Compute neutral flux profile for each species."""

        S_0 = (state.n0 * state.nu_ion - state.ni_full * state.nu_rec) * 1e19  # [m^-3 s^-1]
        Gamma0 = np.zeros_like(state.ni_full)
        #TODO: check axis on cumtrapz below
        #Gamma0_prof = calc.volume_integrate(state.r, S_0.sum(axis=1), state.dVdr) # [m^-2 s^-1]
        Gamma0_prof = calc.integrated_flux(S_0.sum(axis=1), state.r, state.dVdr, state.surfArea)
        Gamma0 = -Gamma0_prof * 1e-20  # [1e20/m^2/s]

        return Gamma0


class KineticNeutralModel(NeutralModel):
    """
    Ballistic neutral transport model.
    
    Solves for neutral density and temperature profiles given:
    - Edge neutral density/temperature
    - Plasma profiles (ne, te, ti, ni)
    - Atomic physics rates (ionization, recombination, CX)
    
    Uses analytical velocity integrals for computational efficiency.
    """

    def __init__(self, options: dict):
        """
        Parameters
        ----------
        n0_edge : list
            Edge neutral density [1e19 m^-3]
        """
        super().__init__(options)

    def solve(self, state,
              picard: bool = False, include_cx_firstgen: bool = True,
              max_cx_iter: int = 20, tol_cx: float = 1e-3) -> None:
        """
        Solve for neutral profiles.
        
        Parameters
        ----------
        state : PlasmaState
            Plasma state with ne, te, ti, ni profiles
        n0_edge : float
            Edge neutral density [1e19 m^-3]
        t0_edge : float
            Edge neutral temperature [keV]
        n_cx_iter : int
            Number of CX iterations
        
        Returns
        -------
        neutral_state : NeutralState
            Neutral profiles and sources
        """
        
        super().solve(state)
        r_m = state.r  # [m]

        for sidx, sp in enumerate(state.species):
            species_name = sp['N']
            if species_name== 'LUMPED':
                continue
            mi_amu = sp.get('A', 2.0)
            mi_kg = mi_amu * m_p
            ni_SI = state.ni_full[:,sidx] * 1e19  # m^-3
            n0_edge_m3 = self.n0_edge[sidx] * 1e19  # m^-3
            Tedge_eV = self.Ti_eV[-1,sidx]  # eV

            nu_ion = state.nu_ion[:,sidx]  # [1/s]
            nu_rec = state.nu_rec[:,sidx]  # [1/s]
            nu_cx = state.nu_cx[:,sidx]  # [1/s]
            S_rec = ni_SI * nu_rec  # [m^-3 s^-1]

            # Compute optical depth G(r) = ∫_r^edge (ν_ion + ν_cx) ds
            # Must integrate OUTWARD from core to edge

            G = _compute_optical_depth(nu_ion, nu_cx, state.r)  # [m/s]
            
            # Dimensionless optical depth for boundary integral
            tau = G / state.vth
            
            # Initial source: recombination only
            q_s = S_rec.copy()

            # Picard iteration for first-generation CX neutrals
            if include_cx_firstgen:
                for it in range(max_cx_iter):
                    n0_m3 = _build_n0_from_sources_picard(n0_edge_m3, q_s) \
                        if picard else _build_n0_from_sources(n0_edge_m3, q_s)
                    S_cx = n0_m3 * nu_cx
                    q_new = S_rec + S_cx
                    rel = np.linalg.norm(q_new - q_s) / (np.linalg.norm(q_s) + 1e-30)
                    q_s = q_new
                    if rel < tol_cx:
                        break
            else:
                n0_m3 = _build_n0_from_sources(n0_edge_m3, q_s)
            
            # Final profiles
            state.n0[:,sidx] = n0_m3 / 1e19  # [1e19/m^3]
        
        state.Gamma0 = self.get_neutral_flux(state)  # [1e20 m^-3 s^-1]



        def _compute_optical_depth(nu_ion: np.ndarray, nu_cx: np.ndarray, r: np.ndarray) -> np.ndarray:
            """
            Compute optical depth G(r) = ∫_r^edge γ(s) ds.
            
            Integrates OUTWARD from each point to edge.
            
            Parameters
            ----------
            gamma : np.ndarray
                Attenuation rate γ = ν_ion + ν_cx [1/s]
            r : np.ndarray
                Radial coordinate [m]
            
            Returns
            -------
            G : np.ndarray
                Optical depth [m/s]
            """
            gamma = nu_ion + nu_cx  # [1/s]

            # Integrate from edge inward, then flip
            G_inward = cumulative_trapezoid(gamma[::-1], r[::-1], initial=0.0)
            G = G_inward[::-1]
            return G
        
        def _Ib_dimless(A):
            """
            I_b = ∫_0^∞ (4/√π) x^2 exp(-x^2 - A/x) dx
            returns dimensionless scalar.
            """
            pref = 4.0 / np.sqrt(pi)
            # integrand in x
            def integrand(x):
                if x <= 0.0:
                    return 0.0
                # avoid overflow/underflow by using np.exp with safe args
                return pref * x*x * np.exp(-x*x - A / x)
            val, err = sp.integrate.quad(integrand, 0.0, np.inf, epsabs=1e-9, epsrel=1e-7, limit=200)
            return val

        def _Id_dimless(A):
            """
            J = ∫_0^∞ (4/√π) x * exp(-x^2 - A/x) dx
            Returns the dimensionless part; actual I_d = (1/v_th) * J.
            """
            pref = 4.0 / np.sqrt(pi)
            def integrand(x):
                if x <= 0.0:
                    return 0.0
                return pref * x * np.exp(-x*x - A / x)
            val, err = sp.integrate.quad(integrand, 0.0, np.inf, epsabs=1e-9, epsrel=1e-7, limit=200)
            return val

        def _boundary_integral_tau(tau, T_eV, mass_kg):
            """Compute I_b(tau) = ∫ φ(v) e^{-tau/v} dv (dimensionless)."""
            T_J = T_eV * e
            v_th = plasma.vthermal(T_eV*1e-3, mass_kg)
            A = tau / v_th
            return _Ib_dimless(A)

        def _distributed_integral_tau(tau, T_eV, mass_kg):
            """Compute I_d(tau) = ∫ φ(v)/v e^{-tau/v} dv  (returns value in 1/v_th units)."""
            v_th = plasma.vthermal(T_eV*1e-3, mass_kg)
            A = tau / v_th
            J = _Id_dimless(A)
            return J / v_th  # because Id_dimless = ∫ (4/√π) x e^{-x^2 - A/x} dx and I_d = J / v_th

        # Build n0 from boundary and distributed sources using proper velocity integrals
        def _build_n0_from_sources_picard(n_edge_val, q_s_array):
            n0_local = np.zeros_like(r_m)
            
            for k in range(self.r.size):
                # 1) Boundary contribution: n_edge * ∫ φ_edge(v) * exp(-τ_ra / v) dv
                tau_ra = G[k]
                #Ib = integral_boundary(tau_ra, Tedge_eV, mi_kg)
                #Ib = integral_boundary_approx(tau_ra, Tedge_eV, mi_kg)  # Use approximation for speed
                Ib = _boundary_integral_tau(tau_ra, Tedge_eV, mi_kg)
                term_b = n_edge_val * Ib
                
                # 2) Distributed source: ∫_r^a q(s) * [∫ φ_s(v)/v * exp(-τ_rs/v) dv] ds
                term_d = 0.0
                for j in range(k+1, self.r.size):
                    q_sj = q_s_array[j]
                    if q_sj <= 0:
                        continue
                    tau_rs = G[k] - G[j]  # ∫_r_k^r_j gamma ds
                    if tau_rs < 0:
                        continue
                    T_s_eV = self.Ti_eV[j]
                    #Jrs = integral_distributed(tau_rs, T_s_eV, mi_kg)
                    #Jrs = integral_distributed_approx(tau_rs, T_s_eV, mi_kg)
                    Jrs = _distributed_integral_tau(tau_rs, T_s_eV, mi_kg)
                    dr = r_m[j] - r_m[j-1] if j > 0 else (r_m[1] - r_m[0] if r_m.size > 1 else 1.0)
                    term_d += q_sj * Jrs * dr
                
                n0_local[k] = term_b + term_d
            return n0_local
        
        # (I-K)*n = M*n = RHS
        def _build_n0_from_sources(n_edge, q_s_array):

            Tedge_eV = self.Ti_eV[-1]
            N = len(self.r)
            dr = np.empty(N)
            dr[0] = self.r[1] - self.r[0]
            dr[1:] = np.diff(self.r)

            # Assemble kernel K (I - K) n = RHS
            K = np.zeros((N,N))
            RHS = np.zeros(N)

            # Precompute I_b at nodes
            I_b_nodes = np.array([_boundary_integral_tau(G[i], Tedge_eV, mi_kg) for i in range(N)])

            # Precompute I_d(tau) table maybe; here compute on the fly
            for i in range(N):
                # RHS boundary term
                RHS[i] = n_edge * I_b_nodes[i]
                # distributed recombination contribution to RHS
                for j in range(i+1, N):
                    tau = G[i] - G[j]
                    Id = _distributed_integral_tau(tau, self.Ti_eV[j], mi_kg)  # s/m
                    RHS[i] += S_rec[j] * Id * dr[j]   # S_rec [m^-3 s^-1] * Id [s/m] * dr [m] -> m^-3
                    # fill K
                    K[i,j] = nu_cx[j] * Id * dr[j]   # dimensionless multiplicative weight for n0[j]
            # Left-hand matrix
            M = np.eye(N) - K

            # Solve linear system
            n0_vec = np.linalg.solve(M, RHS)  # n0 in m^-3
            #n0_vec = spsolve(M, RHS)

            # optional: check condition number
            cond = np.linalg.cond(M)
            print("cond(M) =", cond)

            return n0_vec
        
        return
    

class DiffusiveNeutralModel(NeutralModel):
    """
    Diffusive neutral transport model.
    
    Solves 1D diffusion equation:
        d/dr(D_0 dn_0/dr) - S_ion * n_0 + S_rec = 0
    
    where D_0 = v_th^2 / (2 * (R_ion + R_cx))
    
    Based on MINTneutrals.update_neutral_density_profiles_diff
    """
    def __init__(self, options: dict):
        """
        Parameters
        ----------
        n0_edge : list
            Edge neutral density [1e19 m^-3]
        """
        super().__init__(options)
    
    def solve(self, state,
              bc_type_left: str = 'flux', bc_left_value: float = 0.0,
              method: str = 'implicit', max_iter: int = 50,
              tol: float = 1e-6) -> None:
        """
        Solve diffusive neutral transport.
        
        Parameters
        ----------
        state : PlasmaState
            Plasma profiles
        n0_sep : float
            Separatrix neutral density [1e19 m^-3]
        t0_sep : float
            Separatrix neutral temperature [keV]
        bc_type_left : str
            'flux' for zero flux at axis, 'dirichlet' for fixed value
        bc_left_value : float
            Value for left BC
        method : str
            'implicit' (tridiagonal) or 'log' (log-space iteration)
        max_iter : int
            Maximum iterations for log method
        tol : float
            Convergence tolerance
        
        Returns
        -------
        neutral_state : NeutralState
        """
        
        super().solve(state)
        x = state.roa  # [m]

        for sidx, sp in enumerate(state.species):
            species_name = sp['name']
            if species_name== 'LUMPED':
                continue
            mi_amu = sp.get('A', 2.0)
            mi_kg = mi_amu * m_p
            ni_SI = state.ni_full[:,sidx] * 1e19  # m^-3
            n0_edge_m3 = self.n0_edge[sidx] * 1e19  # m^-3

            nu_ion = state.nu_ion[:,sidx]  # [1/s]
            nu_rec = state.nu_rec[:,sidx]  # [1/s]
            nu_cx = state.nu_cx[:,sidx]  # [1/s]
            S_rec = ni_SI * nu_rec  # [m^-3 s^-1]
        
            # Diffusion coefficient D = v_th^2 / (2*(R_ion + R_cx))
            D0 = state.vth**2 / (2.0 * (nu_ion + nu_cx))  # [m^2/s]
            D0 = np.maximum(D0, 1e-6)  # Avoid division by zero

            x_fine, ys_fine = self._adaptive_grid_interpolate(
                    x, D0, ni_SI, nu_ion, nu_rec, nu_cx, fine_factor=5, adaptive_threshold=0.1, method='akima')
            D0_fine, ni_fine, R_ion_fine, R_rec_fine, R_cx_fine = ys_fine

            # Solve based on method
            if method == 'implicit':
                n0 = self._solve_tridiagonal(
                    x_fine, ni_fine, D0_fine, R_ion_fine, R_rec_fine,
                    bc_type_left=bc_type_left,
                    bc_left_value=bc_left_value * 1e19,  # Convert to SI
                    bc_type_right='dirichlet',
                    bc_right_value=n0_edge_m3
                )
                n0 /= 1e19  # Back to 1e19 m^-3
            elif method == 'lambda':
                R_total = R_ion_fine+R_cx_fine
                lambda_n0 = np.sqrt(D0_fine/R_total) # local mean free path
                n0 = n0_edge_m3*np.exp((x_fine-x[-1])/lambda_n0) # simple decay length approximation, 
                n0 /= 1e19
            else:
                raise ValueError(f"Unknown method: {method}")
            
                        # Final profiles
            state.n0[:,sidx] = self._interp_shape(x_fine,np.maximum(n0,0),x,method='akima')  # [1e19/m^3]
        
        state.Gamma0 = self.get_neutral_flux(state)  # [1e20 m^-3 s^-1]

    
    def _solve_tridiagonal(self, r: np.ndarray, ni: np.ndarray, D: np.ndarray,
                        R_ion: np.ndarray, R_recom: np.ndarray,
                        bc_type_left: str = 'flux', bc_left_value: float = 0.0,
                        bc_type_right: str = 'dirichlet', bc_right_value: float = 1e19):
        """
        Solve d/dr(D dn/dr) - R_ion*n + ni*R_recom = 0 using tridiagonal matrix.
        
        Discretized as:
            a_i * n_{i-1} + b_i * n_i + c_i * n_{i+1} = d_i
        
        Based on MINTneutrals.solve_neutrals_tridiag
        """
        N = len(r)
        dr = np.diff(r)
        dr_mid = 0.5 * (dr[:-1] + dr[1:])
        
        # Allocate tridiagonal arrays
        a = np.zeros(N)  # sub-diagonal
        b = np.zeros(N)  # diagonal
        c = np.zeros(N)  # super-diagonal
        d = np.zeros(N)  # RHS
        
        # Interior points: centered finite difference
        for i in range(1, N-1):
            dr_left = r[i] - r[i-1]
            dr_right = r[i+1] - r[i]
            dr_avg = 0.5 * (dr_left + dr_right)
            
            D_left = 0.5 * (D[i-1] + D[i])
            D_right = 0.5 * (D[i] + D[i+1])
            
            a[i] = -D_left / (dr_left * dr_avg)
            c[i] = -D_right / (dr_right * dr_avg)
            b[i] = -(a[i] + c[i]) - R_ion[i]
            d[i] = -ni[i] * R_recom[i]
        
        # Left boundary (axis)
        if bc_type_left == 'flux':
            # dn/dr = bc_left_value (usually 0)
            # Use forward difference: (n_1 - n_0) / dr_0 = flux
            b[0] = -1.0 / dr[0] - R_ion[0]
            c[0] = 1.0 / dr[0]
            d[0] = bc_left_value - ni[0] * R_recom[0]
        elif bc_type_left == 'dirichlet':
            b[0] = 1.0
            c[0] = 0.0
            d[0] = bc_left_value
        else:
            raise ValueError(f"Unknown bc_type_left: {bc_type_left}")
        
        # Right boundary (separatrix)
        if bc_type_right == 'dirichlet':
            a[N-1] = 0.0
            b[N-1] = 1.0
            d[N-1] = bc_right_value
        elif bc_type_right == 'flux':
            a[N-1] = -1.0 / dr[-1]
            b[N-1] = 1.0 / dr[-1] - R_ion[N-1]
            d[N-1] = bc_right_value - ni[N-1] * R_recom[N-1]
        else:
            raise ValueError(f"Unknown bc_type_right: {bc_type_right}")
        
        # Solve tridiagonal system
        A = diags([a[1:], b, c[:-1]], offsets=[-1, 0, 1], format='csr')
        n0 = spsolve(A, d)
        
        return np.maximum(n0, 0.0)  # Enforce positivity
    
    def _adaptive_grid_interpolate(self, x:np.array, *ys:np.array, fine_factor=5, adaptive_threshold=0.1, method='akima'):
        """
        Adaptive grid with shape-preserving polynomial interpolation for any number of y arrays.

        Parameters
        ----------
        x : ndarray
            1D grid (monotonic)
        *ys : ndarrays
            Any number of 1D arrays to interpolate (same shape as x)
        fine_factor : int
            Base refinement factor
        adaptive_threshold : float
            Threshold for adaptive refinement
        method : str
            Interpolation method ('pchip' or 'akima')

        Returns
        -------
        x_fine : ndarray
            Refined grid
        y_fines : list of ndarrays
            Interpolated arrays on x_fine, one for each input y
        """

        n = len(x)
        ys = [np.nan_to_num(y) for y in ys]

        # Compute refinement indicator from all ys
        def compute_variation_indicator(y:np.array, x:np.array):
            dy_dx = np.gradient(y, x)
            d2y_dx2 = np.gradient(dy_dx, x)
            y_scale = np.maximum(np.abs(y), np.max(np.abs(y)) * 1e-6)
            curvature = np.abs(d2y_dx2) / y_scale
            dx = np.diff(x)
            dx_avg = np.concatenate([[dx[0]], 0.5*(dx[:-1] + dx[1:]), [dx[-1]]])
            gradient_indicator = np.abs(dy_dx) * dx_avg / y_scale
            return np.maximum(curvature, gradient_indicator)

        indicators = [compute_variation_indicator(y, x) for y in ys]
        refine_indicator = np.maximum.reduce([ind/np.max(ind) for ind in indicators])

        # Build adaptive grid
        x_fine_list = []
        for i in range(n - 1):
            x_fine_list.append(x[i])
            n_sub = int(fine_factor * (1 + 3 * refine_indicator[i])) if refine_indicator[i] > adaptive_threshold else 2
            n_sub = min(n_sub, 20)
            if n_sub > 2:
                x_sub = np.linspace(x[i], x[i+1], n_sub + 1)[1:-1]
                x_fine_list.extend(x_sub)
        x_fine_list.append(x[-1])
        x_fine = np.array(x_fine_list)

        y_fines = [self._interp_shape(x, y, x_fine) for y in ys]
        return x_fine, y_fines

    def _interp_shape(self, x_orig:np.array, y_orig:np.array, x_new:np.array, method='akima'):
        valid = np.isfinite(y_orig)
        if np.sum(valid) < 2:
            return np.full_like(x_new, np.nanmean(y_orig))
        if method == 'pchip':
            interp = PchipInterpolator(x_orig[valid], y_orig[valid], extrapolate=True)
        else:
            interp = Akima1DInterpolator(x_orig[valid], y_orig[valid], extrapolate=True)
        y_new = interp(x_new)
        y_min, y_max = np.min(y_orig[valid]), np.max(y_orig[valid])
        y_range = y_max - y_min
        overshoot_limit = 0.2 * y_range
        return np.clip(y_new, y_min - overshoot_limit, y_max + overshoot_limit)

def get_atomic_rates(state) -> dict:
    """
    Get atomic physics rates from Aurora.

    Parameters
    ----------
    state : object
        The simulation state containing species and their properties.

    Returns
    -------
    rates : dict
        {'ionization': nu_ion, 'recombination': nu_rec, 'charge_exchange': nu_cx}
        All in [1/s] including ne factor
    """

    # Conversions
    te_eV = state.te * 1e3
    ti_eV = state.ti * 1e3
    ne_cm3 = state.ne * 1e13  # [cm^-3]

    Rion = np.zeros_like(state.ti_full)
    Rrecom = np.zeros_like(state.ti_full)
    Rcx = np.zeros_like(state.ti_full)

    for sidx, sp in enumerate(state.species):

        atom_data = aurora.get_atom_data(sp['name'], ['acd', 'scd', 'ccd'])
    
        # Get reaction rates [1/s]
        _, R_ion, R_recom, R_cx = aurora.atomic.get_cs_balance_terms(
            atom_data, ne_cm3=ne_cm3, Te_eV=te_eV, Ti_eV=ti_eV, include_cx=True
        )[:4]

        Rion[:,sidx] = R_ion[:,0]
        Rrecom[:,sidx] = R_recom[:,0]
        Rcx[:,sidx] = R_cx[:,0]

    return {
        'ionization': Rion,  # [1/s]
        'recombination': Rrecom,  # [1/s]
        'charge_exchange': Rcx,  # [1/s]
    }

NEUTRAL_MODELS = {
    'kinetic': KineticNeutralModel,
    'diffusive': DiffusiveNeutralModel,
}