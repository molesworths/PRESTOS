"""
Parameterization models for plasma profiles (n_e, T_e, T_i, ...).

Provides a common interface via ParameterModel and concrete implementations:
- SplineParameterModel (Akima or PCHIP), parameterizing a/Ly at user-defined knots
- MTanhParameterModel (stub)
- GaussianRBFParameterModel (stub)

A factory function create_parameter_model(config) instantiates the requested model.

Conventions
-----------
- Coordinate x: by default, models operate on normalized radius x = r/a ("rho").
  In this coordinate, a/Ly = - d(ln y)/d x, which integrates naturally.
  If coord='r' (meters) is used, we convert with a: -d ln y/dr = (a/Ly)/a.
- Boundary condition: to reconstruct y from gradients, a boundary value y_sep at the
  outermost grid point is required. Pass explicitly to .y(..., y_sep=...), or provide
  bc_field in options to read from state.BC.<bc_field> when state is supplied.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union, List
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as linear, Akima1DInterpolator as akima, PchipInterpolator as pchip, CubicSpline
from tools import calc
from scipy.special import erf
from scipy.optimize import curve_fit, brentq
import scipy as sp  
from scipy.optimize import least_squares, minimize
from scipy.special import gamma as Gamma
from scipy.integrate import cumulative_trapezoid

# -------------------------
# Base parameter model
# -------------------------

class ParameterBase:
    """Abstract base class for parameterizing a scalar profile y(x).

    Expected common methods:
    - get_aLy(x_eval, params) -> np.ndarray: return a/Ly on x_eval
    - get_y(x_eval, params) -> np.ndarray: return y on x_eval
    - get_curvature(x_eval, params) -> np.ndarray: return d2y/dx2 on x_eval
    - _build_interpolator(x_data, y_data) -> store interpolator
    - _build_bc_dict(boundary_model, state) -> store dict of BC values from boundary model
    - update_all(boundary_model, state) -> recalculate attributes if dirty flag is set
    
    Attributes:
    - options: Dict[str, Any] of model options
    - interpolator: callable interpolator object
    - bcs: Dict[str, float] of boundary condition values 
    - params: Dict[str, np.ndarray] of model parameters
    - y: Dict[str, np.ndarray] of reconstructed profiles
    - aLy: Dict[str, np.ndarray] of a/Ly profiles
    - dirty: bool flag indicating if model needs re-initialization

    Notes
    -----
    - x_eval is 1D, representing either normalized radius (rho=r/a) or r [m].

    """

    def __init__(self, options: Dict[str, Any]):
        self.verbose = options.get('verbose', False)
        self.predicted_profiles = options.get('predicted_profiles', [])
        self.coord = options.get('coord', 'roa').lower()
        self.include_zero_grad_on_axis = options.get('include_zero_grad_on_axis', True)
        self.bc_dict: Dict[str, List[BCEntry]] = {}
        self.params: Dict[str, np.ndarray] = {}
        self.a = 1.0  # default minor radius, updated in parameterize()
        self.domain = [0.0, 1.0]  # default domain in x (rho), can be overridden
        self.sigma = options.get('sigma', 0.1)  # default relative uncertainty for covariance estimates

        if self.coord not in ('rho', 'roa'):
            raise ValueError("coord must be 'rho' or 'roa'")

        self.lcfs_aLti_in_params = options.get('lcfs_aLti_in_params', False)
        if self.lcfs_aLti_in_params:
            #raise NotImplementedError("lcfs_aLti_in_params=True not implemented for SplineParameterModel.")
            self.param_names.append('aLti_lcfs')

    # ------------------------------
    # Abstract-like interface
    # ------------------------------
    def add_bc(self, key: str, bc: BCEntry):
        self.bc_dict.setdefault(key, []).append({"val": float(bc['val']), "loc": float(bc['loc'])})

    def build_bcs(self, bc_dict: Dict[str, Any]) -> Dict[str, List[BCEntry]]:
        """
        Normalize and store BCs. Accepts:
          bc_dict = {"ne": (1e19, 1.0), "aLne": {"val": -2.0, "loc": 1.0},
                     "aLne": [( -1.5, 0.95), (-2.0, 1.0 )] }
        Stored form: self.bc_dict['aLne'] = [ {'val':..., 'loc':...}, {...} ]
        """
        self.bc_dict = {}

        for key, val in bc_dict.items():
            # allow list of entries
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple, dict)):
                for v in val:
                    self.add_bc(key, _normalize_single_bc(v))
            else:
                # single entry (tuple/list or dict)
                self.add_bc(key, _normalize_single_bc(val))

        # Optionally ensure aL<prof> has an axis BC at 0.0 (append only if no exact axis entry)
        if self.include_zero_grad_on_axis:
            for prof in self.predicted_profiles:
                key = f"aL{prof}"
                entries = self.bc_dict.get(key, [])
                has_axis = any(np.isclose(e['loc'], 0.0) for e in entries)
                if not has_axis:
                    # append axis zero-gradient BC (do not overwrite existing BCs)
                    self.add_bc(key, {"val": 0.0, "loc": 0.0})

        return self.bc_dict

    def get_nearest_bc(self, key: str, location: float) -> Union[BCEntry, None]:
        """Return the BC entry with location nearest to the requested location."""
        entries = self.bc_dict.get(key, [])
        if not entries:
            return None
        locs = np.array([e['loc'] for e in entries], dtype=float)
        idx = int(np.argmin(np.abs(locs - location)))
        return entries[idx]

    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract parameters from a PlasmaState given boundary conditions."""
        raise NotImplementedError

    def get_aLy(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute scale length a/Ly = -a * (dy/dx) / y for each profile."""
        raise NotImplementedError

    def get_y(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profile y(x) from parameter set."""
        raise NotImplementedError

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² for each profile."""
        raise NotImplementedError

    def update(self, params: Dict[str, np.ndarray], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        """Convenience method returning y, aLy, and curvature."""
        self.build_bcs(bc_dict)
        y = self.get_y(params, x_eval)
        aLy = self.get_aLy(params, x_eval)
        curvature = self.get_curvature(params, x_eval)
        return y, aLy, curvature


# -------------------------
# Spline parameter model
# -------------------------


class Spline(ParameterBase):
    """Spline-based parameterization of a/Ly with control points at user-defined knots.

    Design parameters: self.defined_on + i for i in range(len(knots))

    Parameters (options)
    --------------------
    knots : Sequence[float]
        Locations in x (rho=r/a by default) where parameters define a/Ly values.
    spline_type : str
        'akima' (default) or 'pchip'. Determines the interpolator.
    coord : str
        'rho' (default) to treat x as r/a, or 'r' to treat x as meters.
    include_zero_grad_on_axis : bool
        If True (default) and knots do not include x=0, a virtual control point with a/Ly=0 at x=0
        is prepended for smooth behavior at the magnetic axis.
    bc_field : Optional[str]
        Name of boundary condition value on state.BC (e.g., 'ne', 'te', 'ti') to use as y_sep if
        not explicitly provided to y()/curvature().
    """

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.spline_type = options.get('spline_type', 'akima').lower()
        self.knots = np.array(options.get('knots', []) or [])
        self.defined_on = options.get('defined_on', 'aLy')
        if self.spline_type not in ('akima', 'pchip', 'cubic', 'linear'):
            raise ValueError("spline_type must be 'akima', 'pchip', 'cubic', or 'linear'")
        self.param_names = [self.defined_on+str(i) for i in range(len(self.knots))]
        if self.lcfs_aLti_in_params:
            #raise NotImplementedError("lcfs_aLti_in_params=True not implemented for SplineParameterModel.")
            self.param_names.append('aLti_lcfs')
        self.n_params_per_profile = len(self.knots)
        self.splines: Dict[str, Any] = {}
        self.a = 1.0  # Default value, will be updated in parameterize()
        self.sigma = options.get('sigma', 0.1)  # default relative uncertainty for covariance estimates

    # ------------------------------
    # Internal utilities
    # ------------------------------
    def _make_spline(self, x: np.ndarray, y: np.ndarray, prof: str):
        """Return a spline object of chosen type.
        
        If include_zero_grad_on_axis=True and x doesn't start at 0, prepend axis BC.
        """
        x_spline = np.asarray(x)
        y_spline = np.asarray(y)

        if self.include_zero_grad_on_axis and not np.isclose(x_spline[0], 0.0) and self.defined_on == 'aLy':
            x_spline = np.insert(x_spline, 0, 0.0)
            y_spline = np.insert(y_spline, 0, 0.0)
        
        # Build spline
        if self.spline_type == "akima":
            spline = akima(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "pchip":
            spline = pchip(x_spline, y_spline, extrapolate=True)
        elif self.spline_type in ("cubic", "cspline"):
            spline = CubicSpline(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "linear":
            spline = linear(x_spline, y_spline, k=1)
        else:
            raise ValueError(f"Unknown spline_type: {self.spline_type}")
        
        self.splines[prof] = spline
        return spline
    
    def _get_spline(self, prof: str):
        """Retrieve a cached spline."""
        spline = self.splines.get(prof)
        if spline is None:
            raise KeyError(f"Spline for profile '{prof}' not initialized.")
        return spline

    def _integrate_aLy(self, prof: str, x_eval: np.ndarray, spl: Any, bc_value: float, bc_loc: float) -> np.ndarray:
        """
        Integrate spline of a/Ly to recover y(x) via
            dy/dx = -aLy * y   (for x = r/a dimensionless)

            => y(x) = y_bc * exp(-∫[bc_loc to x] aLy dx')

        For coord='roa', x = r/a is dimensionless and aLy ≡ a/Ly = -d(ln y)/dx,
        so no 1/a factor appears in the exponent.
        For bc_loc = 1.0 (edge), integrating inward (decreasing x) gives positive integral.
        For bc_loc = 0.0 (axis), integrating outward (increasing x) gives negative integral.
        """

        if not hasattr(spl, "antiderivative"):
            raise TypeError(f"Spline type {type(spl)} has no .antiderivative()")

        # y(x) = y_bc * exp(-∫[bc_loc→x] aLy dx')
        # aLy = a/Ly = -d(ln y)/d(r/a), so the integral is already dimensionless.
        F = spl.antiderivative()
        phase = -(F(x_eval) - F(bc_loc))

        return bc_value * np.exp(phase)

    # ------------------------------
    # Implement required methods
    # ------------------------------
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract spline coefficients (y values at knots) from a state.
        
        Incorporates boundary conditions by merging BC points into the spline
        construction data before extracting parameters at knots.
        """

        if self.bounds is None:
            self.bounds = {zip(self.param_names, [(0., 100.)]*len(self.param_names))}
        
        self.a = state.a  # store for conversions

        self.build_bcs(bc_dict)
        params = {}
        coord_vals = getattr(state, self.coord)
        
        for prof in self.predicted_profiles:
            prof_name = f"aL{prof}" if self.defined_on == "aLy" else prof
            y_prof = getattr(state, prof_name)
            if y_prof.ndim == 2:
                y_prof = y_prof[:, 0].flatten()
            else:
                y_prof = np.asarray(y_prof).flatten()
            
            # Merge boundary conditions into spline data
            x_data = np.asarray(coord_vals).flatten()
            y_data = np.asarray(y_prof).flatten()
            
            # Get BC entries for this profile
            bc_entries = self.bc_dict.get(prof_name, [])
            
            # Add/replace BC points in the data
            #for bc in bc_entries:
                #bc_loc = bc['loc']
                #bc_val = bc['val']
                
                # Find if this location already exists in data (within tolerance)
                #existing_idx = np.where(np.isclose(x_data, bc_loc, atol=1e-6))[0]
                
                # if len(existing_idx) > 0:
                #     # Replace existing point
                #     y_data[existing_idx[0]] = bc_val
                # else:
                #     # Insert new point in sorted order
                #     insert_idx = np.searchsorted(x_data, bc_loc)
                #     x_data = np.insert(x_data, insert_idx, bc_loc)
                #     y_data = np.insert(y_data, insert_idx, bc_val)
            
            # Build spline with BC-augmented data
            spline = self._make_spline(x_data, y_data, prof)
            params[prof] = dict(zip(self.param_names, spline(self.knots)))
        
        self.params = params
        std_dict = {prof: {name: abs(val)*self.sigma for name, val in params[prof].items()} for prof in params}
        self.param_std = std_dict

        return params, std_dict  # return nominal and std dev
    
    def get_y(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profiles y(x) on x_eval."""
        out = {}
        for prof, prof_params in params.items():
            
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                raise ValueError(f"No boundary condition found for profile '{bc_name}' at x=1.0")
            
            spline = self._make_spline(knots, vals, prof)

            if self.defined_on == "y":
                y = spline(x_eval)
            elif self.defined_on == "aLy":
                bc_y = self.get_nearest_bc(prof, 1.0)
                y = self._integrate_aLy(prof, x_eval, spline, bc_y['val'], bc_y['loc'])
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            
            out[prof] = np.clip(y, a_min=0, a_max=None)
        self.y = out
        return out

    def get_aLy(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute a/Ly(x) on x_eval.
        
        For coord='roa' (normalized): aLy = -a * (dy/dx) / y where x = r/a
        """
        out = {}
        for prof, prof_params in params.items():
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)
        
            # get aLy boundary condition to update vals if needed
            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                raise ValueError(f"No boundary condition found for profile '{bc_name}' at x=1.0")

            spline = self._make_spline(knots, vals, prof)

            if self.defined_on == "aLy":
                aLy = spline(x_eval)
            elif self.defined_on == "y":
                y = spline(x_eval)
                dy = spline.derivative(1)(x_eval)
                # aLy = a/Ly = -(dy/dx)/y  for x = r/a dimensionless
                y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                aLy = -dy / y_safe
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = np.clip(aLy, a_min=0, a_max=None)
        self.aLy = out
        return out

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² on x_eval.

        For coord='roa' (x = r/a dimensionless), aLy = -d(ln y)/dx, so:
            y' = -aLy * y
            y'' = -(aLy' * y + aLy * y') = -y * (aLy' - aLy²)
        """
        out = {}
        for prof, prof_params in params.items():
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                raise ValueError(f"No boundary condition found for profile '{bc_name}' at x=1.0")
            
            spline = self._make_spline(knots, vals, prof)
            
            if self.defined_on == "y":
                curv = spline.derivative(2)(x_eval)
            elif self.defined_on == "aLy":
                # Get boundary condition to integrate aLy → y
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' to compute curvature")
                
                # Get aLy, aLy', and y on x_eval
                aLy_spl = spline
                aLy = aLy_spl(x_eval)
                aLy_prime = aLy_spl.derivative(1)(x_eval)
                y = self._integrate_aLy(prof, x_eval, aLy_spl, bc_y['val'], bc_y['loc'])
                
                # Avoid division issues and NaN propagation
                y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)

                # y'' = -y * (aLy' - aLy²)  for x = r/a dimensionless
                curv = -y_safe * (aLy_prime - aLy**2)

                # Clean up any remaining NaN/inf values
                curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = curv
        self.curv = out
        return out

class Gaussian(ParameterBase):
    """
    Direct Gradient Parameterization for Profile Reconstruction

    This module implements a direct gradient-based parameterization of scalar profiles y(x)
    where the normalized gradient a/Ly is modeled as a Gaussian with offset, and the profile 
    is reconstructed by single integration subject to exact physical boundary conditions.

    Mathematical Formulation
    ========================

    Gradient Model
    --------------
    The normalized gradient is parameterized as:

        a/Ly(x; θ) = b + A · G(x; c, w)

    where:
        G(x; c, w) = exp[-(x-c)²/(2w²)]  is a Gaussian centered at c with width w
        
        θ = (A, b, c, w)  is the parameter vector with:
            A > 0           : Peak gradient amplitude above baseline
            b ≥ 0           : Baseline gradient offset (background level)
            c ∈ [0, 1]      : Peak location (allows edge-localized or interior peaks)
            w ∈ [0.01, 1]   : Characteristic width of gradient structure

    Physical Interpretation
    -----------------------
        b     : Represents the "background" gradient level in flat regions
        A     : Controls the height of the gradient peak above background
        c     : Controls where the steepest gradient occurs
        w     : Controls how localized vs spread out the gradient is

    Regime Coverage
    ---------------
        H-mode profiles (steep edge pedestal):
            - c ∈ [0.85, 0.98]: Peak in pedestal region
            - w ∈ [0.01, 0.1]: Narrow, localized gradient
            - A >> b: Strong peak above background
            - b ≈ 0: Minimal core gradient
            
        L-mode profiles (broad edge gradient):
            - c ∈ [0.95, 1.0]: Peak at or near boundary
            - w ∈ [0.1, 0.5]: Broader gradient structure
            - A ~ b: Moderate peak above background
            - b > 0: Significant background gradient
            
        Transitional profiles:
            - Naturally interpolate by varying (c, w, b)
            - No regime switching or special cases needed

    Profile Reconstruction
    ----------------------
    The profile y(x) is obtained by backward integration from the separatrix:

        y(x) = y₀ - ∫ₓ¹ (a/Ly)(ξ; θ) dξ
        
            = y₀ - ∫ₓ¹ [b + A·G(ξ; c, w)] dξ
            
            = y₀ - b(1-x) - A·∫ₓ¹ G(ξ; c, w) dξ

    Using the Gaussian integral:
        
        ∫ₓ¹ G(ξ; c, w) dξ = (w√(π/2)) · [erf((1-c)/(√2·w)) - erf((x-c)/(√2·w))]

    we obtain:

        y(x) = y₀ - b(1-x) - A·w·√(π/2)·[erf((1-c)/(√2·w)) - erf((x-c)/(√2·w))]


    aLy and y boundary conditions are enforced at their respective locations.

    """

    def __init__(self, options):
        super().__init__(options)
        self.param_names = ["log_A", "b", "c"]
        self.n_params_per_profile = len(self.param_names)
        self.defined_on = 'aLy'
        self.N_xfine = options.get('N_xfine', 101)
        
    # ======================================================
    # --- Gaussian basis and analytic integrals ---
    # ======================================================
    @staticmethod
    def _gaussian(x, mu, w):
        return np.exp(-((x - mu) ** 2) / (2.0 * w**2))
    
    @staticmethod
    def _I1(x, mu, w):
        u = (x - mu) / (np.sqrt(2) * w)
        return np.sqrt(np.pi / 2) * w * erf(u)
    
    @staticmethod
    def _dgdx(x, mu, w):
        G = Gaussian._gaussian(x, mu, w)
        return -((x - mu) / (w**2)) * G

    @staticmethod
    def _gen_sym_gaussian(x, mu, alpha, beta): # generalized symmetric Gaussian
        return (beta/(2*alpha*Gamma(1/beta))*np.exp(-((x - mu) / alpha)**(beta)))
    
    @staticmethod
    def _gen_sym_gaussian_I1(x, mu, alpha, beta):
        G = Gaussian._gen_sym_gaussian(x, mu, alpha, beta)
        return (alpha * Gamma(2 / beta) / beta) * sp.special.gammainc(2 / beta, ((x - mu) / alpha)**beta)
    
    @staticmethod
    def _gen_sym_gaussian_deriv(x, mu, alpha, beta):
        G = Gaussian._gen_sym_gaussian(x, mu, alpha, beta)
        return -((beta * (x - mu)**(beta - 1)) / (2 * alpha**(beta + 1) * Gamma(1 / beta))) * G

    @staticmethod
    def _gen_asym_gaussian(x, mu, alpha, kappa): # generalized asymmetric Gaussian
        return Gaussian._gaussian(x, mu, alpha)/(alpha - kappa*(x - mu))
    
    @staticmethod
    def _gen_sym_gaussian_I1(x, mu, alpha, beta):
        G = Gaussian._gen_sym_gaussian(x, mu, alpha, beta)
        return (alpha * Gamma(2 / beta) / beta) * sp.special.gammainc(2 / beta, ((x - mu) / alpha)**beta)
    
    @staticmethod
    def _gen_sym_gaussian_deriv(x, mu, alpha, beta):
        G = Gaussian._gen_sym_gaussian(x, mu, alpha, beta)
        return -((beta * (x - mu)**(beta - 1)) / (2 * alpha**(beta + 1) * Gamma(1 / beta))) * G

    # ======================================================
    # --- Parameter conversion ---
    # ======================================================
    def _to_physical_params(self, p: Dict[str, float]) -> Dict[str, float]:
        """Convert solver-space parameters to physical parameters.
        
        Parameters
        ----------
        p : dict
            Dictionary with keys ['log_A', 'b', 'c'].
        
        Returns
        -------
        phys : dict
            Dictionary with keys ['A', 'b', 'c'].
        """
        return {
            'A': np.exp(p['log_A']),
            'b': p['b'],
            'c': p['c'],
        }

    def _compute_w_from_bc(self, A, b, c, bc):

        # Rearrange Gaussian expression at bc1['loc'] for w
        # c may be > 1.0, so use absolute distance
        # add small offset to avoid log(0)
        delta = bc['val'] - b
        c_ = c + 1e-6 if bc['loc'] == c else c  # avoid log(0)
        w = (1/(2*(-np.log(delta/A)/(bc['loc']-c_)**2)))**0.5           
        
        return w
    
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract Gaussian parameters from a PlasmaState given boundary conditions.
        Use minimize with least_squares to fit (A,b,c,w) to aLy data while satisfying aLy BC."""

        self.build_bcs(bc_dict)
        params = {}
        coord_vals = getattr(state, self.coord)

        x_data = np.asarray(getattr(state, self.coord))  # e.g., roa
        # Trim x to self.domain if specified
        if hasattr(self, "domain") and self.domain is not None:
            mask = (x_data >= self.domain[0]) & (x_data <= self.domain[1])
            x_data = x_data[mask]
        else:
            mask = slice(None)  # Select all elements
        x_min, x_max = np.min(x_data), np.max(x_data)
        ix_x0 = mask.tolist().index(True) if isinstance(mask, np.ndarray) else 0
        xfine = np.linspace(x_min, x_max, self.N_xfine)
        
        for prof in self.predicted_profiles:
            prof_name = f"aL{prof}"
            aLy_data = getattr(state, prof_name)
            aLy_data = aLy_data[mask]
            aLy_fine = linear(x_data, aLy_data, extrapolate=True)(xfine)

            # Get BC at x=1
            bc1 = self.get_nearest_bc(prof_name, 1.0)


            
            # Fit Gaussian parameters (A, b, c)
            def model_func(x, log_A, b, c):
                A = np.exp(log_A)
                # solve for w such that b+A*G(bc1['loc']) = bc1['val']
                if bc1 is not None:
                    w = self._compute_w_from_bc(A, b, c, bc1)
                
                if self.include_zero_grad_at_axis:
                    x_spl = np.insert(x,0,0.0)
                    y_spl = np.insert(b + A * self._gaussian(x, c, w),0,0.0)
                    spl = akima(x_spl, y_spl, extrapolate=True)
                    return spl(x)
                else: 
                    return b + A * self._gaussian(x, c, w)
              

            # bounds [log_A, b, c]
            bounds = [(np.log(1.01*bc1['val'] if bc1 is not None else 1),np.log(1e4)), (0.0,100.0), (0.9,1.2)]

            # Initial guess
            A0 = bc1['val']*1.01 if bc1 is not None else 1.0
            b0 = 0.
            c0 = bounds[2][1] # avoid exact edge
            #w0 = np.clip(0.25 * (x_max - x_min), bounds[3][0], bounds[3][1])
            p0 = [np.log(A0), b0, c0]
            
            # fit with curve_fit
            popt, pcov = curve_fit(
                model_func,
                xfine,
                aLy_fine,
                p0=p0,
                bounds=tuple(zip(*bounds)),
                maxfev=10000,
                xtol=1e-3,
                ftol=1e-3,
            )
            # Fit via minimize with constraints
            
            # result = minimize(
            #     lambda p: np.sum((aLy_fine - model_func(xfine, *p))**2),
            #     p0,
            #     bounds=bounds,
            #     method='L-BFGS-B'
            # )
            # if result.success is False:
            #     raise RuntimeError(f"Initial Gaussian parameterization failed for profile '{prof}': {result.message}")
            
            # log_A_fit, b_fit, c_fit = result.x

            
            params[prof] = {
                'log_A': popt[0],
                'b': popt[1],
                'c': popt[2],
                        }

            # # std from covariance estimation from inverse Hessian
            # hess = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else result.hess_inv
            # # cov matrix ~ 0.5 x H^-1
            # cov = 0.5 * hess
            # param_std = np.sqrt(np.diag(cov))

            params_std = {
                'log_A': np.sqrt(np.log(pcov[0, 0])),
                'b': np.sqrt(pcov[1, 1]),
                'c': np.sqrt(pcov[2, 2]),
            }
            params_std[prof] = params_std
        self.params = params
        self.params_std = params_std
        return params, params_std
        
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute a/Ly(x) on x_eval."""
        out = {}
        for prof, prof_params in params.items():
            phys = self._to_physical_params(prof_params)
            A = phys['A']
            b = phys['b']
            c = phys['c']

            w = self._compute_w_from_bc(A, b, c, self.get_nearest_bc(f"aL{prof}", 1.0))

            if self.include_zero_grad_at_axis:
                if 0 not in x_eval:
                    x_spl = np.insert(x_eval,0,0.0)
                    y_spl = b + A * self._gaussian(x_spl, c, w)
                else:
                    x_spl = np.insert(x_eval[x_eval>=self.domain[0]],0,0.0)
                    y_spl = b+A * self._gaussian(x_spl, c, w)
                    y_spl[x_spl==0] = 0.0
                spl = akima(x_spl, y_spl, extrapolate=True)
                aLy = spl(x_eval)
            else:
                aLy = b + A * self._gaussian(x_eval, c, w)
            out[prof] = np.clip(aLy, a_min=0, a_max=None)
        self.aLy = out
        return out
    
    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profiles y(x) on x_eval."""
        out = {}
        for prof, prof_params in params.items():
            phys = self._to_physical_params(prof_params)
            A = phys['A']
            b = phys['b']
            c = phys['c']
            w = self._compute_w_from_bc(A, b, c, self.get_nearest_bc(f"aL{prof}", 1.0))
            # Get BC at x=1
            bc = self.get_nearest_bc(f"{prof}", 1.0)
            if bc is None:
                raise ValueError(f"No boundary condition found for profile 'aL{prof}' at x=1.0")
            y_0 = bc['val']
            
            # Compute integral
            I1_1 = self._I1(1.0, c, w)
            I1_x = self._I1(x_eval, c, w)
            
            y = y_0 + b * (1.0 - x_eval) + A * (I1_1 - I1_x)
            out[prof] = np.clip(y, a_min=0, a_max=None)
        self.y = out
        return out

    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² on x_eval."""
        out = {}
        for prof, prof_params in params.items():
            phys = self._to_physical_params(prof_params)
            A = phys['A']
            b = phys['b']
            c = phys['c']
            w = self._compute_w_from_bc(A, b, c, self.get_nearest_bc(f"aL{prof}", 1.0))
            
            # a/Ly = -a * (dy/dx) / y
            # => dy/dx = -(aLy/a) * y
            # => d²y/dx² = -(1/a) * (aLy' * y + aLy * dy/dx)
            #              = -(y/a) * (aLy' - (aLy²/a))
            
            # Get BC at x=1
            bc = self.get_nearest_bc(f"{prof}", 1.0)
            if bc is None:
                raise ValueError(f"No boundary condition found for profile '{prof}' at x=1.0")
            y_0 = bc['val']
            
            # Compute y(x)
            I1_1 = self._I1(1.0, c, w)
            I1_x = self._I1(x_eval, c, w)
            y = y_0 + b * (1.0 - x_eval) + A * (I1_1 - I1_x)
            y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
            
            # Compute aLy and aLy'
            aLy = b + A * self._gaussian(x_eval, c, w)
            aLy_prime = A * (-(x_eval - c) / (w**2)) * self._gaussian(x_eval, c, w)
            
            curv = -(y_safe / self.a) * (aLy_prime - (aLy**2) / self.a)
            curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
            out[prof] = curv
        self.curv = out
        return out

class GaussianDipole(ParameterBase):
    """
    Curvature-Based Profile Parameterization with Morphological Transition

    This module implements a curvature-driven parameterization of scalar profiles y(x)
    where the second derivative is modeled directly and the profile is reconstructed
    by double integration subject to exact physical boundary conditions.

    Mathematical Formulation
    ========================

    Curvature Model
    ---------------
    The curvature k(x) = d²y/dx² is parameterized as:

        k(x; θ) = A · [-S·G(x; x_c-δ/2, w) + S·G(x; x_c+δ/2, w) - (1-S)·G(x; x_c, w)] + b

    where:
        G(x; μ, w) = exp[-(x-μ)²/(2w²)]  is a Gaussian centered at μ with width w
        
        θ = (A, S, x_c, w)  is the parameter vector with:
            A > 0           : Curvature amplitude (scales integrated curvature strength)
            S ∈ [0, 1]      : Morphology/symmetry parameter controlling dipole↔single transition
            x_c ∈ [0.9, 1]    : Center position of the structure
            w ∈ [0.01, 1]   : Characteristic width of Gaussian features
            δ = 2.5w        : Fixed lobe separation (ensures near-sinusoidal dipole)
            b               : Offset derived from input finite-differenced curvature at x0=self.domain[0]

    Morphological Regimes
    ---------------------
        S = 1  : Antisymmetric dipole (two-lobe structure)
                k(x) = A·[G(x; x_c-δ/2, w) - G(x; x_c+δ/2, w)]
                Physically represents separated curvature lobes
                
        S = 0  : Single Gaussian (single-lobe structure)
                k(x) = A·G(x; x_c, w)
                Physically represents localized curvature peak
                
        0 < S < 1 : Smooth interpolation between regimes
                    Allows gradual morphological transitions

    The fixed relationship δ = 2.5w ensures that in dipole mode (S=1), the structure
    exhibits approximately equal peak magnitudes |min(k)| ≈ |max(k)| ≈ A and
    near-uniform derivatives at the zero-crossing, creating a quasi-sinusoidal profile.

    Profile Reconstruction
    ----------------------
    The profile y(x) is obtained by double integration:

        g(x) = dy/dx = ∫₀ˣ k(ξ; θ) dξ + C₁
        
        y(x) = ∫₀ˣ g(η) dη + C₀

    Integration constants (C₀, C₁) are determined by boundary conditions at x = 1:
        
        y(1) = y₀                    (value constraint)
        dy/dx|ₓ₌₁ = -(a/Ly)₀         (gradient constraint)

    This yields:
        C₁ = -(a/Ly)₀ - ∫₀¹ k(ξ; θ) dξ
        C₀ = y₀ - ∫₀¹ ∫₀ᶯ k(ξ; θ) dξ dη - C₁

    The formulation cleanly separates:
        - Shape control via (A, s, x_c, w)
        - Global anchoring via exact boundary conditions (y₀, (a/Ly)₀)

    Implementation Notes
    ====================

    Gaussian Evaluation
    -------------------
    For numerical stability, implement Gaussians as:
        
        def gaussian(x, mu, w):
            return np.exp(-0.5 * ((x - mu) / w)**2)

    Curvature Function
    ------------------
        def curvature(x, A, S, x_c, w):
            delta = 2.5 * w
            G_minus = gaussian(x, x_c - delta/2, w)
            G_plus = gaussian(x, x_c + delta/2, w)
            G_center = gaussian(x, x_c, w)
            return A * (S * G_minus - S * G_plus + (1 - S) * G_center)

    Integration
    -----------
    Use numerical quadrature (scipy.integrate.quad or cumulative trapezoid) for:
        - Computing g(x) from k(x)
        - Computing y(x) from g(x)
        - Evaluating boundary condition integrals

    For gradient-based optimization, consider implementing analytic Jacobians:
        ∂k/∂A = k(x; θ) / A
        ∂k/∂S = A · [G_minus - G_plus - G_center]
        ∂k/∂x_c involves derivatives of Gaussians: G'(x; μ, w) = -(x-μ)/w² · G(x; μ, w)
        ∂k/∂w involves both Gaussian derivatives and δ = 2.5w coupling

    Parameter Bounds
    ----------------
    Enforce constraints during optimization:
        A_bounds = (1e-6, np.inf)     # Positive amplitude
        S_bounds = (0.0, 1.0)          # Morphology transition
        x_c_bounds = (0.9, 1.0)          # Near-boundary positioning
        w_bounds = (0.01, 1.0)         # Finite width resolution

    Physical Interpretation
    =======================
    This parameterization is designed for edge/pedestal transport modeling where:
        - The curvature k(x) represents second-derivative features in profiles
        - S controls whether the structure is a localized peak (S≈0) or a 
        transition layer with separated extrema (S≈1)
        - Boundary conditions ensure physical consistency with separatrix values
        - The model remains well-conditioned and identifiable across all regimes

    Key advantages:
        - No parameter degeneracy at morphological transitions (A has consistent meaning)
        - Smooth, differentiable transitions between single and double-lobe structures
        - Exact boundary condition enforcement separates shape from global constraints
        - Analytic derivatives available for efficient optimization
    """

    def __init__(self, options: Dict[str, Any]):

        super().__init__(options)
        self.param_names = ['log_A', 'x_c', 'w', 'S']
        self.n_params_per_profile = len(self.param_names)
        self.delta_factor = options.get('delta_factor', 2.5)  # Fixed relationship delta = 2.5 * w
        self.N_xfine = options.get('N_xfine', 100) # number of points for fine integration grid
        self.b = {}  # curvature offset from FD at x0
        self.defined_on = "curvature"

        if self.lcfs_aLti_in_params:
            #raise NotImplementedError("lcfs_aLti_in_params=True not implemented for SplineParameterModel.")
            self.param_names.append('aLti_lcfs')

    # ======================================================
    # --- Gaussian basis and analytic integrals ---
    # ======================================================
    @staticmethod
    def _gaussian(x, mu, w):
        return np.exp(-((x - mu) ** 2) / (2.0 * w**2))

    @staticmethod
    def _I1(x, mu, w):
        u = (x - mu) / (np.sqrt(2) * w)
        return np.sqrt(np.pi / 2) * w * erf(u)

    @staticmethod
    def _I2(x, mu, w):
        u = (x - mu) / (np.sqrt(2) * w)
        return (w**2) * (np.sqrt(np.pi) * u * erf(u) + np.exp(-u**2))

    # ======================================================
    # --- Parameterization ---
    # ======================================================

    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Fit the curvature of each predicted profile (y_data) in PlasmaState
        to the composite Gaussian curvature model.

        Returns
        -------
        params : dict of {profile: {"log_A","x_c","log_w","S"}}
        std : dict
            Parameter uncertainties from curve_fit covariance.
        """
        # Define parameter bounds: log_A, log_w in log-space; x_c ∈ [0.9,1], S ∈ [0,1]
        param_bounds = dict(zip(self.param_names, [(0., 10.), (0.9, 1.0), (0.01, 0.1), (0.0, 1.0)]))
        bounds_lower = [param_bounds[name][0] for name in self.param_names]
        bounds_upper = [param_bounds[name][1] for name in self.param_names]

        self.build_bcs(bc_dict)
        params = {}
        stds = {}

        x_data = np.asarray(getattr(state, self.coord))  # e.g., roa
        # Trim x to self.domain if specified
        if hasattr(self, "domain") and self.domain is not None:
            mask = (x_data >= self.domain[0]) & (x_data <= self.domain[1])
            x_data = x_data[mask]
        else:
            mask = slice(None)  # Select all elements
        x_min, x_max = np.min(x_data), np.max(x_data)
        ix_x0 = mask.tolist().index(True) if isinstance(mask, np.ndarray) else 0

        for prof in self.predicted_profiles:
            try:
                # ------------------------------------------------------------------
                # Compute curvature numerically from Akima spline
                # ------------------------------------------------------------------
                y_data = np.asarray(getattr(state, prof))
                aLy_data = np.asarray(getattr(state, f'aL{prof}'))
                dydx_data = - (aLy_data / state.a) * y_data
                self.b[prof] = (dydx_data[ix_x0+1]-dydx_data[ix_x0-1])/(state.roa[ix_x0+1]-state.roa[ix_x0-1]) # 2nd order FD for k(x) at x0
                y_data = y_data[mask]
                spl = akima(x_data, y_data)
                y_curv = spl.derivative(2)(x_data)

                # ------------------------------------------------------------------
                # Define model for fitting
                # ------------------------------------------------------------------
                def model(x, log_A, x_c, w, S):
                    """Composite Gaussian curvature model with morphology parameter S."""
                    A = np.exp(log_A)  # Convert from log space
                    delta = self.delta_factor * w  # Fixed relationship
                    G_minus = np.exp(-((x - (x_c - delta/2)) ** 2) / (2 * w**2))
                    G_plus = np.exp(-((x - (x_c + delta/2)) ** 2) / (2 * w**2))
                    G_center = np.exp(-((x - x_c) ** 2) / (2 * w**2))
                    return A * (-S * G_minus + S * G_plus - (1 - S) * G_center) + self.b[prof]

                # ------------------------------------------------------------------
                # Initial guesses (roughly from profile shape)
                # ------------------------------------------------------------------
                A0 = np.max(np.abs(y_curv)) or 1.0
                log_A0 = np.clip(np.log(A0), param_bounds['log_A'][0], param_bounds['log_A'][1])
                x_c0 = np.clip(x_data[np.argmin(np.abs(y_curv - np.sign(np.mean(y_curv))*A0))], \
                               param_bounds['x_c'][0], param_bounds['x_c'][1])
                w0 = np.clip(0.25 * (x_max - x_min), param_bounds['w'][0], param_bounds['w'][1])
                S0 = 0.0  # Start without pedestal/dipole
                p0 = [log_A0, x_c0, w0, S0]

                # ------------------------------------------------------------------
                # Fit to curvature data
                # ------------------------------------------------------------------
                popt, pcov = curve_fit(
                    model, x_data, y_curv, p0=p0, 
                    bounds=(bounds_lower, bounds_upper), 
                    maxfev=10000,
                    ftol=1e-9,
                )

                # Extract fitted parameters
                params[prof] = dict(zip(self.param_names, popt))
                
                # Estimate standard deviations from covariance matrix
                perr = np.sqrt(np.diag(pcov))
                stds[prof] = dict(zip(self.param_names, perr))

                # Estimate R-squared to assess fit quality
                residuals = y_curv - model(x_data, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_curv - np.mean(y_curv))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                if r_squared < 0.8:
                    Warning(f"[GaussianDipole] Warning: Poor fit for {prof} (R²={r_squared:.3f}).")

            except Exception as err:
                # Fall back to nominal initialization if fit fails
                print(f"[GaussianDipole] Warning: Fit for {prof} failed ({err}); using initial guess.")
                params[prof] = dict(zip(self.param_names, p0))
                stds[prof] = dict(zip(self.param_names, [0.1 * abs(v) for v in p0]))

        self.params = params
        self.param_std = stds
        return params, stds


    # ======================================================
    # --- Parameter conversion ---
    # ======================================================
    def _to_physical_params(self, p: Dict[str, float]) -> Dict[str, float]:
        """Convert solver-space parameters to physical parameters.
        
        Parameters
        ----------
        p : dict
            Dictionary with keys ['log_A', 'x_c', 'w', 'S'].
        
        Returns
        -------
        phys : dict
            Dictionary with keys ['A', 'x_c', 'w', 'S'].
        """
        return {
            'A': np.exp(p['log_A']),
            'x_c': p['x_c'],
            'w': p['w'],
            'S': p['S'],
        }

    # ======================================================
    # --- Core curvature model ---
    # ======================================================
    def _curvature(self, x, A, x_c, w, S, b):
        """Compute composite Gaussian curvature with morphology parameter S.
        
        k(x) = A · [-S·G(x; x_c-δ/2, w) + S·G(x; x_c+δ/2, w) - (1-S)·G(x; x_c, w)] + b
        where δ = 2.5w
        """
        delta = self.delta_factor * w  # Fixed lobe separation
        G_minus = self._gaussian(x, x_c - delta/2, w)
        G_plus = self._gaussian(x, x_c + delta/2, w)
        G_center = self._gaussian(x, x_c, w)
        return A * (-S * G_minus + S * G_plus - (1 - S) * G_center) + b

    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for prof, p in params.items():
            phys = self._to_physical_params(p)
            phys_w_b = phys.copy()
            phys_w_b['b'] = self.b.get(prof, 0.0)
            curv_fine = self._curvature(self.x_fine, **phys_w_b)
            
            # Add axis BC if needed
            x_spline = self.x_fine
            y_spline = curv_fine
            if self.include_zero_grad_on_axis and not np.isclose(self.x_fine[0], 0.0):
                x_spline = np.insert(self.x_fine, 0, 0.0)
                y_spline = np.insert(curv_fine, 0, 0.0)
            
            # down sample to x_eval
            spl = akima(x_spline, y_spline)
            out[prof] = spl(x_eval)
        self.curv = out
        return out

    # ======================================================
    # --- Integrations and BC enforcement ---
    # ======================================================
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute a/Ly(x) from integrated curvature, enforcing boundary conditions."""
        out = {}
        for prof, p in params.items():
            # Get BC for aLy gradient
            bc = self.get_nearest_bc(f"aL{prof}", x_eval[-1])
            if bc is None:
                bc = {'val': 0.0, 'loc': x_eval[-1]}
            
            phys = self._to_physical_params(p)
            A = phys["A"]
            S = phys["S"]
            w = phys["w"]
            x_c = phys["x_c"]
            delta = self.delta_factor * w  # Fixed relationship

            # Integrate curvature to get gradient: aLy = ∫ k dx
            # k(x) = A·[-S·G(x_c-δ/2) + S·G(x_c+δ/2) - (1-S)·G(x_c)] + b
            # ∫k dx = A·[...] + b·x + C1
            I1_minus = self._I1(self.x_fine, x_c - delta/2, w)
            I1_plus = self._I1(self.x_fine, x_c + delta/2, w)
            I1_center = self._I1(self.x_fine, x_c, w)
            b_val = self.b.get(prof, 0.0)
            aLy_raw = A * (-S * I1_minus + S * I1_plus - (1 - S) * I1_center) + b_val * self.x_fine
            
            # Enforce BC at boundary location
            bc_idx = np.argmin(np.abs(self.x_fine - bc['loc']))
            aLy = aLy_raw - aLy_raw[bc_idx] + bc['val']
            
            # Add axis BC if needed
            x_spline = self.x_fine
            y_spline = aLy
            if self.include_zero_grad_on_axis and not np.isclose(self.x_fine[0], 0.0):
                x_spline = np.insert(self.x_fine, 0, 0.0)
                y_spline = np.insert(aLy, 0, 0.0)
            
            spl = akima(x_spline, y_spline)
            out[prof] = np.clip(spl(x_eval), a_min=0, a_max=None)  # Ensure non-negative aLy
        self.aLy = out
        return out

    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute y(x) using analytic double-integral and boundary condition enforcement."""
        out = {}
        for prof, p in params.items():
            # Get boundary conditions
            bc_y = self.get_nearest_bc(prof, self.x_fine[-1])
            if bc_y is None:
                bc_y = {'val': 1.0, 'loc': self.x_fine[-1]}
            
            bc_aLy = self.get_nearest_bc(f"aL{prof}", self.x_fine[-1])
            if bc_aLy is None:
                bc_aLy = {'val': 0.0, 'loc': self.x_fine[-1]}

            x_b = bc_y['loc']
            y_b = bc_y['val']
            aLy_b = bc_aLy['val']

            phys = self._to_physical_params(p)
            A = phys["A"]
            S = phys["S"]
            w = phys["w"]
            x_c = phys["x_c"]
            b_val = self.b.get(prof, 0.0)
            delta = self.delta_factor * w  # Fixed relationship
            mu_minus = x_c - delta/2
            mu_plus = x_c + delta/2
            mu_center = x_c
            
            # Analytic basis functions: first and second antiderivatives of Gaussians
            # For k(x) = A·[-S·G(mu_minus) + S·G(mu_plus) - (1-S)·G(mu_center)] + b
            Phi_minus = self._I1(self.x_fine, mu_minus, w)    # ∫ G(x; mu_minus, w) dx
            Phi_plus = self._I1(self.x_fine, mu_plus, w)      # ∫ G(x; mu_plus, w) dx
            Phi_center = self._I1(self.x_fine, mu_center, w)  # ∫ G(x; mu_center, w) dx
            Psi_minus = self._I2(self.x_fine, mu_minus, w)    # ∫∫ G(x; mu_minus, w) dx² 
            Psi_plus = self._I2(self.x_fine, mu_plus, w)      # ∫∫ G(x; mu_plus, w) dx²
            Psi_center = self._I2(self.x_fine, mu_center, w)  # ∫∫ G(x; mu_center, w) dx²

            # Find BC index
            bc_idx = np.argmin(np.abs(self.x_fine - x_b))
            Phi_minus_b = Phi_minus[bc_idx]
            Phi_plus_b = Phi_plus[bc_idx]
            Phi_center_b = Phi_center[bc_idx]
            Psi_minus_b = Psi_minus[bc_idx]
            Psi_plus_b = Psi_plus[bc_idx]
            Psi_center_b = Psi_center[bc_idx]

            # Compute gradient (dy/dx) = ∫k dx = A·[-S·Φ_minus + S·Φ_plus - (1-S)·Φ_center] + b·x + C1
            # Apply BC: y'(x_b) = aLy_b  (in normalized coords, assumes a=1)
            C1 = aLy_b - A * (-S * Phi_minus_b + S * Phi_plus_b - (1 - S) * Phi_center_b) - b_val * x_b
            
            # Compute y = ∫∫k dx² = A·[-S·Ψ_minus + S·Ψ_plus - (1-S)·Ψ_center] + b·x²/2 + C1·x + C0
            # Apply BC: y(x_b) = y_b
            C0 = y_b - A * (-S * Psi_minus_b + S * Psi_plus_b - (1 - S) * Psi_center_b) - b_val * x_b**2 / 2 - C1 * x_b

            # Construct final profile
            y = A * (-S * Psi_minus + S * Psi_plus - (1 - S) * Psi_center) + b_val * self.x_fine**2 / 2 + C1 * self.x_fine + C0
            spl = akima(self.x_fine, y)
            out[prof] = np.clip(spl(x_eval), a_min=0, a_max=None)  # Ensure non-negative density/temperature
        self.y = out
        return out


    def update(self, params: Dict[str, Dict[str, float]], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        self.build_bcs(bc_dict)
        self.x_fine = np.linspace(x_eval[0], x_eval[-1], self.N_xfine)
        curv = self.get_curvature(params, x_eval)
        aLy = self.get_aLy(params, x_eval)
        y = self.get_y(params, x_eval)
        self.dirty = False
        return y, aLy, curv


# -------------------------
# Polynomial parameter model
# -------------------------


class Polynomial(ParameterBase):
    """Polynomial-based parameterization using weighted orthogonal polynomial expansions.
    
    Represents a/Ly as a sum of weighted polynomials:
        a/Ly(x) = Σ(a_i * P_i(x))
    where a_i are coefficients and P_i are orthogonal polynomials.
    
    Supports Legendre, Chebyshev (1st and 2nd kind), and Hermite polynomials
    via numpy.polynomial.
    
    Parameters
    ----------
    polynomial_class : str
        Type of polynomial basis: 'legendre', 'chebyshev', 'hermite'
    order : int
        Number of polynomial terms (and parameters per profile)
    domain : List[float]
        Domain [x_min, x_max] for polynomial evaluation (default: [0, 1])
    """
    
    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        
        self.polynomial_class = options.get('polynomial_class', 'legendre').lower()
        self.degree = int(options.get('degree', 3))  # Degree of polynomial
        self.defined_on = options.get('defined_on', 'aLy')
        
        # Generate parameter names: a_0, a_1, ..., a_{degree}
        self.param_names = [f"a_{i}" for i in range(self.degree + 1)]
        self.n_params_per_profile = self.degree + 1
        
        # Domain for polynomial evaluation (can be overridden in options)
        if 'domain' in options:
            self.domain = options['domain']
        
        # Map polynomial class name to numpy.polynomial module
        self.poly_module = self._get_polynomial_module()
        
        # Store polynomial representations for each profile
        self.poly_aLy: Dict[str, Any] = {}  # aLy polynomial objects
        self.poly_y: Dict[str, Any] = {}    # y polynomial objects (integrated)
        
    def _get_polynomial_module(self):
        """Get the appropriate numpy.polynomial submodule."""
        poly_map = {
            'legendre': np.polynomial.legendre,
            'chebyshev': np.polynomial.chebyshev,
            'hermite': np.polynomial.hermite,
        }
        
        if self.polynomial_class not in poly_map:
            raise ValueError(
                f"Unknown polynomial_class '{self.polynomial_class}'. "
                f"Choose from: {list(poly_map.keys())}"
            )
        
        return poly_map[self.polynomial_class]
    
    def _rescale_to_canonical(self, x: np.ndarray) -> np.ndarray:
        """Rescale x from user domain [x_min, x_max] to canonical [-1, 1].
        
        Linear transformation: x_canonical = 2*(x - x_min)/(x_max - x_min) - 1
        Maps domain[0] -> -1 and domain[1] -> 1.
        """
        x = np.asarray(x, dtype=float)
        x_min, x_max = self.domain[0], self.domain[1]
        
        # Avoid division by zero
        denom = x_max - x_min
        if np.isclose(denom, 0.0):
            return np.zeros_like(x)
        
        return 2.0 * (x - x_min) / denom - 1.0
    
    def _rescale_from_canonical(self, x_canonical: np.ndarray) -> np.ndarray:
        """Rescale x from canonical [-1, 1] to user domain [x_min, x_max].
        
        Inverse transformation: x = x_min + (x_canonical + 1) * (x_max - x_min) / 2
        Maps -1 -> domain[0] and 1 -> domain[1].
        """
        x_canonical = np.asarray(x_canonical, dtype=float)
        x_min, x_max = self.domain[0], self.domain[1]
        
        return x_min + (x_canonical + 1.0) * (x_max - x_min) / 2.0
    

    def _create_polynomial(self, coeffs: np.ndarray):
        """Create a polynomial object from coefficients.
        
        Polynomials are created on the canonical [-1, 1] domain for numerical stability.
        User domain rescaling is handled in get_aLy, get_y, and get_curvature.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Polynomial coefficients [a_0, a_1, ..., a_n]
        domain : List[float], optional
            Domain [x_min, x_max] - ignored, uses canonical [-1, 1]
            
        Returns
        -------
        poly : Polynomial object from numpy.polynomial (on domain [-1, 1])
        """
        
        # Create the appropriate polynomial class
        # Always use canonical [-1, 1] domain for numerical stability
        canonical_domain = [-1.0, 1.0]
        
        if self.polynomial_class == 'legendre':
            return np.polynomial.Legendre(coeffs, domain=canonical_domain)
        elif self.polynomial_class == 'chebyshev':
            return np.polynomial.Chebyshev(coeffs, domain=canonical_domain)
        elif self.polynomial_class == 'hermite':
            return np.polynomial.Hermite(coeffs, domain=canonical_domain)
    
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Fit polynomial coefficients to state data.
        
        Uses least-squares fitting to match a/Ly data from the state.
        """
        self.build_bcs(bc_dict)
        self.a = getattr(state, 'a', 1.0)
        
        params = {}
        params_std = {}
        
        x_data = np.asarray(getattr(state, self.coord))
        
        # Trim to domain if specified
        if hasattr(self, 'domain') and self.domain is not None:
            mask = (x_data >= self.domain[0]) & (x_data <= self.domain[1])
        else:
            mask = np.ones(len(x_data), dtype=bool)
        
        x_fit = x_data[mask]
        
        # Rescale fit data to canonical [-1, 1] domain
        x_fit_canonical = self._rescale_to_canonical(x_fit)
        
        for prof in self.predicted_profiles:
            # Get a/Ly data from state
            aLy_key = f"aL{prof}"
            
            if hasattr(state, aLy_key):
                y_data = np.asarray(getattr(state, prof))[mask]
                aLy_data = np.asarray(getattr(state, aLy_key))[mask]
            else:
                # Compute a/Ly from profile data if not available
                y_data = np.asarray(getattr(state, prof))[mask]
                dy_dx = np.gradient(y_data, x_fit)
                aLy_data = -self.a * dy_dx / (y_data + 1e-12)

            if self.defined_on == 'aLy':
                fit_data = aLy_data
            else:
                fit_data = y_data

            
            # Fit polynomial coefficients using least squares
            # numpy polynomial fit returns coefficients in increasing degree order
            try:
                if self.polynomial_class == 'legendre':
                    coeffs = np.polynomial.legendre.legfit(x_fit_canonical, fit_data, deg=self.degree, 
                                                          full=False)
                elif self.polynomial_class == 'chebyshev':
                    coeffs = np.polynomial.chebyshev.chebfit(x_fit_canonical, fit_data, deg=self.degree, 
                                                            full=False)
                elif self.polynomial_class == 'hermite':
                    coeffs = np.polynomial.hermite.hermfit(x_fit_canonical, fit_data, deg=self.degree, 
                                                          full=False)
                
                # Store coefficients as parameters
                params[prof] = {f"a_{i}": float(coeffs[i]) for i in range(len(coeffs))}
                
                # Estimate uncertainties (simplified - use residual-based estimate)
                poly_eval = self._create_polynomial(coeffs)(x_fit_canonical)
                residual_std = np.std(fit_data - poly_eval)
                params_std[prof] = {f"a_{i}": residual_std * self.sigma for i in range(len(coeffs))}
                
            except Exception as e:
                print(f"[Polynomial.parameterize] Warning: fitting failed for {prof}: {e}")
                # Fallback to zero coefficients
                params[prof] = {f"a_{i}": 0.0 for i in range(self.degree + 1)}
                params_std[prof] = {f"a_{i}": self.sigma for i in range(self.degree + 1)}
        
        self.params = params
        self.params_std = params_std
        return params, params_std
    
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute a/Ly(x) on x_eval.

        When defined_on='aLy': evaluate the polynomial directly.
        When defined_on='y':   aLy = -a * (dy/dx) / y, using poly.deriv() with chain rule.
        """
        out = {}
        x_eval = np.asarray(x_eval)
        x_canonical = self._rescale_to_canonical(x_eval)
        dx_canonical_dx = 2.0 / (self.domain[1] - self.domain[0])

        for prof, prof_params in params.items():
            coeffs = np.array([prof_params.get(f"a_{i}", 0.0) for i in range(self.degree + 1)])
            poly = self._create_polynomial(coeffs)

            if self.defined_on == 'aLy':
                self.poly_aLy[prof] = poly
                out[prof] = poly(x_canonical)

            elif self.defined_on == 'y':
                self.poly_y[prof] = poly
                # Enforce y BC via constant shift (same as get_y)
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is not None:
                    bc_loc_canonical = self._rescale_to_canonical(np.array([bc_y['loc']]))[0]
                    y_offset = bc_y['val'] - float(poly(bc_loc_canonical))
                else:
                    y_offset = 0.0
                y_eval = poly(x_canonical) + y_offset
                # dy/dx via chain rule; constant offset does not affect derivative
                dy_dx = poly.deriv()(x_canonical) * dx_canonical_dx
                y_safe = np.where(np.abs(y_eval) < 1e-12, 1e-12, y_eval)
                out[prof] = np.clip(-self.a * dy_dx / y_safe, a_min=0, a_max=None)

            else:
                raise ValueError(f"Invalid defined_on: '{self.defined_on}'")

        self.aLy = out
        return out
    
    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute y(x) on x_eval.

        When defined_on='y':   evaluate polynomial directly; enforce y BC via constant shift.
        When defined_on='aLy': y(x) = y_bc * exp(-(1/a) ∫[bc→x] aLy dx) using poly.integ().
        """
        out = {}
        x_eval = np.asarray(x_eval)
        x_canonical = self._rescale_to_canonical(x_eval)

        if self.defined_on == 'y':
            for prof, prof_params in params.items():
                coeffs = np.array([prof_params.get(f"a_{i}", 0.0) for i in range(self.degree + 1)])
                poly = self._create_polynomial(coeffs)
                self.poly_y[prof] = poly
                y_eval = poly(x_canonical)
                # Enforce y BC via constant shift (preserves polynomial shape)
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is not None:
                    bc_loc_canonical = self._rescale_to_canonical(np.array([bc_y['loc']]))[0]
                    y_eval = y_eval + (bc_y['val'] - float(poly(bc_loc_canonical)))
                out[prof] = np.clip(y_eval, a_min=0, a_max=None)

        elif self.defined_on == 'aLy':
            # Populate poly_aLy cache
            self.get_aLy(params, x_eval)
            # Convert canonical integral d(x_canonical) to physical-x integral dx
            # x_canonical = 2*(x - x_min)/(x_max - x_min) - 1  =>  dx = (x_range/2) d(x_canonical)
            integral_scale = (self.domain[1] - self.domain[0]) / 2.0
            for prof in params:
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is None:
                    raise ValueError(f"No boundary condition for profile '{prof}' at x=1.0")
                poly_aLy = self.poly_aLy.get(prof)
                if poly_aLy is None:
                    raise ValueError(f"No aLy polynomial found for profile '{prof}'")
                # Antiderivative F(ξ) = ∫ aLy dξ on canonical domain
                poly_integral = poly_aLy.integ()
                bc_loc_canonical = self._rescale_to_canonical(np.array([bc_y['loc']]))[0]
                # phase = -(1/a) * ∫[bc_loc→x] aLy dx  (scale converts canonical → physical)
                phase = -(1.0 / 1) * integral_scale * (
                    poly_integral(x_canonical) - poly_integral(bc_loc_canonical)
                )
                phase = np.clip(phase, -50, 50)
                out[prof] = bc_y['val'] * np.exp(phase)

        else:
            raise ValueError(f"Invalid defined_on: '{self.defined_on}'")

        self.y = out
        return out
    
    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² using polynomial derivatives.

        When defined_on='y':   d²y/dx² = poly.deriv(2)(ξ) * (dξ/dx)², with chain rule.
        When defined_on='aLy': y'' = -(y/a) * (aLy' - aLy²/a), with aLy' from poly.deriv().
        """
        out = {}
        x_eval = np.asarray(x_eval)
        x_canonical = self._rescale_to_canonical(x_eval)
        # Chain-rule factor: dξ/dx = 2/(x_max - x_min)
        dx_canonical_dx = 2.0 / (self.domain[1] - self.domain[0])

        if self.defined_on == 'y':
            # Populate poly_y cache via get_y
            self.get_y(params, x_eval)
            for prof in params:
                poly = self.poly_y.get(prof)
                if poly is None:
                    raise ValueError(f"No y polynomial found for profile '{prof}'")
                # d²y/dx² = (d²y/dξ²) * (dξ/dx)²
                out[prof] = poly.deriv(2)(x_canonical) * dx_canonical_dx**2

        elif self.defined_on == 'aLy':
            y_dict = self.get_y(params, x_eval)
            aLy_dict = self.get_aLy(params, x_eval)
            for prof in params:
                y_eval = y_dict[prof]
                aLy_eval = aLy_dict[prof]
                poly_aLy = self.poly_aLy.get(prof)
                if poly_aLy is None:
                    raise ValueError(f"No aLy polynomial found for profile '{prof}'")
                # aLy' in physical x via chain rule
                aLy_prime = poly_aLy.deriv()(x_canonical) * dx_canonical_dx
                # y'' = -(y/a) * (aLy' - aLy²/a)
                curvature = -(y_eval / self.a) * (aLy_prime - (aLy_eval**2 / self.a))
                out[prof] = curvature

        else:
            raise ValueError(f"Invalid defined_on: '{self.defined_on}'")

        self.curv = out
        return out


# -------------------------
# Basis function model
# -------------------------


class BasisFunction(ParameterBase):
    """Basis-function parameterization using numpy.polynomial.<family>.<family>.basis.

    Supports Hermite, Legendre, Chebyshev, and Laguerre polynomial families.
    The parameter vector is the coefficient set for the basis expansion of a/Ly.

    Parameters
    ----------
    family : str
        Polynomial family: 'hermite', 'legendre', 'chebyshev', 'laguerre'
    degree : int
        Maximum polynomial degree included in the basis expansion
    domain : List[float]
        Domain [x_min, x_max] for evaluating the basis
    """

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)

        self.family = options.get('family', 'legendre').lower()
        self.degree = int(options.get('degree', 3))
        self.defined_on = 'aLy'

        self.param_names = [f"a_{i}" for i in range(self.degree + 1)]
        self.n_params_per_profile = self.degree + 1

        if 'domain' in options:
            self.domain = options['domain']

        self.poly_module = self._get_polynomial_module()
        self.basis_cache: Dict[int, Any] = {}

    def _get_polynomial_module(self):
        poly_map = {
            'legendre': np.polynomial.legendre.Legendre,
            'chebyshev': np.polynomial.chebyshev.Chebyshev,
            'hermite': np.polynomial.hermite.Hermite,
            'laguerre': np.polynomial.laguerre.Laguerre,
        }

        if self.family not in poly_map:
            raise ValueError(
                f"Unknown basis family '{self.family}'. "
                f"Choose from: {list(poly_map.keys())}"
            )

        return poly_map[self.family]

    def _get_basis(self, i: int):
        """Return cached basis polynomial of degree i."""
        basis = self.basis_cache.get(i)
        if basis is None:
            basis = self.poly_module.basis(i)
            self.basis_cache[i] = basis
        return basis

    def _rescale_to_basis_domain(self, x: np.ndarray) -> np.ndarray:
        """Rescale x to the canonical domain for the selected family."""
        x = np.asarray(x, dtype=float)
        x_min, x_max = self.domain[0], self.domain[1]

        denom = x_max - x_min
        if np.isclose(denom, 0.0):
            return np.zeros_like(x)

        if self.family == 'laguerre':
            # Laguerre basis is defined on [0, ∞); map user domain to [0, 1]
            return (x - x_min) / denom

        # Hermite/Legendre/Chebyshev use canonical [-1, 1]
        return 2.0 * (x - x_min) / denom - 1.0

    def _basis_eval(self, coeffs: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        """Evaluate Σ a_i * basis_i(x) at x_eval (in basis domain)."""
        y = np.zeros_like(x_eval, dtype=float)
        for i, a_i in enumerate(coeffs):
            if a_i == 0.0:
                continue
            y += a_i * self._get_basis(i)(x_eval)
        return y

    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Fit basis coefficients to state a/Ly data using least squares."""
        self.build_bcs(bc_dict)
        self.a = getattr(state, 'a', 1.0)

        params = {}
        params_std = {}

        x_data = np.asarray(getattr(state, self.coord))
        if hasattr(self, 'domain') and self.domain is not None:
            mask = (x_data >= self.domain[0]) & (x_data <= self.domain[1])
        else:
            mask = np.ones(len(x_data), dtype=bool)

        x_fit = x_data[mask]
        x_fit_basis = self._rescale_to_basis_domain(x_fit)

        for prof in self.predicted_profiles:
            aLy_key = f"aL{prof}"

            if hasattr(state, aLy_key):
                aLy_data = np.asarray(getattr(state, aLy_key))[mask]
            else:
                y_data = np.asarray(getattr(state, prof))[mask]
                dy_dx = np.gradient(y_data, x_fit)
                aLy_data = -self.a * dy_dx / (y_data + 1e-12)

            # Build design matrix using basis functions
            Phi = np.column_stack([
                self._get_basis(i)(x_fit_basis) for i in range(self.degree + 1)
            ])

            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(Phi, aLy_data, rcond=None)
                params[prof] = {f"a_{i}": float(coeffs[i]) for i in range(len(coeffs))}

                if residuals.size > 0:
                    residual_std = np.sqrt(residuals[0] / max(len(aLy_data) - len(coeffs), 1))
                else:
                    residual_std = np.std(aLy_data - Phi @ coeffs)

                params_std[prof] = {f"a_{i}": residual_std * self.sigma for i in range(len(coeffs))}
            except Exception as e:
                print(f"[BasisFunction.parameterize] Warning: fitting failed for {prof}: {e}")
                params[prof] = {f"a_{i}": 0.0 for i in range(self.degree + 1)}
                params_std[prof] = {f"a_{i}": self.sigma for i in range(self.degree + 1)}

        self.params = params
        self.params_std = params_std
        return params, params_std

    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        x_eval = np.asarray(x_eval)
        x_basis = self._rescale_to_basis_domain(x_eval)

        for prof, prof_params in params.items():
            coeffs = np.array([prof_params.get(f"a_{i}", 0.0) for i in range(self.degree + 1)])
            aLy_eval = self._basis_eval(coeffs, x_basis)
            if self.defined_on == "aLy":
                shift = -min(0.0, float(np.min(aLy_eval)))
                if shift:
                    aLy_eval = aLy_eval + shift
            out[prof] = aLy_eval

        self.aLy = out
        return out

    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        x_eval = np.asarray(x_eval)

        aLy_dict = self.get_aLy(params, x_eval)

        for prof in params:
            bc_y = self.get_nearest_bc(prof, 1.0)
            if bc_y is None:
                raise ValueError(f"No boundary condition for profile '{prof}' at x=1.0")

            aLy_eval = aLy_dict[prof]
            sort_idx = np.argsort(x_eval)
            x_sorted = x_eval[sort_idx]
            aLy_sorted = aLy_eval[sort_idx]

            integral = cumulative_trapezoid(aLy_sorted, x_sorted, initial=0.0)
            integral_bc = np.interp(bc_y['loc'], x_sorted, integral)
            phase = -(1.0 / self.a) * (integral - integral_bc)
            phase = np.clip(phase, -50, 50)
            y_sorted = bc_y['val'] * np.exp(phase)

            y_eval = np.empty_like(y_sorted)
            y_eval[sort_idx] = y_sorted
            if self.defined_on == "y":
                shift = -min(0.0, float(np.min(y_eval)))
                if shift:
                    y_eval = y_eval + shift
            out[prof] = y_eval

        self.y = out
        return out

    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        x_eval = np.asarray(x_eval)

        y_dict = self.get_y(params, x_eval)
        aLy_dict = self.get_aLy(params, x_eval)

        x_basis = self._rescale_to_basis_domain(x_eval)
        x_min, x_max = self.domain[0], self.domain[1]
        denom = x_max - x_min
        if np.isclose(denom, 0.0):
            dx_basis_dx = 0.0
        else:
            dx_basis_dx = 1.0 / denom if self.family == 'laguerre' else 2.0 / denom

        for prof, prof_params in params.items():
            coeffs = np.array([prof_params.get(f"a_{i}", 0.0) for i in range(self.degree + 1)])

            aLy_prime = np.zeros_like(x_basis, dtype=float)
            for i, a_i in enumerate(coeffs):
                if a_i == 0.0:
                    continue
                aLy_prime += a_i * self._get_basis(i).deriv()(x_basis)

            aLy_prime *= dx_basis_dx

            y_eval = y_dict[prof]
            aLy_eval = aLy_dict[prof]
            curvature = -(y_eval / self.a) * (aLy_prime - (aLy_eval**2 / self.a))
            out[prof] = curvature

        self.curv = out
        return out


# -------------------------
# Other model stubs
# -------------------------


class Mtanh(ParameterBase):
    """Modified-tanh parameter model with asymmetric (position-dependent) width.

    Model
    -----
    y(x) = A * (-tanh(u(x)) + 1) - m*(x-1) + b

    where the argument uses a spatially varying width:

        w(x)  = Delta_0 * (1 + delta*(x - c))
        u(x)  = (x - c) / w(x)
        du/dx = Delta_0 / w(x)^2 = 1 / (Delta_0 * (1 + delta*(x-c))^2)

    So:

        dy/dx = -A * sech^2(u(x)) / (Delta_0*(1+delta*(x-c))^2) - m

    Free parameters (solver space): {log_A, log_Delta_0, delta, m}
        A      > 0  : amplitude
        Delta_0 > 0 : characteristic width (at the pedestal centre x=c)
        delta       : width asymmetry  — w stretches linearly with distance from c
        m      >= 0 : linear slope contribution to gradient

    Boundary conditions enforced deterministically at x = 1
    --------------------------------------------------------
    Define:
        f1  = 1 + delta*(1 - c)         [= w(1)/Delta_0]
        u1  = (c - 1) / (Delta_0 * f1)  [tanh argument at x=1; note tanh(u1) < 0]

    Then the aLy BC gives the root-finding equation for c:

        A * sech^2(u1) / (Delta_0 * f1^2) = y(1)*aLy(1) - m      ... (*)

    and b is solved explicitly:

        b = y(1) - A*(tanh(u1) + 1)

    Note: the sign convention means tanh(u1)+1 plays the role of (-tanh((1-c)/w)+1).

    Existence condition
    -------------------
    The RHS of (*) must be positive, so:

        m < y(1) * aLy(1)

    Given that, the maximum of the LHS over c (attained at c→1 where sech^2→1
    and f1→1) is A/Delta_0, so a sufficient condition for (*) to have a solution is:

        A / Delta_0 >= y(1)*aLy(1) - m

    The upper bound on the RHS depends on c once delta != 0, so existence is
    checked iteratively inside _derive_c_and_b.
    """

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.defined_on = "y"
        self.param_names = ['log_A', 'log_Delta_0', 'delta', 'm']
        self.n_params_per_profile = len(self.param_names)
        self.include_zero_grad_on_axis = False  # BCs enforced analytically

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _to_physical(self, p: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Return (A, Delta_0, delta, m) from solver-space params."""
        return (
            float(np.exp(p['log_A'])),
            float(np.exp(p['log_Delta_0'])),
            float(p['delta']),
            float(p['m']),
        )

    def _resolve_bcs(self, prof: str) -> Tuple[float, float]:
        """Return (y_bc, aLy_bc) at x=1 for the given profile."""
        bc_y   = self.get_nearest_bc(prof, 1.0)
        bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
        if bc_y is None:
            raise ValueError(f"No y BC found for profile '{prof}'")
        if bc_aLy is None:
            raise ValueError(f"No aLy BC found for profile '{prof}'")
        return float(bc_y['val']), float(bc_aLy['val'])

    @staticmethod
    def _f1(c: float, delta: float) -> float:
        """Width factor at x=1:  f1 = 1 + delta*(1-c). Floored at 1e-6."""
        return max(1.0 + delta * (1.0 - c), 1e-6)

    @staticmethod
    def _u1(c: float, Delta_0: float, delta: float) -> float:
        """Tanh argument at x=1:  u1 = (c-1) / (Delta_0 * f1)."""
        f1 = Mtanh._f1(c, delta)
        return (c - 1.0) / (Delta_0 * f1)

    def _derive_c_and_b(
        self,
        A: float, Delta_0: float, delta: float, m: float,
        y_bc: float, aLy_bc: float,
    ) -> Tuple[float, float, float]:
        """Solve for c and b from the BC equations (root-finding on c).

        Root-finding equation (*):
            F(c) = A*sech^2(u1(c)) / (Delta_0*f1(c)^2) - (y_bc*aLy_bc - m) = 0

        b is then:
            b = y_bc - A*(tanh(u1) + 1)

        Returns (c, b, A_used).  A may be raised to satisfy existence.
        """
        rhs = y_bc * aLy_bc - m   # must be > 0

        if rhs <= 0.0:
            # m already accounts for all edge gradient; park tanh plateau far inside
            c = -5.0
            f1 = self._f1(c, delta)
            u1 = self._u1(c, Delta_0, delta)
            b  = y_bc - A * (np.tanh(u1) + 1.0)
            return c, b, A

        # Enforce sufficient existence condition: A/Delta_0 >= rhs  =>  A >= Delta_0*rhs
        A_min = Delta_0 * rhs + 1e-10
        A = max(A, A_min)

        def F(c_):
            f1 = self._f1(c_, delta)
            u1 = (c_ - 1.0) / (Delta_0 * f1)
            sech2 = 1.0 - np.tanh(u1) ** 2
            return A * sech2 / (Delta_0 * f1 ** 2) - rhs

        # F(c→1⁻) ≈ A/Delta_0 - rhs >= 0  (by existence projection above)
        # F(c→-∞) → -rhs < 0   (sech^2→0 faster than denominator grows)
        # So the root is bracketed in [c_lo, 1-eps].
        c_hi = 1.0 - 1e-6
        c_lo = -20.0

        # Verify bracket (F could be non-monotone for large |delta|)
        F_hi = F(c_hi)
        F_lo = F(c_lo)

        if F_hi < 0.0:
            # A/Delta_0 < rhs after all (numerical edge case) — park c far inside
            c = -5.0
        elif F_lo > 0.0:
            # Function positive everywhere in bracket; take leftmost boundary
            c = c_lo
        else:
            try:
                c = float(brentq(F, c_lo, c_hi, xtol=1e-8, maxiter=200))
            except Exception:
                c = -5.0

        f1 = self._f1(c, delta)
        u1 = self._u1(c, Delta_0, delta)
        b  = y_bc - A * (np.tanh(u1) + 1.0)
        return c, b, A

    def _w_eval(self, x: np.ndarray, Delta_0: float, delta: float, c: float) -> np.ndarray:
        """Spatially varying width w(x) = Delta_0*(1+delta*(x-c)), floored at 1e-6."""
        return np.maximum(Delta_0 * (1.0 + delta * (x - c)), 1e-6)

    def _y_eval(self, x: np.ndarray, A: float, Delta_0: float, delta: float,
                m: float, c: float, b: float) -> np.ndarray:
        w  = self._w_eval(x, Delta_0, delta, c)
        u  = (x - c) / w
        return A * (-np.tanh(u) + 1.0) - m * (x - 1.0) + b

    def _dydx_eval(self, x: np.ndarray, A: float, Delta_0: float, delta: float,
                   m: float, c: float) -> np.ndarray:
        """dy/dx = -A * sech^2(u) / (Delta_0*(1+delta*(x-c))^2) - m"""
        f     = np.maximum(1.0 + delta * (x - c), 1e-6)
        u     = (x - c) / (Delta_0 * f)
        sech2 = 1.0 - np.tanh(u) ** 2
        return -A * sech2 / (Delta_0 * f ** 2) - m

    # ─────────────────────────────────────────────────────────
    # Public projection (called by solver after box-clipping)
    # ─────────────────────────────────────────────────────────

    def project_params(
        self,
        params: Dict[str, Dict[str, float]],
        bc_dict: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Project parameters to satisfy the existence condition m < y(1)*aLy(1).

        If violated, A is raised to Delta_0*(y_bc*aLy_bc - m) so that the
        root-finding in _derive_c_and_b succeeds.  All other params unchanged.
        """
        if not bc_dict:
            return params
        self.build_bcs(bc_dict)
        out = {}
        for prof, p in params.items():
            A, Delta_0, delta, m = self._to_physical(p)
            try:
                y_bc, aLy_bc = self._resolve_bcs(prof)
            except ValueError:
                out[prof] = p
                continue
            rhs = y_bc * aLy_bc - m
            if rhs > 0.0:
                A = max(A, Delta_0 * rhs + 1e-10)
            out[prof] = {
                'log_A':      float(np.log(A)),
                'log_Delta_0': p['log_Delta_0'],
                'delta':       delta,
                'm':           m,
            }
        return out

    # ─────────────────────────────────────────────────────────
    # ParameterBase interface
    # ─────────────────────────────────────────────────────────

    def parameterize(
        self,
        state: Any,
        bc_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Fit (A, Delta_0, delta, m) to y(x) in *state* satisfying the BCs exactly.

        c and b are derived via root-finding / explicit formula inside the model
        function, so the fit residual is purely over the free parameters.
        """
        self.build_bcs(bc_dict)
        params: Dict[str, Dict[str, float]] = {}
        params_std: Dict[str, Dict[str, float]] = {}

        x_data = np.asarray(getattr(state, self.coord)).flatten()

        for prof in self.predicted_profiles:
            y_data = np.asarray(getattr(state, prof)).flatten()
            y_bc, aLy_bc = self._resolve_bcs(prof)

            def _model(x, log_A, log_D0, delta_, m_val,
                       _y_bc=y_bc, _aLy_bc=aLy_bc):
                A_  = np.exp(log_A)
                D0_ = np.exp(log_D0)
                c_, b_, A_used = self._derive_c_and_b(A_, D0_, delta_, m_val, _y_bc, _aLy_bc)
                return self._y_eval(x, A_used, D0_, delta_, m_val, c_, b_)

            p0  = [np.log(1.0),  np.log(0.01), 0,  1.]
            blo = [np.log(1e-6), np.log(1e-6),    -100.0,    0.0]
            bhi = [np.log(10.0),  np.log(1.0),      100.0,    10.]

            try:
                popt, pcov = curve_fit(
                    _model, x_data, y_data,
                    p0=p0,
                    bounds=(blo, bhi),
                    maxfev=20000,
                    xtol=1e-12,
                    ftol=1e-12,
                )
                perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            except Exception:
                popt = np.array(p0)
                perr = np.abs(popt) * self.sigma

            params[prof]     = {'log_A': float(popt[0]), 'log_Delta_0': float(popt[1]),
                                 'delta': float(popt[2]), 'm': float(popt[3])}
            params_std[prof] = {'log_A': float(perr[0]), 'log_Delta_0': float(perr[1]),
                                 'delta': float(perr[2]), 'm': float(perr[3])}

        self.params     = params
        self.params_std = params_std
        return params, params_std

    def get_y(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate y(x) on *x_eval* for each profile."""
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, p in params.items():
            A, Delta_0, delta, m = self._to_physical(p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            c, b, A = self._derive_c_and_b(A, Delta_0, delta, m, y_bc, aLy_bc)
            out[prof] = np.clip(self._y_eval(x_eval, A, Delta_0, delta, m, c, b), 0.0, None)
        self.y = out
        return out

    def get_aLy(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate a/Ly(x) = -(dy/dx)/y on *x_eval* for each profile."""
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, p in params.items():
            A, Delta_0, delta, m = self._to_physical(p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            c, b, A = self._derive_c_and_b(A, Delta_0, delta, m, y_bc, aLy_bc)
            y    = self._y_eval(x_eval, A, Delta_0, delta, m, c, b)
            dydx = self._dydx_eval(x_eval, A, Delta_0, delta, m, c)
            y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
            out[prof] = np.clip(-dydx / y_safe, 0.0, None)
        self.aLy = out
        return out

    def get_curvature(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate d²y/dx² on *x_eval*.

        With w(x) = Delta_0*(1+delta*(x-c)) and u(x) = (x-c)/w(x):

            d²y/dx² = 2A*sech^2(u) / Delta_0 * [tanh(u) / (Delta_0*f^4) + delta/f^3]

        where f = 1+delta*(x-c).
        """
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, p in params.items():
            A, Delta_0, delta, m = self._to_physical(p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            c, b, A = self._derive_c_and_b(A, Delta_0, delta, m, y_bc, aLy_bc)
            f     = np.maximum(1.0 + delta * (x_eval - c), 1e-6)
            u     = (x_eval - c) / (Delta_0 * f)
            tanh_ = np.tanh(u)
            sech2 = 1.0 - tanh_ ** 2
            out[prof] = (2.0 * A * sech2 / Delta_0) * (
                tanh_ / (Delta_0 * f ** 4) + delta / f ** 3
            )
        self.curv = out
        return out


class SplineMtanh(ParameterBase):
    """Hybrid: Spline parameter backbone with Mtanh profile constructor.

    The optimizer works with y or a/Ly values at user-defined knot positions
    (identical to the Spline class).  Profile evaluation uses the modified-tanh
    analytic formula and its exact analytic first and second derivatives fitted
    to those knot values, rather than an Akima/PCHIP/Cubic spline.

    Parameters (options)
    --------------------
    knots : Sequence[float]
        Positions in rho = r/a where the free parameters are defined.
    defined_on : str
        'y'   – knot parameters are profile values y(x_k).
        'aLy' – knot parameters are normalised-gradient a/Ly(x_k).
        'mixed' – requires exactly 2 knots [x_0, x_mid].  Parameters are the
                  (y, a/Ly) pair at each knot: (y0, aLy0, y1, aLy1).  The
                  Mtanh is solved exactly (4 equations, 4 unknowns) via
                  nonlinear root-finding.  bc_mode='spline' not supported.
    bc_mode : str
        'mtanh'  (default)
            Boundary conditions (y(1), a/Ly(1)) are sourced entirely from
            the external bc_dict.  The pedestal centre c and offset b are
            derived from them via root-finding, exactly as in the standalone
            Mtanh class — the Mtanh curve passes exactly through the BC
            values at x = 1.  Free optimizer parameters are the knot values
            only.
        'spline'
            The boundary value (y(1) if defined_on='y', a/Ly(1) if
            defined_on='aLy') is appended to the parameter vector as an
            extra entry, making it an optimizer degree of freedom rather than
            a fixed constraint.  c and b are still derived analytically, but
            y_bc (or aLy_bc) is taken from the parameter vector instead of
            bc_dict.
    include_zero_grad_on_axis : bool
        When True (default), and defined_on='aLy', a virtual data point
        (x=0, a/Ly=0) is added to the knot data used for Mtanh fitting.
    coord : str
        'rho' or 'roa' (default).

    Notes
    -----
    For defined_on='y' or 'aLy': the Mtanh is fitted at evaluation time via
    least-squares.  The fit is exact for N ≤ 4 knots and approximate for N > 4.

    For defined_on='mixed': the Mtanh is solved exactly from the two (y, a/Ly)
    pairs via scipy.optimize.least_squares (trust-region, 4 equations/4 unknowns).

    parameterize() always builds initial estimates from an Akima spline on
    state.y (not state.aLy) for robustness, regardless of defined_on.
    """

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.knots = np.array(options.get('knots', []) or [], dtype=float)
        self.defined_on = options.get('defined_on', 'y')
        self.bc_mode = options.get('bc_mode', 'mtanh').lower()

        if self.bc_mode not in ('mtanh', 'spline'):
            raise ValueError("bc_mode must be 'mtanh' or 'spline'")
        if self.defined_on not in ('y', 'aLy', 'mixed'):
            raise ValueError("defined_on must be 'y', 'aLy', or 'mixed'")

        if self.defined_on == 'mixed':
            if len(self.knots) != 2:
                raise ValueError("defined_on='mixed' requires exactly 2 knots: [x_0, x_mid]")
            if self.bc_mode == 'spline':
                raise ValueError("bc_mode='spline' is incompatible with defined_on='mixed'")
            self.param_names = ['y0', 'aLy0', 'y1', 'aLy1']
        else:
            base = self.defined_on
            self.param_names = [f'{base}{i}' for i in range(len(self.knots))]
        self.n_params_per_profile = len(self.param_names)

    # ──────────────────────────────────────────────────────────────────────
    # Mtanh analytic formulae (self-contained; no dependency on Mtanh class)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _f(x, c: float, delta: float) -> np.ndarray:
        """Spatially varying width factor  f(x) = 1 + δ·(x−c), floored at 1e-6."""
        return np.maximum(1.0 + delta * (np.asarray(x, dtype=float) - c), 1e-6)

    @staticmethod
    def _y_mtanh(x, A: float, Delta_0: float, delta: float,
                 m: float, c: float, b: float) -> np.ndarray:
        """y(x) = A·(1 − tanh(u)) − m·(x−1) + b,   u = (x−c)/(Δ₀·f)."""
        x = np.asarray(x, dtype=float)
        f = SplineMtanh._f(x, c, delta)
        u = (x - c) / (Delta_0 * f)
        return A * (1.0 - np.tanh(u)) - m * (x - 1.0) + b

    @staticmethod
    def _dydx_mtanh(x, A: float, Delta_0: float, delta: float,
                    m: float, c: float) -> np.ndarray:
        """dy/dx = −A·sech²(u) / (Δ₀·f²) − m."""
        x = np.asarray(x, dtype=float)
        f = SplineMtanh._f(x, c, delta)
        u = (x - c) / (Delta_0 * f)
        sech2 = 1.0 - np.tanh(u) ** 2
        return -A * sech2 / (Delta_0 * f ** 2) - m

    @staticmethod
    def _d2ydx2_mtanh(x, A: float, Delta_0: float, delta: float,
                      c: float) -> np.ndarray:
        """d²y/dx² = (2A·sech²(u)/Δ₀)·[ tanh(u)/(Δ₀·f⁴) + δ/f³ ]."""
        x = np.asarray(x, dtype=float)
        f = SplineMtanh._f(x, c, delta)
        u = (x - c) / (Delta_0 * f)
        tanh_ = np.tanh(u)
        sech2 = 1.0 - tanh_ ** 2
        return (2.0 * A * sech2 / Delta_0) * (
            tanh_ / (Delta_0 * f ** 4) + delta / f ** 3
        )

    @staticmethod
    def _derive_c_b(
        A: float, Delta_0: float, delta: float, m: float,
        y_bc: float, aLy_bc: float,
    ) -> Tuple[float, float, float]:
        """Solve for pedestal centre c and offset b from edge BCs at x = 1.

        Mirrors Mtanh._derive_c_and_b exactly.  Returns (c, b, A) where A
        may be raised to satisfy the existence condition.
        """
        rhs = y_bc * aLy_bc - m  # must be > 0 for a root to exist

        if rhs <= 0.0:
            c = -5.0
            f1 = max(1.0 + delta * (1.0 - c), 1e-6)
            u1 = (c - 1.0) / (Delta_0 * f1)
            return c, y_bc - A * (np.tanh(u1) + 1.0), A

        # Existence condition: A/Δ₀ ≥ rhs
        A = max(A, Delta_0 * rhs + 1e-10)

        def F(c_: float) -> float:
            f1 = max(1.0 + delta * (1.0 - c_), 1e-6)
            u1 = (c_ - 1.0) / (Delta_0 * f1)
            return A * (1.0 - np.tanh(u1) ** 2) / (Delta_0 * f1 ** 2) - rhs

        c_hi, c_lo = 1.0 - 1e-6, -20.0
        if F(c_hi) < 0.0:
            c = -5.0
        elif F(c_lo) > 0.0:
            c = c_lo
        else:
            try:
                c = float(brentq(F, c_lo, c_hi, xtol=1e-8, maxiter=200))
            except Exception:
                c = -5.0

        f1 = max(1.0 + delta * (1.0 - c), 1e-6)
        u1 = (c - 1.0) / (Delta_0 * f1)
        b = y_bc - A * (np.tanh(u1) + 1.0)
        return c, b, A

    # ──────────────────────────────────────────────────────────────────────
    # Internal: Mtanh fitting to knot data
    # ──────────────────────────────────────────────────────────────────────

    def _fit_mtanh_to_knots(
        self,
        x_k: np.ndarray,
        val_k: np.ndarray,
        y_bc: float,
        aLy_bc: float,
    ) -> Dict[str, float]:
        """Fit Mtanh (A, Δ₀, δ, m) to the knot data (x_k, val_k).

        c and b are derived analytically from (y_bc, aLy_bc) inside the model
        for every candidate parameter set, so the fit residual is purely over
        the four free shape parameters.

        For defined_on='y'   : val_k = y values  → minimise ‖y_mtanh − val_k‖
        For defined_on='aLy' : val_k = a/Ly values → minimise ‖aLy_mtanh − val_k‖
        """
        if self.defined_on == 'y':
            def _model(x, log_A, log_D0, delta_, m_):
                A_ = np.exp(log_A)
                D0_ = np.exp(log_D0)
                c_, b_, A_ = self._derive_c_b(A_, D0_, delta_, m_, y_bc, aLy_bc)
                return self._y_mtanh(x, A_, D0_, delta_, m_, c_, b_)
        else:  # defined_on == 'aLy'
            def _model(x, log_A, log_D0, delta_, m_):
                A_ = np.exp(log_A)
                D0_ = np.exp(log_D0)
                c_, b_, A_ = self._derive_c_b(A_, D0_, delta_, m_, y_bc, aLy_bc)
                y_    = self._y_mtanh(x, A_, D0_, delta_, m_, c_, b_)
                dydx_ = self._dydx_mtanh(x, A_, D0_, delta_, m_, c_)
                y_safe = np.where(np.abs(y_) < 1e-12, 1e-12, y_)
                return np.clip(-dydx_ / y_safe, 0.0, None)

        # Initial guess: amplitude from knot data, moderate width
        p0  = [np.log(1.0),  np.log(0.01), 0,  1.]
        blo = [np.log(1e-6), np.log(1e-6),    -100.0,    0.0]
        bhi = [np.log(10.0),  np.log(1.0),      100.0,    10.]

        try:
            popt, _ = curve_fit(
                _model, x_k, val_k,
                p0=p0, bounds=(blo, bhi),
                maxfev=20000, xtol=1e-10, ftol=1e-10,
            )
        except Exception:
            popt = p0

        return {
            'log_A':       float(popt[0]),
            'log_Delta_0': float(popt[1]),
            'delta':       float(popt[2]),
            'm':           float(popt[3]),
        }

    @staticmethod
    def _solve_mtanh_from_pairs(
        x_0: float, y_0: float, aLy_0: float,
        x_mid: float, y_mid: float, aLy_mid: float,
        y_bc: float, aLy_bc: float,
    ) -> Dict[str, float]:
        """Exactly determine (A, Δ₀, δ, m) from (y, a/Ly) at two interior points.

        c and b are derived from BCs at x=1 inside the residual function.
        Solves the 4×4 nonlinear system:
            y(x_0)      = y_0,    a/Ly(x_0)   = aLy_0
            y(x_mid)    = y_mid,  a/Ly(x_mid) = aLy_mid
        via scipy.optimize.least_squares (trust-region).
        Residuals are normalised by target magnitude to balance y vs a/Ly scales.
        """
        def _residuals(p: np.ndarray) -> np.ndarray:
            A_  = np.exp(p[0])
            D0_ = np.exp(p[1])
            try:
                c_, b_, A_ = SplineMtanh._derive_c_b(A_, D0_, p[2], p[3], y_bc, aLy_bc)
            except Exception:
                return np.array([1e6, 1e6, 1e6, 1e6])
            xs    = np.array([x_0, x_mid])
            y_p   = SplineMtanh._y_mtanh(xs, A_, D0_, p[2], p[3], c_, b_)
            dy_p  = SplineMtanh._dydx_mtanh(xs, A_, D0_, p[2], p[3], c_)
            y_s   = np.where(np.abs(y_p) < 1e-12, 1e-12, y_p)
            aLy_p = np.clip(-dy_p / y_s, 0.0, None)
            return np.array([
                (y_p[0]   - y_0)   / (abs(y_0)   + 1e-12),
                (aLy_p[0] - aLy_0) / (abs(aLy_0) + 1e-12),
                (y_p[1]   - y_mid) / (abs(y_mid)  + 1e-12),
                (aLy_p[1] - aLy_mid) / (abs(aLy_mid) + 1e-12),
            ])

        A0  = max(y_0, y_mid, y_bc, 1e-6)
        m0  = float(np.clip(aLy_0 * y_0, 0.0, 100.0))
        p0  = [np.log(A0), np.log(0.05), 0.0, m0]
        blo = [np.log(1e-6), np.log(1e-6), -100.0,   0.0]
        bhi = [np.log(1e4),  np.log(2.0),   100.0, 100.0]
        try:
            res  = least_squares(_residuals, p0, bounds=(blo, bhi),
                                 method='trf', max_nfev=20000,
                                 xtol=1e-10, ftol=1e-10)
            popt = res.x
        except Exception:
            popt = np.asarray(p0)
        return {
            'log_A':       float(popt[0]),
            'log_Delta_0': float(popt[1]),
            'delta':       float(popt[2]),
            'm':           float(popt[3]),
        }

    def _resolve(
        self,
        prof: str,
        prof_params: Dict[str, float],
    ) -> Tuple[float, float, float, float, float, float]:
        """Assemble knot data, fit/solve Mtanh, derive (c, b) and return full tuple.

        Returns (A, Δ₀, δ, m, c, b).
        """
        # Collect BCs from dict (always needed for _derive_c_b)
        bc_y   = self.get_nearest_bc(prof,        1.0)
        bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
        y_bc   = float(bc_y['val'])   if bc_y   is not None else 1.0
        aLy_bc = float(bc_aLy['val']) if bc_aLy is not None else 1.0

        if self.defined_on == 'mixed':
            # Exact 4×4 solve: (y, aLy) pairs at x_0 and x_mid
            fit = self._solve_mtanh_from_pairs(
                float(self.knots[0]), float(prof_params['y0']),  float(prof_params['aLy0']),
                float(self.knots[1]), float(prof_params['y1']),  float(prof_params['aLy1']),
                y_bc, aLy_bc,
            )
        else:
            base = self.defined_on
            n    = len(self.knots)
            vals = np.array([prof_params[f'{base}{i}'] for i in range(n)], dtype=float)
            x_k  = self.knots.copy()

            if self.bc_mode == 'spline':
                bc_val = float(prof_params[f'{base}_bc'])
                if not np.any(np.isclose(x_k, 1.0)):
                    x_k  = np.append(x_k,  1.0)
                    vals = np.append(vals, bc_val)
                else:
                    idx = int(np.argmin(np.abs(x_k - 1.0)))
                    vals[idx] = bc_val
                if self.defined_on == 'y':
                    y_bc = bc_val
                else:
                    aLy_bc = bc_val

            if self.include_zero_grad_on_axis and self.defined_on == 'aLy':
                if not np.any(np.isclose(x_k, 0.0)):
                    x_k  = np.insert(x_k,  0, 0.0)
                    vals = np.insert(vals, 0, 0.0)

            fit = self._fit_mtanh_to_knots(x_k, vals, y_bc, aLy_bc)

        A  = np.exp(fit['log_A'])
        D0 = np.exp(fit['log_Delta_0'])
        c, b, A = self._derive_c_b(A, D0, fit['delta'], fit['m'], y_bc, aLy_bc)
        return A, D0, fit['delta'], fit['m'], c, b

    # ──────────────────────────────────────────────────────────────────────
    # ParameterBase interface
    # ──────────────────────────────────────────────────────────────────────

    def project_params(
        self,
        params: Dict[str, Dict[str, float]],
        bc_dict: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Snap knot values to what the fitted Mtanh actually predicts at those knots.

        After bound-clipping, individual knot values may be inconsistent with
        the best-fit Mtanh through the full set.  When N > 4 knots the Mtanh
        fit is overdetermined; the optimizer can accumulate 'phantom' degrees
        of freedom where a single knot drifts far from the curve without
        significantly changing the output, because the least-squares fit
        averages across all knots.  This causes inaccurate Jacobian estimates
        and poor solver convergence.

        This projection re-evaluates the Mtanh (fitted from the *current* knot
        values) at every knot location and stores those values back as the
        parameters, keeping the parameter vector consistent with the curve.
        For N ≤ 4 this is a no-op (the Mtanh passes exactly through all knots).
        """
        if not bc_dict:
            return params
        self.build_bcs(bc_dict)
        out = {}
        for prof, pv in params.items():
            try:
                A, D0, delta, m, c, b = self._resolve(prof, pv)
            except Exception:
                out[prof] = pv
                continue

            if self.defined_on == 'mixed':
                x_k    = self.knots
                y_k    = self._y_mtanh(x_k, A, D0, delta, m, c, b)
                dydx_k = self._dydx_mtanh(x_k, A, D0, delta, m, c)
                y_safe = np.where(np.abs(y_k) < 1e-12, 1e-12, y_k)
                aLy_k  = np.clip(-dydx_k / y_safe, 0.0, None)
                out[prof] = {
                    'y0': float(y_k[0]),   'aLy0': float(aLy_k[0]),
                    'y1': float(y_k[1]),   'aLy1': float(aLy_k[1]),
                }
            else:
                base = self.defined_on
                n    = len(self.knots)
                if self.defined_on == 'y':
                    projected = self._y_mtanh(self.knots, A, D0, delta, m, c, b)
                else:  # 'aLy'
                    y_k    = self._y_mtanh(self.knots, A, D0, delta, m, c, b)
                    dydx_k = self._dydx_mtanh(self.knots, A, D0, delta, m, c)
                    y_safe = np.where(np.abs(y_k) < 1e-12, 1e-12, y_k)
                    projected = np.clip(-dydx_k / y_safe, 0.0, None)
                new_pv = {f'{base}{i}': float(projected[i]) for i in range(n)}
                if self.bc_mode == 'spline':
                    x_bc = np.array([1.0])
                    if self.defined_on == 'y':
                        bc_proj = float(self._y_mtanh(x_bc, A, D0, delta, m, c, b)[0])
                    else:
                        y_b  = self._y_mtanh(x_bc, A, D0, delta, m, c, b)
                        dy_b = self._dydx_mtanh(x_bc, A, D0, delta, m, c)
                        y_safe_b = float(max(abs(float(y_b[0])), 1e-12))
                        bc_proj = float(np.clip(-float(dy_b[0]) / y_safe_b, 0.0, None))
                    new_pv[f'{base}_bc'] = bc_proj
                out[prof] = new_pv
        return out

    def parameterize(
        self, state, bc_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Fit Mtanh to the full state.y profile; extract knot parameters per defined_on.

        Workflow (same for all modes):
        1. Fit (A, Δ₀, δ, m) to the complete y(x) profile via least-squares,
           with c and b derived from edge BCs at each candidate evaluation.
        2. Evaluate the fitted Mtanh at self.knots to obtain y_k and aLy_k.
        3. Assemble pdict according to defined_on:
           'y'     → y_k values at knots
           'aLy'   → aLy_k = −(dy/dx)/y at knots
           'mixed' → (y_k, aLy_k) pair at each of the two knots
        """
        self.build_bcs(bc_dict)
        self.a = getattr(state, 'a', 1.0)
        params: Dict[str, Dict[str, float]] = {}
        params_std: Dict[str, Dict[str, float]] = {}

        x_data = np.asarray(getattr(state, self.coord)).flatten()

        for prof in self.predicted_profiles:
            y_raw = np.asarray(getattr(state, prof)).flatten()

            bc_y   = self.get_nearest_bc(prof,        1.0)
            bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
            y_bc   = float(bc_y['val'])   if bc_y   is not None else float(y_raw[-1]) if len(y_raw) else 1.0
            aLy_bc = float(bc_aLy['val']) if bc_aLy is not None else 1.0

            # Fit Mtanh to the full y(x) profile regardless of defined_on
            def _model_y(x, log_A, log_D0, delta_, m_):
                A_  = np.exp(log_A)
                D0_ = np.exp(log_D0)
                c_, b_, A_ = self._derive_c_b(A_, D0_, delta_, m_, y_bc, aLy_bc)
                return self._y_mtanh(x, A_, D0_, delta_, m_, c_, b_)

            log_A0 = np.log(max(float(np.mean(np.abs(y_raw))), 1e-6))
            p0  = [log_A0, np.log(0.01), 0.0, 1.0]
            blo = [np.log(1e-6), np.log(1e-6), -100.0,   0.0]
            bhi = [np.log(1e4),  np.log(2.0),   100.0, 100.0]
            try:
                popt, _ = curve_fit(
                    _model_y, x_data, y_raw,
                    p0=p0, bounds=(blo, bhi),
                    maxfev=20000, xtol=1e-10, ftol=1e-10,
                )
            except Exception:
                popt = np.asarray(p0)

            A     = np.exp(popt[0])
            D0    = np.exp(popt[1])
            delta = float(popt[2])
            m     = float(popt[3])
            c, b, A = self._derive_c_b(A, D0, delta, m, y_bc, aLy_bc)

            # Evaluate the fitted Mtanh at knot positions
            y_k    = self._y_mtanh(self.knots, A, D0, delta, m, c, b)
            dydx_k = self._dydx_mtanh(self.knots, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y_k) < 1e-12, 1e-12, y_k)
            aLy_k  = np.clip(-dydx_k / y_safe, 0.0, None)

            if self.defined_on == 'y':
                pdict: Dict[str, float] = {f'y{i}': float(y_k[i]) for i in range(len(self.knots))}
                if self.bc_mode == 'spline':
                    pdict['y_bc'] = float(self._y_mtanh(np.array([1.0]), A, D0, delta, m, c, b)[0])
            elif self.defined_on == 'aLy':
                pdict = {f'aLy{i}': float(aLy_k[i]) for i in range(len(self.knots))}
                if self.bc_mode == 'spline':
                    y_1  = self._y_mtanh(np.array([1.0]), A, D0, delta, m, c, b)
                    dy_1 = self._dydx_mtanh(np.array([1.0]), A, D0, delta, m, c)
                    pdict['aLy_bc'] = float(np.clip(-float(dy_1[0]) / max(abs(float(y_1[0])), 1e-12), 0.0, None))
            else:  # 'mixed'
                pdict = {
                    'y0':   float(y_k[0]),   'aLy0': float(aLy_k[0]),
                    'y1':   float(y_k[1]),   'aLy1': float(aLy_k[1]),
                }

            params[prof]     = pdict
            params_std[prof] = {k: abs(v) * self.sigma for k, v in pdict.items()}

        self.params     = params
        self.params_std = params_std
        return params, params_std

    def get_y(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate y(x) on *x_eval* using the Mtanh fitted to knot data."""
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            A, D0, delta, m, c, b = self._resolve(prof, pv)
            out[prof] = np.clip(
                self._y_mtanh(x_eval, A, D0, delta, m, c, b), 0.0, None
            )
        self.y = out
        return out

    def get_aLy(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate a/Ly(x) = −(dy/dx)/y on *x_eval* using the Mtanh formula."""
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            A, D0, delta, m, c, b = self._resolve(prof, pv)
            y    = self._y_mtanh(x_eval, A, D0, delta, m, c, b)
            dydx = self._dydx_mtanh(x_eval, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
            out[prof] = np.clip(-dydx / y_safe, 0.0, None)
        self.aLy = out
        return out

    def get_curvature(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Evaluate d²y/dx² on *x_eval* using the analytic Mtanh second derivative."""
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            A, D0, delta, m, c, b = self._resolve(prof, pv)
            out[prof] = self._d2ydx2_mtanh(x_eval, A, D0, delta, c)
        self.curv = out
        return out


class LogSpline(Spline):
    """ Spline parameter model with parameters in log space"""

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.defined_on = "aLy"
        self.param_names = ['log_' + self.defined_on + str(i) for i in range(len(self.knots))]

    def _to_physical_params(self, params: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Exponentiate log-space values in-place under the same log_aLy keys.

        Keys stay as 'log_aLy0', 'log_aLy1', ... so that the parent Spline methods
        (which iterate self.param_names = ['log_aLy0', ...]) can still look them up,
        but the stored values become physical aLy (not log aLy).
        """
        phys = {}
        for prof, prof_params in params.items():
            inner = {}
            for key, val in prof_params.items():
                inner[key] = np.exp(val) if key.startswith('log_aLy') else val
            phys[prof] = inner
        return phys

    def _to_log_params(self, params: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply log transform to values stored under log_aLy keys.

        After Spline.parameterize (called via super()), self.param_names is already
        ['log_aLy0', ...], so the returned dict has log_aLy keys with *physical* aLy
        values.  This method takes the log of those values.
        """
        log_params = {}
        for prof, prof_params in params.items():
            inner = {}
            for key, val in prof_params.items():
                if key.startswith('log_aLy'):
                    inner[key] = np.log(val) if val > 0 else -np.inf
                else:
                    inner[key] = val
            log_params[prof] = inner
        return log_params

    def _to_log_std(self, params: Dict[str, Dict[str, float]], params_std: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Convert physical uncertainties to log-space (σ_log = σ/μ for log-normal)."""
        log_std = {}
        for prof, prof_params in params.items():
            inner = {}
            for key, val in prof_params.items():
                if key.startswith('log_aLy'):
                    inner[key] = (params_std[prof][key] / val) if val > 0 else self.sigma
                else:
                    inner[key] = params_std[prof][key]
            log_std[prof] = inner
        return log_std

    
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """ Fit parameters then transform to log space for storage."""
        params, params_std = super().parameterize(state, bc_dict)
        log_params = self._to_log_params(params)
        log_params_std = self._to_log_std(params, params_std)

        return log_params, log_params_std
    
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_aLy(phys_params, x_eval)
    
    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_y(phys_params, x_eval)
    
    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_curvature(phys_params, x_eval)

class LogSlopeSpline(Spline):
    """ Spline parameter model with slope parameters in log space"""

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.defined_on = "aLy"
        self.param_names = ['log_slope_' + self.defined_on + str(i) for i in range(len(self.knots))]

    def _to_physical_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Convert log-space slope parameters to physical space."""
        phys = params.copy()
        # log_slope_aLy0 = log(aLy0), then log_slope_aLy1 = log(aLy1/aLy0) = log(aLy1) - log(aLy0)
        # So we can reconstruct aLy1 = aLy0 * exp(log_slope_aLy1) = exp(log_slope_aLy0) * exp(log_slope_aLy1) = exp(log_slope_aLy0 + log_slope_aLy1)
        for key in phys:
            if key.startswith('log_slope_aLy'):
                idx = int(key.replace('log_slope_aLy', ''))
                if idx == 0:
                    phys['aLy0'] = np.exp(phys[key])
                else:
                    prev_key = f'log_slope_aLy{idx - 1}'
                    phys[f'aLy{idx}'] = np.exp(phys[key] + phys[prev_key])
        return phys
    
    def _to_log_slope_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Convert physical slope parameters to log space."""
        log_slope = {}
        for key in params:
            if key.startswith('aLy'):
                idx = int(key.replace('aLy', ''))
                if idx == 0:
                    log_slope[f'log_slope_aLy{idx}'] = np.log(params[key])
                else:
                    prev_key = f'aLy{idx - 1}'
                    if prev_key in params and params[prev_key] != 0:
                        log_slope[f'log_slope_aLy{idx}'] = np.log(params[key] / params[prev_key])
                    else:
                        log_slope[f'log_slope_aLy{idx}'] = np.log(params[key])
        return log_slope
    
    def _to_log_slope_std(self, params: Dict[str, float], params_std: Dict[str, float]) -> Dict[str, float]:
        """Convert physical slope parameter uncertainties to log space using error propagation."""
        log_slope_std = {}
        for key in params:
            if key.startswith('aLy'):
                idx = int(key.replace('aLy', ''))
                if idx == 0:
                    log_slope_std[f'log_slope_aLy{idx}'] = (params_std[key] / params[key]) if params[key] != 0 else self.sigma
                else:
                    prev_key = f'aLy{idx - 1}'
                    if prev_key in params and params[prev_key] != 0:
                        log_slope_std[f'log_slope_aLy{idx}'] = np.sqrt((params_std[key] / params[key])**2 + (params_std[prev_key] / params[prev_key])**2)
                    else:
                        log_slope_std[f'log_slope_aLy{idx}'] = (params_std[key] / params[key]) if params[key] != 0 else self.sigma
        return log_slope_std
    
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """ Fit parameters then transform slope parameters to log space for storage."""
        params, params_std = super().parameterize(state, bc_dict)
        log_slope_params = self._to_log_slope_params(params)
        # For uncertainties, we can use error propagation: σ_log_slope = σ_aLy / aLy for the first parameter, and for subsequent parameters, we can use the ratio of uncertainties.
        log_slope_params_std = self._to_log_slope_std(params, params_std)
        
        return log_slope_params, log_slope_params_std
    
    def get_y(self, params, x_eval):
        phys_params = self._to_physical_params(params)
        return super().get_y(phys_params, x_eval)
    
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space slope parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_aLy(phys_params, x_eval)
    
    def get_curvature(self, params, x_eval):
        phys_params = self._to_physical_params(params)
        return super().get_curvature(phys_params, x_eval)

class LogPolynomial(Polynomial):
    """ Polynomial parameter model with parameters in log space"""

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.defined_on = "aLy"
        self.param_names = ['log_' + self.defined_on + str(i) for i in range(self.degree + 1)]

    def _to_physical_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Convert log-space parameters to physical space."""
        phys = params.copy()
        for key in phys:
            if key.startswith('log_aLy'):
                phys[key.replace('log_', '')] = np.exp(phys[key])
        return phys
    
    def _to_log_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Convert physical parameters to log space."""
        log_params = {}
        for key in params:
            if key.startswith('aLy'):
                log_params[f'log_{key}'] = np.log(params[key]) if params[key] > 0 else -np.inf
        return log_params
    
    def _to_log_std(self, params: Dict[str, float], params_std: Dict[str, float]) -> Dict[str, float]:
        """Convert physical parameter uncertainties to log space using error propagation."""
        log_std = {}
        for key in params:
            if key.startswith('aLy'):
                log_std[f'log_{key}'] = (params_std[key] / params[key]) if params[key] > 0 else self.sigma
        return log_std

    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:  
        """ Fit parameters then transform to log space for storage."""
        params, params_std = super().parameterize(state, bc_dict)
        log_params = self._to_log_params(params)
        log_params_std = self._to_log_std(params, params_std)

        return log_params, log_params_std
    
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_aLy(phys_params, x_eval)
    
    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_y(phys_params, x_eval)

    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Override to convert log-space parameters back to physical space before evaluation."""
        phys_params = self._to_physical_params(params)
        return super().get_curvature(phys_params, x_eval)


# -------------------------
# Factory and registry
# -------------------------


PARAMETER_MODELS = {
    'spline': Spline,
    'mtanh': Mtanh,
    'spline_mtanh': SplineMtanh,
    'gaussian': Gaussian,
    'polynomial': Polynomial,
    'basis': BasisFunction,
    'log_spline': LogSpline,
    'log_slope_spline': LogSlopeSpline,
    'log_polynomial': LogPolynomial,
}


def create_parameter_model(config: Dict[str, Any]) -> ParameterBase:
    """Create a parameter model instance from config.

    Expected config format:
    {"type": "spline"|"mtanh"|"gaussian"|"log_spline"|"log_slope_spline", "kwargs": { ... model options ... }}
    """
    model_type = (config or {}).get('type', 'spline')
    kwargs = (config or {}).get('kwargs', {})
    cls = PARAMETER_MODELS.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown parameter model type: {model_type}")
    return cls(kwargs)


BCEntry = Dict[str, float]  # {'val': float, 'loc': float}

def _normalize_single_bc(val: Union[tuple, list, dict]) -> BCEntry:
    """Accept (value, loc) tuple/list or {'value':..., 'location':...}"""
    if isinstance(val, dict):
        v = float(val.get('val', 0.0))
        loc = float(val.get('loc', 1.0))
    elif isinstance(val, (tuple, list)) and len(val) == 2:
        v, loc = val
        v, loc = float(v), float(loc)
    else:
        raise ValueError("BC must be (value,location) or dict{'value','location'}")
    return {'val': v, 'loc': loc}