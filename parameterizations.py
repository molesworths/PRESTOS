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
from scipy.interpolate import interp1d, Akima1DInterpolator as akima, PchipInterpolator as pchip, CubicSpline
from tools import calc
from scipy.special import erf
from scipy.optimize import curve_fit

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
        self.predicted_profiles = options.get('predicted_profiles', [])
        self.coord = options.get('coord', 'roa').lower()
        self.include_zero_grad_at_axis = options.get('include_zero_grad_at_axis', True)
        self.bc_dict: Dict[str, List[BCEntry]] = {}
        self.params: Dict[str, np.ndarray] = {}

        if self.coord not in ('rho', 'roa'):
            raise ValueError("coord must be 'rho' or 'roa'")

    # ------------------------------
    # Abstract-like interface
    # ------------------------------
    def add_bc(self, key: str, bc: BCEntry):
        self.bc_dict.setdefault(key, []).append({"value": float(bc["value"]), "location": float(bc["location"])})

    def build_bcs(self, bc_dict: Dict[str, Any]) -> Dict[str, List[BCEntry]]:
        """
        Normalize and store BCs. Accepts:
          bc_dict = {"ne": (1e19, 1.0), "aLne": {"value": -2.0, "location": 1.0},
                     "aLne": [( -1.5, 0.95), (-2.0, 1.0 )] }
        Stored form: self.bc_dict['aLne'] = [ {'value':..., 'location':...}, {...} ]
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
        if self.include_zero_grad_at_axis:
            for prof in self.predicted_profiles:
                key = f"aL{prof}"
                entries = self.bc_dict.get(key, [])
                has_axis = any(np.isclose(e["location"], 0.0) for e in entries)
                if not has_axis:
                    # append axis zero-gradient BC (do not overwrite existing BCs)
                    self.add_bc(key, {"value": 0.0, "location": 0.0})

        return self.bc_dict

    def get_nearest_bc(self, key: str, location: float) -> Union[BCEntry, None]:
        """Return the BC entry with location nearest to the requested location."""
        entries = self.bc_dict.get(key, [])
        if not entries:
            return None
        locs = np.array([e["location"] for e in entries], dtype=float)
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
        y = self.get_y(params, x_eval)
        aLy = self.get_aLy(params, x_eval)
        curvature = self.get_curvature(params, x_eval)
        return y, aLy, curvature


# -------------------------
# Spline parameter model
# -------------------------


class SplineParameterModel(ParameterBase):
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
    include_zero_grad_at_axis : bool
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
        if self.spline_type not in ('akima', 'pchip', 'cubic'):
            raise ValueError("spline_type must be 'akima', 'pchip', or 'cubic'")
        self.param_names = [self.defined_on+str(i) for i in range(len(self.knots))]
        self.n_params_per_profile = len(self.knots)
        self._splines: Dict[str, Any] = {}
        self.a = 1.0  # Default value, will be updated in parameterize()
        self.sigma = options.get('sigma', 0.1)  # default relative uncertainty for covariance estimates

    # ------------------------------
    # Internal utilities
    # ------------------------------
    def _make_spline(self, x: np.ndarray, y: np.ndarray, prof: str):
        """Return a spline object of chosen type.
        
        If include_zero_grad_at_axis=True and x doesn't start at 0, prepend axis BC.
        """
        x_spline = np.asarray(x)
        y_spline = np.asarray(y)

        if self.include_zero_grad_at_axis and not np.isclose(x_spline[0], 0.0) and self.defined_on == 'aLy':
            x_spline = np.insert(x_spline, 0, 0.0)
            y_spline = np.insert(y_spline, 0, 0.0)
        
        # Build spline
        if self.spline_type == "akima":
            spline = akima(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "pchip":
            spline = pchip(x_spline, y_spline, extrapolate=True)
        elif self.spline_type in ("cubic", "cspline"):
            spline = CubicSpline(x_spline, y_spline, extrapolate=True)
        else:
            raise ValueError(f"Unknown spline_type: {self.spline_type}")
        
        self._splines[prof] = spline
        return spline
    
    def _get_spline(self, prof: str):
        """Retrieve a cached spline."""
        spline = self._splines.get(prof)
        if spline is None:
            raise KeyError(f"Spline for profile '{prof}' not initialized.")
        return spline

    def _integrate_aLy(self, prof: str, x_eval: np.ndarray, spl: Any, bc_value: float, bc_loc: float) -> np.ndarray:
        """
        Integrate spline of a/Ly to recover y(x) via
            dy/dx = -(aLy/a) * y

            => y(x) = y_bc * exp(-∫[bc_loc to x] (aLy/a) dx')
        
        For bc_loc = 1.0 (edge), integrating inward (decreasing x) gives positive integral.
        For bc_loc = 0.0 (axis), integrating outward (increasing x) gives negative integral.
        """

        if not hasattr(spl, "antiderivative"):
            raise TypeError(f"Spline type {type(spl)} has no .antiderivative()")

        # Antiderivative F(x) = ∫ aLy dx
        #F = spl.antiderivative()
        #F_bc = float(F(bc_loc))
        #F_eval = F(x_eval)
        
        # Integral from bc_loc to x_eval: ∫[bc_loc to x] aLy dx' = F(x) - F(bc_loc)
        #integral = np.nan_to_num(F_eval - F_bc, nan=0.0)
        
        # y(x) = y_bc * exp(-∫[bc_loc to x] (aLy/a) dx')
        # Note: aLy spline is already in units of a/Ly, so we need to divide by a
        # But since we're in normalized coords where a=1 effectively, the formula becomes:
        # y(x) = y_bc * exp(-∫[bc_loc to x] aLy dx')
        #phase = -integral / self.a  # Divide by a to convert aLy → gradient

        phase = -spl.antiderivative()(x_eval) + spl.antiderivative()(bc_loc)
        
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
            for bc in bc_entries:
                bc_loc = bc["location"]
                bc_val = bc["value"]
                
                # Find if this location already exists in data (within tolerance)
                existing_idx = np.where(np.isclose(x_data, bc_loc, atol=1e-6))[0]
                
                if len(existing_idx) > 0:
                    # Replace existing point
                    y_data[existing_idx[0]] = bc_val
                else:
                    # Insert new point in sorted order
                    insert_idx = np.searchsorted(x_data, bc_loc)
                    x_data = np.insert(x_data, insert_idx, bc_loc)
                    y_data = np.insert(y_data, insert_idx, bc_val)
            
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
            spline = self._make_spline(self.knots, vals, prof)
            
            if self.defined_on == "y":
                y = spline(x_eval)
            elif self.defined_on == "aLy":
                # Need boundary condition to integrate aLy → y
                bc = self.get_nearest_bc(prof, x_eval[-1])
                if bc is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' at x={x_eval[-1]}")
                y = self._integrate_aLy(prof, x_eval, spline, bc["value"], bc["location"])
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = y
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
            spline = self._make_spline(self.knots, vals, prof)
            if self.defined_on == "aLy":
                aLy = spline(x_eval)
            elif self.defined_on == "y":
                y = spline(x_eval)
                dy = spline.derivative(1)(x_eval)
                # aLy = -a * (dy/dx) / y
                # Avoid division by zero
                y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                aLy = -self.a * dy / y_safe
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = aLy
        self.aLy = out
        return out

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² on x_eval.
        
        When defined_on='aLy':
            y' = -(aLy/a) * y
            y'' = -(1/a) * (aLy' * y + aLy * y')
            y'' = -(y/a) * (aLy' - (aLy²/a))
        """
        out = {}
        for prof, prof_params in params.items():
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)
            spline = self._make_spline(self.knots, vals, prof)
            
            if self.defined_on == "y":
                curv = spline.derivative(2)(x_eval)
            elif self.defined_on == "aLy":
                # Get boundary condition to integrate aLy → y
                bc = self.get_nearest_bc(prof, x_eval[-1])
                if bc is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' to compute curvature")
                
                # Get aLy, aLy', and y on x_eval
                aLy_spl = spline
                aLy = aLy_spl(x_eval)
                aLy_prime = aLy_spl.derivative(1)(x_eval)
                y = self._integrate_aLy(prof, x_eval, spline, bc["value"], bc["location"])
                
                # Avoid division issues and NaN propagation
                y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                
                # y'' = -(y/a) * (aLy' - (aLy²/a))
                curv = -(y_safe / self.a) * (aLy_prime - (aLy**2) / self.a)
                
                # Clean up any remaining NaN/inf values
                curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = curv
        self.curv = out
        return out


class RbfParameterModel(ParameterBase):
    """
    Two-Gaussian curvature parameterization using a separation factor B such that δ = B * σ.

    Curvature:
        f(x) = A [ G(x; x0 - Bσ, σ) - G(x; x0 + Bσ, σ) ]

    When B <= 1, f(x) ~ 0 (the two Gaussians overlap and cancel).

    Design parameters: A, roa_center, sigma, B
    """

    def __init__(self, options: Dict[str, Any]):

        super().__init__(options)
        self.param_names = ['A', 'roa_center', 'sigma', 'B']

    # ======================================================
    # --- Gaussian basis and analytic integrals ---
    # ======================================================
    @staticmethod
    def _gaussian(x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

    @staticmethod
    def _I1(x, mu, sigma):
        u = (x - mu) / (np.sqrt(2) * sigma)
        return np.sqrt(np.pi / 2) * sigma * erf(u)

    @staticmethod
    def _I2(x, mu, sigma):
        u = (x - mu) / (np.sqrt(2) * sigma)
        return (sigma**2) * (np.sqrt(np.pi) * u * erf(u) + np.exp(-u**2))

    # ======================================================
    # --- Parameterization ---
    # ======================================================

    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Fit the curvature of each predicted profile (y_data) in PlasmaState
        to the composite Gaussian curvature model.

        Returns
        -------
        params : dict of {profile: {"A","roa_center","sigma","B"}}
        """
        if self.bounds is None:
            self.bounds = {zip(self.param_names, [(0., 1000.), (0.9, 1.0), (0.01, 0.1), (1.0, 3.0)])}

        self.build_bcs(bc_dict)
        params = {}

        x_data = getattr(state, self.coord)  # e.g., roa
        x_min, x_max = np.min(x_data), np.max(x_data)

        for prof in self.predicted_profiles:
            try:
                # ------------------------------------------------------------------
                # Compute curvature numerically from Akima spline
                # ------------------------------------------------------------------
                y_data = np.asarray(getattr(state, prof))
                spl = akima(x_data, y_data)
                y_curv = spl.derivative(2)(x_data)

                # ------------------------------------------------------------------
                # Define model for fitting
                # ------------------------------------------------------------------
                def model(x, A, roa_center, sigma, B):
                    delta = B * sigma
                    f1 = np.exp(-((x - (roa_center - delta)) ** 2) / (2 * sigma**2))
                    f2 = np.exp(-((x - (roa_center + delta)) ** 2) / (2 * sigma**2))
                    return -A * (f1 - f2) if B >= 1.0 else -A * f1

                # ------------------------------------------------------------------
                # Initial guesses (roughly from profile shape)
                # ------------------------------------------------------------------
                A0 = np.max(np.abs(y_curv)) or 1.0
                roa_center0 = x_data[np.argmin(np.abs(y_curv - np.sign(np.mean(y_curv))*A0))]
                sigma0 = 0.15 * (x_max - x_min)
                B0 = 2.0
                

                # ------------------------------------------------------------------
                # Bounds (reasonable physical limits)
                # ------------------------------------------------------------------
                bounds = (
                    [self.bounds[i][0] for i in self.param_names],   # lower
                    [self.bounds[i][1] for i in self.param_names],    # upper
                )
                p0 = [(bounds[0] + bounds[1]) / 2 for i in range(len(self.param_names))]

                # ------------------------------------------------------------------
                # Fit to curvature data
                # ------------------------------------------------------------------
                popt, _ = curve_fit(model, x_data, y_curv, p0=p0, bounds=bounds, maxfev=10000)
                A, roa_center, sigma, B = popt

                # Store fit parameters
                params[prof] = dict(zip(self.param_names, popt))

            except Exception as err:
                # Fall back to nominal initialization if fit fails
                roa_center = np.mean(x_data)
                sigma = 0.15 * (x_max - x_min)
                params[prof] = dict(zip(self.param_names, p0))
                print(f"[RbfParameter] Warning: Fit for {prof} failed ({err}); using nominal params.")

        self.params = params
        return params


    # ======================================================
    # --- Core curvature model ---
    # ======================================================
    def _curvature(self, x, A, roa_center, sigma, B):
        """Compute curvature; collapses if B<=1."""
        delta = B * sigma
        f1 = self._gaussian(x, roa_center - delta, sigma)
        f2 = self._gaussian(x, roa_center + delta, sigma)
        return -A * (f1 - f2) if B >= 1.0 else -A * f1

    def get_curvature(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for prof, p in params.items():
            out[prof] = self._curvature(x_eval, **p)
        self.curv = out
        return out

    # ======================================================
    # --- Integrations and BC enforcement ---
    # ======================================================
    def get_aLy(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for prof, p in params.items():
            bc = self.bc_dict.get(f"aL{prof}", self.bc_dict.get(prof, {"value": 0.0, "location": x_eval[-1]}))
            B = max(p["B"], 1.0)
            delta = B * p["sigma"]

            I1_1 = self._I1(x_eval, p["roa_center"] - delta, p["sigma"])
            I1_2 = self._I1(x_eval, p["roa_center"] + delta, p["sigma"])
            aLy = p["A"] * (I1_1 - I1_2)

            # Normalize to BC
            aLy -= np.interp(bc["location"], x_eval, aLy) - bc["value"]
            out[prof] = aLy
        self.aLy = out
        return out

    def get_y(self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute y(x) using analytic double-integral and algebraic BC enforcement."""
        out = {}
        for prof, p in params.items():
            bc_y = self.get_nearest_bc(prof, x_eval[-1])
            bc_aLy = self.get_nearest_bc(f"aL{prof}", x_eval[-1])
            x_b = bc_y["location"]
            y_b = bc_y["value"]
            aLy_b = bc_aLy["value"]

            B = max(p["B"], 1.0)
            delta = B * p["sigma"]
            mu1 = p["roa_center"] - delta
            mu2 = p["roa_center"] + delta
            A = p["A"]
            sigma = p["sigma"]

            # Analytic basis functions
            Phi1 = self._I1(x_eval, mu1, sigma)
            Phi2 = self._I1(x_eval, mu2, sigma)
            Psi1 = self._I2(x_eval, mu1, sigma)
            Psi2 = self._I2(x_eval, mu2, sigma)

            # Evaluate at BC location
            Phi1_b, Phi2_b = np.interp(x_b, x_eval, Phi1), np.interp(x_b, x_eval, Phi2)
            Psi1_b, Psi2_b = np.interp(x_b, x_eval, Psi1), np.interp(x_b, x_eval, Psi2)

            # Solve for constants C1, C0 algebraically
            yprime_b = -aLy_b * y_b  # in normalized a/Ly units (a=1 scaling)
            C1 = yprime_b - A * (Phi1_b - Phi2_b)
            C0 = y_b - A * (Psi1_b - Psi2_b) - C1 * x_b

            # Construct final profile
            y = A * (Psi1 - Psi2) + C1 * x_eval + C0
            out[prof] = y
        self.y = out
        return out


    def update(self, params: Dict[str, Dict[str, float]], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        self.build_bcs(bc_dict)
        curv = self.get_curvature(params, x_eval)
        aLy = self.get_aLy(params, x_eval)
        y = self.get_y(params, x_eval)
        self.dirty = False
        return y, aLy, curv
    

# -------------------------
# Other model stubs
# -------------------------


class MTanhParameterModel(ParameterBase):
    """Modified-tanh parameter model (stub)."""

    def __init__(self, options: Dict[str, Any]):
        self.options = options or {}

    def aLy(self, x_eval: np.ndarray, params: np.ndarray, state: Any = None) -> np.ndarray:
        raise NotImplementedError("MTanhParameterModel.aLy not yet implemented")

    def y(
        self,
        x_eval: np.ndarray,
        params: np.ndarray,
        y_sep: Optional[float] = None,
        state: Any = None,
    ) -> np.ndarray:
        raise NotImplementedError("MTanhParameterModel.y not yet implemented")

    def curvature(
        self,
        x_eval: np.ndarray,
        params: np.ndarray,
        y_sep: Optional[float] = None,
        state: Any = None,
    ) -> np.ndarray:
        raise NotImplementedError("MTanhParameterModel.curvature not yet implemented")
    

# -------------------------
# Factory and registry
# -------------------------


PARAMETER_MODELS = {
    'spline': SplineParameterModel,
    'mtanh': MTanhParameterModel,
    'rbf': RbfParameterModel,
}


def create_parameter_model(config: Dict[str, Any]) -> ParameterBase:
    """Create a parameter model instance from config.

    Expected config format:
    {"type": "spline"|"mtanh"|"gaussian_rbf", "kwargs": { ... model options ... }}
    """
    model_type = (config or {}).get('type', 'spline')
    kwargs = (config or {}).get('kwargs', {})
    cls = PARAMETER_MODELS.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown parameter model type: {model_type}")
    return cls(kwargs)


BCEntry = Dict[str, float]  # {"value": float, "location": float}

def _normalize_single_bc(val: Union[tuple, list, dict]) -> BCEntry:
    """Accept (value, loc) tuple/list or {'value':..., 'location':...}"""
    if isinstance(val, dict):
        v = float(val.get("value", 0.0))
        loc = float(val.get("location", 1.0))
    elif isinstance(val, (tuple, list)) and len(val) == 2:
        v, loc = val
        v, loc = float(v), float(loc)
    else:
        raise ValueError("BC must be (value,location) or dict{'value','location'}")
    return {"value": v, "location": loc}