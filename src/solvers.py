"""Concise solver framework.

This refactor provides:
 - A lightweight SolverData container with consistent fields.
 - A simplified SolverBase.run loop handling parameters → state → model/targets → residuals.
 - Optional surrogate usage, Jacobian assistance (kept for RelaxSolver), and bounds projection.
 - Removal of legacy duplicate run paths and broken temporary classes.

Design assumptions (can be revisited):
 - parameters.parameterize(state, bc_dict) returns an initial parameter vector (1D array) or dict.
 - parameters.get_y / get_aLy produce profiles sampled on state.roa.
 - transport.evaluate(state) and targets.evaluate(state) each return dict-like outputs.
 - boundary.get_boundary_conditions(state) populates boundary.bc_dict used in parameterization.
 - Residuals are built by matching predicted_profiles → target_vars mapping; if absent, intersection of keys.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import re
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pickle
from surrogates import SurrogateManager
import scipy as sp
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SolverData:
    iterations: List[int] = field(default_factory=list)
    X: List[Dict[str, Dict[str, float]]] = field(default_factory=list)  # parameter dicts
    X_std: List[Dict[str, Dict[str, float]]] = field(default_factory=list)  # parameter std dicts
    R: List[Optional[np.ndarray]] = field(default_factory=list)  # residual vector
    R_std: List[Optional[np.ndarray]] = field(default_factory=list)  # residual std vector
    R_dict: List[Optional[Dict[str, np.ndarray]]] = field(default_factory=list)  # residual blocks
    Z: List[Optional[float]] = field(default_factory=list)       # objective scalar
    Z_std: List[Optional[float]] = field(default_factory=list)       # objective std scalar
    Y: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_std: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_target: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_target_std: List[Dict[str, np.ndarray]] = field(default_factory=list)
    used_surrogate: List[bool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, i: int, X: Dict[str, Dict[str, float]], X_std: Dict[str, Dict[str, float]], R: Optional[np.ndarray], R_std: Optional[np.ndarray],
            R_dict: Optional[Dict[str, np.ndarray]], Z: Optional[float], Z_std: Optional[float], Y: Dict[str, np.ndarray],
            Y_std: Dict[str, np.ndarray], Y_target: Dict[str, np.ndarray], Y_target_std: Dict[str, np.ndarray], used_surr: bool):
        self.iterations.append(int(i))
        # Deep copy dicts to avoid mutation issues
        self.X.append(copy.deepcopy(X))
        self.X_std.append({k: copy.deepcopy(v) for k, v in (X_std or {}).items()})
        self.R.append(None if R is None else np.asarray(R).copy())
        self.R_std.append(None if R_std is None else np.asarray(R_std).copy())
        self.R_dict.append({k: np.asarray(v).copy() for k, v in (R_dict or {}).items()} if R_dict is not None else {})
        self.Z.append(None if Z is None else float(Z))
        self.Z_std.append(None if Z_std is None else float(Z_std))
        self.Y.append({k: np.asarray(v).copy() for k, v in (Y or {}).items()})
        self.Y_std.append({k: np.asarray(v).copy() for k, v in (Y_std or {}).items()})
        self.Y_target.append({k: np.asarray(v).copy() for k, v in (Y_target or {}).items()})
        self.Y_target_std.append({k: np.asarray(v).copy() for k, v in (Y_target_std or {}).items()})
        self.used_surrogate.append(bool(used_surr))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert solver history to readable DataFrame format.
        
        Flattens nested dict structures into columns for CSV export.
        """
        rows = []
        for i, iter_num in enumerate(self.iterations):
            row = {
                "iter": iter_num,
                "Z": self.Z[i],
                "Z_std": self.Z_std[i],
                "used_surrogate": self.used_surrogate[i],
            }
            
            # Flatten X parameters: profile__param -> value
            if i < len(self.X) and self.X[i]:
                for prof, params in self.X[i].items():
                    for pname, pval in params.items():
                        row[f"X_{prof}_{pname}"] = pval
                        row[f"X_std_{prof}_{pname}"] = self.X_std[i].get(prof, {}).get(pname, 0.0)
            
            # Flatten R residuals: R_0, R_1, ...
            if i < len(self.R) and self.R[i] is not None:
                for j, rval in enumerate(self.R[i]):
                    row[f"R_{j}"] = rval
                    row[f"R_std_{j}"] = self.R_std[i][j]

            # Flatten residual blocks
            if i < len(self.R_dict) and self.R_dict[i]:
                for block, arr in self.R_dict[i].items():
                    arr_flat = np.asarray(arr).flatten()
                    for j, val in enumerate(arr_flat):
                        row[f"R_{block}_{j}"] = val
            
            # Flatten Y model outputs: var__idx -> value
            if i < len(self.Y) and self.Y[i]:
                for var, arr in self.Y[i].items():
                    arr_flat = np.asarray(arr).flatten()
                    for j, val in enumerate(arr_flat):
                        row[f"model_{var}_{j}"] = val
                        row[f"model_{var}_std_{j}"] = self.Y_std[i].get(var, np.zeros_like(arr_flat))[j]
            
            # Flatten Y_target: var__idx -> value
            if i < len(self.Y_target) and self.Y_target[i]:
                for var, arr in self.Y_target[i].items():
                    arr_flat = np.asarray(arr).flatten()
                    for j, val in enumerate(arr_flat):
                        row[f"target_{var}_{j}"] = val
                        row[f"target_{var}_std_{j}"] = self.Y_target_std[i].get(var, np.zeros_like(arr_flat))[j]
            
            rows.append(row)
        
        return pd.DataFrame(rows)

    def save(self, path: str):
        """Save solver history to CSV with flattened structure."""
        self.metadata["last_saved"] = datetime.now(timezone.utc).isoformat()
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        #print(f"Saved solver history to {path}")


# ---------------------------------------------------------------------------
# Objective Functions
# ---------------------------------------------------------------------------

class ObjectiveFunction:
    def __init__(self, scale: bool = False):
        self.scale = scale

    def __call__(self, residual: np.ndarray) -> float:
        raise NotImplementedError

    def _scale(self, value: float, residual: np.ndarray) -> float:
        return value / residual.size if self.scale and residual.size else value
    
    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        """
        Delta-method variance estimate for scalar objective function g(R).
        Default uses gradient computed numerically: var(g) ≈ grad_g^T C_R grad_g.
        """
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        # numeric gradient of objective w.r.t. residual vector (finite diff)
        eps = 1e-6 * (np.maximum(1.0, np.abs(r)))
        grad = np.zeros_like(r)
        base = float(self.__call__(r))
        for i in range(r.size):
            rp = r.copy(); rm = r.copy()
            rp[i] += eps[i]; rm[i] -= eps[i]
            gp = float(self.__call__(rp)); gm = float(self.__call__(rm))
            grad[i] = (gp - gm) / (2.0 * eps[i])
        varg = float(grad.T @ (C_R @ grad))
        return varg

class SumSquares(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = float(np.sum(r * r))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        # grad_g = 2 * r (before scaling)
        grad = 2.0 * r
        if self.scale and r.size:
            grad = grad / r.size
        return float(grad.T @ (C_R @ grad))
    
    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        grad = 2.0 * r
        if self.scale and r.size:
            grad = grad / r.size
        return grad


class MeanSquares(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = float(np.mean(r * r))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = (2.0 * r) / float(N)
        if self.scale:
            grad = grad / N
        return float(grad.T @ (C_R @ grad))
    
    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = (2.0 * r) / float(N)
        if self.scale:
            grad = grad / N
        return grad


class MeanAbsolute(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = float(np.mean(np.abs(r)))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        """
        Approximate variance: grad ≈ sign(r)/N (delta-method).
        Note: MAE gradient is nondifferentiable at r==0; this is an approximation.
        """
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = np.sign(r) / float(N)
        if self.scale:
            grad = grad / N
        return float(grad.T @ (C_R @ grad))
    
    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = np.sign(r) / float(N)
        if self.scale:
            grad = grad / N
        return grad
    
class RootMeanSquareError(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = np.sqrt(float(np.mean(r * r)))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        rmse = np.sqrt(float(np.mean(r * r))) + 1e-12  # avoid div by zero
        grad = (r / (rmse * N))
        if self.scale:
            grad = grad / N
        return float(grad.T @ (C_R @ grad))
    
    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        rmse = np.sqrt(float(np.mean(r * r))) + 1e-12  # avoid div by zero
        grad = (r / (rmse * N))
        if self.scale:
            grad = grad / N
        return grad

OBJECTIVE_FUNCTIONS = {
    "sse": SumSquares,
    "sum_squares": SumSquares,
    "mse": MeanSquares,
    "mean_squares": MeanSquares,
    "mae": MeanAbsolute,
    "mean_absolute": MeanAbsolute,
    "rmse": RootMeanSquareError,
    "root_mean_square_error": RootMeanSquareError,
}

def create_objective_function(cfg: Any) -> ObjectiveFunction:
    if isinstance(cfg, str):
        cls = OBJECTIVE_FUNCTIONS.get(cfg.lower())
        if cls is None:
            raise ValueError(f"Unknown objective '{cfg}'")
        return cls()
    if isinstance(cfg, dict):
        key = str(cfg.get("type", "mse")).lower()
        scale = bool(cfg.get("scale", True))
        cls = OBJECTIVE_FUNCTIONS.get(key)
        if cls is None:
            raise ValueError(f"Unknown objective '{key}'")
        return cls(scale=scale)
    return MeanSquares()


# ---------------------------------------------------------------------------
# Solver Base
# ---------------------------------------------------------------------------

class SolverBase:
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = dict(options or {})
        self.objective = create_objective_function({
            "type": self.options.get("objective", "mse"),
            "scale": self.options.get("scale_objective", True)
        })
        self.tol = float(self.options.get("tol", 1e-6))
        self.step_size = float(self.options.get("step_size", 1e-2))
        self.min_step_size = float(self.options.get("min_step_size", 1e-3))
        self.adaptive_step = bool(self.options.get("adaptive_step", True))

        self.converged = False
        self.stalled = False
        self.iter = 0
        self.model_iter_to_stall = int(self.options.get("model_iter_to_stall", 10))
        self.Z = None               
        self._last_model_Z: Optional[float] = None
        self._last_model_iter: Optional[int] = None
        self._model_eval_nondec = 0
        self.iter_between_save = int(self.options.get("iter_between_save", 5))
        self.cwd = Path.cwd()
        self._active_sandbox = None  # single reusable clone for batched evals
        
        # Best model evaluation tracking (for stall recovery)
        self._best_model_Z: Optional[float] = None
        self._best_model_iter: Optional[int] = None
        self._best_model_X: Optional[Dict[str, Dict[str, float]]] = None
        self._best_model_Y: Optional[Dict[str, np.ndarray]] = None
        self._best_model_Y_target: Optional[Dict[str, np.ndarray]] = None
        self._best_model_R: Optional[np.ndarray] = None
        self._best_model_R_dict: Optional[Dict[str, np.ndarray]] = None
        self.R: Optional[np.ndarray] = None
        self.R_dict: Dict[str, np.ndarray] = {}

        # Mapping predicted profiles to target variables
        self.predicted_profiles = list(self.options.get("predicted_profiles", []))
        self.target_vars = list(self.options.get("target_vars", []))
        self.transport_vars = [f"{t}_neo" for t in self.target_vars] + [f"{t}_turb" for t in self.target_vars]
        self.model_vars = self.transport_vars+self.target_vars
        self.n_params_per_profile = int(self.options.get("n_params_per_profile", 0))

        # Evaluation grid, #TODO: push eval_coord through. eval_loc instead of roa_eval, etc.
        self.eval_coord = options.get('eval_coord', 'roa').lower()
        self.domain = self.options.get("domain", [0.85, 1.0])
        self.roa_eval = np.asarray(self.options.get("roa_eval", []), float)
        if self.roa_eval.size == 0:
            self.n_eval = int(self.options.get("n_eval", 32))
            self.roa_eval = np.linspace(self.domain[0], self.domain[1], self.n_eval)
        else: self.n_eval = self.roa_eval.size

        # Residual controls
        self.residual_weights = self.options.get("residual_weights", {'flux': 0.5, 
                                        'boundary_conditions': 0.25, 'constraints': 0.25})  # dict of var -> weight
        self.normalize_residual = bool(self.options.get("normalize_residual", True))
        self.residual_on_lcfs = bool(self.options.get("residual_on_lcfs", False))

        # Bounds & iteration controls
        self.bounds = self._parse_bounds(self.options.get("bounds"))
        self.bounds_dict: Dict[str, Dict[str, np.ndarray]] = self._build_bounds_dict()
        self.max_iter = int(self.options.get("max_iter", 100))

        # Surrogate configuration / datasets
        self.use_surrogate = bool(self.options.get("use_surrogate", False))
        self.surr_warmup = int(self.options.get("surrogate_warmup", 5))
        self.surr_retrain_every = int(self.options.get("surrogate_retrain_every", 10))
        self.surr_verify_on_converge = bool(self.options.get("surrogate_verify_on_converge", True))
        self._surr_trained = False
        self._use_surr_iter = False
        #self._surrogate_models = {}        # var_name / var_name__col -> model
        #self._surrogate_sigmas = {}        # var_name / var_name__col -> sigma vector

        # Current parameter vector
        self.X = None             # type: Optional[np.ndarray]
        
        
        # Module references (set during run)
        self._state = None
        self._parameters = None
        self._boundary = None
        self._neutrals = None
        self._transport = None
        self._targets = None

        # Jacobian config
        self.J = None
        self.use_jacobian = bool(self.options.get("use_jacobian", True))
        self.jacobian_reg = float(self.options.get("jacobian_reg", 1e-8))
        self.fd_epsilon = float(self.options.get("fd_epsilon", 0.05))
        self.gradient_clip = float(self.options.get("gradient_clip", 1e6))  # max norm for gradient clipping
        
        # Jacobian caching & adaptive recomputation
        self._J_cache = None              # cached Jacobian matrix
        self._J_cache_iter = -1           # iteration when last computed
        self._R_cache = None              # cached residual for change detection
        self.jacobian_cache_enabled = bool(self.options.get("jacobian_cache_enabled", True))
        self.jacobian_recompute_every = int(self.options.get("jacobian_recompute_every", 5))  # recompute every k iters
        self.jacobian_recompute_tol = float(self.options.get("jacobian_recompute_tol", 0.1))  # ||R_new - R_old|| / ||R_old|| threshold
        self.jacobian_use_surrogate_grads = bool(self.options.get("jacobian_use_surrogate_grads", True))  # use SurrogateManager.get_grads if available
        
        # Parameter scaling/preconditioning
        self.use_parameter_preconditioning = bool(self.options.get("use_parameter_preconditioning", True))
        self.param_scales: Optional[Dict[str, Dict[str, float]]] = None  # set during run
        
        # solution constraints
        self.constraints = self.options.get("constraints", None)
        self.VAR_MAP = {
            "Ti": "state.ti",
            "Te": "state.te",
            "Pi": "state.Pi",
            "Pe": "state.Pe",
            "ne": "state.ne",
            "ni": "state.ni",
        }
        self.ALIASES = {
            "T_i": "Ti",
            "T_e": "Te",
            "n_e": "ne",
            "n_i": "ni",
            "P_i": "Pi",
            "P_e": "Pe",
        }
        self.compiled_constraints: List[Dict[str, Any]] = []
        self.constraint_compilation()


    def constraint_compilation(self):
        """Compile user-defined constraints into evaluable functions."""
        if self.constraints is not None:
            for i, c in enumerate(self.constraints):

                loc = float(c["location"])
                weight = float(c.get("weight", 1.0))
                norm = c.get("norm", None)
                enforcement = c.get("enforcement", "exact")

                user_expr = c["expression"]
                expr_norm = self._normalize_expression(user_expr, self.ALIASES)

                lhs, rhs, op = self._parse_constraint(expr_norm)

                # canonical form
                if op == "=":
                    canon = f"({lhs}) - ({rhs})"
                elif op == ">=":
                    canon = f"({lhs}) - ({rhs})"
                elif op == "<=":
                    canon = f"({rhs}) - ({lhs})"

                # backend mapping
                backend = self._backend_expr(canon, self.VAR_MAP)

                # optional log normalization
                if norm == "log":
                    backend = f"np.log(np.abs({backend}) + 1e-12)"

                # compile callable
                code = compile(backend, "<constraint>", "eval")

                def _make_eval(compiled_code):
                    def _eval(state):
                        return eval(compiled_code, {"np": np}, {"state": state})
                    return _eval

                eval_fn = _make_eval(code)

                # print transparency
                print(f"[PRESTOS] Constraint {i}")
                print(f"  read:        {user_expr}")
                print(f"  normalized:  {expr_norm}")
                print(f"  enforcing:   {backend}")
                print(f"  location:    r/a = {loc}")
                print(f"  weight:      {weight}")
                print(f"  enforcement: {enforcement}")

                self.compiled_constraints.append({
                    "location": loc,
                    "weight": weight,
                    "op": op,
                    "eval": eval_fn,
                    "enforcement": enforcement,
                })


    # --------------------- helpers ---------------------
    def get_initial_parameters(self):
        _ = self._targets.evaluate(self._state)
        self._boundary.get_boundary_conditions(self._state, self._targets)
        self.X, self.X_std = self._parameters.parameterize(self._state, self._boundary.bc_dict)
        self.X = self._project_bounds(self.X)
        self._flatten_params(self.X)  # sets self.schema
        self._build_bounds_dict()
        initial_profiles_list = [
            (str(prof),getattr(self._state, prof)) for prof in self.predicted_profiles
        ] + [(f'aL{prof}', getattr(self._state, f'aL{prof}')) for prof in self.predicted_profiles]
        self._state.initial_profiles = dict(initial_profiles_list)  # store initial state for reference


    def _update_from_params(self, X: np.ndarray):
        """
        Update state from parameters and evaluate targets with a self-consistent state.

        PlasmaState.process is responsible for handling neutrals.
        """
        self._boundary.get_boundary_conditions(self._state, self._targets)
        # Delegate to parameters to reconstruct profiles
        if isinstance(X, np.ndarray):
            X = self._unflatten_params(X)
        self._parameters.update(X, self._boundary.bc_dict, self.roa_eval)
        self._state.update(X, self._parameters, neutrals=self._neutrals)
        
        # Evaluate targets with self-consistent state
        _ = self._targets.evaluate(self._state)

    # --------------------- bounds helpers (schema + BC) ---------------------
    def _build_bounds_dict(self):
        """Convert self.bounds (profile -> list of (lo,hi)) into nested dict using schema.

        Result stored in self.bounds_dict as: bounds_dict[profile][param_name] = np.array([lo, hi])
        If self.bounds is None, initialize wide bounds (-1e6,1e6).
        Requires self.schema to be defined (set by _flatten_params)."""
        if not hasattr(self, 'schema') or self.schema is None:
            return
        bd: Dict[str, Dict[str, np.ndarray]] = {}
        # Build an index counter per profile to map param order in self.bounds
        profile_counts: Dict[str, int] = {}
        for prof, pname in self.schema:
            profile_counts[prof] = profile_counts.get(prof, 0)
            idx = profile_counts[prof]
            # fetch bounds
            if self.bounds is None or prof not in self.bounds or idx >= len(self.bounds.get(prof, [])):
                lo, hi = -1e6, 1e6
            else:

                
                lo, hi = self.bounds[prof][idx]
                lo = float(lo) if np.isfinite(lo) else -1e6
                hi = float(hi) if np.isfinite(hi) else 1e6
            bd.setdefault(prof, {})[pname] = np.asarray([lo, hi], dtype=float)
            profile_counts[prof] += 1
        self.bounds_dict = bd


    def _parse_bounds(self, bounds):
        """Normalize bounds specification from config.
        
        Converts bounds into dict format: {profile: [[low, high], ...]}.
        Supports uniform bounds, per-profile bounds, or per-parameter bounds.
        """

        if bounds is None:
            return None

        bounds_dict = {}

        # --- Uniform scalar tuple ---
        if isinstance(bounds, list) and len(bounds) == 2:
            try:
                low = float(bounds[0])
                high = float(bounds[1])
                return {n: [[low, high] for i in range(self.n_params_per_profile)] for n in self.predicted_profiles}
            except Exception:
                pass

        # --- Dict for each profile with [low, high] ---
        if isinstance(bounds, dict) and all(k in bounds.keys() for k in self.predicted_profiles):
            for k in self.predicted_profiles:
                if not (isinstance(bounds[k], list) and len(bounds[k]) == 2):
                    return None
                try:
                    low = np.asarray([float(bounds[k][0]) for k in self.predicted_profiles], float)
                    high = np.asarray([float(bounds[k][1]) for k in self.predicted_profiles], float)
                    bounds_dict.update({k: [[low, high] for i in range(self.n_params_per_profile)] for k in self.predicted_profiles})
                except Exception:
                    return None
            return bounds_dict

        # --- Per-parameter list of [low,high] ---
        if isinstance(bounds, list) and all(isinstance(b, (list, tuple)) and len(b) == 2 for b in bounds) and len(bounds) == self.n_params_per_profile:
            try:
                bounds_dict = {k: [[float(a), float(b)] for [a, b] in bounds] for k in self.predicted_profiles}
                return bounds_dict
            except Exception:
                return None

        print('Unable to parse parameter bounds specification; ignoring.')
        return None

    def _project_bounds(self, X):
        """Clip parameters using bounds. """
        
        if self.bounds is None:
            return X

        if isinstance(X, dict):
            # Get cached bc_dict from previous iteration (already computed in _update_from_params)
            bc_dict = getattr(self._boundary, 'bc_dict', {}) if self._boundary is not None else {}
            
            Xc = {}
            for prof, pvals in X.items():
                if prof not in self.bounds:
                    # No bounds for this profile, keep unchanged
                    Xc[prof] = pvals
                    continue
                
                bounds_list = self.bounds[prof]  # List of (lower, upper) tuples from static config
                Xc[prof] = {}
                
                # Get parameter names in sorted order for consistent indexing
                param_names = list(pvals.keys())
                
                for idx, pname in enumerate(param_names):
                    val = pvals[pname]
                    
                    # Start with static bounds
                    if idx < len(bounds_list):
                        lo, hi = bounds_list[idx]
                    else:
                        lo, hi = -1e6, 1e6  # Wide default bounds
                    
                    # Clip parameter to final bounds
                    Xc[prof][pname] = float(np.clip(val, lo, hi))

            return Xc

        return X

    def _get_bounds_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return flat lower/upper bound arrays aligned with a schema.

        Falls back to wide bounds when bounds_dict is missing an entry.
        """
        schema = self.schema
        lo_list = []
        hi_list = []
        for prof, pname in schema:
            if self.bounds_dict and prof in self.bounds_dict and pname in self.bounds_dict[prof]:
                lo, hi = self.bounds_dict[prof][pname]
            else:
                lo, hi = -1e6, 1e6
            lo_list.append(float(lo))
            hi_list.append(float(hi))
        return np.asarray(lo_list, float), np.asarray(hi_list, float)

    def _use_surrogate_iteration(self) -> bool:
        """Check if we should use surrogate for prediction (not training).
        
        Returns False during:
        - Warmup period
        - Retraining iterations
        - When converged
        - When surrogate is disabled/unavailable
        """
        if self._surrogate is None:
            return False
        if self.use_surrogate==False:
            return False
        if self.iter < self.surr_warmup-1:
            return False
        if self.converged==True: 
            return False
        # Don't use surrogate on retraining iterations (full model evaluation)
        if (self.iter % self.surr_retrain_every) == 0:
            return False
        
        return True
    
    def _train_surrogate(self):
        """Train surrogate model if enabled and we have enough data."""
        if self._surrogate is None or self._use_surr_iter==False:
            return
        if (self.iter % self.surr_retrain_every) == 0 or (self.iter == self.surr_warmup-1):
            self._surrogate.fit()
            self._surr_trained = True
        return


    def _evaluate(self, X: Optional[Dict[str, Dict[str, float]]] = None,
                      in_place: bool = True, is_fd_sample: bool = False):
        """Evaluate transport + target (or surrogate) optionally without mutating solver attributes.

        Parameters
        ----------
        state : PlasmaState
            State instance to evaluate on (can be a copy).
        X : dict, optional
            Parameter dict associated with evaluation (used for surrogate sample when in_place).
        in_place : bool
            When True, store results on self; otherwise just return them.
        is_fd_sample : bool
            If True, don't add to surrogate training (preserves diversity vs FD perturbations).

        Returns
        -------
        Y_model, Y_model_std, Y_target, Y_target_std : tuple(dict,dict,dict,dict)
        """
        X_current = self.X if X is None else X
        if self._use_surr_iter and self._surrogate is not None:
            Y, Y_std = self._surrogate.evaluate(X_current, self._state)
            Y_model = self._surrogate.transport
            Y_model_std = getattr(self._surrogate, 'transport_std', {})
            Y_target = self._surrogate.targets
            Y_target_std = getattr(self._surrogate, 'targets_std', {})
        else:
            Y_model, Y_model_std = self._transport.evaluate(self._state)
            Y_target, Y_target_std = self._targets.evaluate(self._state)
            if in_place and self._surrogate is not None:
                self._surrogate.add_sample(self._state, X_current, Y_model, Y_target, is_fd_sample=is_fd_sample)
        if in_place:
            self.Y = Y_model
            self.Y_std = Y_model_std
            self.Y_target = Y_target
            self.Y_target_std = Y_target_std
        return Y_model, Y_model_std, Y_target, Y_target_std


    def _compute_residuals(self, Y_model: Dict[str, Any], Y_target: Dict[str, Any]) -> Optional[np.ndarray]:
        normalize = self.normalize_residual
        R_bc = np.array([], float)
        R_constraints = np.array([], float)

        R_flux_dict = {k: np.array(Y_model[k]) - np.array(Y_target[k]) \
                  for k in Y_target.keys() if k in Y_model.keys()}
        R_flux = np.concatenate([R_flux_dict[k] for k in sorted(R_flux_dict.keys())])
        Y_target_concat = np.concatenate([np.array(Y_target[k]) for k in Y_target.keys() if k in Y_model])
        if normalize:
            R_flux = R_flux / (np.abs(Y_target_concat) + 1e-8)
        if not self.residual_on_lcfs:
            idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
            if idx.size == 1:
                n = len(self.roa_eval)
                k = int(idx[0])
                nblocks = R_flux.size // n if n > 0 else 0
                # Zero the LCFS residual element for each variable block
                # for b in range(nblocks):
                #     R_flux[b * n + k] = 0.0

                # Drop the LCFS residual element for each variable block
                R_flux = np.concatenate([R_flux[b * n : b * n + k] for b in range(nblocks)] +
                                  [R_flux[b * n + k + 1 : (b + 1) * n] for b in range(nblocks)])
                
        # Apply flux residual weighting  
        R = R_flux * self.residual_weights.get('flux',0.5)

        # ---------------------------------------------------------
        # Boundary condition residuals (optional block)
        # ---------------------------------------------------------
        bc_residuals = []   
        bc_dict = getattr(self._boundary, "bc_dict", {}) if hasattr(self, "_boundary") else {}
        if bc_dict and self.residual_weights.get('boundary_conditions',0.0) != 0.0:
            x_arr = np.asarray(self._state.roa, float).ravel()
            for name, entry in bc_dict.items():
                # Entry can be dict-like {"val": v, "loc": l} or a list/tuple [v, l]
                if isinstance(entry, dict):
                    target_val = entry.get("val")
                    target_loc = entry.get("loc")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    target_val, target_loc = entry[0], entry[1]
                else:
                    continue

                if target_val is None or target_loc is None:
                    continue
                if not hasattr(self._state, name):
                    continue
                if not any(name in i or f'aL{i}' in name for i in self.predicted_profiles):
                    continue

                y_arr = np.asarray(getattr(self._state,name), float).ravel()
                if y_arr.size != x_arr.size or y_arr.size == 0:
                    continue

                pred_val = float(np.interp(float(target_loc), x_arr, y_arr))
                denom = abs(float(target_val)) + 1e-8 if normalize else 1.0
                resid = (pred_val - float(target_val)) / denom
                
                bc_residuals.append(resid * self.residual_weights.get('boundary_conditions',0.25))

        R_bc = np.asarray(bc_residuals, float)
        if bc_residuals:
            R = np.concatenate([R, R_bc])

        # ---------------------------------------------------------
        # Constraint residuals
        # ---------------------------------------------------------
        if hasattr(self, "compiled_constraints") and self.compiled_constraints:

            R_constraints = []

            for c in self.compiled_constraints:

                # nearest radial index
                k = int(np.argmin(np.abs(self.roa_eval - c["location"])))

                try:
                    val = c["eval"](self.state)
                except Exception:
                    val = np.nan

                # inequality handling (hinge)
                if c["op"] in (">=", "<="):
                    val = min(0.0, val)

                # ramp enforcement
                if c["enforcement"] == "ramp":
                    ramp = min(1.0, self.iter / max(1, self.max_iter // 5))
                    val *= ramp

                R_constraints.append(np.sqrt(c["weight"]) * val)

            if R_constraints:
                R_constraints = np.array(R_constraints) * self.residual_weights.get('constraints',0.25)
                R = np.concatenate([R, R_constraints])

        self.R = np.nan_to_num(R, nan=1e6, posinf=1e6, neginf=-1e6)
        self.R_dict = {'flux': R_flux, 'boundary_conditions': R_bc, 'constraints': R_constraints}

        return R
    
    # --------------------- constraint helpers ---------------------
    def _normalize_expression(self, expr: str, aliases: Dict[str, str]) -> str:
        expr_norm = expr
        for k, v in aliases.items():
            expr_norm = re.sub(rf"\b{k}\b", v, expr_norm)
        return expr_norm
    
    def _parse_constraint(self, expr: str):
        if ">=" in expr:
            lhs, rhs = expr.split(">=")
            return lhs.strip(), rhs.strip(), ">="
        if "<=" in expr:
            lhs, rhs = expr.split("<=")
            return lhs.strip(), rhs.strip(), "<="
        if "=" in expr:
            lhs, rhs = expr.split("=")
            return lhs.strip(), rhs.strip(), "="
        raise ValueError(f"Unsupported constraint expression: {expr}")

    def _backend_expr(self, expr: str, var_map: Dict[str, str]) -> str:
        out = expr
        for k, v in var_map.items():
            out = re.sub(rf"\b{k}\b", v, out)
        return out
    
    # --------------------- Parameter dict helpers ---------------------
    def _flatten_params(self, X_dict: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Convert dict-of-dicts parameters to flat array with schema.
        
        Returns
        -------
        X_flat : np.ndarray
            Flattened parameter vector
        schema : List[Tuple[str, str]]
            List of (profile, param_name) tuples defining the order
        """
        if not hasattr(self, 'schema'):
            schema = []
            values = []
            for prof in list(X_dict.keys()):
                param_dict = X_dict[prof]
                for pname in list(param_dict.keys()):
                    schema.append((prof, pname))
                    values.append(float(param_dict[pname]))
            self.schema = schema
        else:
            schema = self.schema
            values = []
            for prof, pname in schema:
                values.append(float(X_dict[prof][pname]))

        return np.array(values, dtype=float), schema
    
    def _unflatten_params(self, X_flat: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Convert flat array back to dict-of-dicts using schema.
        
        Parameters
        ----------
        X_flat : np.ndarray
            Flattened parameter vector

        Returns
        -------
        X_dict : Dict[str, Dict[str, float]]
            Reconstructed dict-of-dicts parameters
        """
        X_dict = {}
        for i, (prof, pname) in enumerate(self.schema):
            if prof not in X_dict:
                X_dict[prof] = {}
            X_dict[prof][pname] = float(X_flat[i])
        return X_dict
    
    # --------------------- Jacobian helpers ---------------------
    def _objective_at(self, X: Dict[str, Dict[str, float]]) -> float:
        """Compute objective at params X on the current solver object.

        When called inside `with self.sandbox():`, this operates on the sandboxed
        clone and does not affect the main solver. Outside a sandbox, it will
        update and use the main solver instance.
        """
        solver = self._active_sandbox if self._active_sandbox is not None else self
        solver._update_from_params(X)
        Y_model, _, Y_target, _ = solver._evaluate(X, in_place=False)
        R = solver._compute_residuals(Y_model, Y_target)
        if R is None or R.size == 0:
            return float("inf")
        return solver.objective(R)

    @contextmanager
    def sandbox(self):
        """Context manager to reuse a single cloned solver for batched evaluations."""
        if self._active_sandbox is not None:
            yield self._active_sandbox
            return
        clone = copy.deepcopy(self)
        clone._active_sandbox = None
        clone._state = copy.deepcopy(self._state)
        self._active_sandbox = clone
        try:
            yield clone
        finally:
            self._active_sandbox = None
                # Rp = sb2._compute_residuals(Ymp, Ytp)

    def _attempt_get_jacobian(self, X: np.ndarray, R: np.ndarray):
        """Compute Jacobian with caching and staged fallbacks.
        
        Priority order:
        1. Return cached J if recompute check passes
        2. Surrogate analytic gradients (if available)
        3. FD on surrogate (if requested and trained)
        4. FD on full model
        """
        surrogate = self._surrogate if (self._use_surr_iter and self._surrogate is not None) else None

        # Handle dict-of-dicts parameters: flatten for Jacobian computation
        if isinstance(X, dict):
            X_flat, _ = self._flatten_params(X)
            X_dict = X  # Keep original for reconstruction
        else:
            X_flat = np.asarray(X, float)
            X_dict = self._unflatten_params(X)
        
        # Check if we should reuse cached Jacobian
        if self.jacobian_cache_enabled and self._J_cache is not None:
            should_recompute = self._should_recompute_jacobian(R)
            if not should_recompute:
                return self._J_cache
        
        J = None

        # Priority 1: Surrogate gradients (analytic, fastest if available)
        if (self.jacobian_use_surrogate_grads and surrogate is not None and 
            surrogate.trained):
            try:
                J = self._jacobian_from_surrogate_grads(X_flat, X_dict, surrogate)
            except Exception:
                J = None
        
        # Priority 2: Finite difference using surrogate evaluations (if trained)
        if J is None and surrogate is not None and surrogate.trained:
            J = self._fd_jacobian(X_flat, R)

        # Priority 3: Finite difference on the full, expensive models
        if J is None:
            J = self._fd_jacobian(X_flat, R)
        
        # Cache the result
        if J is not None and self.jacobian_cache_enabled:
            self._J_cache = J
            self._J_cache_iter = self.iter
            self._R_cache = R.copy() if R is not None else None
        
        return J

    def _jacobian_from_surrogate_grads(self, X_flat: np.ndarray, X_dict: Dict[str, Dict[str, float]], surrogate) -> Optional[np.ndarray]:
        """Extract Jacobian directly from surrogate.get_grads() without finite differences.
        
        This is the fastest path when surrogate is trained. Maps surrogate output
        gradients (w.r.t. features) to residual gradients (w.r.t. parameters).
        
        Returns None if surrogate lacks get_grads or gradient assembly fails.
        """
        try:
            if not hasattr(surrogate, 'get_grads'):
                return None
            
            # Get surrogate gradients: transport_grads, target_grads
            transport_grads, target_grads = surrogate.get_grads(X_dict, self._state)
            
            if transport_grads is None or target_grads is None:
                return None
            
            # Determine residual ordering (same as _compute_residuals)
            y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
            if not y_keys:
                return None
            
            # Build Jacobian row by row (one row per residual element)
            J_rows = []
            for var_key in y_keys:
                # Get gradient for this variable
                grad_dict = transport_grads if var_key in transport_grads else target_grads
                if var_key not in grad_dict:
                    continue
                
                grad_var = np.asarray(grad_dict[var_key], dtype=float)  # shape (n_roa, n_features)
                if grad_var.ndim == 3:
                    grad_var = grad_var[0, :, :]  # take first batch if batched
                
                # Map each roa point's gradient to Jacobian rows
                for roa_idx in range(grad_var.shape[0]):
                    grad_features = grad_var[roa_idx, :]  # (n_features,)
                    
                    # Map feature gradients to parameter gradients
                    # features = state_features + param_features + [roa]
                    # We need gradients w.r.t. param_features only
                    if hasattr(surrogate, 'param_features') and surrogate.param_features:
                        param_indices = [surrogate.all_features.index(pf) for pf in surrogate.param_features 
                                        if pf in surrogate.all_features]
                        grad_params = grad_features[param_indices]
                    else:
                        grad_params = grad_features
                    
                    J_rows.append(grad_params)
            
            if not J_rows:
                return None
            
            J = np.vstack(J_rows).astype(float)
            return J if J.shape[1] == X_flat.size else None
            
        except Exception:
            # Silently return None; caller will fall back to FD
            return None

    def _should_recompute_jacobian(self, R: Optional[np.ndarray]) -> bool:
        """Check if cached Jacobian should be recomputed based on residual change.
        
        Returns True (recompute) if:
        - No cache exists
        - Enough iterations have passed (jacobian_recompute_every)
        - Residual has changed significantly (jacobian_recompute_tol)
        """
        if self._J_cache is None or self._R_cache is None:
            return True
        
        # Check iteration count
        iters_since_cache = self.iter - self._J_cache_iter
        if iters_since_cache >= self.jacobian_recompute_every:
            return True
        
        # Check residual change
        if R is None:
            return True
        
        R_new = np.asarray(R, dtype=float)
        R_old = np.asarray(self._R_cache, dtype=float)
        
        if R_old.size != R_new.size:
            return True
        
        R_norm_old = np.linalg.norm(R_old)
        if R_norm_old < 1e-14:
            return iters_since_cache > 0  # Always recompute if norm is essentially zero
        
        rel_change = np.linalg.norm(R_new - R_old) / R_norm_old
        if rel_change > self.jacobian_recompute_tol:
            return True
        
        return False
    
    def _fd_jacobian(self, X: np.ndarray, R: np.ndarray):
        """Finite-difference Jacobian with bound projection and adaptive step sizing."""
        X_flat = np.asarray(X, float)
        m = R.size
        n = X_flat.size
        if m == 0 or n == 0:
            return None
        
        J = np.zeros((m, n), float)
        lo_arr, hi_arr = self._get_bounds_vectors()
        X_base_dict = self._unflatten_params(X_flat)

        def _project_flat(x_flat: np.ndarray) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
            x_dict = self._unflatten_params(x_flat)
            x_proj = self._project_bounds(x_dict)
            x_proj_flat, _ = self._flatten_params(x_proj)
            return x_proj_flat, x_proj

        def _eval_residual(x_flat_eval: np.ndarray) -> Optional[np.ndarray]:
            try:
                self._update_from_params(x_flat_eval)
                self._evaluate(in_place=True, is_fd_sample=True)
                return self._compute_residuals(self.Y, self.Y_target)
            except Exception:
                return None

        R_base = np.asarray(R, float) if R is not None else _eval_residual(X_flat)
        if R_base is None or not np.all(np.isfinite(R_base)):
            return None

        for j in range(n):
            span = hi_arr[j] - lo_arr[j]
            scale_j = max(abs(X_flat[j]), 0.5 * span if np.isfinite(span) and span > 0 else 0.0, 1.0)
            h = self.fd_epsilon * scale_j

            Xp = X_flat.copy(); Xm = X_flat.copy()
            Xp[j] += h; Xm[j] -= h

            Xp_clip, _ = _project_flat(Xp)
            Xm_clip, _ = _project_flat(Xm)

            eps_eff = Xp_clip[j] - Xm_clip[j]
            eps_floor = 1e-8 * max(scale_j, 1.0)

            Rp = _eval_residual(Xp_clip) if np.any(Xp_clip != X_flat) else None
            Rm = _eval_residual(Xm_clip) if np.any(Xm_clip != X_flat) else None

            if Rp is not None and Rm is not None and abs(eps_eff) >= eps_floor:
                J[:, j] = (Rp - Rm) / eps_eff
                continue

            if Rp is not None:
                step = Xp_clip[j] - X_flat[j]
                if abs(step) >= eps_floor:
                    J[:, j] = (Rp - R_base) / step
                    continue

            if Rm is not None:
                step = X_flat[j] - Xm_clip[j]
                if abs(step) >= eps_floor:
                    J[:, j] = (R_base - Rm) / step
                    continue

            # If all evaluations failed or step collapsed, leave column as zeros

        # Restore state to baseline X
        self._update_from_params(X_base_dict)

        return J

    # --------------------- Main methods to override ---------------------
    def propose_parameters(self) -> np.ndarray:
        # Default: keep current (no change)
        return self.X.copy()
    
    def check_convergence(self, y_model, y_target):
        """
        Compute residuals, objective Z, and objective variance varZ.
        Declare converged if upper confidence bound on Z is below tol:
            Z + k_sigma * sqrt(varZ) < tol
        where k_sigma corresponds to self.options['convergence_confidence'] (default 0.95 -> 1.96).
        """
        _ = self._compute_residuals(y_model, y_target)
        self.R_std = np.zeros_like(self.R) # placeholder
        if self.R is None:
            self.Z = float("inf")
            self.converged = False
            return

        # Objective value
        self.Z = self.objective(self.R)
        self.Z_std = np.zeros_like(self.Z) # placeholder

        # initial check without uncertainty
        self.converged = bool(self.Z < self.tol)

        if self.converged:

            # Build residual covariance (tries Jacobian linearization if param sigmas present)
            X_dict = self.X if isinstance(self.X, dict) else None
            C_R = self._build_residual_cov(self.R, X_dict, use_jacobian=self.use_jacobian)
            
            # Propagate parameter uncertainty to model output variables (uses cached Jacobian from _build_residual_cov)
            if self.J is not None:
                self._propagate_param_uncertainty_to_outputs(self.J)

            # If residuals were normalized (division by Y_target), propagate that transform:
            if self.normalize_residual and C_R is not None:
                # determine Y_target concatenation ordering (same used in _compute_residuals)
                try:
                    y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
                    Yt_vec = np.concatenate([np.asarray(self.Y_target[k]).ravel() for k in y_keys]) if y_keys else np.ones(self.R.size)
                    Dinv = np.diag(1.0 / (np.abs(Yt_vec) + 1e-12))
                    C_R = Dinv @ C_R @ Dinv
                except Exception:
                    # if something goes wrong, leave C_R as is
                    pass

            # Objective variance via objective-specific method
            varZ = float(self.objective.variance(self.R, C_R)) if C_R is not None else 0.0
            varZ = max(0.0, varZ)

            # Decide threshold based on confidence interval
            confidence = float(self.options.get("convergence_confidence", 0.9))
            # clamp
            confidence = max(0.5, min(confidence, 0.9999))
            # two-sided normal quantile -> one-sided multiplier
            # approximate z for two-sided interval: z = norm.ppf((1+confidence)/2)
            k_sigma = float(sp.stats.norm.ppf((1.0 + confidence) / 2.0))

            # Upper confidence bound
            ub = self.Z + k_sigma * (np.sqrt(varZ) if varZ > 0 else 0.0)

            # convergence decision
            self.converged = bool(ub < self.tol)

            # store uncertainty metrics for diagnostics
            self.Z_std = np.sqrt(varZ)
            self.R_std = np.sqrt(np.diag(C_R))

    def save(self, bundle, filename="solver_checkpoint.pkl"):
        """Save all key objects from current solver run into one pickle file."""

        # Save solver history as CSV
        bundle['data'].save(self.cwd / "solver_history.csv")

        # Build a lightweight, serializable spec to reconstruct modules and solver state

        def _class_path(obj):
            return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

        def _to_basic(x):
            if x is None or isinstance(x, (bool, int, float, str)):
                return x
            if isinstance(x, (list, tuple, set)):
                return [_to_basic(v) for v in x]
            if isinstance(x, dict):
                return {k: _to_basic(v) for k, v in x.items()}
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer, np.floating)):
                return x.item()
            if isinstance(x, Path):
                return str(x)
            return repr(x)

        module_specs = {}
        for name in ("solver", "state", "surrogate", "transport", "targets", "boundary",
                 "parameters", "neutrals"):
            obj = bundle.get(name) if name != "solver" else self
            if obj is None:
                continue
            # Serialize only public (non "_" prefixed) attributes
            public_attrs = {}
            for attr, val in vars(obj).items():
                if attr.startswith("_"):
                    continue
                # Skip obviously callable or class objects
                if callable(val):
                    continue
                try:
                    public_attrs[attr] = (_to_basic(val), type(val).__name__)
                except Exception:
                    public_attrs[attr] = repr(val)

            module_specs[name] = {
                "class_path": _class_path(obj),
                "attributes": public_attrs,
            }

        module_specs['timestamp'] = datetime.now(timezone.utc).isoformat()

        with open(self.cwd / filename, "wb") as fh:
            pickle.dump(module_specs, fh)

    
    # --------------------- uncertainty helpers ---------------------

    def _propagate_param_uncertainty_to_outputs(self, J_R: Optional[np.ndarray] = None) -> None:
        """
        Propagate parameter uncertainty to model output variables via Jacobian linearization.
        
        Updates self.Y_std by adding parameter contribution: var_param = J_Y @ Cx @ J_Y^T
        This is called from check_convergence to populate Y_std with parameter uncertainty
        without requiring repeated Jacobian computations.
        
        Parameters
        ----------
        J_R : np.ndarray, optional
            Jacobian of residuals w.r.t. parameters (nR x nX). If None, uses cached self.J.
            The flux portion of J_R corresponds to dY_model/dX.
        """
        if not self.use_jacobian:
            return
        
        # Get Jacobian (use provided or cached)
        J = J_R if J_R is not None else self.J
        if J is None:
            return
        
        # Get parameter std flattened
        try:
            if self._parameters is None or not hasattr(self._parameters, 'param_std'):
                return
            sigma_X_dict = self._parameters.param_std
            if not sigma_X_dict:
                return
            
            sigma_X, _ = self._flatten_params(sigma_X_dict)
            if sigma_X is None or sigma_X.size == 0:
                return
        except Exception:
            return
        
        # Build parameter covariance matrix
        Cx = np.diag(sigma_X**2)
        
        # Extract flux block of Jacobian and propagate to outputs
        try:
            y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
            
            row_idx = 0
            for k in y_keys:
                y_array = np.asarray(self.Y[k], dtype=float)
                n_elements = y_array.size
                
                # Extract Jacobian block for this output variable
                if row_idx + n_elements > J.shape[0]:
                    break
                J_Y_k = J[row_idx:row_idx + n_elements, :]
                
                # Propagate: Cov_param = J_Y @ Cx @ J_Y^T, then extract diagonal for std
                try:
                    cov_k = J_Y_k @ Cx @ J_Y_k.T
                    std_param_k = np.sqrt(np.maximum(0.0, np.diag(cov_k)))
                except Exception:
                    # If covariance computation fails, skip this variable
                    row_idx += n_elements
                    continue
                
                # Add parameter contribution to existing model uncertainty
                y_std_existing = np.asarray(self.Y_std.get(k, np.zeros_like(y_array)), dtype=float).ravel()
                y_std_combined = np.sqrt(y_std_existing**2 + std_param_k**2)
                
                # Store back with original shape
                self.Y_std[k] = y_std_combined.reshape(y_array.shape)
                
                row_idx += n_elements
        except Exception:
            # If anything fails, just leave Y_std as is
            pass

    def _build_residual_cov(self,
                            R: np.ndarray,
                            X_dict: Optional[Dict[str, Dict[str, float]]],
                            use_jacobian: bool = True) -> Optional[np.ndarray]:
        """
        Build covariance matrix C_R for residual vector R.

        Sources:
          - parameter uncertainty via linearization: J Cx J^T
          - model uncertainty (self.Y_std): per-output std returned from transport/surrogate
          - target uncertainty (self.Y_target_std)
        Returns dense covariance matrix (nR x nR) or None when nothing available.
        """
        # Build measurement/model covariance C_meas from all residual blocks
        # R consists of: flux + boundary_conditions + constraints (from self.R_dict keys)
        try:
            sigma_list = []
            
            # --- Flux block: model + target uncertainty ---
            if 'flux' in (self.R_dict or {}):
                y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
                model_sig_list = []
                target_sig_list = []
                for k in y_keys:
                    mstd = np.asarray(self.Y_std.get(k, np.zeros_like(self.Y[k])), dtype=float)
                    tstd = np.asarray(self.Y_target_std.get(k, np.zeros_like(self.Y_target[k])), dtype=float)
                    model_sig_list.append(mstd.ravel())
                    target_sig_list.append(tstd.ravel())
                sigma_model_vec = np.concatenate(model_sig_list) if model_sig_list else np.empty(0)
                sigma_target_vec = np.concatenate(target_sig_list) if target_sig_list else np.empty(0)
                sigma_flux = np.sqrt(np.maximum(0.0, sigma_model_vec**2 + sigma_target_vec**2))
                if not self.residual_on_lcfs:
                    idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
                    if idx.size == 1:
                        n = len(self.roa_eval)
                        k = int(idx[0])
                        nblocks = len(sigma_flux) // n if n > 0 else 0
                        # Drop the LCFS sigma element for each variable block
                        sigma_flux = np.concatenate([sigma_flux[b * n : b * n + k] for b in range(nblocks)] +
                                                  [sigma_flux[b * n + k + 1 : (b + 1) * n] for b in range(nblocks)])
                sigma_list.append(sigma_flux)
            
            # --- Boundary conditions block: fixed sigma from self._boundary ---
            if 'boundary_conditions' in (self.R_dict or {}):
                bc_size = np.asarray(self.R_dict['boundary_conditions']).size
                bc_sigma = float(getattr(self._boundary, 'sigma', 0.01)) if hasattr(self, '_boundary') else 0.01
                sigma_bc = np.full(bc_size, bc_sigma, dtype=float)
                sigma_list.append(sigma_bc)
            
            # --- Constraints block: sigma from each compiled constraint ---
            if 'constraints' in (self.R_dict or {}):
                constraints_size = np.asarray(self.R_dict['constraints']).size
                if constraints_size > 0 and hasattr(self, 'compiled_constraints'):
                    sigma_constraints = []
                    for c in self.compiled_constraints:
                        # Get sigma from constraint dict or use default
                        c_sigma = float(c.get('sigma', 0.01)) if isinstance(c, dict) else 0.01
                        sigma_constraints.append(c_sigma)
                    # Replicate to match constraints_size if needed
                    sigma_constraints_arr = np.array(sigma_constraints)
                    if sigma_constraints_arr.size < constraints_size:
                        # Pad or replicate if fewer sigmas than constraints
                        sigma_constraints_arr = np.tile(sigma_constraints_arr, 
                                                         (constraints_size // sigma_constraints_arr.size) + 1)[:constraints_size]
                    sigma_constraints_vec = sigma_constraints_arr[:constraints_size]
                else:
                    sigma_constraints_vec = np.full(constraints_size, 0.01, dtype=float)
                sigma_list.append(sigma_constraints_vec)
            
            # Concatenate all sigma vectors and build diagonal covariance
            if sigma_list:
                sigma_resid = np.concatenate(sigma_list)
                C_meas = np.diag(sigma_resid**2) if sigma_resid.size > 0 else None
            else:
                C_meas = None
        except Exception:
            C_meas = None
        self.C_meas = C_meas

        # Parameter contribution via Jacobian * Cx * J^T
        C_param = None
        if use_jacobian and X_dict is not None:
            # try to compute Jacobian (this will attempt training surrogate / FD as configured)
            try:
                X_flat, schema = self._flatten_params(X_dict)
                J = self._attempt_get_jacobian(X_flat, R)
                if J is not None:
                    self.J = J
                    # get sigma_X diag
                    sigma_X, _ = self._flatten_params(self._parameters.param_std)
                    if sigma_X is not None and sigma_X.size == J.shape[1]:
                        # build diag Cx
                        Cx = np.diag(sigma_X**2)
                        C_param = J @ Cx @ J.T
            except Exception:
                C_param = None
        
        # Combine
        if C_meas is None and C_param is None:
            return None
        if C_meas is None:
            C_total = C_param
        if C_param is None:
            C_total = C_meas
        else:
            C_total = C_param + C_meas

        # Zero out where R is zero (no residual, no uncertainty)
        if not self.residual_on_lcfs:
            idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
            if idx.size == 1:
                n = len(self.roa_eval)
                k = int(idx[0])
                # Zero the LCFS residual element for each variable block
                nblocks = C_total.shape[0] // n if n > 0 else 0
                C_total[:, [b * n + k for b in range(nblocks)]] = 0.0
                C_total[[b * n + k for b in range(nblocks)], :] = 0.0

        # Compute posterior parameter uncertainty (may populate self.J as side effect)
        _ = self._compute_posterior_parameter_uncertainty()

        return C_total
    
    def _compute_posterior_parameter_uncertainty(self):
        """Compute posterior parameter covariance via linearized Bayesian update.
        
        Formula:
            Cx_post = (J^T (C_meas)^-1 J + Cx_prior^-1)^-1
        where C_meas includes model and target uncertainty.
        Using Moore-Penrose pseudoinverses for non-square matrices.
        Returns Cx_post or None if not computable.
        """
        if self.J is None and self.use_jacobian:
            self.J = self._attempt_get_jacobian(
                self.X if isinstance(self.X, dict) else self.X,
                self.R
            )
        else: return None
        # Posterior covariance
        try:
            Cx_post = sp.linalg.pinv(self.J) @ self.C_meas @ sp.linalg.pinv(self.J).T #+ Cx_prior_inv
            return Cx_post
        except np.linalg.LinAlgError:
            return None


    def check_stalled(self) -> None:
        """Set self.stalled when model Z shows < self.tol improvement over model_iter_to_stall consecutive evaluations.
        
        Special case: model_iter_to_stall=0 means stall immediately on any non-improvement.
        """
        if self._use_surr_iter or self.Z is None or self.iter < self.surr_warmup:
            return

        # Skip stall check if counter disabled
        if self.model_iter_to_stall is None or self.model_iter_to_stall < 0:
            return

        if self._last_model_Z is not None:
            delta = self._last_model_Z - float(self.Z)  # > 0 means improvement
            if delta < self.tol:
                self._model_eval_nondec += 1
                # Trigger stall if either:
                # - model_iter_to_stall=0 and any non-improvement
                # - model_iter_to_stall>0 and threshold met
                if self.model_iter_to_stall == 0 or self._model_eval_nondec >= self.model_iter_to_stall:
                    self.stalled = True
            else:
                # Reset counter on improvement
                self._model_eval_nondec = 0
        else:
            # First evaluation: initialize
            self._model_eval_nondec = 0

        self._last_model_Z = float(self.Z)
        self._last_model_iter = self.iter


    # --------------------- main loop ---------------------
    def run(self, state, boundary, parameters, neutrals, transport, targets, surrogate=None) -> SolverData:

        # migrate info to modules
        state.domain = self.domain
        state._trim()
        transport.roa_eval = self.roa_eval
        transport.output_vars = self.transport_vars + self.target_vars
        targets.roa_eval = self.roa_eval
        targets.output_vars = self.target_vars
        parameters.predicted_profiles = self.predicted_profiles
        parameters.bounds = self.bounds
        parameters.domain = self.domain
        parameters.roa_eval = self.roa_eval
        data = SolverData()

        # Store module references for Jacobian calls
        self._state = state
        self._parameters = parameters
        self._boundary = boundary
        self._neutrals = neutrals
        self._transport = transport
        self._targets = targets
        self._surrogate = surrogate

        # initialize parameter vector
        self.get_initial_parameters()

        # Build surrogate models
        if surrogate is not None:
            surrogate._initialize(self.transport_vars, self.target_vars, self.roa_eval, state, self.X)

        module_bundle = {
            "solver_options": self.options,
            "data": data,
            "state": state,
            "surrogate": surrogate,
            "transport": transport,
            "targets": targets,
            "boundary": boundary,
            "parameters": parameters,
            "neutrals": neutrals,
        }

        while not self.converged and not self.stalled and self.iter <= self.max_iter:
            # apply parameters → state
            X_current, it = self.X, self.iter
            if it > 0: self._update_from_params(X_current)

            # On retraining iterations: run full model and add sample for training
            # On other iterations: use surrogate for prediction if available
            self._use_surr_iter = self._use_surrogate_iteration()
            self._train_surrogate()
            self._evaluate(in_place=True)
        
            # convergence
            self.check_convergence(self.Y, self.Y_target)
            
            # Track best model evaluation (not surrogate)
            if not self._use_surr_iter and self.Z is not None and np.isfinite(self.Z):
                if self._best_model_Z is None or self.Z < self._best_model_Z:
                    self._best_model_Z = float(self.Z)
                    self._best_model_iter = self.iter
                    self._best_model_X = copy.deepcopy(self.X)
                    self._best_model_Y = {k: np.asarray(v).copy() for k, v in self.Y.items()}
                    self._best_model_Y_target = {k: np.asarray(v).copy() for k, v in self.Y_target.items()}
                    self._best_model_R = np.asarray(self.R).copy() if self.R is not None else None
                    self._best_model_R_dict = {k: np.asarray(v).copy() for k, v in (self.R_dict or {}).items()}
            
            self.check_stalled()

            if self.converged or self.stalled:
                if self._use_surr_iter and self.surr_verify_on_converge:
                    # verify on full model
                    self._use_surr_iter = False
                    self._evaluate(in_place=True)
                    self.check_convergence(self.Y, self.Y_target)
                    if self.converged or (self.Z >= data.Z[-1]):
                        self.stalled = True
                
                if self.converged:
                    print('Convergence achieved. Solver run complete.')
                elif self.stalled:
                    print('Convergence stalled. Solver run complete.')
            else:
                # iteration count check
                if self.iter < self.max_iter: 
                    if self.iter % self.iter_between_save == 0: self.save(module_bundle)
                else:
                    if self._use_surr_iter:
                        # final evaluation on full model
                        self._use_surr_iter = False
                        self._evaluate(in_place=True)
                        self.check_convergence(self.Y, self.Y_target)
                    print('Max iterations reached. Solver run complete.')

            # Restore best model evaluation if stalled or max_iter reached
            if self.stalled and self._best_model_X is not None:
                if self._best_model_Z < self.Z:
                    print(f'Restoring best model evaluation from iteration {self._best_model_iter} (Z={self._best_model_Z:.6e} < current Z={self.Z:.6e})')
                    self.X = copy.deepcopy(self._best_model_X)
                    self.Y = {k: v.copy() for k, v in self._best_model_Y.items()}
                    self.Y_target = {k: v.copy() for k, v in self._best_model_Y_target.items()}
                    self.R = self._best_model_R.copy() if self._best_model_R is not None else None
                    self.R_dict = {k: np.asarray(v).copy() for k, v in (self._best_model_R_dict or {}).items()}
                    self.Z = self._best_model_Z
                    self._update_from_params(self.X)

            data.add(self.iter, self.X, self.X_std, self.R, self.R_std, self.R_dict, \
                     self.Z, self.Z_std, self.Y, self.Y_std, self.Y_target, self.Y_target_std, self._use_surr_iter)
            self.save(module_bundle)

            # propose new parameters
            X, X_std = self.propose_parameters()
            self.X = self._project_bounds(X)
            self.X_std = X_std

            self.iter += 1

        return data

# ---------------------------------------------------------------------------
# Derived Solvers
# ---------------------------------------------------------------------------

class RelaxSolver(SolverBase):
    def __init__(self, options=None):
        super().__init__(options)

    # --- proposal logic ---
    def propose_parameters(self):
        """Relaxation-based parameter update with gradient clipping for numerical stability."""

        R = np.asarray(self.R, float)

        if self.adaptive_step:
            # adaptive step size based on residual norm
            res_norm = np.linalg.norm(R)
            step_size = max(self.min_step_size, self.step_size * (1.0 / (1.0 + res_norm)))
        else:
            step_size = self.step_size
        
        # Flatten for Jacobian computation if enabled
        if self.use_jacobian:
            X_flat, schema = self._flatten_params(self.X)
            direction = np.sign(np.nanmean(R)) * np.ones_like(X_flat)

            J = getattr(self, 'J', None)
            if J is None:
                try:
                    J = self._attempt_get_jacobian(
                        X_flat, R
                    )
                    self.J = J
                except Exception:
                    J = None
        
            if J is not None:
                # Column preconditioning: scale columns of J to unit norm
                col_norms = np.linalg.norm(J, axis=0)
                col_scale = 1.0 / np.maximum(col_norms, 1e-12)  # inv scales
                J_hat = J * col_scale  # scale columns

                JTJ = J_hat.T @ J_hat + self.jacobian_reg * np.eye(J.shape[1])
                rhs = -J_hat.T @ R
                z = np.linalg.solve(JTJ, rhs)
                delta = col_scale * z  # map back to original scaling

                if np.all(np.isfinite(delta)):
                    direction = delta
            
                # Apply step in flat space

                X_new_flat = X_flat + step_size * direction
                # Unflatten back to dict
                X_new_wo_bounds = self._unflatten_params(X_new_flat)
            else:
                X_new_wo_bounds = copy.deepcopy(self.X)

        else:
            # Simple relaxation: uniform step on all parameters
            sign_term = float(np.sign(np.nanmean(R)))
            step = -self.step_size * sign_term
            X_new_wo_bounds = {}
            for prof, param_dict in self.X.items():
                new_inner = {}
                for pname, pval in (param_dict or {}).items():
                    try:
                        new_inner[pname] = float(pval) + step
                    except Exception:
                        new_inner[pname] = pval  # leave non-numeric untouched
                X_new_wo_bounds[prof] = new_inner
        # Apply bounds projection
        X_new = self._project_bounds(X_new_wo_bounds)

        X_new_std = {prof: {name: abs(val)*self._parameters.sigma for name, val in X_new[prof].items()} for prof in X_new}
        for prof in X_new.keys():
            for param in X_new[prof].keys():
                if param.startswith('log_'):
                    # stay in log space for std
                    X_new_std[prof][param] = X_new[prof][param] + np.log(self._parameters.sigma)
        return X_new, X_new_std

class BayesianOptSolver(SolverBase):
    """
    BO solver that optimizes a Monte-Carlo estimate of E[Z] (expected objective)
    finite difference gradients and a small gradient optimizer (Adam) with multiple restarts.

    Options (self.options):
      - surr_warmup: int  (use RelaxSolver until this iter)
      - n_restarts: int
      - n_steps: int (optimizer steps per restart)
      - lr: float (adam learning rate)
      - n_mc: int (MC samples per objective estimate)
      - batch_size: int (number of initial candidate starting points / restarts)
      - adam_beta1, adam_beta2, adam_eps: Adam params
      - seed: int random seed
    """
    def __init__(self, options: Optional[dict] = None):
        super().__init__(options)
        self.n_restarts = int(self.options.get("n_restarts", 8))
        self.n_steps = int(self.options.get("n_steps", 80))
        self.lr = float(self.options.get("lr", 1e-1))
        self.n_mc = int(self.options.get("n_mc", 128))
        self.batch_size = int(self.options.get("batch_size", 64))
        self.seed = int(self.options.get("seed", 0))
        self.adam_beta1 = float(self.options.get("adam_beta1", 0.9))
        self.adam_beta2 = float(self.options.get("adam_beta2", 0.999))
        self.adam_eps = float(self.options.get("adam_eps", 1e-8))

        # Acquisition configuration
        self.acquisition = str(self.options.get("acquisition", "ei")).lower()  # ei | pi | ucb
        self.ucb_k = float(self.options.get("ucb_k", 2.0))  # exploration weight for UCB
        self.pi_k = float(self.options.get("pi_k", 10.0))   # sharpness for smooth PI sigmoid

        # cache
        self._schema = None
        self._rng = np.random.default_rng(self.seed)

    # ----------------------
    # propose_parameters
    # ----------------------
    def propose_parameters(self):
        """Batch MC BO using surrogate gradients when available.

        Acquisition on inverse objective -Z:
            - EI  = E[max(0, Z_best - Z)]
            - PI  = E[sigmoid(pi_k * (Z_best - Z))]
            - UCB = -(E[Z]) - ucb_k * Std(Z)
        Primary gradient source: `SurrogateManager.get_grads`.
        Fallback: finite-difference gradient on acquisition value.
        """
        # --- Warmup fallback --------------------------------------------------
        if getattr(self, 'iter', 0) < getattr(self, 'surr_warmup', 5):
            return RelaxSolver.propose_parameters(self)

        # --- Flatten current parameters & schema ------------------------------
        X_flat0, schema = self._flatten_params(self.X)
        n_params = X_flat0.size

        # --- Build bounds matrix ----------------------------------------------
        bounds_mat = np.zeros((n_params, 2), float)
        for i, (prof, pname) in enumerate(schema):
            lo, hi = self.bounds_dict[prof][pname]
            bounds_mat[i, 0] = -1e6 if lo is None else float(lo)
            bounds_mat[i, 1] =  1e6 if hi is None else float(hi)
        lo_arr = bounds_mat[:, 0]
        hi_arr = bounds_mat[:, 1]

        # --- Configuration -----------------------------------------------------
        acquisition = str(getattr(self, 'acquisition', 'ei')).lower()
        n_restarts  = int(getattr(self, 'n_restarts', 5))
        n_steps     = int(getattr(self, 'n_steps', 50))
        n_mc        = int(getattr(self, 'n_mc', 50))
        lr          = float(getattr(self, 'lr', 1e-1))
        pi_k        = float(getattr(self, 'pi_k', 1.0))
        ucb_k       = float(getattr(self, 'ucb_k', 1.0))
        fd_eps_rel  = float(getattr(self, 'fd_acq_epsilon', 1e-3))
        seed        = int(getattr(self, 'seed', 0)) + int(self.iter)
        rng         = np.random.default_rng(seed)
        normalize_resid = bool(self.normalize_residual)
        self._use_surr_iter = True if self._surrogate is not None else False

        # --- Current best objective -------------------------------------------
        Z_best = float(self.Z) if (hasattr(self, 'Z') and self.Z is not None and np.isfinite(self.Z)) else float('inf')
        target_keys = sorted(self.Y_target.keys())
        n_target_vars = len(target_keys)
        n_transport_vars = len(self.transport_vars)
        n_roa = len(self.roa_eval)

        Yt_stack = np.vstack([np.asarray(self.Y_target[k]).ravel().reshape(1, n_roa) for k in target_keys])
        denom_norm = np.abs(Yt_stack) + 1e-8 if normalize_resid else None
        feature_index = {name: idx for idx, name in enumerate(getattr(self._surrogate, 'all_features', []))}
        #param_feature_names = [f"{prof}_{pname}" for prof, pname in schema]


        def objective_grad(residual_vec):
            if hasattr(self.objective, 'gradient'):
                try:
                    return np.asarray(self.objective.gradient(residual_vec), float)
                except Exception:
                    pass
            # Numeric fallback
            r = np.asarray(residual_vec, float)
            eps = 1e-6 * np.maximum(1.0, np.abs(r))
            grad = np.zeros_like(r)
            base = float(self.objective(r))
            for i in range(r.size):
                rp = r.copy(); rm = r.copy()
                rp[i] += eps[i]; rm[i] -= eps[i]
                gp = float(self.objective(rp)); gm = float(self.objective(rm))
                grad[i] = (gp - gm) / (2.0 * eps[i])
            return grad

        def _acq_single(x_flat):
            # Convert flat -> dict for surrogate call
            X_dict = self._unflatten_params(x_flat)
            Y_model, Y_model_std, Y_target, Y_target_std = self._evaluate(in_place=True)

            eps = rng.normal(size=(n_mc, len(Y_model), n_roa))
            y_samples = np.array(list(Y_model.values()))[None, :, :] + np.array(list(Y_model_std.values()))[None, :, :] * eps  # (n_mc, n_tgt, n_roa)
            # Compute residuals for each MC sample
            Z_samples = np.zeros(n_mc, dtype=float)
            for mc_i in range(n_mc):
                Y_mc = y_samples[mc_i, :, :]
                Y_mc_dict = {k: Y_mc[i, :] for i, k in enumerate(self.transport_vars+self.target_vars)}
                R_mc = self._compute_residuals(Y_mc_dict, Y_target)
                Z_samples[mc_i] = self.objective(R_mc)
            
            if acquisition == 'ei':
                improv = np.maximum(0.0, Z_best - Z_samples)
                return float(np.mean(improv))
            if acquisition == 'pi':
                probs = 1.0 / (1.0 + np.exp(-pi_k * (Z_best - Z_samples)))
                return float(np.mean(probs))
            if acquisition == 'ucb':
                muZ = float(np.mean(Z_samples))
                stdZ = float(np.std(Z_samples) + 1e-8)
                return float(-(muZ) - ucb_k * stdZ)
            improv = np.maximum(0.0, Z_best - Z_samples)

            return float(np.mean(improv))

        def _acq_batch(x_batch):
            """Batched acquisition using surrogate.evaluate with 3D params array.

            x_batch: (n_batch, n_params)
            Returns: (n_batch,) acquisition values
            """
            x_batch = np.asarray(x_batch, float)
            if x_batch.ndim != 2:
                raise ValueError(f"_acq_batch expects 2D array, got shape {x_batch.shape}")

            n_batch = x_batch.shape[0]
            
            # make unique state for each sample
            states = []
            for b in range(n_batch):
                self._update_from_params(x_batch[b])
                state_b = copy.deepcopy(self._state)
                states.append(state_b)
            Y_model, Y_model_std, Y_target, Y_target_std = self._evaluate(x_batch, states, in_place=False)

            # return to original state
            self._update_from_params(self.X)

            # Stack model means/stds in fixed ordering matching transport_vars + target_vars
            M = np.stack([np.asarray(Y_model[k]) for k in Y_model.keys()], axis=1)      # (n_batch, n_transport_vars, n_roa)
            S = np.stack([np.asarray(Y_model_std[k]) for k in Y_model_std.keys()], axis=1)  # (n_batch, n_transport_vars, n_roa)

            # Monte Carlo sampling per batch
            eps = rng.normal(size=(n_mc, n_batch, M.shape[1], n_roa))
            Ys = M[None, :, :, :] + S[None, :, :, :] * eps  # (n_mc, n_batch, n_out, n_roa)

            # Build target array in same ordering
            T = np.stack([np.asarray(Y_target[k]) for k in self.target_vars], axis=1)  # (n_batch, n_out, n_roa)

            # Compute residuals and objective per MC sample, per batch
            Z_samples = np.zeros((n_mc, n_batch), dtype=float)
            Z_samples  = np.array([[self.objective(self._compute_residuals(dict(zip(self.model_vars, Ys[mc_i, b])), dict(zip(self.target_vars, T[b])))) \
                                for b in range(n_batch)] for mc_i in range(n_mc)]) 

            if acquisition == 'ei':
                improv = np.maximum(0.0, Z_best - Z_samples)
                return np.mean(improv, axis=0)
            if acquisition == 'pi':
                probs = 1.0 / (1.0 + np.exp(-pi_k * (Z_best - Z_samples)))
                return np.mean(probs, axis=0)
            if acquisition == 'ucb':
                muZ = np.mean(Z_samples, axis=0)
                stdZ = np.std(Z_samples, axis=0) + 1e-8
                return -(muZ) - ucb_k * stdZ
            improv = np.maximum(0.0, Z_best - Z_samples)
            return np.mean(improv, axis=0)

        # --- Acquisition value at x (Monte Carlo) -----------------------------
        def acquisition_value(x_flat):
            x_arr = np.asarray(x_flat, float)
            if x_arr.ndim == 1:
                return _acq_single(x_arr)
            if x_arr.ndim == 2:
                # Batched path using surrogate.evaluate to reduce overhead
                return _acq_batch(x_arr)
            raise ValueError(f"acquisition_value expects 1D or 2D input, got shape {x_arr.shape}")

        def surrogate_grad_batch(x_batch):
            """Batched acquisition gradient via surrogate.get_grads; None on failure.

            x_batch: (n_batch, n_params)
            returns: (n_batch, n_params)
            """
            try:
                x_batch = np.asarray(x_batch, float)
                if x_batch.ndim != 2:
                    return None
                n_batch = x_batch.shape[0]

                # Surrogate prediction for current batch
                # make unique state for each sample
                            # make unique state for each sample
                states = []
                for b in range(n_batch):
                    self._update_from_params(x_batch[b])
                    state_b = copy.deepcopy(self._state)
                    states.append(state_b)
                
                Y_model, Y_model_std, Y_target, Y_target_std = self._evaluate(x_batch, states, in_place=False)

                # return to original state
                self._update_from_params(self.X)

                # create a list of dicts for batch residual computation
                Y_model_list = [dict(zip(Y_model.keys(), [np.asarray(Y_model[k])[b] for k in Y_model.keys()])) for b in range(n_batch)]
                Y_target_list = [dict(zip(Y_target.keys(), [np.asarray(Y_target[k])[b] for k in Y_target.keys()])) for b in range(n_batch)]

                # Compute residuals and objective gradients per batch
                grad_r_list = []
                Z_det_list = []
                # Keys ordering for residual assembly is handled inside _compute_residuals
                R_array = np.array([self._compute_residuals(Y_model_list[b], Y_target_list[b]) for b in range(n_batch)])
                Z_det_list = [self.objective(R_array[b]) for b in range(n_batch)]
                grad_r_list = [objective_grad(R_array[b].reshape(-1)) for b in range(n_batch)]

                # Surrogate gradients wrt features
                transport_grads, target_grads = self._surrogate.get_grads(x_batch, states)
                if transport_grads is None:
                    return None

                # Build gradient of mean prediction wrt parameters for target vars
                G_full = np.zeros((n_batch, n_params), dtype=float)
                # Pre-compute schema parameter -> feature index mapping
                param_to_feature = {}
                for j, (prof, pname) in enumerate(schema):
                    if pname.startswith('aL'):
                        feature_name = f"aL{prof}"
                        f_idx = feature_index.get(feature_name, None)
                        if f_idx is not None:
                            param_to_feature[j] = (f_idx, float(self._surrogate.x_scaler.scale_[f_idx]))

                for b in range(n_batch):
                    grad_mu_flat = np.zeros((n_target_vars * n_roa, n_params), dtype=float)
                    for t_idx, tname in enumerate(target_keys):
                        g_arr_b = transport_grads.get(tname)
                        if g_arr_b is None:
                            continue
                        g_b = np.asarray(g_arr_b)[b]  # (n_roa, n_features)
                        # Only iterate over parameters with valid feature mappings
                        for j, (f_idx, scale) in param_to_feature.items():
                            # per-roa feature grad → param grad via chain rule
                            grad_unscaled = g_b[:, f_idx] / (scale if scale != 0 else 1.0)  # (n_roa,)
                            # place into flattened block for this target
                            start = t_idx * n_roa
                            stop = (t_idx + 1) * n_roa
                            grad_mu_flat[start:stop, j] = grad_unscaled

                    # Normalize residuals if enabled
                    if normalize_resid and denom_norm is not None:
                        grad_mu_flat = grad_mu_flat / denom_norm.reshape(-1, 1)

                    # Chain with objective gradient
                    gradZ_b = grad_mu_flat.T @ grad_r_list[b]

                    # Acquisition gradient transform
                    Z_det_b = Z_det_list[b]
                    if acquisition == 'ei':
                        G_full[b] = (-gradZ_b) if Z_det_b < Z_best else np.zeros_like(gradZ_b)
                    elif acquisition == 'pi':
                        sig = 1.0 / (1.0 + np.exp(-pi_k * (Z_best - Z_det_b)))
                        G_full[b] = sig * (1.0 - sig) * (-pi_k) * gradZ_b
                    elif acquisition == 'ucb':
                        G_full[b] = -gradZ_b
                    else:
                        G_full[b] = (-gradZ_b) if Z_det_b < Z_best else np.zeros_like(gradZ_b)

                return G_full
            except Exception:
                return None

        # --- Finite-difference gradient ---------------------------------------
        def fd_grad_batch(x_batch):
            """Batched central FD gradient of acquisition.

            x_batch: (n_batch, n_params)
            returns: (n_batch, n_params)
            """
            x_batch = np.asarray(x_batch, float)
            n_batch, n_params_b = x_batch.shape
            G = np.zeros_like(x_batch)
            for j in range(n_params_b):
                span = hi_arr[j] - lo_arr[j]
                h = fd_eps_rel * (span if np.isfinite(span) and span > 0 else 1.0)
                x_p = x_batch.copy(); x_m = x_batch.copy()
                x_p[:, j] = np.minimum(hi_arr[j], x_p[:, j] + h)
                x_m[:, j] = np.maximum(lo_arr[j], x_m[:, j] - h)
                fp = acquisition_value(x_p)  # (n_batch,)
                fm = acquisition_value(x_m)  # (n_batch,)
                denom = (x_p[:, j] - x_m[:, j])
                # avoid division by zero
                safe = denom != 0
                gcol = np.zeros(n_batch, float)
                gcol[safe] = (fp[safe] - fm[safe]) / denom[safe]
                G[:, j] = gcol
            return G

        # --- Multi-restart optimization ---------------------------------------
        seed_batch = max(n_restarts, int(getattr(self, 'batch_size', 1)))
        # Local perturbation around current X: small steps to avoid BC issues.
        # Sample candidates via small Gaussian perturbations clamped to bounds.
        step_scale = float(getattr(self, 'bo_step_scale', 0.05))  # relative to bounds width
        width = hi_arr - lo_arr
        sigma = step_scale * np.maximum(width, 1e-12)
        candidate_batch = X_flat0 + rng.normal(scale=sigma, size=(seed_batch, n_params))
        candidate_batch = np.clip(candidate_batch, lo_arr, hi_arr)
        batch_vals = acquisition_value(candidate_batch)
        if np.ndim(batch_vals) == 0:
            batch_vals = np.array([batch_vals])
        seed_order = np.argsort(batch_vals)[::-1][:n_restarts]

        # Optimize all selected seeds in parallel
        x_batch = candidate_batch[seed_order].copy()  # (n_restarts, n_params)
        for step in range(n_steps):
            # Try batched surrogate gradients first; fall back to batched FD
            G = surrogate_grad_batch(x_batch)
            if G is None or not np.all(np.isfinite(G)):
                G = fd_grad_batch(x_batch)
            # Gradient ascent step
            x_batch = x_batch + lr * G
            x_batch = np.clip(x_batch, lo_arr, hi_arr)

        # Evaluate final batch and select best
        vals = acquisition_value(x_batch)
        best_idx = int(np.argmax(vals))
        best_x = x_batch[best_idx].copy()

        if best_x is None:
            best_x = X_flat0.copy()

        # --- Convert to dict & build std dict ---------------------------------
        X_new = self._unflatten_params(best_x)
        X_new = self._project_bounds(X_new)
        X_new_std = {prof: {pname: abs(val) * getattr(self._parameters, 'sigma', 0.0)
                            for pname, val in prof_dict.items()}
                     for prof, prof_dict in X_new.items()}
        return X_new, X_new_std


class IvpSolver(SolverBase):
    """SolverBase child that integrates parameter evolution in pseudo-time using BDF for stiff systems."""

    def __init__(self, options=None):
        super().__init__(options)
        self.dt = float(self.options.get("step_size", 1e-2))
        self.method = self.options.get("method", "BDF")
        self.vectorized = bool(self.options.get("vectorized", False))


    def _ode_rhs(self, t, X_flat):
        """Compute dX/dt = -α*R for residual-driven dynamics.
        
        Uses direct residual feedback rather than gradient descent.
        The time integrator handles adaptive step sizing.
        """
        # Update state with current parameters
        self._update_from_params(X_flat)
        self._evaluate(in_place=True)
        R = self._compute_residuals(self.Y, self.Y_target)
        if R is None:
            return np.zeros_like(X_flat)

        # Cache Jacobian for potential use by solve_ivp's jac callback
        # Only compute if not already cached for this timestep
        if not hasattr(self, '_J_cache') or self._J_cache is None:
            self._J_cache = self._attempt_get_jacobian(X_flat, R)
        
        if self._J_cache is not None:
            # J is (m_residuals, n_params), we want to map R (m,) -> dX (n,)
            # Use Moore-Penrose pseudo-inverse: dX = -α * J^+ @ R
            dX = -sp.linalg.pinv(self._J_cache) @ R
        else:
            # Fallback: uniform damping
            dX = -np.sign(R).mean() * np.ones_like(X_flat) * self.step_size
        
        return dX

    def _jacobian(self, t, X_flat):
        """Jacobian ∂(dX/dt)/∂X for implicit ODE methods.
        
        Reuses cached J from _ode_rhs to avoid redundant computation.
        For dX/dt = -J^+ @ R, the Jacobian is -J^+ @ J.
        """
        # Reuse cached Jacobian if available from same evaluation
        if hasattr(self, '_J_cache') and self._J_cache is not None:
            J = self._J_cache
        else:
            # Compute if not cached (shouldn't happen if _ode_rhs called first)
            self._update_from_params(X_flat)
            self._evaluate(in_place=True)
            R = self._compute_residuals(self.Y, self.Y_target)
            if R is None:
                return np.zeros((X_flat.size, X_flat.size))
            J = self._attempt_get_jacobian(X_flat, R)
            if J is None:
                return np.zeros((X_flat.size, X_flat.size))
            self._J_cache = J
        
        # Jacobian of RHS = ∂(-J^+ @ R)/∂X ≈ -J^+ @ J
        # This is (n_params, n_params) symmetric negative semi-definite
        J_pinv = sp.linalg.pinv(J)
        return -J_pinv @ J

    def propose_parameters(self):
        """Advance one pseudo-time step (Δt) using residual-driven dynamics."""
        if self.X is None:
            raise RuntimeError("Parameters are not initialized; run get_initial_parameters() first.")

        # Ensure we have a flat parameter vector and schema
        if isinstance(self.X, dict):
            X0_flat,_= self._flatten_params(self.X)
        else:
            X0_flat = np.asarray(self.X, dtype=float)

        X0 = np.asarray(X0_flat, float)
        t_curr = self.iter * self.dt
        t_next = t_curr + self.dt

        # Clear Jacobian cache before step
        self._J_cache = None

        # ----------------------------
        # Integrate one Δt step
        # ----------------------------
        sol = sp.integrate.solve_ivp(
            fun=self._ode_rhs,
            t_span=(t_curr, t_next),
            y0=X0,
            method=self.method,
            t_eval=[t_next],
            args=(),
            max_step=self.dt,
            first_step=self.dt*0.5,
            vectorized=self.vectorized,
            jac=self._jacobian,
            dense_output=False,
            rtol=float(self.options.get('ts_rtol', 1e-3)),
            atol=float(self.options.get('ts_atol', 1e-6)),
        )

        X_new_flat = np.asarray(sol.y[:, -1])
        X_new = self._unflatten_params(X_new_flat)
        X_new = self._project_bounds(X_new)
        X_new_std = {prof: {pname: abs(val) * getattr(self._parameters, "sigma", 0.0)
                            for pname, val in prof_dict.items()}
                     for prof, prof_dict in X_new.items()}
        return X_new, X_new_std


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SOLVER_MODELS = {
    "relax": RelaxSolver,
    "bayesian_opt": BayesianOptSolver,
    "ivp": IvpSolver,
}

def create_solver(config: Any) -> SolverBase:
    if isinstance(config, str):
        key = config.lower(); kwargs = {}
    else:
        key = str((config or {}).get("type", "relax")).lower()
        kwargs = (config or {}).get("kwargs", {})
    cls = SOLVER_MODELS.get(key)
    if cls is None:
        raise ValueError(f"Unknown solver type '{key}'. Available: {list(SOLVER_MODELS)}")
    return cls(kwargs)


# (obsolete solver definitions removed)

