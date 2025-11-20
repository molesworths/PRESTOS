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
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pickle
from surrogates import SurrogateManager
import scipy as sp
from math import sqrt


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
    Z: List[Optional[float]] = field(default_factory=list)       # objective scalar
    Z_std: List[Optional[float]] = field(default_factory=list)       # objective std scalar
    Y: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_std: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_target: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_target_std: List[Dict[str, np.ndarray]] = field(default_factory=list)
    used_surrogate: List[bool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, i: int, X: Dict[str, Dict[str, float]], X_std: Dict[str, Dict[str, float]], R: Optional[np.ndarray], R_std: Optional[np.ndarray], Z: Optional[float],
            Z_std: Optional[float], Y: Dict[str, np.ndarray], Y_std: Dict[str, np.ndarray], Y_target: Dict[str, np.ndarray],
            Y_target_std: Dict[str, np.ndarray], used_surr: bool):
        self.iterations.append(int(i))
        # Deep copy dicts to avoid mutation issues
        self.X.append(copy.deepcopy(X))
        self.X_std.append({k: copy.deepcopy(v) for k, v in (X_std or {}).items()})
        self.R.append(None if R is None else np.asarray(R).copy())
        self.R_std.append(None if R_std is None else np.asarray(R_std).copy())
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

    def __call__(self, residual: np.ndarray) -> float:  # pragma: no cover
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

OBJECTIVE_FUNCTIONS = {
    "sse": SumSquares,
    "sum_squares": SumSquares,
    "mse": MeanSquares,
    "mean_squares": MeanSquares,
    "mae": MeanAbsolute,
    "mean_absolute": MeanAbsolute,
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
        self.normalize_residual = bool(self.options.get("normalize_residual", True))
        self.residual_on_lcfs = bool(self.options.get("residual_on_lcfs", False))
        self.converged = False
        self.iter = 0
        self.iter_between_save = int(self.options.get("iter_between_save", 5))
        self.cwd = Path.cwd()

        # Mapping predicted profiles to target variables
        self.predicted_profiles = list(self.options.get("predicted_profiles", []))
        self.target_vars = list(self.options.get("target_vars", []))
        self.transport_vars = [f"{t}_neo" for t in self.target_vars] + [f"{t}_turb" for t in self.target_vars]
        self.n_params_per_profile = int(self.options.get("n_params_per_profile", 0))

        # Evaluation grid
        self.domain = self.options.get("domain", [0.85, 1.0])
        self.roa_eval = np.asarray(self.options.get("roa_eval", []), float)
        if self.roa_eval.size == 0:
            self.n_eval = int(self.options.get("n_eval", 32))
            self.roa_eval = np.linspace(self.domain[0], self.domain[1], self.n_eval)
        else: self.n_eval = self.roa_eval.size

        # Bounds & iteration controls
        self.bounds = self._parse_bounds(self.options.get("bounds"))
        self.max_iter = int(self.options.get("max_iter", 100))

        # Surrogate configuration / datasets
        self.surr_warmup = int(self.options.get("surrogate_warmup", 5))
        self.surr_retrain_every = int(self.options.get("surrogate_retrain_every", 10))
        self.surr_verify_on_converge = bool(self.options.get("surrogate_verify_on_converge", True))
        self._surr_trained = False
        self._surr_Xtrain = []             # list of (1, n_features_total)
        self._surr_Ytrain = {}             # var_name -> list of (1, n_outputs_var)
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
        self.jacobian_reg = float(self.options.get("jacobian_reg", 1e-8))
        self.fd_epsilon = float(self.options.get("fd_epsilon", 5e-2))
        self.jacobian_methods = [
            "jacobian_wrt_parameters",
            "compute_jacobian",
            "residual_jacobian",
            "jacobian",
            "get_jacobian",
        ]

    # --------------------- helpers ---------------------
    def get_initial_parameters(self, state, parameters, boundary, neutrals, targets):
        neutrals.solve(state)
        _ = targets.evaluate(state)
        boundary.get_boundary_conditions(state, targets)
        self.X, self.X_std = parameters.parameterize(state, boundary.bc_dict)
        _, _ = self._flatten_params(self.X)

    def _update_from_params(self, X: np.ndarray, state, parameters, boundary, neutrals, targets):
        boundary.get_boundary_conditions(state, targets)
        # delegate to parameters to reconstruct profiles
        if isinstance(X, np.ndarray):
            X = self._unflatten_params(X, self.schema)
        parameters.update(X, boundary.bc_dict, self.roa_eval)
        state.update(X,parameters)
        neutrals.solve(state)
        _ = targets.evaluate(state)

    def _parse_bounds(self, bounds):
        """Normalize bounds specification from setup.yaml.

        Returns one of:
        - None
        - ("uniform", low, high)
        - ("per_param", list_of_tuples)
        - ("arrays", low_array, high_array)
        """

        if bounds is None:
            return None

        bounds_dict = {}

        # --- Uniform scalar tuple ---
        if isinstance(bounds, list) and len(bounds) == 2:
            try:
                low = float(bounds[0])
                high = float(bounds[1])
                return {n: [(low, high) for i in range(self.n_params_per_profile)] for n in self.predicted_profiles}
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
                    bounds_dict.update({k: [(low, high) for i in range(self.n_params_per_profile)] for k in self.predicted_profiles})
                except Exception:
                    return None
            return bounds_dict

        # --- Per-parameter list of [low,high] ---
        if isinstance(bounds, list[list]) and all(len(b) == 2 for b in bounds) and len(bounds) == self.n_params_per_profile:
            try:
                bounds_dict = {k: [(float(a), float(b)) for [a, b] in bounds] for k in self.predicted_profiles}
                return bounds_dict
            except Exception:
                return None

        print('Unable to parse parameter bounds specification; ignoring.')
        return None

    def _project_bounds(self, X):
        """Clip parameters using bounds.
        
        Expects self.bounds as dict with keys = self.predicted_profiles
        and values = [(lower, upper), ...] for each parameter.
        """
        if self.bounds is None:
            return X

        if isinstance(X, dict):
            Xc = {}
            for prof, pvals in X.items():
                if prof not in self.bounds:
                    # No bounds for this profile, keep unchanged
                    Xc[prof] = pvals
                    continue
                
                bounds_list = self.bounds[prof]  # List of (lower, upper) tuples
                Xc[prof] = {}
                
                # Get parameter names in sorted order for consistent indexing
                param_names = sorted(pvals.keys())
                
                for idx, pname in enumerate(param_names):
                    val = pvals[pname]
                    if idx < len(bounds_list):
                        lo, hi = bounds_list[idx]
                        Xc[prof][pname] = float(np.clip(val, lo, hi))
                    else:
                        # No bounds for this parameter index
                        Xc[prof][pname] = val
            return Xc

        return X

    def _use_surrogate_iteration(self, surrogate) -> bool:
        # Surrogates used after warm-up on scheduled evaluation iterations
        if surrogate is None:
            return False
        if self.iter < self.surr_warmup:
            return False
        if self.converged==True: 
            return False
        if (self.iter % self.surr_retrain_every) == 0:
            return False
        return True
    
    def _evaluate(self,use_surr: bool):
        if use_surr:
            train = (self.iter % self.surr_retrain_every) == 0 or (self.iter == self.surr_warmup)
            self._surrogate.evaluate(self.X, self._state, train=train)
            self.Y = self._surrogate.transport # dict of List[float]
            self.Y_std = getattr(self._surrogate, 'transport_std', np.zeros_like(self.Y)) # dict of List[float]
            self.Y_cov = getattr(self._surrogate, 'transport_cov', None) # dict of Lists[2D array]
            self.Y_target = self._surrogate.targets # dict of List[float]
            self.Y_target_std = getattr(self._surrogate, 'targets_std', np.zeros_like(self.Y_target)) # dict of List[float]
            self.Y_target_cov = getattr(self._surrogate, 'targets_cov', None) # dict of Lists[2D array]
        else:
            self.Y, self.Y_std = self._transport.evaluate(self._state) # dict of arrays/lists
            # assuming linearly independent model uncertainty, Cov = diag(sigma[i]^2)
            self.Y_target, self.Y_target_std = self._targets.evaluate(self._state) # dict of arrays/lists
            # assuming linearly independent target uncertainty, Cov = diag(sigma^2)
            if self._surrogate is not None:
                self._surrogate.add_sample(self._state, self.X, self.Y, self.Y_target)


    def _compute_residuals(self, Y_model: Dict[str, Any], Y_target: Dict[str, Any]) -> Optional[np.ndarray]:
        normalize = self.normalize_residual
        R_dict = {k: np.array(Y_model[k]) - np.array(Y_target[k]) \
                  for k in Y_target.keys() if k in Y_model.keys()}
        R = np.concatenate([R_dict[k] for k in sorted(R_dict.keys())])
        Y_target_concat = np.concatenate([np.array(Y_target[k]) for k in Y_target.keys() if k in Y_model])
        if normalize:
            R = R / (np.abs(Y_target_concat) + 1e-8)
        if not self.residual_on_lcfs:
            idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
            if idx.size == 1:
                n = len(self.roa_eval)
                k = int(idx[0])
                # Zero the LCFS residual element for each variable block
                nblocks = R.size // n if n > 0 else 0
                for b in range(nblocks):
                    R[b * n + k] = 0.0
        self.R = np.nan_to_num(R, nan=1e6, posinf=1e6, neginf=-1e6)
        return R
    
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
            for prof in sorted(X_dict.keys()):
                param_dict = X_dict[prof]
                for pname in sorted(param_dict.keys()):
                    schema.append((prof, pname))
                    values.append(float(param_dict[pname]))
            self.schema = schema
        else:
            schema = self.schema
            values = []
            for prof, pname in schema:
                values.append(float(X_dict[prof][pname]))

        return np.array(values, dtype=float), schema
    
    def _unflatten_params(self, X_flat: np.ndarray, schema: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """Convert flat array back to dict-of-dicts using schema.
        
        Parameters
        ----------
        X_flat : np.ndarray
            Flattened parameter vector
        schema : List[Tuple[str, str]]
            List of (profile, param_name) tuples defining the order
            
        Returns
        -------
        X_dict : Dict[str, Dict[str, float]]
            Reconstructed dict-of-dicts parameters
        """
        X_dict = {}
        for i, (prof, pname) in enumerate(schema):
            if prof not in X_dict:
                X_dict[prof] = {}
            X_dict[prof][pname] = float(X_flat[i])
        return X_dict
    
    # --------------------- Jacobian helpers ---------------------
    def _objective_at(self, state, X: np.ndarray, parameters, boundary, transport, targets, neutrals) -> float:
        # temporary update
        st = copy.deepcopy(state)  # assume in-place update acceptable
        self._update_from_params(X, st, parameters, boundary, neutrals, targets)
        Y_model, _ = transport.evaluate(st)
        Y_target, _ = targets.evaluate(st)
        R = self._compute_residuals(Y_model, Y_target)
        if R is None:
            return float("inf")
        return self.objective(R)
        

    def _attempt_get_jacobian(self, state, X: np.ndarray, R: np.ndarray, parameters, boundary, transport, targets, neutrals, surrogate):
        # Handle dict-of-dicts parameters: flatten for Jacobian computation

        if isinstance(X, dict):
            X_flat, schema = self._flatten_params(X)
            X_dict = X  # Keep original for reconstruction
        else:
            X_flat = X
            X_dict = self._unflatten_params(X, self.schema) if hasattr(self, 'schema') else None
        
        # Priority 1: Analytic Jacobian from transport/targets models
        # for comp in (transport, targets):
        #     if comp is None:
        #         continue
        #     for name in self.jacobian_methods:
        #         if hasattr(comp, name):
        #             try:
        #                 # Pass original dict if available, else flat array
        #                 J = getattr(comp, name)(state, X_dict if X_dict else X)
        #                 J = np.atleast_2d(np.asarray(J, float))
        #                 if J.shape[0] == R.size:
        #                     return J
        #             except Exception:
        #                 pass
        
        # Priority 2: Finite difference on a trained surrogate model
        # if surrogate is not None:
        #     try:
        #         # Pass flat X and schema to FD method
        #         J = self._fd_jacobian_on_surrogate(state, X_flat, R, parameters, boundary, transport, targets, neutrals, surrogate)
        #         if J is not None:
        #             return J
        #     except Exception:
        #         pass

        # Priority 3: Finite difference on the full, expensive models
        J = self._fd_jacobian_on_full_model(state, X_flat, R, parameters, boundary, transport, targets, neutrals, surrogate)

        return J

    def _fd_jacobian_on_surrogate(self, state, X: np.ndarray, R: np.ndarray, parameters, boundary, transport, targets, neutrals, surrogate=None):
        """Fast Jacobian via finite differences on the surrogate model.
        
        Parameters
        ----------
        schema : List[Tuple[str, str]], optional
            If provided, X is flat and needs to be unflattened to dict for evaluate calls
        """
        m = R.size
        n = X.size
        J = np.zeros((m, n), float)
        eps_rel = self.fd_epsilon

        # Convert X to dict
        schema = self.schema
        X_dict = self._unflatten_params(X, schema)

        # Get baseline surrogate predictions
        st = copy.deepcopy(state)
        self._update_from_params(X_dict, st, parameters, boundary, neutrals, targets)
        surrogate.evaluate(X_dict, st)
        Y_model_base = {**surrogate.transport, **surrogate.targets}
        
        # Get baseline targets (assuming they don't depend on X)
        Y_target_base, _ = targets.evaluate(st)
        R_base = self._compute_residuals(Y_model_base, Y_target_base)
        if R_base is None:
            return None

        for j in range(n):
            xj = X[j]
            eps = eps_rel * abs(xj) if xj != 0 else eps_rel
            
            # Perturb parameters (in flat space)
            Xp = X.copy()
            Xp[j] += eps
            
            # Convert to dict if needed
            Xp_dict = self._unflatten_params(Xp, schema) if schema else Xp
            
            # Update state and get surrogate prediction
            st_p = copy.deepcopy(state)
            self._update_from_params(Xp_dict, st_p, parameters, boundary, neutrals, targets)
            surrogate.evaluate(Xp_dict, st_p)
            Y_model_p = {**surrogate.transport, **surrogate.targets}
            Rp = self._compute_residuals(Y_model_p, Y_target_base)

            if Rp is None: return None

            J[:, j] = (Rp - R_base) / eps
            
        return J

    def _fd_jacobian_on_full_model(self, state, X: np.ndarray, R: np.ndarray, parameters, boundary, transport, targets, neutrals, surrogate=None):
        """Central finite-difference Jacobian on full transport/targets models.
        
        Parameters
        ----------
        schema : List[Tuple[str, str]], optional
            If provided, X is flat and needs to be unflattened to dict for evaluate calls
        """
        m = R.size
        n = X.size
        J = np.zeros((m, n), float)
        eps_rel = self.fd_epsilon

        # Convert X to dict
        schema = self.schema
        X_dict = self._unflatten_params(X, schema)

        base_obj = self._objective_at(state, X, parameters, boundary, transport, targets, neutrals)
        if not np.isfinite(base_obj):
            return None
            
        for j in range(n):
            xj = X[j]
            eps = eps_rel * abs(xj)
            if eps == 0:
                eps = eps_rel
                
            # Perturb in flat space
            Xp = X.copy()
            Xm = X.copy()
            Xp[j] += eps
            Xm[j] -= eps
            
            # Convert to dict if needed
            Xp_dict = self._unflatten_params(Xp, schema) if schema else Xp
            Xm_dict = self._unflatten_params(Xm, schema) if schema else Xm
            
            # Forward perturbation
            self._update_from_params(Xp_dict, state, parameters, boundary, neutrals, targets)
            Ymp, _ = transport.evaluate(state)
            Ytp, _ = targets.evaluate(state)
            Rp = self._compute_residuals(Ymp, Ytp)
            
            # Backward perturbation
            self._update_from_params(Xm_dict, state, parameters, boundary, neutrals, targets)
            Ymm, _ = transport.evaluate(state)
            Ytm, _ = targets.evaluate(state)
            Rm = self._compute_residuals(Ymm, Ytm)

            # Store samples in surrogate if provided
            if surrogate is not None:
                surrogate.add_sample(state, Xp_dict, Ymp, Ytp)
                surrogate.add_sample(state, Xm_dict, Ymm, Ytm)
            
            if Rp is None or Rm is None or Rp.size != Rm.size:
                return None
            J[:, j] = (Rp - Rm) / (2.0 * eps)
            
        # Restore state to baseline X
        self._update_from_params(X_dict, state, parameters, boundary, neutrals, targets)

        return J

    # --------------------- Main methods to override ---------------------
    def propose_parameters(self, state) -> np.ndarray:
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
        if self.R is None:
            self.Z = float("inf")
            self.converged = False
            return

        # Objective value
        self.Z = self.objective(self.R)

        # Build residual covariance (tries Jacobian linearization if param sigmas present)
        X_dict = self.X if isinstance(self.X, dict) else None
        C_R = self._build_residual_cov(self.R, X_dict, use_jacobian=self.use_jacobian)

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
        confidence = float(self.options.get("convergence_confidence", 0.95))
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

        bundle['data'].save(self.cwd / "solver_history.csv")
        bundle['timestamp'] =  datetime.now(timezone.utc).isoformat()

    
    # --------------------- uncertainty helpers ---------------------

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
        # Collect model and target stds in the same ordering as R
        # R ordering is based on sorted keys of Y_target intersection Y_model (see _compute_residuals)
        # Build arrays of stds for each block
        try:
            y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
            # build model std vector and target std vector in same order
            model_sig_list = []
            target_sig_list = []
            for k in y_keys:
                mstd = np.asarray(self.Y_std.get(k, np.zeros_like(self.Y[k])), dtype=float)
                tstd = np.asarray(self.Y_target_std.get(k, np.zeros_like(self.Y_target[k])), dtype=float)
                # flatten to 1D and append
                model_sig_list.append(mstd.ravel())
                target_sig_list.append(tstd.ravel())
            sigma_model_vec = np.concatenate([s.ravel() for s in model_sig_list]) if model_sig_list else np.zeros(R.size)
            sigma_target_vec = np.concatenate([s.ravel() for s in target_sig_list]) if target_sig_list else np.zeros(R.size)
            # total measurement/model std for residual
            sigma_resid = np.sqrt(np.maximum(0.0, sigma_model_vec**2 + sigma_target_vec**2))
            C_meas = np.diag(sigma_resid**2)
        except Exception:
            C_meas = None
        self.C_meas = C_meas

        # Parameter contribution via Jacobian * Cx * J^T
        C_param = None
        if use_jacobian and X_dict is not None:
            # try to compute Jacobian (this will attempt training surrogate / FD as configured)
            try:
                X_flat, schema = self._flatten_params(X_dict)
                J = self._attempt_get_jacobian(self._state, X_flat, R,
                                               self._parameters, self._boundary,
                                               self._transport, self._targets,
                                               self._neutrals, self._surrogate)
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
        self.C_param = C_param
        
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

        self.C_Xpost = self._compute_posterior_parameter_uncertainty()

        return C_total
    
    def _compute_posterior_parameter_uncertainty(self):
        """
        CALLED THROUGH _build_residual_cov TO SET self.C_Xposterior

        Compute posterior parameter covariance via linearized update:
            Cx_post = (J^T (C_meas)^-1 J + Cx_prior^-1)^-1
        where C_meas includes model and target uncertainty.
        Using Moore-Penrose pseudoinverses for non-square matrices.
        Returns Cx_post or None if not computable.
        """
        if self.J is None and self.use_jacobian:
            self.J = self._attempt_get_jacobian(self._state,
                                                 self.X if isinstance(self.X, dict) else self.X,
                                                 self.R,
                                                 self._parameters,
                                                 self._boundary,
                                                 self._transport,
                                                 self._targets,
                                                 self._neutrals,
                                                 self._surrogate)
        else: return None
        # Posterior covariance
        try:
            Cx_post = sp.linalg.pinv(self.J) @ self.C_meas @ sp.linalg.pinv(self.J).T #+ Cx_prior_inv
            return Cx_post
        except np.linalg.LinAlgError:
            return None

    def propagate_uncertainty_mc(self,
                                 n_samples: int = 200,
                                 mode: str = "param",
                                 random_state: Optional[int] = None):
        """
        Monte-Carlo uncertainty propagation using the surrogate only.

        Parameters
        ----------
        n_samples : int
            Number of Monte-Carlo samples.
        mode : str
            "param" → sample parameter uncertainty: X ~ N(X, sigma_X)
            "model" → sample model uncertainty:   Y ~ surrogate(X) + eps
            "both"  → sample both parameter and model uncertainty
        random_state : int or None
            Optional seed.

        Returns
        -------
        results : dict with:
            'Y_samples' : list of dicts of surrogate outputs
            'R_samples' : array of residual vectors
            'Z_samples' : array of objective values
            'param_samples' : list of parameter dicts (only if mode='param'/'both')
        """
        rng = np.random.default_rng(random_state)

        # --- Check surrogate availability --------------------------------------
        surrogate = getattr(self, "_surrogate", None)
        if surrogate is None or not surrogate.is_trained:
            raise RuntimeError("Surrogate model unavailable or untrained.")

        # --- Determine base parameter set --------------------------------------
        if not isinstance(self.X, dict):
            raise RuntimeError("MC uncertainty requires dict-format parameters (X is not dict).")

        X0 = self.X
        X0_flat, schema = self._flatten_params(X0)

        # --- Parameter uncertainties -------------------------------------------
        sigma_X = None
        if mode.lower() in ("param", "both"):
            sigma_X = self._parameters.param_std
            if sigma_X is None:
                raise RuntimeError("Parameter uncertainties not provided.")

        # --- Storage -----------------------------------------------------------
        Y_samples = []
        R_samples = []
        Z_samples = []
        param_samples = [] if mode.lower() in ("param", "both") else None

        # For computing residuals in consistent ordering
        y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])

        # --- Monte-Carlo loop ---------------------------------------------------
        for _ in range(n_samples):

            # --- 1) Sample parameters if requested -----------------------------
            if sigma_X is not None:
                Xs_flat = X0_flat + rng.normal(scale=sigma_X)
                Xs = self._unflatten_params(Xs_flat, schema)
            else:
                Xs = X0

            # --- 2) Surrogate evaluation ---------------------------------------
            # The surrogate manager already returns (Y, Y_std)
            Ys, Ys_std = surrogate.evaluate(Xs)

            # --- 3) Add model noise if requested -------------------------------
            if mode.lower() in ("model", "both"):
                Ys_noisy = {}
                for k in Ys:
                    std = np.asarray(Ys_std.get(k, 0.0), float)
                    eps = rng.normal(scale=std)
                    Ys_noisy[k] = np.asarray(Ys[k]) + eps
                Ys = Ys_noisy

            # --- 4) Compute residual and objective -----------------------------
            # Build residual vector (same as solver logic)
            r_list = []
            for k in y_keys:
                ym = np.asarray(Ys[k]).ravel()
                yt = np.asarray(self.Y_target[k]).ravel()
                r_list.append(ym - yt)
            R = np.concatenate(r_list)

            # Normalize if requested
            if self.normalize_residual:
                yt_vec = np.concatenate([np.asarray(self.Y_target[k]).ravel() for k in y_keys])
                R = R / (np.abs(yt_vec) + 1e-12)

            # Objective
            Z = float(self.objective(R))

            # --- 5) Store -------------------------------------------------------
            Y_samples.append(Ys)
            R_samples.append(R)
            Z_samples.append(Z)
            if param_samples is not None:
                param_samples.append(Xs)

        return {
            "Y_samples": Y_samples,
            "R_samples": np.vstack(R_samples),
            "Z_samples": np.asarray(Z_samples),
            "param_samples": param_samples,
        }


    # --------------------- main loop ---------------------
    def run(self, state, boundary, parameters, neutrals, transport, targets, surrogate=None) -> SolverData:

        # Store module references for Jacobian calls
        self._state = state
        self._parameters = parameters
        self._boundary = boundary
        self._neutrals = neutrals
        self._transport = transport
        self._targets = targets
        self._surrogate = surrogate

        # migrate info to modules
        transport.roa_eval = self.roa_eval
        transport.output_vars = self.transport_vars + self.target_vars
        targets.roa_eval = self.roa_eval
        targets.output_vars = self.target_vars
        parameters.predicted_profiles = self.predicted_profiles
        parameters.bounds = self.bounds
        data = SolverData()

        # initialize parameter vector
        self.get_initial_parameters(state, parameters, boundary, neutrals, targets)

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

        while not self.converged and self.iter <= self.max_iter:
            # apply parameters → state
            X_current, it = self.X, self.iter
            self._update_from_params(X_current, state, parameters, boundary, neutrals, targets)

            # surrogate or full model
            use_surr = self._use_surrogate_iteration(surrogate)
            self._evaluate(use_surr)

            # convergence
            self.check_convergence(self.Y, self.Y_target)
            if self.converged:
                if use_surr and self.surr_verify_on_converge:
                    # verify on full model
                    use_surr = False
                    self._evaluate(use_surr)
                    self.check_convergence(self.Y, self.Y_target)
                if self.converged: # check for convergence on verified model
                    self.save(module_bundle)
                    print('Convergence achieved. Solver run complete.')
            else:
                # iteration count check
                if self.iter < self.max_iter: 
                    if self.iter % self.iter_between_save == 0: self.save(module_bundle)
                    self.iter += 1
                else:
                    if use_surr:
                        # final evaluation on full model
                        use_surr = False
                        self._evaluate(use_surr)
                        self.check_convergence(self.Y, self.Y_target)
                    self.save(module_bundle)
                    print('Max iterations reached. Solver run complete.')
                    break

            data.add(self.iter, self.X, self.X_std, self.R, self.R_std, \
                     self.Z, self.Z_std, self.Y, self.Y_std, self.Y_target, self.Y_target_std, use_surr)

            # propose next parameters, dict of keys = self.predicted profiles, values = parameter arrays
            self.X, self.X_std = self.propose_parameters(state, surrogate)
            self.J = None  # reset Jacobian cache

        print('Solver run complete.')  # placeholder return
        return data

# ---------------------------------------------------------------------------
# Derived Solvers
# ---------------------------------------------------------------------------

class RelaxSolver(SolverBase):
    def __init__(self, options=None):
        super().__init__(options)
        self.alpha = float(self.options.get("alpha", self.step_size))
        self.use_jacobian = bool(self.options.get("use_jacobian", True))

    # --- proposal logic ---
    def propose_parameters(self, state, surrogate=None):
        """Generate new parameter proposal.

        Supports dict-of-dicts parameter format. When Jacobian is enabled and
        parameters are dict-of-dicts, they are flattened for Jacobian computation
        and then unflattened for the result.
        """
        residual = self.R

        if self.X is None:
            # Should have been initialized in run(); safeguard anyway
            return None

        if residual is None or np.size(residual) == 0:
            return copy.deepcopy(self.X)

        R = np.asarray(residual, float)
        
        # Handle dict-of-dicts parameters
        if isinstance(self.X, dict):
            # Flatten for Jacobian computation if enabled
            if self.use_jacobian:
                X_flat, schema = self._flatten_params(self.X)
                direction = np.sign(np.nanmean(R)) * np.ones_like(X_flat)

                if hasattr(self,'J'):
                    J = self.J
                else:

                    try:
                        J = self._attempt_get_jacobian(
                            self._state, self.X, R,
                            self._parameters, self._boundary,
                            self._transport, self._targets,
                            self._neutrals, surrogate
                        )
                    except Exception:
                        # Jacobian computation failed, fall back to simple step
                        pass

                if J is not None:
                    # Column preconditioning: scale columns of J to unit norm
                    col_norms = np.linalg.norm(J, axis=0)
                    col_scale = 1.0 / np.maximum(col_norms, 1e-12)  # inv scales
                    J_hat = J * col_scale  # scale columns

                    JTJ = J_hat.T @ J_hat + self.jacobian_reg * np.eye(J.shape[1])
                    rhs = J_hat.T @ R
                    z = np.linalg.solve(JTJ, rhs)
                    delta = col_scale * z  # map back to original scaling

                    if np.all(np.isfinite(delta)):
                        direction = delta
                
                # Apply step in flat space
                X_new_flat = X_flat - self.alpha * direction
                # Unflatten back to dict
                X_new_wo_bounds = self._unflatten_params(X_new_flat, schema)
                # Apply bounds projection
                X_new = self._project_bounds(X_new_wo_bounds)
            else:
                # Simple relaxation: uniform step on all parameters
                sign_term = float(np.sign(np.nanmean(R)))
                step = self.alpha * sign_term
                X_new_wo_bounds = {}
                for prof, param_dict in self.X.items():
                    new_inner = {}
                    for pname, pval in (param_dict or {}).items():
                        try:
                            new_inner[pname] = float(pval) - step
                        except Exception:
                            new_inner[pname] = pval  # leave non-numeric untouched
                    X_new_wo_bounds[prof] = new_inner
                # Apply bounds projection
                X_new = self._project_bounds(X_new_wo_bounds)

        X_new_std = {prof: {name: abs(val)*self._parameters.sigma for name, val in X_new[prof].items()} for prof in X_new}
        return X_new, X_new_std


class FiniteDifferenceSolver(SolverBase):

    def __init__(self, options=None):
        super().__init__(options)
        self.alpha = float(self.options.get("alpha", self.step_size))
        self.eps = float(self.options.get("fd_epsilon", 1e-6))

    def propose_parameters(self, state, surrogate=None):
        """Finite-difference gradient descent in parameter space."""
        if self.X is None:
            return None

        # Flatten structure: dict-of-dicts → 1D vector
        X_flat, schema = self._flatten_params(self.X)
        n = len(X_flat)

        # Compute baseline objective
        base_obj = self.objective(self.R)

        grad = np.zeros_like(X_flat)

        for i in range(n):
            # Perturb parameter i
            d = self.eps * max(1.0, abs(X_flat[i]))
            Xp = X_flat.copy()
            Xp[i] += d

            # Unflatten and project bounds
            Xp_dict = self._unflatten_params(Xp, schema)
            Xp_dict = self._project_bounds(Xp_dict)

            # Evaluate objective at perturbed parameters
            self._evaluate(use_surr=surrogate)
            Rp = self._compute_residuals(Xp_dict, state, surrogate)
            obj_p = self.objective(Rp)

            # Finite difference slope
            grad[i] = (obj_p - base_obj) / d

        # Gradient descent step
        X_new_flat = X_flat - self.alpha * grad

        # Convert back to dict and apply bounds
        X_new_dict = self._unflatten_params(X_new_flat, schema)
        X_new_dict = self._project_bounds(X_new_dict)

        return X_new_dict


class BayesianOptSolver(SolverBase):
    def __init__(self, options=None):
        super().__init__(options)
        self.n_candidates = int(self.options.get("n_candidates", 32))
        self.proposal_sigma = float(self.options.get("proposal_sigma", 0.1))
    def propose_parameters(self, iteration, state, residual):
        if self.X is None:
            return np.zeros(1, float)
        X0 = self.X
        C = X0 + self.proposal_sigma * np.random.randn(self.n_candidates, X0.size)
        # score by synthetic objective: distance to zero residual direction
        if residual is None:
            return X0
        scores = -np.abs(np.mean(residual)) * np.ones(self.n_candidates)
        idx = int(np.argmax(scores))
        return C[idx]

class TimeStepperSolver(SolverBase):
    """SolverBase child that integrates parameter evolution in pseudo-time."""

    def __init__(self, options=None):
        super().__init__(options)
        self.dt = float(self.options.get("dt", 1e-3))
        self.method = self.options.get("method", "BDF")
        self.vectorized = bool(self.options.get("vectorized", False))


    def _evaluate_with_residual(self, t, X, use_surr: bool):
        self._update_from_params(X, self._state, self._parameters, self._boundary, self._neutrals, self._targets)
        self._evaluate(use_surr)
        R = self._compute_residuals(self.Y, self.Y_target)
        return R
    
    def _jacobian_as_array(self, X, R, use_surr: bool):
        J = self._attempt_get_jacobian(
            self._state, X, R,
            self._parameters, self._boundary,
            self._transport, self._targets,
            self._neutrals, None
        )
        if J is None:
            raise RuntimeError("Unable to compute Jacobian for TimeStepperSolver.")
        return J

    def propose_parameters(self, iteration, state, residual, surrogate=None):
        """Advance one pseudo-time step (Δt)."""

        X0 = self.X
        t_curr = iteration * self.dt
        t_next = t_curr + self.dt

        # convert to half grid
        if getattr(self, "roa_half", None) is None:
            self.roa_half = 0.5 * (self.roa_eval[:-1] + self.roa_eval[1:])
            self.roa_half = np.pad(self.roa_half, ((1, 1), (0, 0)), mode='edge')

        # ----------------------------
        # Integrate one Δt step
        # ----------------------------
        sol = sp.integrate.solve_ivp(
            self._evaluate_with_residual,
            (t_curr, t_next),
            X0,
            method=self.method,
            t_eval=[t_next],
            vectorized=self.vectorized,
        )

        X_new = np.asarray(sol.y[:, -1])
        self.X = X_new
        return X_new


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SOLVER_MODELS = {
    "relax": RelaxSolver,
    "finite_difference": FiniteDifferenceSolver,
    "bayesian_opt": BayesianOptSolver,
    "timestepper": TimeStepperSolver,
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

