from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np


class JacobianMixin:
    """Jacobian helpers shared by solvers."""

    def _objective_at(self, X: Dict[str, Dict[str, float]]) -> float:
        """Compute objective at params X on the current solver object."""
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

    def _attempt_get_jacobian(self, X: np.ndarray, R: np.ndarray):
        """Compute Jacobian with caching and staged fallbacks."""
        surrogate = self._surrogate if (self._use_surr_iter and self._surrogate is not None) else None

        if isinstance(X, dict):
            X_flat, _ = self._flatten_params(X)
            X_dict = X
        else:
            X_flat = np.asarray(X, float)
            X_dict = self._unflatten_params(X)

        if self.jacobian_cache_enabled and self._J_cache is not None:
            should_recompute = self._should_recompute_jacobian(R)
            if not should_recompute:
                return self._J_cache

        J = None

        if (
            self.jacobian_use_surrogate_grads
            and surrogate is not None
            and surrogate.trained
        ):
            try:
                J = self._jacobian_from_surrogate_grads(X_flat, X_dict, surrogate)
            except Exception:
                J = None

        if J is None and surrogate is not None and surrogate.trained:
            J = self._fd_jacobian(X_flat, R)

        if J is None:
            J = self._fd_jacobian(X_flat, R)

        if J is not None and self.jacobian_cache_enabled:
            self._J_cache = J
            self._J_cache_iter = self.iter
            self._R_cache = R.copy() if R is not None else None

        return J

    def _jacobian_from_surrogate_grads(
        self,
        X_flat: np.ndarray,
        X_dict: Dict[str, Dict[str, float]],
        surrogate,
    ) -> Optional[np.ndarray]:
        """Extract Jacobian directly from surrogate.get_grads() without finite differences."""
        try:
            if not hasattr(surrogate, "get_grads"):
                return None

            transport_grads, target_grads = surrogate.get_grads(X_dict, self._state)

            if transport_grads is None or target_grads is None:
                return None

            y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
            if not y_keys:
                return None

            J_rows = []
            for var_key in y_keys:
                grad_dict = transport_grads if var_key in transport_grads else target_grads
                if var_key not in grad_dict:
                    continue

                grad_var = np.asarray(grad_dict[var_key], dtype=float)
                if grad_var.ndim == 3:
                    grad_var = grad_var[0, :, :]

                for roa_idx in range(grad_var.shape[0]):
                    grad_features = grad_var[roa_idx, :]

                    if hasattr(surrogate, "param_features") and surrogate.param_features:
                        param_indices = [
                            surrogate.all_features.index(pf)
                            for pf in surrogate.param_features
                            if pf in surrogate.all_features
                        ]
                        grad_params = grad_features[param_indices]
                    else:
                        grad_params = grad_features

                    J_rows.append(grad_params)

            if not J_rows:
                return None

            J = np.vstack(J_rows).astype(float)
            return J if J.shape[1] == X_flat.size else None

        except Exception:
            return None

    def _should_recompute_jacobian(self, R: Optional[np.ndarray]) -> bool:
        """Check if cached Jacobian should be recomputed based on residual change."""
        if self._J_cache is None or self._R_cache is None:
            return True

        iters_since_cache = self.iter - self._J_cache_iter
        if iters_since_cache >= self.jacobian_recompute_every:
            return True

        if R is None:
            return True

        R_new = np.asarray(R, dtype=float)
        R_old = np.asarray(self._R_cache, dtype=float)

        if R_old.size != R_new.size:
            return True

        R_norm_old = np.linalg.norm(R_old)
        if R_norm_old < 1e-14:
            return iters_since_cache > 0

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
                self._evaluate(in_place=True)
                return self._compute_residuals(self.Y, self.Y_target)
            except Exception:
                return None

        R_base = np.asarray(R, float) if R is not None else _eval_residual(X_flat)
        if R_base is None or not np.all(np.isfinite(R_base)):
            return None

        for j in range(n):
            # Scale by the parameter value itself; avoid the bounds span which can be
            # huge (e.g. [0,100]) and would produce ~50% perturbations that push TGLF
            # into completely different turbulent regimes, making the Jacobian useless.
            scale_j = max(abs(X_flat[j]), 1.0)
            h = self.fd_epsilon * scale_j

            Xp = X_flat.copy()
            Xm = X_flat.copy()
            Xp[j] += h
            Xm[j] -= h

            Xp_clip, _ = _project_flat(Xp)
            Xm_clip, _ = _project_flat(Xm)

            eps_eff = Xp_clip[j] - Xm_clip[j]
            eps_floor = 1e-8 * max(scale_j, 1.0)

            # Check only element j: np.any(Xp_clip != X_flat) would be True
            # whenever *any* unrelated element drifted due to floating-point round-trip
            # through unflatten/flatten, which would silently trigger an unnecessary
            # full model evaluation for the wrong column.
            Rp = _eval_residual(Xp_clip) if Xp_clip[j] != X_flat[j] else None
            Rm = _eval_residual(Xm_clip) if Xm_clip[j] != X_flat[j] else None

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

        _eval_residual(X_flat)

        # Replace any remaining NaN/Inf entries (e.g. from a single failed FD
        # evaluation) with zero so they don't corrupt the subsequent lstsq solve.
        J = np.where(np.isfinite(J), J, 0.0)

        return J
