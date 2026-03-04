from __future__ import annotations

import copy

import numpy as np

from .solver_base import SolverBase


class RelaxSolver(SolverBase):
    def __init__(self, options=None):
        super().__init__(options)

    def propose_parameters(self):
        """Relaxation-based parameter update with gradient clipping for stability."""
        R = np.asarray(self.R, float)

        if self.adaptive_step:
            res_norm = np.linalg.norm(R)
            step_size = max(self.min_step_size, self.step_size * (1.0 / (1.0 + res_norm)))
        else:
            step_size = self.step_size

        if self.use_jacobian:
            X_flat, schema = self._flatten_params(self.X)
            direction = np.sign(np.nanmean(R)) * np.ones_like(X_flat)

            J = getattr(self, "J", None)
            if J is None:
                try:
                    J = self._attempt_get_jacobian(X_flat, R)
                    self.J = J
                except Exception:
                    J = None

            if J is not None:
                col_norms = np.linalg.norm(J, axis=0)
                col_scale = 1.0 / np.maximum(col_norms, 1e-12)
                J_hat = J * col_scale

                JTJ = J_hat.T @ J_hat + self.jacobian_reg * np.eye(J.shape[1])
                rhs = -J_hat.T @ R
                z = np.linalg.solve(JTJ, rhs)
                delta = col_scale * z

                if np.all(np.isfinite(delta)):
                    direction = delta

                X_new_flat = X_flat + step_size * direction
                X_new_wo_bounds = self._unflatten_params(X_new_flat)
            else:
                X_new_wo_bounds = copy.deepcopy(self.X)

        else:
            sign_term = float(np.sign(np.nanmean(R)))
            step = -self.step_size * sign_term
            X_new_wo_bounds = {}
            for prof, param_dict in self.X.items():
                new_inner = {}
                for pname, pval in (param_dict or {}).items():
                    try:
                        new_inner[pname] = float(pval) + step
                    except Exception:
                        new_inner[pname] = pval
                X_new_wo_bounds[prof] = new_inner

        X_new = self._project_bounds(X_new_wo_bounds)

        X_new_std = {
            prof: {name: abs(val) * self._parameters.sigma for name, val in X_new[prof].items()}
            for prof in X_new
        }
        for prof in X_new.keys():
            for param in X_new[prof].keys():
                if param.startswith("log_"):
                    X_new_std[prof][param] = X_new[prof][param] + np.log(self._parameters.sigma)
        return X_new, X_new_std
