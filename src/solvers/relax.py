from __future__ import annotations

import numpy as np

from .solver_base import SolverBase


class Relax(SolverBase):
    def __init__(self, options=None):
        super().__init__(options)
        opts = options or {}
        # Fractional clamp on the normalised step dx (applied before scaling by |x|).
        # PORTALS uses ~0.5; None disables.
        self.dx_max = float(opts.get("dx_max", 1.0))
        # Absolute clamp on x_step = dx * |x| after scaling (None = disabled).
        self.dx_max_abs = opts.get("dx_max_abs", None)
        if self.dx_max_abs is not None:
            self.dx_max_abs = float(self.dx_max_abs)
        # Absolute minimum on |x_step| (enforces a floor on each move).
        self.dx_min_abs = opts.get("dx_min_abs", None)
        if self.dx_min_abs is not None:
            self.dx_min_abs = float(self.dx_min_abs)

    def propose_parameters(self):
        """Relaxation-based parameter update with gradient clipping for stability."""
        R = np.asarray(self.R, float)

        if self._adaptive_step_ctrl is not None:
            step_size = self._adaptive_step_ctrl.compute_step_size(
                X=self.X, R=R, J=getattr(self, "J", None), iteration=self.iter
            )
        else:
            step_size = self.step_size

        # PORTALS simple relaxation:
        #   denom = sqrt(Q^2 + QT^2).clamp(min=1e-10)
        #   dx    = relax * (QT - Q) / denom        [bounded in (-sqrt(2), sqrt(2))]
        #   dx    = clamp(dx, -dx_max, dx_max)
        #   x_step = dx * abs(x)
        #   optionally clamp x_step with dx_max_abs / dx_min_abs
        X_flat, schema = self._flatten_params(self.X)

        Y = getattr(self, "Y", None)
        Y_target = getattr(self, "Y_target", None)
        y_keys = sorted([k for k in (Y or {}).keys() if k in (Y_target or {})])

        if y_keys:
            Q = np.concatenate([np.asarray(Y[k], float) for k in y_keys])
            QT = np.concatenate([np.asarray(Y_target[k], float) for k in y_keys])

            denom = np.maximum(np.sqrt(Q ** 2 + QT ** 2), 1e-10)
            dx = step_size * (QT - Q) / denom          # relax * (QT - Q) / denom
            dx = np.clip(dx, -self.dx_max, self.dx_max)

            n_x = len(X_flat)
            n_q = len(Q)

            if n_x == n_q:
                # Direct element-wise application (typical case: knots == roa_eval)
                x_step = dx * np.abs(X_flat)
            else:
                # Sizes differ (e.g. spline knots < n_eval).
                # Compute mean dx per profile channel then broadcast to parameters.
                n_profiles = len(y_keys)
                n_pts = n_q // n_profiles if n_profiles else n_q
                dx_per_profile = dx.reshape(n_profiles, n_pts).mean(axis=1)
                n_params_each = n_x // n_profiles if n_profiles else n_x
                dx_broadcast = np.repeat(dx_per_profile, n_params_each)[:n_x]
                x_step = dx_broadcast * np.abs(X_flat)

            if self.dx_max_abs is not None:
                x_step = np.sign(x_step) * np.minimum(np.abs(x_step), self.dx_max_abs)
            if self.dx_min_abs is not None:
                # sign_or_plus_one: treat zero-step as positive direction
                sign_or_plus = np.where(x_step != 0, np.sign(x_step), 1.0)
                x_step = sign_or_plus * np.maximum(np.abs(x_step), self.dx_min_abs)

            X_new_flat = X_flat + x_step
            X_new_wo_bounds = self._unflatten_params(X_new_flat)
        else:
            # No flux data yet: fall back to sign-based uniform step
            sign_term = float(np.sign(np.nanmean(R)))
            step = -step_size * sign_term
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
