from __future__ import annotations

import numpy as np
import scipy as sp

from .solver_base import SolverBase


class IvpSolver(SolverBase):
    """Integrate parameter evolution in pseudo-time using BDF for stiff systems."""

    def __init__(self, options=None):
        super().__init__(options)
        self.dt = float(self.options.get("step_size", 1e-2))
        self.method = self.options.get("method", "BDF")
        self.vectorized = bool(self.options.get("vectorized", False))

    def _ode_rhs(self, t, X_flat):
        """Compute dX/dt = -alpha*R for residual-driven dynamics."""
        self._update_from_params(X_flat)
        self._evaluate(in_place=True)
        R = self._compute_residuals(self.Y, self.Y_target)
        if R is None:
            return np.zeros_like(X_flat)

        if not hasattr(self, "_J_cache") or self._J_cache is None:
            self._J_cache = self._attempt_get_jacobian(X_flat, R)

        if self._J_cache is not None:
            dX = -sp.linalg.pinv(self._J_cache) @ R
        else:
            dX = -np.sign(R).mean() * np.ones_like(X_flat) * self.step_size

        return dX

    def _jacobian(self, t, X_flat):
        """Jacobian for implicit ODE methods."""
        if hasattr(self, "_J_cache") and self._J_cache is not None:
            J = self._J_cache
        else:
            self._update_from_params(X_flat)
            self._evaluate(in_place=True)
            R = self._compute_residuals(self.Y, self.Y_target)
            if R is None:
                return np.zeros((X_flat.size, X_flat.size))
            J = self._attempt_get_jacobian(X_flat, R)
            if J is None:
                return np.zeros((X_flat.size, X_flat.size))
            self._J_cache = J

        J_pinv = sp.linalg.pinv(J)
        return -J_pinv @ J

    def propose_parameters(self):
        """Advance one pseudo-time step using residual-driven dynamics."""
        if self.X is None:
            raise RuntimeError("Parameters are not initialized; run get_initial_parameters() first.")

        if isinstance(self.X, dict):
            X0_flat, _ = self._flatten_params(self.X)
        else:
            X0_flat = np.asarray(self.X, dtype=float)

        X0 = np.asarray(X0_flat, float)
        t_curr = self.iter * self.dt
        t_next = t_curr + self.dt

        self._J_cache = None

        sol = sp.integrate.solve_ivp(
            fun=self._ode_rhs,
            t_span=(t_curr, t_next),
            y0=X0,
            method=self.method,
            t_eval=[t_next],
            args=(),
            max_step=self.dt,
            first_step=self.dt * 0.5,
            vectorized=self.vectorized,
            jac=self._jacobian,
            dense_output=False,
            rtol=float(self.options.get("ts_rtol", 1e-3)),
            atol=float(self.options.get("ts_atol", 1e-6)),
        )

        X_new_flat = np.asarray(sol.y[:, -1])
        X_new = self._unflatten_params(X_new_flat)
        X_new = self._project_bounds(X_new)
        X_new_std = {
            prof: {pname: abs(val) * getattr(self._parameters, "sigma", 0.0) for pname, val in prof_dict.items()}
            for prof, prof_dict in X_new.items()
        }
        return X_new, X_new_std
