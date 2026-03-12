"""Newton-direction iterative solver for transport fixed-point problems.

Solves F(X) = 0 by computing the Newton direction d = -J^{-1}R at each
iteration and applying it as X_{k+1} = X_k + alpha * step_size * d_normalized.
The SolverBase iteration loop handles convergence checking, best-model tracking,
and checkpointing.

Robustness measures for stiff / ill-conditioned Jacobians
---------------------------------------------------------
1. Column equilibration
     J_hat = J * diag(1 / ||J[:,j]||)
   Normalises column scales before the lstsq solve so that a parameter with
   large sensitivity (large J column) does not dominate and a parameter with
   near-zero sensitivity does not produce an unbounded pseudo-inverse step.

2. SVD truncation (lstsq rcond)
   Discards singular directions of J_hat whose singular value is below
   rcond * sigma_max.  This zeroes out directions where J is rank-deficient
   instead of producing enormous steps in the null-space.

3. Trust-region direction normalisation
   After solving, rescale d so max(|d|) = 1.  step_size then controls the
   actual move magnitude independently of J's scale.

4. Bound-distance protection  (bound_safe_frac, default 0.5)
   Computes a global multiplier alpha in (0, 1] after the direction is known:
     alpha = min over all i of  bound_safe_frac * room_i / move_i
   where room_i = distance from x_i to the nearest bound in direction d_i,
   and move_i = |step_size * d_i|.  This guarantees that no component travels
   more than bound_safe_frac of its remaining room in one iteration, preventing
   the cascade failure where a parameter is pinned at a bound and corrupts
   subsequent FD Jacobian evaluations.

5. Steepest-descent fallback  (direction = -J^T R)
   Used when the linear solve fails (singular J) or J is unavailable.

References
----------
Nocedal & Wright (2006): Numerical Optimization, Ch. 4 (trust region), Ch. 7
More, Garbow, Hillstrom (1980): MINPACK — Levenberg-Marquardt trust region
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from .solver_base import SolverBase


class RootSolver(SolverBase):
    """Iterative Newton-direction solver for transport fixed-point problems.

    Each call to propose_parameters() computes one Newton step
    (direction d = -J^{-1}R, magnitude controlled by step_size and alpha).
    The SolverBase loop drives the iterations, so best-model tracking,
    surrogate switching, convergence detection, and checkpointing are
    automatically inherited.

    Options
    -------
    step_size : float, default 1.0
        Base step multiplier applied to the normalised Newton direction.
        Values < 1 damp the step (safer for highly nonlinear problems).
    bound_safe_frac : float, default 0.5
        Maximum fraction of remaining distance to a bound that one step may
        consume.  Prevents cascade-to-bounds failures on stiff problems.
    jacobian_reg : float, default 1e-3
        Effective rcond for the lstsq direction solve.  Singular values of J_hat
        below this fraction of sigma_max are truncated to zero.  Increase (e.g.
        1e-2) for more aggressive regularisation when the Jacobian is noisy.
    adaptive_step : bool, default False
        If True, uses AdaptiveStepControl to modulate step_size each iteration.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        opts = options or {}
        self.bound_safe_frac = float(opts.get("bound_safe_frac", 0.5))

    def propose_parameters(self):
        """Newton direction step with trust-region + bound-distance protection."""
        R = np.asarray(self.R, float)
        X_flat, schema = self._flatten_params(self.X)

        if self._adaptive_step_ctrl is not None:
            step_size = self._adaptive_step_ctrl.compute_step_size(
                X=self.X, R=R, J=getattr(self, "J", None), iteration=self.iter
            )
        else:
            step_size = self.step_size

        # ---- Newton direction ----------------------------------------
        delta = None
        J = None
        try:
            J = self._attempt_get_jacobian(X_flat, R)
            self.J = J
        except Exception:
            pass

        if J is not None and np.all(np.isfinite(J)):
            try:
                # Column equilibration: normalise each column of J to unit norm.
                # A large-norm column (highly sensitive parameter) is scaled to 1,
                # preventing it from trivially dominating the lstsq solution.
                # A near-zero-norm column (insensitive parameter) is also scaled
                # to 1, but its corresponding singular value will be small so
                # lstsq with rcond truncates that direction instead of inverting it.
                col_norms = np.linalg.norm(J, axis=0)
                col_scale = 1.0 / np.maximum(col_norms, 1e-12)
                J_hat = J * col_scale

                # SVD-least-squares: truncate null-/near-null-space directions.
                # rcond = max(jacobian_reg, 1e-3) is the relative singular-value
                # cutoff.  1e-3 handles condition numbers up to ~1000 cleanly
                # without producing the enormous pseudo-inverse steps that arise
                # when a near-zero singular value is naively inverted.
                rcond = max(self.jacobian_reg, 1e-3)
                z, _, _, _ = np.linalg.lstsq(J_hat, -R, rcond=rcond)
                d = col_scale * z

                if np.all(np.isfinite(d)):
                    # Trust-region normalisation: cap the largest component at 1.
                    # step_size then sets the actual move magnitude irrespective
                    # of J's scale, making it a well-defined tuning parameter.
                    max_abs = float(np.max(np.abs(d)))
                    if max_abs > 1.0:
                        d /= max_abs
                    delta = d
            except (np.linalg.LinAlgError, ValueError):
                pass

        if delta is None:
            # Steepest-descent fallback: d = -J^T R / ||J^T R||.
            # Always reduces the residual norm locally even when the Newton
            # direction is unavailable (rank-deficient J, failed FD, etc.).
            if J is not None and np.all(np.isfinite(J)):
                grad = J.T @ R
                g_norm = np.linalg.norm(grad)
                delta = -grad / g_norm if g_norm > 1e-12 else np.zeros_like(X_flat)
            else:
                delta = np.zeros_like(X_flat)

        # ---- Bound-distance protection --------------------------------
        # Find a global multiplier alpha in (0, 1] such that no component of
        # X_flat + alpha*step_size*delta travels more than bound_safe_frac of
        # its remaining distance to the nearest bound in the direction of travel.
        # This is the primary guard against the jump-to-bounds / cascade failure
        # mode: once a parameter hits its limit, every subsequent FD Jacobian
        # evaluation either clamps to the same bound or reflects off it, making
        # the column garbage and causing a diverging cascade on following iters.
        lo_arr, hi_arr = self._get_bounds_vectors()
        alpha = 1.0
        if lo_arr is not None and hi_arr is not None and len(lo_arr) == len(X_flat):
            for i in range(len(X_flat)):
                d_i = float(delta[i])
                if d_i > 0 and np.isfinite(hi_arr[i]):
                    room = float(hi_arr[i]) - float(X_flat[i])
                    if room > 1e-12:
                        move = step_size * d_i
                        if move > self.bound_safe_frac * room:
                            alpha = min(alpha, self.bound_safe_frac * room / move)
                elif d_i < 0 and np.isfinite(lo_arr[i]):
                    room = float(X_flat[i]) - float(lo_arr[i])
                    if room > 1e-12:
                        move = -step_size * d_i
                        if move > self.bound_safe_frac * room:
                            alpha = min(alpha, self.bound_safe_frac * room / move)

        X_new_flat = X_flat + alpha * step_size * delta
        X_new_wo_bounds = self._unflatten_params(X_new_flat)
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
