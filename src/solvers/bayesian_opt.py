"""Surrogate-based Bayesian optimization solver.

Acquisition optimiser chain (PORTALS-aligned):
  1. Latin Hypercube seeding over effective bounds (replaces Gaussian ball).
  2. Simple-relax (SR) on the GP surrogate for each candidate (cheap; no physics).
  3. scipy.root(method='lm') from the best SR candidate (replaces manual gradient ascent).

Default acquisition: ``posterior_mean`` — maximise the GP posterior mean of the
scalar residual.  Pure exploitation; appropriate for flux-matching root-finding where
physics evaluations are expensive.  MC acquisitions (``ei``, ``ucb``) remain available
but are not the default.

Trust-region (``apply_trust_region=True``, enabled by default): axis-aligned geometric
bounds shrink/expand driven by the PORTALS BO fitness metric (ratio of actual vs
surrogate-predicted improvement).  No external package required — pure numpy.

Removed from original implementation
--------------------------------------
- Gradient-ascent inner loop (``n_steps``, ``lr``) → replaced by SR + scipy LM.
- ``surrogate_grad_batch`` / ``fd_grad_batch`` → scipy LM computes its own Jacobian.
- Adam parameters (``adam_beta1``, ``adam_beta2``, ``adam_eps``) — were dead code.
- ``batch_size`` — merged into ``n_restarts``.
- ``pi_k`` / PI acquisition — niche; poorly tuned relative to posterior_mean.
- Gaussian-ball candidate seeding → Latin Hypercube for better coverage.
- ``import copy`` — state restored via ``_update_from_params`` instead of deepcopy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.optimize as _sopt
from scipy.stats.qmc import LatinHypercube

from .relax import Relax
from .solver_base import SolverBase


class Bayesian(SolverBase):
    """Surrogate-based BO solver: LHS seeding → SR on surrogate → LM root-finder."""

    def __init__(self, options: Optional[dict] = None):
        super().__init__(options)
        self.n_restarts  = int(self.options.get("n_restarts", 5))
        self.seed        = int(self.options.get("seed", 0))
        self.acquisition = str(self.options.get("acquisition", "posterior_mean")).lower()
        self.n_mc        = int(self.options.get("n_mc", 256))    # MC acquisitions only
        self.ucb_k       = float(self.options.get("ucb_k", 2.0))
        self.sr_relax    = float(self.options.get("sr_relax", 0.1))
        self.sr_maxiter  = int(self.options.get("sr_maxiter", 200))
        self.lm_maxiter  = int(self.options.get("lm_maxiter", 500))

        # Trust-region (disabled by default)
        self.apply_trust_region = bool(self.options.get("apply_trust_region", True))
        self.tr_shrink   = float(self.options.get("tr_shrink", 0.75))
        self.tr_expand   = float(self.options.get("tr_expand", 1.33))
        self.tr_n_bad    = int(self.options.get("tr_n_bad", 3))
        self.tr_n_good   = int(self.options.get("tr_n_good", 3))

        self._tr_lo: Optional[np.ndarray] = None
        self._tr_hi: Optional[np.ndarray] = None
        self._tr_lo_init: Optional[np.ndarray] = None
        self._tr_hi_init: Optional[np.ndarray] = None
        self._bo_metrics: list = []
        self._Z_surr_predicted: Optional[float] = None
        self._Z_at_proposal: Optional[float] = None
        # Fraction of the full bounds range used as the *initial* half-width of the
        # trust region, centred on the starting parameters.  A value of 0.3 means the
        # initial TR spans 30 % of [lo, hi] on each side of X0.  This prevents the LHS
        # from seeding across the entire (often very wide) parameter bounds before the
        # surrogate has been validated against physics.
        self.tr_init_frac = float(self.options.get("tr_init_frac", 0.1))

        # Related to relaxation
        # Fractional clamp on the normalised step dx (applied before scaling by |x|).
        # PORTALS uses ~0.5; None disables.
        self.dx_max = float(self.options.get("dx_max", 1.0))
        # Absolute clamp on x_step = dx * |x| after scaling (None = disabled).
        self.dx_max_abs = self.options.get("dx_max_abs", None)
        if self.dx_max_abs is not None:
            self.dx_max_abs = float(self.dx_max_abs)
        # Absolute minimum on |x_step| (enforces a floor on each move).
        self.dx_min_abs = self.options.get("dx_min_abs", None)
        if self.dx_min_abs is not None:
            self.dx_min_abs = float(self.dx_min_abs)

    # ------------------------------------------------------------------
    # Surrogate evaluation (does not mutate self.R / self.Z)
    # ------------------------------------------------------------------

    def _eval_surr_at(self, x_flat: np.ndarray):
        """Evaluate surrogate at x_flat. Returns (transport, transport_std, targets, R).

        Uses ``_compute_residuals`` for exact consistency with the solver objective
        (normalisation, lcfs exclusion, weights).  self.R is saved and restored so
        inner-loop calls do not corrupt the solver's residual state.

        Returns None if state reconstruction or surrogate evaluation fails (e.g.,
        non-finite profile values triggering interpolation errors in a LHS candidate).
        """
        R_saved      = self.R
        R_dict_saved = getattr(self, "R_dict", None)
        try:
            self._update_from_params(x_flat)
            transport, transport_std, targets, _ = self._surrogate.evaluate(
                self._unflatten_params(x_flat), self._state
            )
            R = self._compute_residuals(transport, targets)
            return transport, transport_std, targets, R
        except Exception:
            return None
        finally:
            self.R = R_saved
            if R_dict_saved is not None:
                self.R_dict = R_dict_saved
            # Restore state to current best so a failed candidate leaves no side-effects
            try:
                self._update_from_params(self.X)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Acquisition (lower = better for all types)
    # ------------------------------------------------------------------

    def _acq(self, x_flat: np.ndarray, rng: np.random.Generator) -> float:
        """Scalar acquisition value — always lower is better.

        posterior_mean : objective(mu)  — direct surrogate Z; no sampling.
        ei             : -E[max(0, Z_best - Z_sample)]  (negated; minimise).
        ucb            : mean(Z) - k*std(Z)  (lower-confidence bound).

        Returns inf for candidates that produce non-finite state reconstructions
        so they are always ranked last.
        """
        result = self._eval_surr_at(x_flat)
        if result is None:
            return float("inf")
        transport, transport_std, targets, R = result
        if R is None or not np.all(np.isfinite(R)):
            return float("inf")
        Z = float(self.objective(R))

        if self.acquisition == "posterior_mean":
            return Z

        # MC path: sample transport, recompute residual in correct units
        keys = sorted(self.target_vars)
        M = {k: np.asarray(transport[k])                                  for k in keys if k in transport}
        S = {k: np.asarray(transport_std.get(k, np.zeros_like(M[k])))    for k in keys if k in M}
        T = {k: np.asarray(targets[k])                                    for k in keys if k in targets}
        n_roa = len(self.roa_eval)

        R_saved      = self.R
        R_dict_saved = getattr(self, "R_dict", None)
        Z_samp = np.empty(self.n_mc)
        try:
            for i in range(self.n_mc):
                y_s = {k: M[k] + S[k] * rng.normal(size=n_roa) for k in keys if k in M}
                Z_samp[i] = float(self.objective(self._compute_residuals(y_s, T)))
        finally:
            self.R = R_saved
            if R_dict_saved is not None:
                self.R_dict = R_dict_saved

        if self.acquisition == "ei":
            Z_best = float(self.Z) if (self.Z is not None and np.isfinite(self.Z)) else float("inf")
            return float(-np.mean(np.maximum(0.0, Z_best - Z_samp)))
        if self.acquisition == "ucb":
            return float(np.mean(Z_samp) - self.ucb_k * np.std(Z_samp))
        return Z  # fallback

    # ------------------------------------------------------------------
    # Simple-relax on surrogate (PORTALS Stage-1 acquisition optimiser)
    # ------------------------------------------------------------------

    def _sr_on_surrogate(self, x0: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """PORTALS-style closed-form relaxation applied to the surrogate.

        Each step:  dx_k = relax * (Q_tar - Q_tr) / sqrt(Q_tar^2 + Q_tr^2)
                    x    = clip(x + dx * |x|, lo, hi)
        """
        x    = x0.copy()
        keys = sorted(self.target_vars)
        n_x  = x.size

        for _ in range(self.sr_maxiter):
            try:
                self._update_from_params(x)
                transport, _, targets, _ = self._surrogate.evaluate(
                    self._unflatten_params(x), self._state
                )
            except Exception:
                break
            Q_tr  = np.concatenate([np.asarray(transport.get(k, np.zeros(len(self.roa_eval)))) for k in keys])
            Q_tar = np.concatenate([np.asarray(targets.get(k,   np.zeros(len(self.roa_eval)))) for k in keys])
            denom = np.maximum(np.sqrt(Q_tr**2 + Q_tar**2), 1e-10)
            dx    = np.clip(self.sr_relax * (Q_tar - Q_tr) / denom, -1.0, 1.0)

            # Broadcast dx (n_chan*n_roa) → x_step (n_x): mirrors Relax.propose_parameters
            n_q = dx.size
            if n_x == n_q:
                x_step = dx * np.abs(x)
            else:
                n_ch      = len(keys)
                dx_per_ch = dx.reshape(n_ch, -1).mean(axis=1)
                n_each    = n_x // n_ch if n_ch else n_x
                x_step    = np.repeat(dx_per_ch, n_each)[:n_x] * np.abs(x)

            x_prev = x.copy()
            x = np.clip(x + x_step, lo, hi)
            if np.max(np.abs(x - x_prev)) < 1e-8:
                break
        return x

    # ------------------------------------------------------------------
    # Trust-region
    # ------------------------------------------------------------------

    def _compute_bo_metric(self, Z_prev: float,
                           Z_actual: float, Z_surr: float) -> float:
        """Signed BO fitness metric (PORTALS §6.2).

        +ve = physics improved; magnitude encodes surrogate accuracy (1–3).
        """
        if not (np.isfinite(Z_prev) and Z_prev != 0.0 and np.isfinite(Z_surr)):
            return 0.0
        delta_A = (Z_prev - Z_actual) / abs(Z_prev)
        delta_M = (Z_prev - Z_surr)   / abs(Z_prev)
        if delta_M == 0.0:
            return 0.0
        r        = delta_A / delta_M
        improved = delta_A > 0
        # Lucky/unlucky: sign of r disagrees with improvement direction
        if improved and r <= 0:
            return  0.2
        if not improved and r <= 0:
            return -0.2
        sign = 1 if improved else -1
        mag  = 1 if abs(r) > 2.0 else (2 if abs(r) > 1.5 else 3)
        return sign * mag

    def _update_trust_region(self, x_best: np.ndarray):
        lo_init, hi_init = self._tr_lo_init, self._tr_hi_init
        cur_lo,  cur_hi  = self._tr_lo,      self._tr_hi

        # Consecutive tail failures/successes since the last resize event
        metrics     = self._bo_metrics
        last_resize = next((i for i in range(len(metrics) - 1, -1, -1)
                            if metrics[i] == 0.0), -1)
        window = metrics[last_resize + 1:]
        if not window:
            return

        n_c_fail = next((i for i, m in enumerate(reversed(window)) if m >= 0), len(window))
        n_c_good = next((i for i, m in enumerate(reversed(window)) if m <  0), len(window))

        rho = None
        if n_c_fail >= self.tr_n_bad:
            rho = self.tr_shrink
            self._bo_metrics.append(0.0)   # mark resize
        elif n_c_good >= self.tr_n_good:
            rho = self.tr_expand
            self._bo_metrics.append(0.0)
        if rho is None:
            return

        if rho < 1.0:   # shrink — center on best point
            half         = (cur_hi - cur_lo) * rho / 2.0
            self._tr_lo  = np.clip(x_best - half, lo_init, hi_init)
            self._tr_hi  = np.clip(x_best + half, lo_init, hi_init)
        else:           # expand — center on midpoint of current bounds
            mid          = (cur_lo + cur_hi) / 2.0
            half         = (cur_hi - cur_lo) * rho / 2.0
            self._tr_lo  = np.clip(mid - half, lo_init, hi_init)
            self._tr_hi  = np.clip(mid + half, lo_init, hi_init)

        # Hard stop when all dims shrunk to < 0.5 % of initial range
        d_init = hi_init - lo_init
        d_cur  = self._tr_hi - self._tr_lo
        if np.all((d_cur / np.maximum(d_init, 1e-12)) < 0.005):
            self.converged = True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def propose_parameters(self):
        """Surrogate BO: LHS seeding → SR on surrogate → LM root-finder.

        Falls back to Relax during warm-up (iter < surr_warmup).
        """
        if getattr(self, "iter", 0) < self.surr_warmup:
            return Relax.propose_parameters(self)

        X_flat0, schema = self._flatten_params(self.X)
        n_params = X_flat0.size

        lo_arr = np.array([
            -1e6 if self.bounds_dict[p][n][0] is None else float(self.bounds_dict[p][n][0])
            for p, n in schema
        ])
        hi_arr = np.array([
            1e6  if self.bounds_dict[p][n][1] is None else float(self.bounds_dict[p][n][1])
            for p, n in schema
        ])

        # ── Trust-region bookkeeping ─────────────────────────────────────────
        if self.apply_trust_region:
            if self._tr_lo is None:
                self._tr_lo_init, self._tr_hi_init = lo_arr.copy(), hi_arr.copy()
                # Initialise TR to tr_init_frac of the full bounds range centred on X0.
                # This stops the LHS from seeding across the entire [0, 100] parameter
                # space before the surrogate has been validated against real physics.
                half = (hi_arr - lo_arr) * (self.tr_init_frac / 2.0)
                self._tr_lo = np.clip(X_flat0 - half, lo_arr, hi_arr)
                self._tr_hi = np.clip(X_flat0 + half, lo_arr, hi_arr)
            # Retrospective BO metric: only meaningful when self.Z is an actual physics
            # value, not a surrogate prediction.  During surrogate-only iterations the
            # metric is trivially ~+3 (the surrogate predicts its own output), which
            # causes n_c_good to accumulate and the TR to expand spuriously.
            if not self._use_surr_iter:
                if (self._Z_at_proposal is not None and self._Z_surr_predicted is not None
                        and self.Z is not None and np.isfinite(self.Z)):
                    m = self._compute_bo_metric(
                        self._Z_at_proposal, float(self.Z), self._Z_surr_predicted
                    )
                    self._bo_metrics.append(m)
                    self._update_trust_region(X_flat0)

        eff_lo = self._tr_lo if (self.apply_trust_region and self._tr_lo is not None) else lo_arr
        eff_hi = self._tr_hi if (self.apply_trust_region and self._tr_hi is not None) else hi_arr

        int_seed = (self.seed + getattr(self, "iter", 0)) % (2**31)
        rng = np.random.default_rng(int_seed)

        # ── Stage 1: Latin Hypercube seeding ────────────────────────────────
        lhs        = LatinHypercube(d=n_params, seed=int_seed).random(n=self.n_restarts)
        candidates = eff_lo + lhs * (eff_hi - eff_lo)
        candidates[0] = np.clip(X_flat0, eff_lo, eff_hi)   # always include current best

        # ── Stage 2: SR on surrogate for each candidate ─────────────────────
        sr_results = [self._sr_on_surrogate(c, eff_lo, eff_hi) for c in candidates]

        # ── Stage 3: LM root-finder from the best SR candidate ───────────────
        acq_sr        = np.array([self._acq(x, rng) for x in sr_results])
        x_lm_start    = sr_results[int(np.argmin(acq_sr))]

        def _mismatch(x_flat):
            result = self._eval_surr_at(x_flat)
            if result is None:
                return np.full(n_params, 1e6)
            _, _, _, R = result
            return R if (R is not None and np.all(np.isfinite(R))) else np.full(n_params, 1e6)

        try:
            res  = _sopt.root(_mismatch, x_lm_start, method="lm",
                              options={"maxiter": self.lm_maxiter,
                                       "xtol": 1e-6, "ftol": 1e-6})
            x_lm = np.clip(res.x, eff_lo, eff_hi) if np.all(np.isfinite(res.x)) else x_lm_start
        except Exception:
            x_lm = x_lm_start

        # ── Stage 4: select best across all SR + LM candidates ──────────────
        all_candidates = np.vstack([sr_results, x_lm[None]])
        acq_all        = np.array([self._acq(x, rng) for x in all_candidates])
        best_idx       = int(np.argmin(acq_all))
        best_x         = all_candidates[best_idx]

        # Store surrogate-predicted Z for next iteration's BO metric
        result_best = self._eval_surr_at(best_x)
        if result_best is not None:
            _, _, _, R_best = result_best
            self._Z_surr_predicted = float(self.objective(R_best)) if (R_best is not None and np.all(np.isfinite(R_best))) else None
        else:
            self._Z_surr_predicted = None
        # Only snapshot Z_at_proposal when the most recent evaluation was a physics call.
        # Surrogate Z values are self-consistent but not physics ground truth, so using
        # them as the "before" baseline corrupts the BO trust-region metric.
        if not self._use_surr_iter:
            self._Z_at_proposal = float(self.Z) if (self.Z is not None and np.isfinite(self.Z)) else None

        # Restore solver state to current best point before returning
        self._update_from_params(self.X)

        return self._project_bounds(self._unflatten_params(best_x)), {}
