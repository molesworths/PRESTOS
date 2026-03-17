"""Surrogate-based Bayesian optimization solver.

Acquisition optimiser chain (PORTALS-aligned):
  1. Latin Hypercube seeding over effective bounds (replaces Gaussian ball).
  2. Dynamic simple-relax (SR) on the GP surrogate for each candidate.
  3. Bounded nonlinear mismatch solve from the best SR candidates
     (scipy.least_squares + LM fallback).

Default acquisition: ``posterior_mean`` — maximise the GP posterior mean of the
scalar residual.  Pure exploitation; appropriate for flux-matching root-finding where
physics evaluations are expensive.  MC acquisitions (``ei``, ``ucb``) remain available
but are not the default.

Trust-region (``apply_trust_region=True``, enabled by default): axis-aligned geometric
bounds shrink/expand driven by the PORTALS BO fitness metric (ratio of actual vs
surrogate-predicted improvement).  No external package required — pure numpy.

Removed from original implementation
--------------------------------------
- Gradient-ascent inner loop (``n_steps``, ``lr``) → replaced by SR + nonlinear solves.
- ``surrogate_grad_batch`` / ``fd_grad_batch`` → scipy solvers compute Jacobians.
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
    """Surrogate-based BO solver: LHS seeding → dynamic SR → nonlinear root solve."""

    def __init__(self, options: Optional[dict] = None):
        super().__init__(options)
        self.n_restarts  = int(self.options.get("n_restarts", 5))
        self.seed        = int(self.options.get("seed", 0))
        self.acquisition = str(self.options.get("acquisition", "posterior_mean")).lower()
        self.n_mc        = int(self.options.get("n_mc", 256))    # MC acquisitions only
        self.ucb_k       = float(self.options.get("ucb_k", 2.0))
        # Bias SR defaults toward quickly delivering a strong root-solve start.
        self.sr_relax    = float(self.options.get("sr_relax", 0.4))
        self.sr_maxiter  = int(self.options.get("sr_maxiter", 250))
        self.sr_relax_dyn = bool(self.options.get("sr_relax_dyn", True))
        self.sr_relax_min = float(self.options.get("sr_relax_min", 0.02))
        self.sr_relax_max = float(self.options.get("sr_relax_max", 1.0))
        self.sr_relax_grow = float(self.options.get("sr_relax_grow", 1.6))
        self.sr_relax_shrink = float(self.options.get("sr_relax_shrink", 0.5))
        self.sr_backtrack = int(self.options.get("sr_backtrack", 3))
        self.sr_rel_improve_stop = float(self.options.get("sr_rel_improve_stop", 5e-3))
        self.sr_plateau_patience = int(self.options.get("sr_plateau_patience", 8))

        # PORTALS-like dynamic stepping controls (per-channel, history-based).
        self.sr_dyn_num = int(self.options.get("sr_dyn_num", 100))
        self.sr_dyn_decrease = float(self.options.get("sr_dyn_decrease", 2.0))
        self.sr_dyn_window = int(self.options.get("sr_dyn_window", 32))
        self.sr_dyn_single_freq_thresh = float(self.options.get("sr_dyn_single_freq_thresh", 0.3))
        self.sr_dyn_high_freq_thresh = float(self.options.get("sr_dyn_high_freq_thresh", 0.5))
        self.sr_dyn_flat_std = float(self.options.get("sr_dyn_flat_std", 1e-6))
        self.sr_min_relax = float(self.options.get("sr_min_relax", 1e-6))
        self._sr_relax_channels: Optional[np.ndarray] = None
        self._sr_relax_last_applied_iter = -1
        self._sr_channel_signal_hist: list = []

        # Legacy threshold retained for config compatibility only.
        # Stage 3 now always runs from top SR candidates.
        self.acq_rel_improvement_for_stopping = float(
            self.options.get("acq_rel_improvement_for_stopping", 0.01)
        )

        # Nonlinear root stage
        self.lm_maxiter  = int(self.options.get("lm_maxiter", 1000))
        self.root_solver = str(self.options.get("root_solver", "least_squares")).lower()
        self.root_method = str(self.options.get("root_method", "trf")).lower()
        self.root_tol = float(self.options.get("root_tol", 1e-6))
        self.root_max_nfev = int(self.options.get("root_max_nfev", max(self.lm_maxiter, 1000)))
        self.root_n_restarts = int(self.options.get("root_n_restarts", 12))
        self.root_rel_improve_stop = float(self.options.get("root_rel_improve_stop", 1e-4))
        self.root_use_lm_fallback = bool(self.options.get("root_use_lm_fallback", True))
        self.root_loss = str(self.options.get("root_loss", "linear"))
        self.root_f_scale = float(self.options.get("root_f_scale", 1.0))

        # Cache repeated surrogate/acquisition evaluations inside one BO propose call.
        self.acq_cache_round_decimals = int(self.options.get("acq_cache_round_decimals", 12))
        self._acq_eval_cache: dict = {}
        self._acq_value_cache: dict = {}

        # Trust-region (enabled by default)
        self.apply_trust_region = bool(self.options.get("apply_trust_region", True))
        self.tr_shrink   = float(self.options.get("tr_shrink", 0.75))
        self.tr_expand   = float(self.options.get("tr_expand", 1.33))
        self.tr_n_bad    = int(self.options.get("tr_n_bad", 3))
        self.tr_n_good   = int(self.options.get("tr_n_good", 3))
        # If True, TR widths are expressed relative to |x| (with floors) rather
        # than only as fractions of the global bounds span.
        self.tr_scale_with_x = bool(self.options.get("tr_scale_with_x", True))
        # Relative-floor fraction of full bounds span used when |x| is small.
        self.tr_x_floor_frac = max(0.0, float(self.options.get("tr_x_floor_frac", 0.02)))
        # Optional absolute floor for local TR scaling (None disables).
        self.tr_x_floor_abs = self.options.get("tr_x_floor_abs", None)
        if self.tr_x_floor_abs is not None:
            self.tr_x_floor_abs = float(self.tr_x_floor_abs)

        self._tr_lo: Optional[np.ndarray] = None
        self._tr_hi: Optional[np.ndarray] = None
        self._tr_lo_init: Optional[np.ndarray] = None
        self._tr_hi_init: Optional[np.ndarray] = None
        self._bo_metrics: list = []
        self._Z_surr_predicted: Optional[float] = None
        self._Z_at_proposal: Optional[float] = None
        # Initial TR size control.
        # - tr_scale_with_x=True: half-width = tr_init_frac * local_scale(|x|),
        #   where local_scale has floors from bounds span / absolute floor.
        # - tr_scale_with_x=False (legacy): half-width = tr_init_frac * (hi-lo)/2.
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

        # BO uses the surrogate only inside propose_parameters(); the main
        # solver loop must always perform a real transport/target evaluation.
        # Track the training-set size used in the last BO surrogate fit so we
        # only refit when a new physics sample has been appended.
        self._bo_last_fit_n_samples = -1

    # ------------------------------------------------------------------
    # Solver loop integration
    # ------------------------------------------------------------------

    def _use_surrogate_iteration(self) -> bool:
        """BayesianOpt always evaluates real physics in SolverBase.run.

        Surrogates are used exclusively for acquisition optimisation inside
        propose_parameters(). This prevents surrogate-on-surrogate feedback.
        """
        return False

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

    def _acq(self, x_flat: np.ndarray, rng: Optional[np.random.Generator] = None) -> float:
        """Scalar acquisition value — always lower is better.

        posterior_mean : objective(mu)  — direct surrogate Z; no sampling.
        ei             : -E[max(0, Z_best - Z_sample)]  (negated; minimise).
        ucb            : mean(Z) - k*std(Z)  (lower-confidence bound).

        Returns inf for candidates that produce non-finite state reconstructions
        so they are always ranked last.
        """
        x_arr = np.asarray(x_flat, float)
        key = (self.acquisition, self._x_cache_key(x_arr))
        if key in self._acq_value_cache:
            return self._acq_value_cache[key]

        result = self._eval_surr_cached(x_arr)
        if result is None:
            self._acq_value_cache[key] = float("inf")
            return float("inf")
        transport, transport_std, targets, R = result
        if R is None or not np.all(np.isfinite(R)):
            self._acq_value_cache[key] = float("inf")
            return float("inf")
        Z = float(self.objective(R))

        if self.acquisition == "posterior_mean":
            self._acq_value_cache[key] = Z
            return Z

        # MC path: sample transport, recompute residual in correct units
        cand_rng = rng if rng is not None else self._candidate_rng(x_arr)
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
                y_s = {k: M[k] + S[k] * cand_rng.normal(size=n_roa) for k in keys if k in M}
                Z_samp[i] = float(self.objective(self._compute_residuals(y_s, T)))
        finally:
            self.R = R_saved
            if R_dict_saved is not None:
                self.R_dict = R_dict_saved

        if self.acquisition == "ei":
            Z_best = float(self.Z) if (self.Z is not None and np.isfinite(self.Z)) else float("inf")
            acq_value = float(-np.mean(np.maximum(0.0, Z_best - Z_samp)))
            self._acq_value_cache[key] = acq_value
            return acq_value
        if self.acquisition == "ucb":
            acq_value = float(np.mean(Z_samp) - self.ucb_k * np.std(Z_samp))
            self._acq_value_cache[key] = acq_value
            return acq_value
        self._acq_value_cache[key] = Z
        return Z  # fallback

    # ------------------------------------------------------------------
    # Simple-relax on surrogate (PORTALS Stage-1 acquisition optimiser)
    # ------------------------------------------------------------------

    def _x_cache_key(self, x_flat: np.ndarray) -> bytes:
        """Stable cache key for candidate vectors in one BO iteration."""
        arr = np.asarray(x_flat, float).ravel()
        if self.acq_cache_round_decimals >= 0:
            arr = np.round(arr, self.acq_cache_round_decimals)
        return arr.tobytes()

    def _eval_surr_cached(self, x_flat: np.ndarray):
        """Cached wrapper around surrogate evaluation for repeated candidates."""
        key = self._x_cache_key(x_flat)
        if key not in self._acq_eval_cache:
            self._acq_eval_cache[key] = self._eval_surr_at(np.asarray(x_flat, float))
        return self._acq_eval_cache[key]

    def _candidate_rng(self, x_flat: np.ndarray) -> np.random.Generator:
        """Deterministic RNG per candidate to avoid MC ranking noise."""
        key = self._x_cache_key(x_flat)
        key_seed = int.from_bytes(key[:8].ljust(8, b"\0"), byteorder="little", signed=False)
        iter_seed = (int(self.seed) + 1000003 * int(getattr(self, "iter", 0))) % (2**32)
        return np.random.default_rng((key_seed ^ iter_seed) % (2**32))

    def _init_sr_channel_relax(self, n_channels: int):
        """Initialize persistent channel-wise relaxation memory."""
        base = float(np.clip(self.sr_relax, self.sr_min_relax, self.sr_relax_max))
        self._sr_relax_channels = np.full(n_channels, base, dtype=float)
        self._sr_channel_signal_hist = [[] for _ in range(n_channels)]

    def _update_sr_channel_history(self, channel_signal: np.ndarray):
        """Append per-channel SR signal history used for oscillation detection."""
        if self._sr_relax_channels is None or self._sr_relax_channels.size != channel_signal.size:
            self._init_sr_channel_relax(channel_signal.size)

        while len(self._sr_channel_signal_hist) < channel_signal.size:
            self._sr_channel_signal_hist.append([])

        for i, val in enumerate(np.asarray(channel_signal, float).ravel()):
            hist = self._sr_channel_signal_hist[i]
            hist.append(float(val))
            if len(hist) > self.sr_dyn_window:
                del hist[:-self.sr_dyn_window]

    def _is_oscillatory_or_stuck(self, hist: np.ndarray) -> bool:
        """PORTALS-style channel detector: periodic oscillation or flat-lined signal."""
        h = np.asarray(hist, float).ravel()
        if h.size < 8:
            return False

        centered = h - np.mean(h)
        std = float(np.std(centered))
        if std < self.sr_dyn_flat_std:
            return True

        power = np.abs(np.fft.rfft(centered))**2
        if power.size <= 1:
            return False
        p = power[1:]
        p_sum = float(np.sum(p))
        if p_sum <= 0.0:
            return False

        single_frequency_power = float(np.max(p) / p_sum)
        n_hi = max(1, p.size // 2)
        high_frequency_power = float(np.sum(p[-n_hi:]) / p_sum)
        return (
            (single_frequency_power > self.sr_dyn_single_freq_thresh)
            or (high_frequency_power > self.sr_dyn_high_freq_thresh)
        )

    def _maybe_apply_dynamic_channel_relax(self):
        """Periodically damp only unstable channels, preserving fast stable channels."""
        if (not self.sr_relax_dyn) or (self._sr_relax_channels is None):
            return

        iter_now = int(getattr(self, "iter", 0))
        if (self._sr_relax_last_applied_iter >= 0
                and (iter_now - self._sr_relax_last_applied_iter) <= self.sr_dyn_num):
            return

        changed = False
        for i, hist in enumerate(self._sr_channel_signal_hist):
            if self._is_oscillatory_or_stuck(np.asarray(hist, float)):
                self._sr_relax_channels[i] = max(
                    self.sr_min_relax,
                    self._sr_relax_channels[i] / max(self.sr_dyn_decrease, 1.0),
                )
                changed = True

        if changed:
            self._sr_relax_last_applied_iter = iter_now

    def _surrogate_objective(self, x_flat: np.ndarray) -> float:
        """Deterministic surrogate objective at x_flat (lower is better)."""
        result = self._eval_surr_cached(np.asarray(x_flat, float))
        if result is None:
            return float("inf")
        _, _, _, R = result
        if R is None:
            return float("inf")
        R_arr = np.asarray(R, float)
        if not np.all(np.isfinite(R_arr)):
            return float("inf")
        return float(self.objective(R_arr))

    def _dx_to_x_step(self, dx: np.ndarray, x: np.ndarray, n_channels: int) -> np.ndarray:
        """Broadcast normalized SR step to parameter space with optional clamps."""
        n_x = x.size
        n_q = dx.size

        if n_x == n_q:
            x_step = dx * np.abs(x)
        else:
            n_each = n_x // n_channels if n_channels else n_x
            dx_per_ch = dx.reshape(n_channels, -1).mean(axis=1) if n_channels else dx
            x_step = np.repeat(dx_per_ch, n_each)[:n_x] * np.abs(x)

        if self.dx_max_abs is not None:
            x_step = np.sign(x_step) * np.minimum(np.abs(x_step), self.dx_max_abs)
        if self.dx_min_abs is not None:
            sign_or_plus = np.where(x_step != 0.0, np.sign(x_step), 1.0)
            x_step = sign_or_plus * np.maximum(np.abs(x_step), self.dx_min_abs)
        return x_step

    def _sr_on_surrogate(self, x0: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """PORTALS-style SR with dynamic relaxation and backtracking.

        The update direction follows SR; relaxation is adapted based on
        surrogate-objective improvement, with line-search style shrink on failure.
        """
        x = np.clip(x0.copy(), lo, hi)
        keys = sorted(self.target_vars)
        n_ch = len(keys)
        if n_ch > 0 and (self._sr_relax_channels is None or self._sr_relax_channels.size != n_ch):
            self._init_sr_channel_relax(n_ch)

        z_now = self._surrogate_objective(x)
        if not np.isfinite(z_now):
            return x

        best_x = x.copy()
        best_z = z_now
        plateau_steps = 0

        for _ in range(max(1, self.sr_maxiter)):
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
            direction = np.clip((Q_tar - Q_tr) / denom, -self.dx_max, self.dx_max)

            if n_ch > 0 and direction.size > 0:
                parts = np.array_split(direction, n_ch)
                channel_signal = np.array([
                    float(np.mean(p)) if p.size > 0 else 0.0
                    for p in parts
                ])
                self._update_sr_channel_history(channel_signal)

            trial_scale = 1.0
            n_bt = max(1, self.sr_backtrack if self.sr_relax_dyn else 1)
            accepted = False
            rel_gain = 0.0

            for _ in range(n_bt):
                if n_ch > 0 and self._sr_relax_channels is not None:
                    relax_channels = np.clip(
                        self._sr_relax_channels * trial_scale,
                        self.sr_min_relax,
                        self.sr_relax_max,
                    )
                    if direction.size % n_ch == 0:
                        n_per = max(1, direction.size // n_ch)
                        relax_vec = np.repeat(relax_channels, n_per)[:direction.size]
                    else:
                        relax_vec = np.full(direction.size, float(np.mean(relax_channels)))
                else:
                    relax_scalar = float(np.clip(self.sr_relax * trial_scale,
                                                 self.sr_min_relax,
                                                 self.sr_relax_max))
                    relax_vec = np.full(direction.size, relax_scalar)

                dx = relax_vec * direction
                x_step = self._dx_to_x_step(dx, x, n_ch)
                x_trial = np.clip(x + x_step, lo, hi)

                if np.max(np.abs(x_trial - x)) < 1e-10:
                    break

                z_trial = self._surrogate_objective(x_trial)
                if np.isfinite(z_trial) and z_trial <= z_now:
                    rel_gain = (z_now - z_trial) / max(abs(z_now), 1e-12)
                    x = x_trial
                    z_now = z_trial
                    accepted = True
                    if z_now < best_z:
                        best_z = z_now
                        best_x = x.copy()
                    if self.sr_relax_dyn and self._sr_relax_channels is not None:
                        grow = 1.0 + 0.1 * max(self.sr_relax_grow - 1.0, 0.0)
                        self._sr_relax_channels = np.minimum(
                            self.sr_relax_max,
                            self._sr_relax_channels * grow,
                        )
                    break

                if not self.sr_relax_dyn:
                    break
                trial_scale *= self.sr_relax_shrink

            if accepted:
                plateau_steps = plateau_steps + 1 if rel_gain <= self.sr_rel_improve_stop else 0
            else:
                plateau_steps += 1

            if plateau_steps >= self.sr_plateau_patience:
                break
        return best_x

    def _solve_root_candidate(self, x0: np.ndarray,
                              lo: np.ndarray, hi: np.ndarray,
                              residual_dim: int) -> np.ndarray:
        """Bounded mismatch solve from a single start; returns best candidate."""

        x0 = np.clip(np.asarray(x0, float), lo, hi)
        residual_dim = max(1, int(residual_dim))

        def _penalty_residual() -> np.ndarray:
            return np.full(residual_dim, 1e6, dtype=float)

        def _mismatch(x_flat: np.ndarray) -> np.ndarray:
            x_clip = np.clip(np.asarray(x_flat, float), lo, hi)
            result = self._eval_surr_cached(x_clip)
            if result is None:
                return _penalty_residual()
            _, _, _, R = result
            if R is None:
                return _penalty_residual()
            R_arr = np.asarray(R, float).ravel()
            if R_arr.size != residual_dim:
                return _penalty_residual()
            if not np.all(np.isfinite(R_arr)):
                return _penalty_residual()
            return np.nan_to_num(R_arr, nan=1e6, posinf=1e6, neginf=-1e6)

        candidates = [x0]
        z0 = self._surrogate_objective(x0)

        if self.root_solver in {"least_squares", "lsq", "auto"}:
            try:
                lsq = _sopt.least_squares(
                    _mismatch,
                    x0,
                    bounds=(lo, hi),
                    method=self.root_method,
                    max_nfev=self.root_max_nfev,
                    xtol=self.root_tol,
                    ftol=self.root_tol,
                    gtol=self.root_tol,
                    loss=self.root_loss,
                    f_scale=self.root_f_scale,
                )
                if np.all(np.isfinite(lsq.x)):
                    candidates.append(np.clip(lsq.x, lo, hi))
            except Exception:
                pass

        if self.root_use_lm_fallback:
            try:
                lm = _sopt.root(
                    _mismatch,
                    x0,
                    method="lm",
                    options={"maxiter": self.lm_maxiter, "xtol": self.root_tol, "ftol": self.root_tol},
                )
                if np.all(np.isfinite(lm.x)):
                    candidates.append(np.clip(lm.x, lo, hi))
            except Exception:
                pass

        best_x = x0
        best_z = z0
        for cand in candidates[1:]:
            z_c = self._surrogate_objective(cand)
            if np.isfinite(z_c) and z_c < best_z:
                best_z = z_c
                best_x = cand

        rel_improve = (z0 - best_z) / max(abs(z0), 1e-12) if np.isfinite(z0) and np.isfinite(best_z) else 0.0
        if rel_improve < self.root_rel_improve_stop:
            return x0
        return best_x

    # ------------------------------------------------------------------
    # Trust-region
    # ------------------------------------------------------------------

    def _tr_local_scale(self, x_center: np.ndarray,
                        lo_ref: np.ndarray, hi_ref: np.ndarray) -> np.ndarray:
        """Per-dimension local scaling for value-relative trust regions."""
        x_abs = np.abs(np.asarray(x_center, float))
        span = np.maximum(np.asarray(hi_ref, float) - np.asarray(lo_ref, float), 0.0)
        floor = self.tr_x_floor_frac * span
        if self.tr_x_floor_abs is not None:
            floor = np.maximum(floor, self.tr_x_floor_abs)
        return np.maximum(x_abs, np.maximum(floor, 1e-12))

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

        if self.tr_scale_with_x:
            cur_mid = (cur_lo + cur_hi) / 2.0
            cur_half = (cur_hi - cur_lo) / 2.0
            prev_scale = self._tr_local_scale(cur_mid, lo_init, hi_init)
            rel_half = cur_half / np.maximum(prev_scale, 1e-12)
            new_rel_half = rel_half * rho

            # Keep existing centering semantics: shrink around best; expand around mid.
            center = np.asarray(x_best, float) if rho < 1.0 else cur_mid
            center_scale = self._tr_local_scale(center, lo_init, hi_init)
            max_half = (hi_init - lo_init) / 2.0
            half = np.minimum(new_rel_half * center_scale, max_half)
            half = np.clip(np.nan_to_num(half, nan=0.0, posinf=0.0, neginf=0.0), 0.0, max_half)

            self._tr_lo = np.clip(center - half, lo_init, hi_init)
            self._tr_hi = np.clip(center + half, lo_init, hi_init)
        else:
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
        """Surrogate BO: LHS seeding → dynamic SR → bounded nonlinear solve.

        Falls back to Relax during warm-up (iter < surr_warmup).
        """
        if getattr(self, "iter", 0) < self.surr_warmup:
            return Relax.propose_parameters(self)

        # BO requires a fitted surrogate, but we only refit after new physics
        # samples are appended by the real-model evaluation path.
        if self._surrogate is None:
            return Relax.propose_parameters(self)

        n_samples = len(getattr(self._surrogate, "X_train", []))
        if n_samples == 0:
            return Relax.propose_parameters(self)

        if (not getattr(self._surrogate, "trained", False)) or (n_samples != self._bo_last_fit_n_samples):
            self._surrogate.fit()
            if getattr(self._surrogate, "trained", False):
                self._bo_last_fit_n_samples = n_samples

        if not getattr(self._surrogate, "trained", False):
            return Relax.propose_parameters(self)

        # Fresh per-iteration caches: surrogate fit, X, and TR may have changed.
        self._acq_eval_cache.clear()
        self._acq_value_cache.clear()
        self._maybe_apply_dynamic_channel_relax()

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
                # Initialise TR around X0.
                # Default: value-relative width (with floors) so very wide bounds do
                # not dominate when parameters are small. Legacy bounds-only mode can
                # be restored via tr_scale_with_x=False.
                if self.tr_scale_with_x:
                    local_scale = self._tr_local_scale(X_flat0, lo_arr, hi_arr)
                    half = np.minimum((hi_arr - lo_arr) / 2.0, self.tr_init_frac * local_scale)
                else:
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

        # ── Stage 1: Latin Hypercube seeding ────────────────────────────────
        lhs        = LatinHypercube(d=n_params, seed=int_seed).random(n=self.n_restarts)
        candidates = eff_lo + lhs * (eff_hi - eff_lo)
        candidates[0] = np.clip(X_flat0, eff_lo, eff_hi)   # always include current best

        # ── Stage 2: dynamic SR on surrogate for each candidate ─────────────
        sr_results = [self._sr_on_surrogate(c, eff_lo, eff_hi) for c in candidates]

        # ── Stage 3: bounded nonlinear solve from best SR candidate(s) ──────
        # Always run to make acquisition optimization more aggressive.
        acq_sr        = np.array([self._acq(x) for x in sr_results])
        sr_order       = np.argsort(acq_sr)

        root_candidates = []
        n_root = max(1, min(self.root_n_restarts, len(sr_results)))
        for idx in sr_order[:n_root]:
            x_start = sr_results[int(idx)]
            r_eval = self._eval_surr_cached(x_start)
            r_dim = len(np.asarray(r_eval[3]).ravel()) if (r_eval is not None and r_eval[3] is not None) else n_params
            root_candidates.append(self._solve_root_candidate(x_start, eff_lo, eff_hi, r_dim))

        # ── Stage 4: select best across all SR + root candidates ────────────
        if root_candidates:
            all_candidates = np.vstack([sr_results, np.vstack(root_candidates)])
        else:
            all_candidates = np.vstack(sr_results)
        acq_all        = np.array([self._acq(x) for x in all_candidates])
        best_idx       = int(np.argmin(acq_all))
        best_x         = all_candidates[best_idx]

        # Store surrogate-predicted Z for next iteration's BO metric
        result_best = self._eval_surr_cached(best_x)
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
