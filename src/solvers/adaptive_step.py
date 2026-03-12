"""
Residual-ratio adaptive step size control (C_eta-style dynamic relaxation).

Analogous to TGYRO's LOC_RELAX parameter: the step is shrunk when residuals
improve slowly and grows when they improve quickly, with a floor/ceiling.

Core update rule (applied each iteration):

    eta_{k+1} = clip(eta_k * (||R_{k-1}|| / ||R_k||)^gamma, eta_min, eta_max)

When ||R|| drops by half, eta grows by 2^(-gamma) (reward good steps).
When ||R|| rises (bad step), eta shrinks (penalise overshoots).
gamma=0.5 gives moderate tracking; gamma=1.0 tracks residual ratio exactly.
gamma=0.75 (default) is a good balance for monotone-improving objectives.

Additional modulation:
  - Early-iteration warmup: linear ramp over the first few iterations.
  - Oscillation damping: reduces step when residual norms alternate up/down.
  - Sustained-improvement accelerator: applies a small multiplicative bonus
    each iteration after `accel_streak` consecutive improving iterations,
    compounding upward during monotone descent without overshoot risk.

References:
  - TGYRO LOC_RELAX / C_eta: gacode.io/tgyro/solver.html
  - Nocedal & Wright (2006): Numerical Optimization, §3.1 (step-length rules)
"""

import numpy as np
from typing import Dict, Optional


class AdaptiveStepControl:
    """Residual-ratio adaptive step controller (C_eta-style dynamic relaxation).

    The current step eta is updated each call via:

        eta = clip(eta * (||R_prev|| / ||R_curr||)^gamma, eta_min, eta_max)

    Options
    -------
    step_size : float
        Starting (and maximum if decay_gamma=0) step size.
    min_step_size : float
        Hard floor on eta.
    max_step_size : float
        Hard ceiling on eta.
    decay_gamma : float, default 0.5
        Exponent on the residual ratio.  0 = fixed step, 1 = exact ratio,
        0.5 (default) = square-root tracking (moderate pursue/retreat).
    warmup_iters : int, default 3
        Number of iterations over which to linearly ramp from 0.5*eta to eta.
        Set to 0 to disable.
    oscillation_damping : bool, default True
        Halve the step when residual norms have been alternating for the
        last `oscillation_window` iterations.
    oscillation_window : int, default 5
        Rolling window used for oscillation detection.
    accel_streak : int, default 4
        Number of consecutive improving iterations before the sustained-
        improvement accelerator kicks in.  Set to 0 to disable.
    accel_factor : float, default 1.15
        Per-iteration multiplicative bonus applied once the streak is met.
        E.g. 1.15 grows the step by 15% each additional improving iteration.
    """

    def __init__(self, options: Dict = None):
        opts = options or {}
        self.base_step = float(opts.get("step_size", 1e-2))
        self.min_step = float(opts.get("min_step_size", 1e-6))
        self.max_step = float(opts.get("max_step_size", 5e-2))
        self.decay_gamma = float(opts.get("decay_gamma", 0.75))
        self.warmup_iters = int(opts.get("warmup_iters", 3))
        self.oscillation_damping = bool(opts.get("oscillation_damping", True))
        self.oscillation_window = int(opts.get("oscillation_window", 5))
        self.accel_streak = int(opts.get("accel_streak", 4))
        self.accel_factor = float(opts.get("accel_factor", 1.15))

        # Running state
        self._eta = self.base_step          # current adaptive step
        self._prev_norm: Optional[float] = None
        self._norm_history: list = []
        self._improving_streak: int = 0     # consecutive improving iterations

    def compute_step_size(
        self,
        X=None,
        R: Optional[np.ndarray] = None,
        J=None,                             # accepted for interface compat, unused
        iteration: int = 0,
    ) -> float:
        """Return the adaptive step size for this iteration.

        Call once per iteration *before* the parameter update.  The residual
        norm ||R|| from the just-evaluated state is used to update eta.
        """
        curr_norm: Optional[float] = None
        if R is not None:
            arr = np.asarray(R, float)
            curr_norm = float(np.linalg.norm(arr)) if arr.size > 0 else None

        # ---- Residual-ratio decay (C_eta rule) -----------------------
        if curr_norm is not None and self._prev_norm is not None and curr_norm > 1e-14:
            ratio = self._prev_norm / curr_norm   # > 1 when improving, < 1 when worsening
            self._eta = float(np.clip(
                self._eta * ratio ** self.decay_gamma,
                self.min_step,
                self.max_step,
            ))
            # Track consecutive improving iterations for the accelerator
            if ratio > 1.0:
                self._improving_streak += 1
            else:
                self._improving_streak = 0

        # ---- Sustained-improvement accelerator -----------------------
        if (
            self.accel_streak > 0
            and self._improving_streak >= self.accel_streak
            and curr_norm is not None
        ):
            self._eta = float(np.clip(
                self._eta * self.accel_factor,
                self.min_step,
                self.max_step,
            ))

        # ---- Oscillation damping -------------------------------------
        if self.oscillation_damping and curr_norm is not None:
            self._norm_history.append(curr_norm)
            if len(self._norm_history) > self.oscillation_window:
                self._norm_history.pop(0)
            if len(self._norm_history) >= 4:
                diffs = np.diff(self._norm_history)
                sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
                # More than half the steps reversed direction → oscillating
                if sign_changes >= len(diffs) // 2:
                    self._eta = max(self.min_step, self._eta * 0.5)

        # ---- Early-iteration warmup ----------------------------------
        step = self._eta
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            ramp = 0.5 + 0.5 * (iteration / self.warmup_iters)
            step = step * ramp

        step = float(np.clip(step, self.min_step, self.max_step))

        # Advance history for next call
        self._prev_norm = curr_norm

        return step

    def reset(self):
        """Reset to base step (call on solver restart)."""
        self._eta = self.base_step
        self._prev_norm = None
        self._norm_history = []
        self._improving_streak = 0
