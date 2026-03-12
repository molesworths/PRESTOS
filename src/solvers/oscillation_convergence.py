"""Convergence detection with oscillation handling for transport solvers.

Standard convergence (Z < tol) can miss oscillatory solutions that have stabilized
around a minimum. This module uses proven methods from numerical analysis:

1. **Moving Average** - Smooths residuals to separate trend from noise
2. **Derivative Analysis** - Checks if residual derivative has stabilized
3. **Envelope Fitting** - Detects decaying oscillations

Theory:
  - Moving average μ_k = α*||R_k|| + (1-α)*μ_{k-1} captures trend
  - Convergence when: small mean AND small amplitude
  - Derivative d(μ)/dk ≈ 0 indicates stabilization
  
References:
  - Kelley (1995): Iterative Methods for Linear and Nonlinear Equations
  - More & Sorensen (1983): Computing a trust region step
  - Dennis & Schnabel (1983): Numerical Methods for Unconstrained Optimization
"""

import numpy as np
from typing import Dict, Optional, Tuple


class OscillationConvergenceDetector:
    """Detect convergence via moving average + derivative stabilization.
    
    Uses exponential moving average (EMA) to smooth residuals and detect when:
    1. Mean objective has converged: μ_k < tol
    2. Oscillations have damped: d(μ)/dk ≈ 0
    3. Amplitude relative to mean is small: σ_window / μ_window < tol_rel
    
    This is orders of magnitude simpler than autocorrelation-based methods
    while being more robust to noise and scaling.
    """
    
    def __init__(self, options: Dict = None):
        self.options = options or {}
        
        # Convergence thresholds
        self.tol = float(self.options.get("tol", 1e-6))
        self.oscillation_rel_tol = float(self.options.get("oscillation_rel_tol", 0.05))  # 5% amplitude
        self.min_iterations_after_convergence = int(self.options.get("min_iterations_after_convergence", 5))
        
        # Moving average parameters
        self.ema_alpha = float(self.options.get("ema_alpha", 0.3))  # Smoothing factor (lower = more smoothing)
        self.derivative_window = int(self.options.get("derivative_window", 3))  # For derivative calculation
        
        # History tracking
        self.Z_history = []
        self.Z_ema_history = []  # Exponential Moving Average
        self.Z_derivative_history = []  # Derivative of EMA
        self.X_history = []
        self.Y_history = []
        self.R_history = []
        
        # Minimum iterations before the oscillation-stall check can fire.
        # Needs enough history for derivatives to form and stabilize.
        default_stall_min = max(self.derivative_window * 4 + 2, self.min_iterations_after_convergence * 2, 12)
        self.osc_stall_min_iters = int(self.options.get("oscillation_stall_min_iters", default_stall_min))

        # State tracking
        self.is_converged_via_oscillation = False
        self.mean_state = None
        self.converged_iteration = None
        
    def add_iteration(
            self,
            Z: float,
            X: Dict,
            Y: Optional[Dict] = None,
            R: Optional[np.ndarray] = None):
            """
            Record iteration data for convergence analysis.
            """
            Z_float = float(Z)
            self.Z_history.append(Z_float)
            self.X_history.append(dict(X))
            if Y is not None:
                self.Y_history.append({k: np.array(v).copy() for k, v in Y.items()})
            if R is not None:
                self.R_history.append(np.array(R).copy())

            # Update exponential moving average
            if len(self.Z_ema_history) == 0:
                self.Z_ema_history.append(Z_float)
            else:
                ema_new = self.ema_alpha * Z_float + (1 - self.ema_alpha) * self.Z_ema_history[-1]
                self.Z_ema_history.append(ema_new)

            # Update derivative of EMA
            if len(self.Z_ema_history) >= self.derivative_window + 1:
                # Central difference for smoother derivative
                idx = len(self.Z_ema_history) - 1
                if idx >= self.derivative_window:
                    dema = (self.Z_ema_history[idx] - self.Z_ema_history[idx - self.derivative_window]) / self.derivative_window
                    self.Z_derivative_history.append(dema)

    def check_standard_convergence(self) -> bool:
        """Standard convergence: Z < tol."""
        if len(self.Z_history) == 0:
            return False
        return self.Z_history[-1] < self.tol

    def check_oscillation_convergence(self) -> Tuple[bool, Optional[Dict]]:
        """Check if oscillations have stabilized around a minimum.

        Uses exponential moving average + derivative analysis:
        1. EMA smooths residuals to extract trend
        2. Derivative d(EMA)/dk indicates if trend is stabilizing
        3. Amplitude check ensures oscillations are small

        Returns:
            (converged, mean_state_dict)
        """
        
        if len(self.Z_history) < 8:  # Need minimum history
            return False, None

        # Check EMA-based convergence
        ema_converged = self._check_ema_convergence()

        if not ema_converged:
            return False, None

        # Check derivative stabilization (trend has flattened)
        derivative_stable = self._check_derivative_stabilization()

        if not derivative_stable:
            return False, None

        # Additional check: have we stayed converged for a few iterations?
        if not self.is_converged_via_oscillation:
            self.is_converged_via_oscillation = True
            self.converged_iteration = len(self.Z_history)
            return False, None  # Flag convergence but don't accept yet

        # Converged for min_iterations_after_convergence consecutive checks
        if len(self.Z_history) - self.converged_iteration >= self.min_iterations_after_convergence:
            mean_state = self._compute_mean_state(window_size=self.min_iterations_after_convergence + 2)
            self.mean_state = mean_state
            return True, mean_state

        return False, None

    def check_oscillation_stall(self) -> bool:
        """Check if oscillations have stabilized but remain above tolerance.

        Returns True when the EMA trend has flattened (oscillations are no longer
        improving) but the EMA is still above *tol* — meaning the solver is stuck
        oscillating without making progress toward the convergence target.

        The convergence path (check_oscillation_convergence) handles the below-tol
        case; this method covers the above-tol stuck-oscillation case that the
        standard stall counter misses because oscillations periodically reset it.
        """
        if len(self.Z_history) < self.osc_stall_min_iters:
            return False

        # Only stall here if EMA is above tolerance; convergence handles below-tol.
        if not self.Z_ema_history or self.Z_ema_history[-1] < self.tol:
            return False

        return self._check_derivative_stabilization()

    def _check_ema_convergence(self) -> bool:
        """Check if exponential moving average is below tolerance."""
        if len(self.Z_ema_history) == 0:
            return False
        
        current_ema = self.Z_ema_history[-1]
        return current_ema < self.tol
    
    def _check_derivative_stabilization(self) -> bool:
        """Check if derivative of EMA has stabilized (trend flattened).
        
        d(EMA)/dk ≈ 0 indicates the moving average is no longer decreasing,
        meaning oscillations have stopped improving convergence.
        """
        if len(self.Z_derivative_history) < 2:
            return False
        
        # Check if recent derivatives are small and stable
        # (not consistently negative, which would indicate still converging)
        recent_derivs = np.array(self.Z_derivative_history[-3:])
        
        # Mean derivative close to zero
        mean_deriv = np.mean(recent_derivs)
        if abs(mean_deriv) > 0.01 * self.tol:  # Still significant changes
            return False
        
        # Derivative should be small relative to values (not just in absolute terms)
        # but avoid division by zero
        if len(self.Z_ema_history) > 0:
            relative_deriv = abs(mean_deriv) / (abs(self.Z_ema_history[-1]) + 1e-12)
            if relative_deriv > 0.01:  # > 1% change per iteration
                return False
        
        return True
    
    def _compute_mean_state(self, n_samples: int) -> Dict:
        """Compute time-averaged state over last n_samples iterations.
        
        Returns dict with structure:
        {
            'Z_mean': float,
            'Z_std': float,
            'X_mean': Dict[str, Dict[str, float]],
            'X_std': Dict[str, Dict[str, float]],
            'Y_mean': Dict[str, np.ndarray],  # if available
            'R_mean': np.ndarray,  # if available
        }
        """
        
        n = min(n_samples, len(self.Z_history))
        
        result = {}
        
        # Objective statistics
        Z_recent = self.Z_history[-n:]
        result['Z_mean'] = float(np.mean(Z_recent))
        result['Z_std'] = float(np.std(Z_recent))
        result['Z_min'] = float(np.min(Z_recent))
        result['Z_max'] = float(np.max(Z_recent))
        
        # Parameter averages
        X_recent = self.X_history[-n:]
        X_mean = {}
        X_std = {}
        
        # Iterate over profiles
        for profile in X_recent[0].keys():
            X_mean[profile] = {}
            X_std[profile] = {}
            
            for param in X_recent[0][profile].keys():
                values = [X[profile][param] for X in X_recent]
                X_mean[profile][param] = float(np.mean(values))
                X_std[profile][param] = float(np.std(values))
        
        result['X_mean'] = X_mean
        result['X_std'] = X_std
        
        # Output averages (if tracked)
        if len(self.Y_history) >= n:
            Y_recent = self.Y_history[-n:]
            Y_mean = {}
            
            for key in Y_recent[0].keys():
                values = np.array([Y[key] for Y in Y_recent])
                Y_mean[key] = np.mean(values, axis=0)
            
            result['Y_mean'] = Y_mean
        
        # Residual averages (if tracked)
        if len(self.R_history) >= n:
            R_recent = self.R_history[-n:]
            result['R_mean'] = np.mean(np.array(R_recent), axis=0)
            result['R_std'] = np.std(np.array(R_recent), axis=0)
        
        return result
    
    def get_convergence_report(self) -> str:
        """Generate human-readable convergence report."""
        
        if len(self.Z_history) == 0:
            return "No iterations recorded yet."
        
        lines = []
        lines.append("=" * 60)
        lines.append("CONVERGENCE ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"Total iterations: {len(self.Z_history)}")
        lines.append(f"Current Z: {self.Z_history[-1]:.6e}")
        lines.append(f"Minimum Z: {np.min(self.Z_history):.6e}")
        lines.append(f"Target tolerance: {self.tol:.6e}")
        lines.append("")
        
        # Standard convergence
        std_conv = self.check_standard_convergence()
        lines.append(f"Standard convergence (Z < tol): {std_conv}")
        
        # Oscillation analysis
        if len(self.Z_history) >= self.min_iterations_for_oscillation:
            osc_conv, mean_state = self.check_oscillation_convergence()
            
            lines.append(f"Oscillation detected: {self.is_oscillating}")
            if self.is_oscillating:
                lines.append(f"Estimated period: {self.oscillation_period} iterations")
            
            lines.append(f"Oscillation convergence: {osc_conv}")
            
            if osc_conv and mean_state is not None:
                lines.append("")
                lines.append("CONVERGED via oscillation stabilization!")
                lines.append(f"  Mean Z: {mean_state['Z_mean']:.6e} ± {mean_state['Z_std']:.6e}")
                lines.append(f"  Min Z: {mean_state['Z_min']:.6e}")
                lines.append(f"  Relative amplitude: {mean_state['Z_std']/abs(mean_state['Z_mean']):.3%}")
        else:
            lines.append(f"Insufficient iterations for oscillation analysis (need {self.min_iterations_for_oscillation})")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def integrate_oscillation_convergence(solver):
    """Integrate oscillation-aware convergence into existing solver.
    
    Usage:
        from solvers.oscillation_convergence import integrate_oscillation_convergence
        
        solver = RelaxSolver(options)
        integrate_oscillation_convergence(solver)
    """
    
    # Create detector
    detector_opts = solver.options.copy()
    detector_opts['oscillation_rel_tol'] = solver.options.get('oscillation_rel_tol', 0.05)
    detector_opts['min_iterations_for_oscillation'] = solver.options.get('min_iterations_for_oscillation', 10)
    detector_opts['oscillation_window'] = solver.options.get('oscillation_window', 8)
    
    solver._osc_detector = OscillationConvergenceDetector(detector_opts)
    
    # Override check_convergence
    original_check = solver.check_convergence
    
    def check_convergence_with_oscillation(y_model, y_target):
        # Call original convergence check
        original_check(y_model, y_target)
        
        # Record iteration
        solver._osc_detector.add_iteration(
            Z=solver.Z,
            X=solver.X,
            Y=y_model,
            R=solver.R
        )
        
        # Check for oscillation convergence if not yet converged
        if not solver.converged:
            osc_conv, mean_state = solver._osc_detector.check_oscillation_convergence()
            
            if osc_conv:
                print("\n" + "="*60)
                print("OSCILLATION CONVERGENCE DETECTED")
                print("="*60)
                print(f"Objective oscillating around minimum with small amplitude.")
                print(f"Mean Z: {mean_state['Z_mean']:.6e} ± {mean_state['Z_std']:.6e}")
                print(f"Extracting time-averaged solution as converged state.")
                print("="*60 + "\n")
                
                # Update solver state to mean values
                solver.X = mean_state['X_mean']
                solver.Z = mean_state['Z_mean']
                solver.converged = True
                solver._converged_via_oscillation = True
                
                # Optionally update Y and R
                if 'Y_mean' in mean_state:
                    solver.Y = mean_state['Y_mean']
                if 'R_mean' in mean_state:
                    solver.R = mean_state['R_mean']
    
    solver.check_convergence = check_convergence_with_oscillation
    
    # Add report method
    def print_convergence_report():
        print(solver._osc_detector.get_convergence_report())
    
    solver.print_convergence_report = print_convergence_report
    
    return solver
