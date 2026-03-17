"""Concise solver framework.

This refactor provides:
 - A lightweight SolverData container with consistent fields.
 - A simplified SolverBase.run loop handling parameters -> state -> model/targets -> residuals.
 - Optional surrogate usage, Jacobian assistance (kept for RelaxSolver), and bounds projection.
 - Removal of legacy duplicate run paths and broken temporary classes.
"""

from __future__ import annotations

import copy
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy as sp

from .jacobian import JacobianMixin
from .objectives import create_objective_function
from .solver_data import SolverData
from .uncertainty import UncertaintyMixin


class SolverBase(JacobianMixin, UncertaintyMixin):
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = dict(options or {})
        self.verbose = self.options.get("verbose", False)
        self.objective = create_objective_function(
            {
                "type": self.options.get("objective", "mse"),
                "scale": self.options.get("scale_objective", True),
            }
        )
        self.tol = float(self.options.get("tol", 1e-6))
        self.step_size = float(self.options.get("step_size", 1e-2))
        self.min_step_size = float(self.options.get("min_step_size", 1e-6))
        self.adaptive_step = bool(self.options.get("adaptive_step", True))

        # Advanced adaptive step control (Jacobian-based, triggered by adaptive_step option)
        if self.adaptive_step:
            from .adaptive_step import AdaptiveStepControl
            self._adaptive_step_ctrl = AdaptiveStepControl({
                "step_size": self.step_size,
                "min_step_size": self.min_step_size,
                "max_step_size": float(self.options.get("max_step_size", 5e-2)),
                "decay_gamma": float(self.options.get("decay_gamma", 0.75)),
                "warmup_iters": int(self.options.get("warmup_iters", 3)),
                "oscillation_damping": bool(self.options.get("oscillation_damping", True)),
                "oscillation_window": int(self.options.get("oscillation_window", 5)),
            })
        else:
            self._adaptive_step_ctrl = None

        # Oscillation-aware convergence detection (triggered by use_oscillation_convergence option)
        self.use_oscillation_convergence = bool(self.options.get("use_oscillation_convergence", True))
        if self.use_oscillation_convergence:
            from .oscillation_convergence import OscillationConvergenceDetector
            self._osc_detector = OscillationConvergenceDetector({
                "tol": self.tol,
                "oscillation_rel_tol": float(self.options.get("oscillation_rel_tol", 0.05)),
                "min_iterations_after_convergence": int(self.options.get("oscillation_min_iters", 5)),
                "ema_alpha": float(self.options.get("ema_alpha", 0.3)),
                "derivative_window": int(self.options.get("derivative_window", 3)),
                # Allow stall check to fire after slightly more iters than the
                # standard stall counter, so the two mechanisms don't race.
                "oscillation_stall_min_iters": int(self.options.get("model_iter_to_stall", 10)) + 5,
            })
        else:
            self._osc_detector = None

        self.converged = False
        self.stalled = False
        self.iter = 0
        self.model_iter_to_stall = int(self.options.get("model_iter_to_stall", 10))
        self.Z = None
        self._last_model_Z: Optional[float] = None
        self._last_model_iter: Optional[int] = None
        self._model_eval_nondec = 0
        self._Z_window: list = []   # rolling window for stall detection
        self.iter_between_save = int(self.options.get("iter_between_save", 5))
        self.cwd = Path.cwd()
        self._active_sandbox = None

        self._best_model_Z: Optional[float] = None
        self._best_model_iter: Optional[int] = None
        self._best_model_X: Optional[Dict[str, Dict[str, float]]] = None
        self._best_model_Y: Optional[Dict[str, np.ndarray]] = None
        self._best_model_Y_target: Optional[Dict[str, np.ndarray]] = None
        self._best_model_R: Optional[np.ndarray] = None
        self._best_model_R_dict: Optional[Dict[str, np.ndarray]] = None
        self.R: Optional[np.ndarray] = None
        self.R_dict: Dict[str, np.ndarray] = {}

        self.predicted_profiles = list(self.options.get("predicted_profiles", []))

        # Unit system for flux/flow matching in the residual.
        # These options control how transport/target outputs are mapped before
        # computing the residual R = Y_model - Y_target.
        #   flux_or_flow        : 'flux' (per m²) | 'flow' (surface-integrated, MW)
        #   gB_or_real          : 'gB' (dimensionless gyroBohm) | 'real' (physical)
        #   total_or_conduction : 'total' | 'conduction' (heat channels only)
        _output_units_cfg = self.options.get("output_units", {})
        self.solver_output_units: Dict[str, str] = {
            "flux_or_flow": str(_output_units_cfg.get("flux_or_flow", "flow")),
            "gB_or_real": str(_output_units_cfg.get("gB_or_real", "real")),
            "total_or_conduction": str(_output_units_cfg.get("total_or_conduction", "total")),
        }

        # target_vars: explicit override or auto-derived from output_units.
        #   flux_or_flow=flow,  total_or_conduction=total      → [Ce, Pe, Pi]
        #   flux_or_flow=flow,  total_or_conduction=conduction → [Ce, De, Di]
        #   flux_or_flow=flux,  total_or_conduction=total      → [Ge, Qe, Qi]
        #   flux_or_flow=flux,  total_or_conduction=conduction → [Ge, Qe, Qi]  (no separate conduction flux key)
        _explicit_vars = self.options.get("target_vars", [])
        if _explicit_vars:
            self.target_vars = list(_explicit_vars)
        else:
            _fof = self.solver_output_units["flux_or_flow"]
            _toc = self.solver_output_units["total_or_conduction"]
            if _fof == "flow":
                self.target_vars = ["Ce", "De" if _toc == "conduction" else "Pe", "Di" if _toc == "conduction" else "Pi"]
            else:
                self.target_vars = ["Ge", "Qe", "Qi"]

        self.transport_vars = [f"{t}_neo" for t in self.target_vars] + [f"{t}_turb" for t in self.target_vars]
        exchange_supported = {"Qe", "Qi", "Pe", "Pi", "De", "Di"}
        exchange_targets = [t for t in self.target_vars if t in exchange_supported]
        self.transport_vars += [f"{t}_exch_neo" for t in exchange_targets]
        self.transport_vars += [f"{t}_exch_turb" for t in exchange_targets]
        self.model_vars = self.transport_vars + self.target_vars

        self.n_params_per_profile = int(self.options.get("n_params_per_profile", 0))

        self.eval_coord = self.options.get("eval_coord", "roa").lower()
        self.domain = self.options.get("domain", [0.85, 1.0])
        self.roa_eval = np.asarray(self.options.get("roa_eval", []), float)
        if self.roa_eval.size == 0:
            self.n_eval = int(self.options.get("n_eval", 32))
            self.roa_eval = np.linspace(self.domain[0], self.domain[1], self.n_eval)
        else:
            self.n_eval = self.roa_eval.size

        self.residual_weights = self.options.get(
            "residual_weights",
            {"flux": 0.5, "boundary_conditions": 0.25, "constraints": 0.25},
        )
        self.normalize_residual = bool(self.options.get("normalize_residual", True))
        self.residual_on_lcfs = bool(self.options.get("residual_on_lcfs", False))

        self.bounds = self._parse_bounds(self.options.get("bounds"))
        self.bounds_dict: Dict[str, Dict[str, np.ndarray]] = self._build_bounds_dict()
        self.max_iter = int(self.options.get("max_iter", 100))

        self.use_surrogate = bool(self.options.get("use_surrogate", False))
        self.surr_warmup = int(self.options.get("surrogate_warmup", 5))
        self.surr_retrain_every = int(self.options.get("surrogate_retrain_every", 10))
        self.surr_verify_on_converge = bool(self.options.get("surrogate_verify_on_converge", True))
        self._use_surr_iter = False

        self.X = None

        self._state = None
        self._parameters = None
        self._boundary = None
        self._neutrals = None
        self._transport = None
        self._targets = None

        self.J = None
        self.use_jacobian = bool(self.options.get("use_jacobian", True))
        self.jacobian_reg = float(self.options.get("jacobian_reg", 1e-8))
        self.fd_epsilon = float(self.options.get("fd_epsilon", 0.02))
        self.gradient_clip = float(self.options.get("gradient_clip", 1e6))

        self._J_cache = None
        self._J_cache_iter = -1
        self._R_cache = None
        self.jacobian_cache_enabled = bool(self.options.get("jacobian_cache_enabled", True))
        self.jacobian_recompute_every = int(self.options.get("jacobian_recompute_every", 5))
        self.jacobian_recompute_tol = float(self.options.get("jacobian_recompute_tol", 0.1))
        self.jacobian_use_surrogate_grads = bool(self.options.get("jacobian_use_surrogate_grads", True))

        self.use_parameter_preconditioning = bool(self.options.get("use_parameter_preconditioning", True))
        self.param_scales: Optional[Dict[str, Dict[str, float]]] = None

        self.constraints = self.options.get("constraints", None)
        self.VAR_MAP = {
            "Ti": "state.ti",
            "Te": "state.te",
            "Pi": "state.Pi",
            "Pe": "state.Pe",
            "ne": "state.ne",
            "ni": "state.ni",
        }
        self.ALIASES = {
            "T_i": "Ti",
            "T_e": "Te",
            "n_e": "ne",
            "n_i": "ni",
            "P_i": "Pi",
            "P_e": "Pe",
        }
        self.compiled_constraints: List[Dict[str, Any]] = []
        self.constraint_compilation()

    def constraint_compilation(self):
        """Compile user-defined constraints into evaluable functions."""
        if self.constraints is not None:
            for i, c in enumerate(self.constraints):
                loc = float(c["location"])
                weight = float(c.get("weight", 1.0))
                norm = c.get("norm", None)
                enforcement = c.get("enforcement", "exact")

                user_expr = c["expression"]
                expr_norm = self._normalize_expression(user_expr, self.ALIASES)

                lhs, rhs, op = self._parse_constraint(expr_norm)

                if op == "=":
                    canon = f"({lhs}) - ({rhs})"
                elif op == ">=":
                    canon = f"({lhs}) - ({rhs})"
                elif op == "<=":
                    canon = f"({rhs}) - ({lhs})"

                backend = self._backend_expr(canon, self.VAR_MAP)

                if norm == "log":
                    backend = f"np.log(np.abs({backend}) + 1e-12)"

                code = compile(backend, "<constraint>", "eval")

                def _make_eval(compiled_code):
                    def _eval(state):
                        return eval(compiled_code, {"np": np}, {"state": state})

                    return _eval

                eval_fn = _make_eval(code)

                print(f"[PRESTOS] Constraint {i}")
                print(f"  read:        {user_expr}")
                print(f"  normalized:  {expr_norm}")
                print(f"  enforcing:   {backend}")
                print(f"  location:    r/a = {loc}")
                print(f"  weight:      {weight}")
                print(f"  enforcement: {enforcement}")

                self.compiled_constraints.append(
                    {
                        "location": loc,
                        "weight": weight,
                        "op": op,
                        "eval": eval_fn,
                        "enforcement": enforcement,
                    }
                )

    def get_initial_parameters(self):
        _ = self._targets.evaluate(self._state)
        self._boundary.get_boundary_conditions(self._state, self._targets)
        self.X, self.X_std = self._parameters.parameterize(self._state, self._boundary.bc_dict)
        self.X = self._project_bounds(self.X)
        self._flatten_params(self.X)
        self._build_bounds_dict()
        initial_profiles_list = [
            (str(prof), getattr(self._state, prof)) for prof in self.predicted_profiles
        ] + [(f"aL{prof}", getattr(self._state, f"aL{prof}")) for prof in self.predicted_profiles]
        self._state.initial_profiles = dict(initial_profiles_list)

    def _update_from_params(self, X: np.ndarray):
        """Update state from parameters and evaluate targets with a self-consistent state."""
        self._boundary.get_boundary_conditions(self._state, self._targets)
        if isinstance(X, np.ndarray):
            X = self._unflatten_params(X)
        self._parameters.update(X, self._boundary.bc_dict, self.roa_eval)
        self._state.update(X, self._parameters, neutrals=self._neutrals)
        _ = self._targets.evaluate(self._state)

    def _build_bounds_dict(self):
        """Convert bounds into nested dict using schema."""
        if not hasattr(self, "schema") or self.schema is None:
            return
        bd: Dict[str, Dict[str, np.ndarray]] = {}
        profile_counts: Dict[str, int] = {}
        for prof, pname in self.schema:
            profile_counts[prof] = profile_counts.get(prof, 0)
            idx = profile_counts[prof]
            if self.bounds is None or prof not in self.bounds or idx >= len(self.bounds.get(prof, [])):
                lo, hi = -1e6, 1e6
            else:
                lo, hi = self.bounds[prof][idx]
                lo = float(lo) if np.isfinite(lo) else -1e6
                hi = float(hi) if np.isfinite(hi) else 1e6
            bd.setdefault(prof, {})[pname] = np.asarray([lo, hi], dtype=float)
            profile_counts[prof] += 1
        self.bounds_dict = bd

    def _parse_bounds(self, bounds):
        """Normalize bounds specification from config."""
        if bounds is None:
            return None

        bounds_dict = {}

        if isinstance(bounds, list) and len(bounds) == 2:
            try:
                low = float(bounds[0])
                high = float(bounds[1])
                return {
                    n: [[low, high] for _ in range(self.n_params_per_profile)]
                    for n in self.predicted_profiles
                }
            except Exception:
                pass

        if isinstance(bounds, dict) and all(k in bounds.keys() for k in self.predicted_profiles):
            for k in self.predicted_profiles:
                if not (isinstance(bounds[k], list) and len(bounds[k]) == 2):
                    return None
                try:
                    low = np.asarray([float(bounds[k][0]) for k in self.predicted_profiles], float)
                    high = np.asarray([float(bounds[k][1]) for k in self.predicted_profiles], float)
                    bounds_dict.update(
                        {
                            k: [[low, high] for _ in range(self.n_params_per_profile)]
                            for k in self.predicted_profiles
                        }
                    )
                except Exception:
                    return None
            return bounds_dict

        if (
            isinstance(bounds, list)
            and all(isinstance(b, (list, tuple)) and len(b) == 2 for b in bounds)
            and len(bounds) == self.n_params_per_profile
        ):
            try:
                bounds_dict = {
                    k: [[float(a), float(b)] for [a, b] in bounds]
                    for k in self.predicted_profiles
                }
                return bounds_dict
            except Exception:
                return None

        print("Unable to parse parameter bounds specification; ignoring.")
        return None

    def _project_bounds(self, X):
        """Clip parameters using bounds."""
        if self.bounds is None:
            return X

        if isinstance(X, dict):
            bc_dict = getattr(self._boundary, "bc_dict", {}) if self._boundary is not None else {}

            Xc = {}
            for prof, pvals in X.items():
                if prof not in self.bounds:
                    Xc[prof] = pvals
                    continue

                bounds_list = self.bounds[prof]
                Xc[prof] = {}

                param_names = list(pvals.keys())

                for idx, pname in enumerate(param_names):
                    val = pvals[pname]

                    if idx < len(bounds_list):
                        lo, hi = bounds_list[idx]
                    else:
                        lo, hi = -1e6, 1e6

                    Xc[prof][pname] = float(np.clip(val, lo, hi))

            return Xc

        return X

    def _get_bounds_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return flat lower/upper bound arrays aligned with a schema."""
        schema = self.schema
        lo_list = []
        hi_list = []
        for prof, pname in schema:
            if self.bounds_dict and prof in self.bounds_dict and pname in self.bounds_dict[prof]:
                lo, hi = self.bounds_dict[prof][pname]
            else:
                lo, hi = -1e6, 1e6
            lo_list.append(float(lo))
            hi_list.append(float(hi))
        return np.asarray(lo_list, float), np.asarray(hi_list, float)

    def _use_surrogate_iteration(self) -> bool:
        """Check if we should use surrogate for prediction (not training)."""
        if self._surrogate is None:
            return False
        if self.use_surrogate is False:
            return False
        if self.iter + 1 < self.surr_warmup:
            return False
        if self.converged is True:
            return False
        if ((self.iter + 1) % self.surr_retrain_every) == 0:
            return False

        self._surrogate.fit()
        return True

    def _convert_output_units(
        self,
        Y_dict: Dict[str, Any],
        from_units: Dict[str, str],
        Gamma_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Convert a variable dict from *from_units* to ``self.solver_output_units``.

        Currently handles gB↔real and flux↔flow conversions via
        ``tools.plasma.convert_output_units``.  The total↔conduction
        decomposition requires particle-flux arrays in *Gamma_dict* and will be
        fully activated once transport/targets are refactored to expose them.

        Parameters
        ----------
        Y_dict : dict
            Mapping of channel name → 1-D array at ``self.roa_eval``.
        from_units : dict
            Unit specification of the incoming data (same keys as
            ``solver_output_units``).
        Gamma_dict : dict, optional
            Particle-flux arrays keyed by species (``'Ge'``, ``'Gi'``)
            [1e19/m²/s] at ``self.roa_eval``, used to compute conductive flows.

        Returns
        -------
        dict
            Same keys as *Y_dict*, values converted to ``solver_output_units``.
        """
        from tools import plasma as _plasma

        to_units = self.solver_output_units
        # Fast path: no conversion needed
        if from_units == to_units:
            return Y_dict

        state = self._state
        roa_points = self.roa_eval
        out = {}
        for channel, value in Y_dict.items():
            Gamma = None
            if Gamma_dict is not None:
                spec = 'e' if 'e' in channel.lower() else 'i'
                Gamma = Gamma_dict.get(f'G{spec}')
            out[channel] = _plasma.convert_output_units(
                value=np.asarray(value, dtype=float),
                channel=channel,
                from_units=from_units,
                to_units=to_units,
                state=state,
                roa_points=roa_points,
                Gamma_1e19=Gamma,
            )
        return out

    def _evaluate(
        self,
        X: Optional[Dict[str, Dict[str, float]]] = None,
        in_place: bool = True,
    ):
        """Evaluate transport + target (or surrogate) optionally without mutating solver attributes."""
        X_current = self.X if X is None else X
        if self._use_surr_iter and self._surrogate is not None:
            Y_model, Y_model_std, Y_target, Y_target_std = self._surrogate.evaluate(X_current, self._state)
        else:
            Y_model, Y_model_std = self._transport.evaluate(self._state)
            Y_target, Y_target_std = self._targets.evaluate(self._state)
            transport_dict = getattr(self._transport, "transport_dict", None)
            if in_place and self._surrogate is not None:
                # Pass full nested transport_dict for turb/neo breakdown;
                # targets only have totals so the flat Y_gB dict suffices.
                # Use totals-only Y_model here so the surrogate training set
                # is not polluted with derived component keys.
                Y_target_gB = getattr(self._targets, "Y_gB", None)
                self._surrogate.add_sample(
                    self._state, X_current, Y_model, Y_target,
                    transport_dict=transport_dict, targets_gB=Y_target_gB,
                )
            # Augment Y_model with turb/neo component breakdown so that
            # SolverData stores per-mechanism values alongside totals.
            if transport_dict:
                Y_model, Y_model_std = self._merge_transport_components(
                    Y_model, Y_model_std, transport_dict
                )
        if in_place:
            self.Y = Y_model
            self.Y_std = Y_model_std
            self.Y_target = Y_target
            self.Y_target_std = Y_target_std
        return Y_model, Y_model_std, Y_target, Y_target_std

    def _get_channel_components(
        self,
        var: str,
        transport_dict: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return available mechanism components for *var* from *transport_dict*.

        Handles both direct flux channels (Ge, Gi, Qe, Qi) and derived flow
        channels (Pe, Pi, Ce, Ci, De, Di) whose breakdowns live under nested
        sub-keys in the flows dict.
        """
        fluxes_real = transport_dict.get("fluxes", {}).get("real", {})
        flows_real  = transport_dict.get("flows",  {}).get("real", {})
        out: Dict[str, np.ndarray] = {}

        def _collect_components(src: Dict[str, Any]) -> None:
            mapping = {
                "neo": "neo",
                "turb": "turb",
                "neo_exch": "exch_neo",
                "turb_exch": "exch_turb",
            }
            for src_key, dst_key in mapping.items():
                if src_key in src:
                    out[dst_key] = np.asarray(src[src_key], float)

        # Direct flux channels (Ge, Gi, Qe, Qi)
        if var in fluxes_real:
            _collect_components(fluxes_real[var])
            if out:
                return out

        # Total flow channels (Pe, Pi)
        if var in flows_real:
            _collect_components(flows_real[var].get("total", {}))
            if out:
                return out

        # Derived flow channels (Ce/De come from flows['Pe'], Ci/Di from flows['Pi'])
        _derived_map: Dict[str, Tuple[str, str]] = {
            "Ce": ("Pe", "conv"), "De": ("Pe", "cond"),
            "Ci": ("Pi", "conv"), "Di": ("Pi", "cond"),
        }
        if var in _derived_map:
            parent, level = _derived_map[var]
            if parent in flows_real and level in flows_real[parent]:
                _collect_components(flows_real[parent][level])
                if out:
                    return out

        return None

    def _merge_transport_components(
        self,
        Y_model: Dict[str, Any],
        Y_model_std: Dict[str, Any],
        transport_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Augment flat Y_model with turb/neo breakdown from transport_dict.

        For each target variable (e.g. 'Pe') the method looks up per-mechanism
        arrays in transport_dict and inserts them as additional keys
        (e.g. 'Pe_neo', 'Pe_turb') so that SolverData / analysis can access
        the component breakdown without reprocessing the nested dict.

        Arrays that are not at the expected roa_eval size are skipped silently.
        """
        sigma = float(getattr(self._transport, "sigma", 0.1))
        Y_out     = dict(Y_model)
        Y_std_out = dict(Y_model_std)

        for var in self.target_vars:
            comps = self._get_channel_components(var, transport_dict)
            if comps is None:
                continue
            for comp, arr in comps.items():
                if arr.size != self.n_eval:
                    continue
                key = f"{var}_{comp}"
                Y_out[key]     = arr.tolist()
                Y_std_out[key] = [sigma * abs(float(v)) for v in Y_out[key]]

        return Y_out, Y_std_out

    def _compute_residuals(self, Y_model: Dict[str, Any], Y_target: Dict[str, Any]) -> Optional[np.ndarray]:
        normalize = self.normalize_residual
        R_bc = np.array([], float)
        R_constraints = np.array([], float)

        # Residuals are computed only on the target channels (totals), not on
        # the augmented component keys (Pe_neo, Pe_turb, etc.).
        R_flux_dict = {
            k: np.array(Y_model[k]) - np.array(Y_target[k])
            for k in Y_target.keys()
            if k in Y_model.keys()
        }
        R_flux = np.concatenate([R_flux_dict[k] for k in sorted(R_flux_dict.keys())])
        Y_target_concat = np.concatenate([np.array(Y_target[k]) for k in Y_target.keys() if k in Y_model])
        if normalize:
            R_flux = R_flux / (np.abs(Y_target_concat) + 1e-8)
        if not self.residual_on_lcfs:
            idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
            if idx.size == 1:
                n = len(self.roa_eval)
                k = int(idx[0])
                nblocks = R_flux.size // n if n > 0 else 0
                R_flux = np.concatenate(
                    [R_flux[b * n : b * n + k] for b in range(nblocks)]
                    + [R_flux[b * n + k + 1 : (b + 1) * n] for b in range(nblocks)]
                )

        R = R_flux * self.residual_weights.get("flux", 0.5)

        bc_residuals = []
        bc_dict = getattr(self._boundary, "bc_dict", {}) if hasattr(self, "_boundary") else {}
        if bc_dict and self.residual_weights.get("boundary_conditions", 0.0) != 0.0:
            x_arr = np.asarray(self._state.roa, float).ravel()
            for name, entry in bc_dict.items():
                if isinstance(entry, dict):
                    target_val = entry.get("val")
                    target_loc = entry.get("loc")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    target_val, target_loc = entry[0], entry[1]
                else:
                    continue

                if target_val is None or target_loc is None:
                    continue
                if not hasattr(self._state, name):
                    continue
                if not any(name in i or f"aL{i}" in name for i in self.predicted_profiles):
                    continue

                y_arr = np.asarray(getattr(self._state, name), float).ravel()
                if y_arr.size != x_arr.size or y_arr.size == 0:
                    continue

                pred_val = float(np.interp(float(target_loc), x_arr, y_arr))
                denom = abs(float(target_val)) + 1e-8 if normalize else 1.0
                resid = (pred_val - float(target_val)) / denom

                bc_residuals.append(resid * self.residual_weights.get("boundary_conditions", 0.25))

        R_bc = np.asarray(bc_residuals, float)
        if bc_residuals:
            R = np.concatenate([R, R_bc])

        if hasattr(self, "compiled_constraints") and self.compiled_constraints:
            R_constraints = []

            for c in self.compiled_constraints:
                k = int(np.argmin(np.abs(self.roa_eval - c["location"])))

                try:
                    val = c["eval"](self.state)
                except Exception:
                    val = np.nan

                if c["op"] in (">=", "<="):
                    val = min(0.0, val)

                if c["enforcement"] == "ramp":
                    ramp = min(1.0, self.iter / max(1, self.max_iter // 5))
                    val *= ramp

                R_constraints.append(np.sqrt(c["weight"]) * val)

            if R_constraints:
                R_constraints = np.array(R_constraints) * self.residual_weights.get("constraints", 0.25)
                R = np.concatenate([R, R_constraints])

        self.R = np.nan_to_num(R, nan=1e6, posinf=1e6, neginf=-1e6)
        self.R_dict = {"flux": R_flux, "boundary_conditions": R_bc, "constraints": R_constraints}

        return R

    def _normalize_expression(self, expr: str, aliases: Dict[str, str]) -> str:
        expr_norm = expr
        for k, v in aliases.items():
            expr_norm = re.sub(rf"\b{k}\b", v, expr_norm)
        return expr_norm

    def _parse_constraint(self, expr: str):
        if ">=" in expr:
            lhs, rhs = expr.split(">=")
            return lhs.strip(), rhs.strip(), ">="
        if "<=" in expr:
            lhs, rhs = expr.split("<=")
            return lhs.strip(), rhs.strip(), "<="
        if "=" in expr:
            lhs, rhs = expr.split("=")
            return lhs.strip(), rhs.strip(), "="
        raise ValueError(f"Unsupported constraint expression: {expr}")

    def _backend_expr(self, expr: str, var_map: Dict[str, str]) -> str:
        out = expr
        for k, v in var_map.items():
            out = re.sub(rf"\b{k}\b", v, out)
        return out

    def _flatten_params(self, X_dict: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Convert dict-of-dicts parameters to flat array with schema."""
        if not hasattr(self, "schema"):
            schema = []
            values = []
            for prof in list(X_dict.keys()):
                param_dict = X_dict[prof]
                for pname in list(param_dict.keys()):
                    schema.append((prof, pname))
                    values.append(float(param_dict[pname]))
            self.schema = schema
        else:
            schema = self.schema
            values = []
            for prof, pname in schema:
                values.append(float(X_dict[prof][pname]))

        return np.array(values, dtype=float), schema

    def _unflatten_params(self, X_flat: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Convert flat array back to dict-of-dicts using schema."""
        X_dict = {}
        for i, (prof, pname) in enumerate(self.schema):
            if prof not in X_dict:
                X_dict[prof] = {}
            X_dict[prof][pname] = float(X_flat[i])
        return X_dict

    def propose_parameters(self) -> np.ndarray:
        return self.X.copy()

    def check_convergence(self, y_model, y_target):
        """Compute residuals, objective Z, and objective variance varZ."""
        _ = self._compute_residuals(y_model, y_target)
        self.R_std = np.zeros_like(self.R)
        if self.R is None:
            self.Z = float("inf")
            self.converged = False
            return

        self.Z = self.objective(self.R)
        self.Z_std = np.zeros_like(self.Z)

        self.converged = bool(self.Z < self.tol)

        if self.converged:
            X_dict = self.X if isinstance(self.X, dict) else None
            C_R = self._build_residual_cov(self.R, X_dict, use_jacobian=self.use_jacobian)

            if self.J is not None:
                self._propagate_param_uncertainty_to_outputs(self.J)

            if self.normalize_residual and C_R is not None:
                try:
                    y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
                    Yt_vec = (
                        np.concatenate([np.asarray(self.Y_target[k]).ravel() for k in y_keys])
                        if y_keys
                        else np.ones(self.R.size)
                    )
                    Dinv = np.diag(1.0 / (np.abs(Yt_vec) + 1e-12))
                    C_R = Dinv @ C_R @ Dinv
                except Exception:
                    pass

            varZ = float(self.objective.variance(self.R, C_R)) if C_R is not None else 0.0
            varZ = max(0.0, varZ)

            confidence = float(self.options.get("convergence_confidence", 0.9))
            confidence = max(0.5, min(confidence, 0.9999))
            k_sigma = float(sp.stats.norm.ppf((1.0 + confidence) / 2.0))

            ub = self.Z + k_sigma * (np.sqrt(varZ) if varZ > 0 else 0.0)

            self.converged = bool(ub < self.tol)

            self.Z_std = np.sqrt(varZ)
            self.R_std = np.sqrt(np.diag(C_R))

        # Oscillation-aware convergence: feed history and check EMA stabilization
        if self._osc_detector is not None:
            self._osc_detector.add_iteration(
                Z=float(self.Z),
                X=self.X if isinstance(self.X, dict) else {},
                Y=getattr(self, "Y", None),
                R=self.R,
            )
            if not self.converged:
                osc_converged, mean_state = self._osc_detector.check_oscillation_convergence()
                if osc_converged and mean_state is not None:
                    print("Oscillation convergence detected. Applying time-averaged state.")
                    self.X = mean_state["X_mean"]
                    self.converged = True

    def save(self, bundle, filename="solver_checkpoint.pkl"):
        """Save all key objects from current solver run into one pickle file."""
        bundle["data"].save(self.cwd / "solver_history.csv")

        def _class_path(obj):
            return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

        def _to_basic(x):
            if x is None or isinstance(x, (bool, int, float, str)):
                return x
            if isinstance(x, (list, tuple, set)):
                return [_to_basic(v) for v in x]
            if isinstance(x, dict):
                return {k: _to_basic(v) for k, v in x.items()}
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer, np.floating)):
                return x.item()
            if isinstance(x, Path):
                return str(x)
            return repr(x)

        module_specs = {}
        for name in (
            "solver",
            "state",
            "surrogate",
            "transport",
            "targets",
            "boundary",
            "parameters",
            "neutrals",
        ):
            obj = bundle.get(name) if name != "solver" else self
            if obj is None:
                continue
            public_attrs = {}
            for attr, val in vars(obj).items():
                if attr.startswith("_"):
                    continue
                if callable(val):
                    continue
                try:
                    public_attrs[attr] = (_to_basic(val), type(val).__name__)
                except Exception:
                    public_attrs[attr] = repr(val)

            module_specs[name] = {
                "class_path": _class_path(obj),
                "attributes": public_attrs,
            }

        module_specs["timestamp"] = datetime.now(timezone.utc).isoformat()

        with open(self.cwd / filename, "wb") as fh:
            pickle.dump(module_specs, fh)

    def check_stalled(self) -> None:
        """Set stalled when total Z improvement over a model_iter_to_stall-wide window is < tol.

        Uses a sliding window of the last model_iter_to_stall model evaluations.  Stall
        fires when the best Z in the window hasn't improved over the oldest entry by at
        least tol, regardless of step-to-step fluctuations.  This prevents the previous
        consecutive-reset behaviour from making the argument effectively unreachable
        during monotone-but-slow descent.
        """
        if self._use_surr_iter or self.Z is None or self.iter < self.surr_warmup:
            return

        if self.model_iter_to_stall is None or self.model_iter_to_stall < 0:
            return

        self._Z_window.append(float(self.Z))
        if len(self._Z_window) > self.model_iter_to_stall:
            self._Z_window.pop(0)

        self._last_model_Z = float(self.Z)
        self._last_model_iter = self.iter

        # Fire stall once the window is full and the best value in it hasn't
        # improved on the oldest (window-start) value by more than tol.
        if self.model_iter_to_stall == 0:
            self.stalled = True
        elif len(self._Z_window) >= self.model_iter_to_stall:
            window_improvement = self._Z_window[0] - min(self._Z_window)
            if window_improvement < self.tol:
                self.stalled = True

        # Oscillation-aware stall: standard counter resets on every downswing, so
        # it never accumulates when Z oscillates.  Fire stall when the EMA trend
        # has flattened but is still above tolerance.
        if self._osc_detector is not None and not self.stalled:
            if self._osc_detector.check_oscillation_stall():
                print(
                    f"Oscillation stall detected at iter {self.iter}: "
                    f"EMA={self._osc_detector.Z_ema_history[-1]:.4e} is stable above "
                    f"tol={self.tol:.4e}. Solver is stuck oscillating."
                )
                self.stalled = True

    def run(self, state, boundary, parameters, neutrals, transport, targets, surrogate=None) -> SolverData:
        state.domain = self.domain
        state._trim()
        self.ix_eval = np.searchsorted(state.roa, self.roa_eval)
        transport.roa_eval = self.roa_eval
        transport.output_vars = self.target_vars  # totals only; surrogate gets turb/neo via transport_dict
        targets.roa_eval = self.roa_eval
        targets.output_vars = self.target_vars
        parameters.predicted_profiles = self.predicted_profiles
        parameters.bounds = self.bounds
        parameters.domain = self.domain
        parameters.roa_eval = self.roa_eval
        data = SolverData()

        self._state = state
        self._parameters = parameters
        self._boundary = boundary
        self._neutrals = neutrals
        self._transport = transport
        self._targets = targets
        self._surrogate = surrogate

        self.get_initial_parameters()

        if surrogate is not None:
            surrogate._initialize(
                self.transport_vars, self.target_vars, self.roa_eval,
                state, self.X,
                solver_output_units=self.solver_output_units,
            )

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

        while not self.converged and not self.stalled and self.iter <= self.max_iter:
            X_current, it = self.X, self.iter
            if it > 0:
                self._update_from_params(X_current)

            self._use_surr_iter = self._use_surrogate_iteration()
            self._evaluate(in_place=True)

            self.check_convergence(self.Y, self.Y_target)

            if not self._use_surr_iter and self.Z is not None and np.isfinite(self.Z):
                if self._best_model_Z is None or self.Z < self._best_model_Z:
                    self._best_model_Z = float(self.Z)
                    self._best_model_iter = self.iter
                    self._best_model_X = copy.deepcopy(self.X)
                    self._best_model_Y = {k: np.asarray(v).copy() for k, v in self.Y.items()}
                    self._best_model_Y_target = {k: np.asarray(v).copy() for k, v in self.Y_target.items()}
                    self._best_model_R = np.asarray(self.R).copy() if self.R is not None else None
                    self._best_model_R_dict = {
                        k: np.asarray(v).copy() for k, v in (self.R_dict or {}).items()
                    }

            self.check_stalled()

            if self.converged or self.stalled:
                if self._use_surr_iter and self.surr_verify_on_converge:
                    self._use_surr_iter = False
                    self._evaluate(in_place=True)
                    self.check_convergence(self.Y, self.Y_target)
                    if self.converged or (self.Z >= data.Z[-1]):
                        self.stalled = True

                if self.converged:
                    print("Convergence achieved. Solver run complete.")
                elif self.stalled:
                    print("Convergence stalled. Solver run complete.")
            else:
                if self.iter < self.max_iter:
                    if self.iter % self.iter_between_save == 0:
                        self.save(module_bundle)
                else:
                    if self._use_surr_iter:
                        self._use_surr_iter = False
                        self._evaluate(in_place=True)
                        self.check_convergence(self.Y, self.Y_target)
                    print("Max iterations reached. Solver run complete.")

            if (self.stalled or self.iter >= self.max_iter) and self._best_model_X is not None:
                if self._best_model_Z < self.Z:
                    print(
                        "Restoring best model evaluation from iteration "
                        f"{self._best_model_iter} (Z={self._best_model_Z:.6e} < current Z={self.Z:.6e})"
                    )
                    self.X = copy.deepcopy(self._best_model_X)
                    self.Y = {k: v.copy() for k, v in self._best_model_Y.items()}
                    self.Y_target = {k: v.copy() for k, v in self._best_model_Y_target.items()}
                    self.R = self._best_model_R.copy() if self._best_model_R is not None else None
                    self.R_dict = {
                        k: np.asarray(v).copy() for k, v in (self._best_model_R_dict or {}).items()
                    }
                    self.Z = self._best_model_Z
                    self._update_from_params(self.X)

            data.add(
                self.iter,
                self.X,
                self.X_std,
                self.R,
                self.R_std,
                self.R_dict,
                self.Z,
                self.Z_std,
                self.Y,
                self.Y_std,
                self.Y_target,
                self.Y_target_std,
                self._use_surr_iter,
            )
            self.save(module_bundle)

            X, X_std = self.propose_parameters()
            self.X = self._project_bounds(X)
            self.X_std = X_std

            self.iter += 1

            final_profiles_list = [
                (str(prof), getattr(self._state, prof)) for prof in self.predicted_profiles
                ] + [(f"aL{prof}", getattr(self._state, f"aL{prof}")) for prof in self.predicted_profiles]
            self._state.final_profiles = dict(final_profiles_list)

            # for tuning work, add normalized MSE between initial and final predicted profiles at roa_eval 
            self.mod_Z = self.Z + self.objective(
                np.concatenate([
                    (self._state.final_profiles.get(prof, np.zeros_like(self.roa_eval))[self.ix_eval] - \
                     self._state.initial_profiles.get(prof, np.zeros_like(self.roa_eval))[self.ix_eval]) / \
                        (np.abs(self._state.initial_profiles.get(prof, np.zeros_like(self.roa_eval))[self.ix_eval]) + 1e-8)
                    for prof in self.predicted_profiles
                ])
            )

        return data

    # def mapping(self):
    #     {ne : Ge : Ce,
    #      ni : Gi : Ci,
    #      Te : Qe : Pe,
    #      Ti : Qi : Pi,}