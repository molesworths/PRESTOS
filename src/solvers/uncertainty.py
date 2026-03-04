from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy as sp


class UncertaintyMixin:
    """Uncertainty propagation helpers shared by solvers."""

    def _propagate_param_uncertainty_to_outputs(self, J_R: Optional[np.ndarray] = None) -> None:
        """Propagate parameter uncertainty to model output variables via Jacobian linearization."""
        if not self.use_jacobian:
            return

        J = J_R if J_R is not None else self.J
        if J is None:
            return

        try:
            if self._parameters is None or not hasattr(self._parameters, "param_std"):
                return
            sigma_X_dict = self._parameters.param_std
            if not sigma_X_dict:
                return

            sigma_X, _ = self._flatten_params(sigma_X_dict)
            if sigma_X is None or sigma_X.size == 0:
                return
        except Exception:
            return

        Cx = np.diag(sigma_X**2)

        try:
            y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])

            row_idx = 0
            for k in y_keys:
                y_array = np.asarray(self.Y[k], dtype=float)
                n_elements = y_array.size

                if row_idx + n_elements > J.shape[0]:
                    break
                J_Y_k = J[row_idx : row_idx + n_elements, :]

                try:
                    cov_k = J_Y_k @ Cx @ J_Y_k.T
                    std_param_k = np.sqrt(np.maximum(0.0, np.diag(cov_k)))
                except Exception:
                    row_idx += n_elements
                    continue

                y_std_existing = np.asarray(self.Y_std.get(k, np.zeros_like(y_array)), dtype=float).ravel()
                y_std_combined = np.sqrt(y_std_existing**2 + std_param_k**2)

                self.Y_std[k] = y_std_combined.reshape(y_array.shape)

                row_idx += n_elements
        except Exception:
            pass

    def _build_residual_cov(
        self,
        R: np.ndarray,
        X_dict: Optional[Dict[str, Dict[str, float]]],
        use_jacobian: bool = True,
    ) -> Optional[np.ndarray]:
        """Build covariance matrix C_R for residual vector R."""
        try:
            sigma_list = []

            if "flux" in (self.R_dict or {}):
                y_keys = sorted([k for k in (self.Y_target or {}).keys() if k in (self.Y or {})])
                model_sig_list = []
                target_sig_list = []
                for k in y_keys:
                    mstd = np.asarray(self.Y_std.get(k, np.zeros_like(self.Y[k])), dtype=float)
                    tstd = np.asarray(self.Y_target_std.get(k, np.zeros_like(self.Y_target[k])), dtype=float)
                    model_sig_list.append(mstd.ravel())
                    target_sig_list.append(tstd.ravel())
                sigma_model_vec = np.concatenate(model_sig_list) if model_sig_list else np.empty(0)
                sigma_target_vec = np.concatenate(target_sig_list) if target_sig_list else np.empty(0)
                sigma_flux = np.sqrt(np.maximum(0.0, sigma_model_vec**2 + sigma_target_vec**2))
                if not self.residual_on_lcfs:
                    idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
                    if idx.size == 1:
                        n = len(self.roa_eval)
                        k = int(idx[0])
                        nblocks = len(sigma_flux) // n if n > 0 else 0
                        sigma_flux = np.concatenate(
                            [sigma_flux[b * n : b * n + k] for b in range(nblocks)]
                            + [sigma_flux[b * n + k + 1 : (b + 1) * n] for b in range(nblocks)]
                        )
                sigma_list.append(sigma_flux)

            if "boundary_conditions" in (self.R_dict or {}):
                bc_size = np.asarray(self.R_dict["boundary_conditions"]).size
                bc_sigma = float(getattr(self._boundary, "sigma", 0.01)) if hasattr(self, "_boundary") else 0.01
                sigma_bc = np.full(bc_size, bc_sigma, dtype=float)
                sigma_list.append(sigma_bc)

            if "constraints" in (self.R_dict or {}):
                constraints_size = np.asarray(self.R_dict["constraints"]).size
                if constraints_size > 0 and hasattr(self, "compiled_constraints"):
                    sigma_constraints = []
                    for c in self.compiled_constraints:
                        c_sigma = float(c.get("sigma", 0.01)) if isinstance(c, dict) else 0.01
                        sigma_constraints.append(c_sigma)
                    sigma_constraints_arr = np.array(sigma_constraints)
                    if sigma_constraints_arr.size < constraints_size:
                        sigma_constraints_arr = np.tile(
                            sigma_constraints_arr,
                            (constraints_size // sigma_constraints_arr.size) + 1,
                        )[:constraints_size]
                    sigma_constraints_vec = sigma_constraints_arr[:constraints_size]
                else:
                    sigma_constraints_vec = np.full(constraints_size, 0.01, dtype=float)
                sigma_list.append(sigma_constraints_vec)

            if sigma_list:
                sigma_resid = np.concatenate(sigma_list)
                C_meas = np.diag(sigma_resid**2) if sigma_resid.size > 0 else None
            else:
                C_meas = None
        except Exception:
            C_meas = None
        self.C_meas = C_meas

        C_param = None
        if use_jacobian and X_dict is not None:
            try:
                X_flat, schema = self._flatten_params(X_dict)
                J = self._attempt_get_jacobian(X_flat, R)
                if J is not None:
                    self.J = J
                    sigma_X, _ = self._flatten_params(self._parameters.param_std)
                    if sigma_X is not None and sigma_X.size == J.shape[1]:
                        Cx = np.diag(sigma_X**2)
                        C_param = J @ Cx @ J.T
            except Exception:
                C_param = None

        if C_meas is None and C_param is None:
            return None
        if C_meas is None:
            C_total = C_param
        if C_param is None:
            C_total = C_meas
        else:
            C_total = C_param + C_meas

        if not self.residual_on_lcfs:
            idx = np.where(np.isclose(self.roa_eval, 1.0, atol=1e-3))[0]
            if idx.size == 1:
                n = len(self.roa_eval)
                k = int(idx[0])
                nblocks = C_total.shape[0] // n if n > 0 else 0
                C_total[:, [b * n + k for b in range(nblocks)]] = 0.0
                C_total[[b * n + k for b in range(nblocks)], :] = 0.0

        _ = self._compute_posterior_parameter_uncertainty()

        return C_total

    def _compute_posterior_parameter_uncertainty(self):
        """Compute posterior parameter covariance via linearized Bayesian update."""
        if self.J is None and self.use_jacobian:
            self.J = self._attempt_get_jacobian(
                self.X if isinstance(self.X, dict) else self.X,
                self.R,
            )
        else:
            return None
        try:
            Cx_post = sp.linalg.pinv(self.J) @ self.C_meas @ sp.linalg.pinv(self.J).T
            return Cx_post
        except np.linalg.LinAlgError:
            return None
