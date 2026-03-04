from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from .relax import RelaxSolver
from .solver_base import SolverBase


class BayesianOptSolver(SolverBase):
    """BO solver that optimizes a Monte-Carlo estimate of E[Z]."""

    def __init__(self, options: Optional[dict] = None):
        super().__init__(options)
        self.n_restarts = int(self.options.get("n_restarts", 8))
        self.n_steps = int(self.options.get("n_steps", 80))
        self.lr = float(self.options.get("lr", 1e-1))
        self.n_mc = int(self.options.get("n_mc", 128))
        self.batch_size = int(self.options.get("batch_size", 64))
        self.seed = int(self.options.get("seed", 0))
        self.adam_beta1 = float(self.options.get("adam_beta1", 0.9))
        self.adam_beta2 = float(self.options.get("adam_beta2", 0.999))
        self.adam_eps = float(self.options.get("adam_eps", 1e-8))

        self.acquisition = str(self.options.get("acquisition", "ei")).lower()
        self.ucb_k = float(self.options.get("ucb_k", 2.0))
        self.pi_k = float(self.options.get("pi_k", 10.0))

        self._schema = None
        self._rng = np.random.default_rng(self.seed)

    def propose_parameters(self):
        """Batch MC BO using surrogate gradients when available."""
        if getattr(self, "iter", 0) < getattr(self, "surr_warmup", 5):
            return RelaxSolver.propose_parameters(self)

        X_flat0, schema = self._flatten_params(self.X)
        n_params = X_flat0.size

        bounds_mat = np.zeros((n_params, 2), float)
        for i, (prof, pname) in enumerate(schema):
            lo, hi = self.bounds_dict[prof][pname]
            bounds_mat[i, 0] = -1e6 if lo is None else float(lo)
            bounds_mat[i, 1] = 1e6 if hi is None else float(hi)
        lo_arr = bounds_mat[:, 0]
        hi_arr = bounds_mat[:, 1]

        acquisition = str(getattr(self, "acquisition", "ei")).lower()
        n_restarts = int(getattr(self, "n_restarts", 5))
        n_steps = int(getattr(self, "n_steps", 50))
        n_mc = int(getattr(self, "n_mc", 50))
        lr = float(getattr(self, "lr", 1e-1))
        pi_k = float(getattr(self, "pi_k", 1.0))
        ucb_k = float(getattr(self, "ucb_k", 1.0))
        fd_eps_rel = float(getattr(self, "fd_acq_epsilon", 1e-3))
        seed = int(getattr(self, "seed", 0)) + int(self.iter)
        rng = np.random.default_rng(seed)
        normalize_resid = bool(self.normalize_residual)
        self._use_surr_iter = True if self._surrogate is not None else False

        Z_best = (
            float(self.Z)
            if (hasattr(self, "Z") and self.Z is not None and np.isfinite(self.Z))
            else float("inf")
        )
        target_keys = sorted(self.Y_target.keys())
        n_target_vars = len(target_keys)
        n_transport_vars = len(self.transport_vars)
        n_roa = len(self.roa_eval)

        Yt_stack = np.vstack([np.asarray(self.Y_target[k]).ravel().reshape(1, n_roa) for k in target_keys])
        denom_norm = np.abs(Yt_stack) + 1e-8 if normalize_resid else None
        feature_index = {name: idx for idx, name in enumerate(getattr(self._surrogate, "all_features", []))}

        def objective_grad(residual_vec):
            if hasattr(self.objective, "gradient"):
                try:
                    return np.asarray(self.objective.gradient(residual_vec), float)
                except Exception:
                    pass
            r = np.asarray(residual_vec, float)
            eps = 1e-6 * np.maximum(1.0, np.abs(r))
            grad = np.zeros_like(r)
            base = float(self.objective(r))
            for i in range(r.size):
                rp = r.copy()
                rm = r.copy()
                rp[i] += eps[i]
                rm[i] -= eps[i]
                gp = float(self.objective(rp))
                gm = float(self.objective(rm))
                grad[i] = (gp - gm) / (2.0 * eps[i])
            return grad

        def _acq_single(x_flat):
            X_dict = self._unflatten_params(x_flat)
            Y_model, Y_model_std, Y_target, Y_target_std = self._evaluate(in_place=True)

            eps = rng.normal(size=(n_mc, len(Y_model), n_roa))
            y_samples = (
                np.array(list(Y_model.values()))[None, :, :]
                + np.array(list(Y_model_std.values()))[None, :, :] * eps
            )
            Z_samples = np.zeros(n_mc, dtype=float)
            for mc_i in range(n_mc):
                Y_mc = y_samples[mc_i, :, :]
                Y_mc_dict = {k: Y_mc[i, :] for i, k in enumerate(self.transport_vars + self.target_vars)}
                R_mc = self._compute_residuals(Y_mc_dict, Y_target)
                Z_samples[mc_i] = self.objective(R_mc)

            if acquisition == "ei":
                improv = np.maximum(0.0, Z_best - Z_samples)
                return float(np.mean(improv))
            if acquisition == "pi":
                probs = 1.0 / (1.0 + np.exp(-pi_k * (Z_best - Z_samples)))
                return float(np.mean(probs))
            if acquisition == "ucb":
                muZ = float(np.mean(Z_samples))
                stdZ = float(np.std(Z_samples) + 1e-8)
                return float(-(muZ) - ucb_k * stdZ)
            improv = np.maximum(0.0, Z_best - Z_samples)

            return float(np.mean(improv))

        def _acq_batch(x_batch):
            """Batched acquisition using surrogate.evaluate with 3D params array."""
            x_batch = np.asarray(x_batch, float)
            if x_batch.ndim != 2:
                raise ValueError(f"_acq_batch expects 2D array, got shape {x_batch.shape}")

            n_batch = x_batch.shape[0]

            states = []
            for b in range(n_batch):
                self._update_from_params(x_batch[b])
                state_b = copy.deepcopy(self._state)
                states.append(state_b)
            Y_model, Y_model_std, Y_target, Y_target_std = self._evaluate(x_batch, states, in_place=False)

            self._update_from_params(self.X)

            M = np.stack([np.asarray(Y_model[k]) for k in Y_model.keys()], axis=1)
            S = np.stack([np.asarray(Y_model_std[k]) for k in Y_model_std.keys()], axis=1)

            eps = rng.normal(size=(n_mc, n_batch, M.shape[1], n_roa))
            Ys = M[None, :, :, :] + S[None, :, :, :] * eps

            T = np.stack([np.asarray(Y_target[k]) for k in self.target_vars], axis=1)

            Z_samples = np.zeros((n_mc, n_batch), dtype=float)
            Z_samples = np.array(
                [
                    [
                        self.objective(
                            self._compute_residuals(
                                dict(zip(self.model_vars, Ys[mc_i, b])),
                                dict(zip(self.target_vars, T[b])),
                            )
                        )
                        for b in range(n_batch)
                    ]
                    for mc_i in range(n_mc)
                ]
            )

            if acquisition == "ei":
                improv = np.maximum(0.0, Z_best - Z_samples)
                return np.mean(improv, axis=0)
            if acquisition == "pi":
                probs = 1.0 / (1.0 + np.exp(-pi_k * (Z_best - Z_samples)))
                return np.mean(probs, axis=0)
            if acquisition == "ucb":
                muZ = np.mean(Z_samples, axis=0)
                stdZ = np.std(Z_samples, axis=0) + 1e-8
                return -(muZ) - ucb_k * stdZ
            improv = np.maximum(0.0, Z_best - Z_samples)
            return np.mean(improv, axis=0)

        def acquisition_value(x_flat):
            x_arr = np.asarray(x_flat, float)
            if x_arr.ndim == 1:
                return _acq_single(x_arr)
            if x_arr.ndim == 2:
                return _acq_batch(x_arr)
            raise ValueError(f"acquisition_value expects 1D or 2D input, got shape {x_arr.shape}")

        def surrogate_grad_batch(x_batch):
            """Batched acquisition gradient via surrogate.get_grads; None on failure."""
            try:
                x_batch = np.asarray(x_batch, float)
                if x_batch.ndim != 2:
                    return None
                n_batch = x_batch.shape[0]

                states = []
                for b in range(n_batch):
                    self._update_from_params(x_batch[b])
                    state_b = copy.deepcopy(self._state)
                    states.append(state_b)

                Y_model, Y_model_std, Y_target, Y_target_std = self._evaluate(x_batch, states, in_place=False)

                self._update_from_params(self.X)

                Y_model_list = [
                    dict(zip(Y_model.keys(), [np.asarray(Y_model[k])[b] for k in Y_model.keys()]))
                    for b in range(n_batch)
                ]
                Y_target_list = [
                    dict(zip(Y_target.keys(), [np.asarray(Y_target[k])[b] for k in Y_target.keys()]))
                    for b in range(n_batch)
                ]

                grad_r_list = []
                Z_det_list = []
                R_array = np.array(
                    [self._compute_residuals(Y_model_list[b], Y_target_list[b]) for b in range(n_batch)]
                )
                Z_det_list = [self.objective(R_array[b]) for b in range(n_batch)]
                grad_r_list = [objective_grad(R_array[b].reshape(-1)) for b in range(n_batch)]

                transport_grads, target_grads = self._surrogate.get_grads(x_batch, states)
                if transport_grads is None:
                    return None

                G_full = np.zeros((n_batch, n_params), dtype=float)
                param_to_feature = {}
                for j, (prof, pname) in enumerate(schema):
                    if pname.startswith("aL"):
                        feature_name = f"aL{prof}"
                        f_idx = feature_index.get(feature_name, None)
                        if f_idx is not None:
                            param_to_feature[j] = (f_idx, float(self._surrogate.x_scaler.scale_[f_idx]))

                for b in range(n_batch):
                    grad_mu_flat = np.zeros((n_target_vars * n_roa, n_params), dtype=float)
                    for t_idx, tname in enumerate(target_keys):
                        g_arr_b = transport_grads.get(tname)
                        if g_arr_b is None:
                            continue
                        g_b = np.asarray(g_arr_b)[b]
                        for j, (f_idx, scale) in param_to_feature.items():
                            grad_unscaled = g_b[:, f_idx] / (scale if scale != 0 else 1.0)
                            start = t_idx * n_roa
                            stop = (t_idx + 1) * n_roa
                            grad_mu_flat[start:stop, j] = grad_unscaled

                    if normalize_resid and denom_norm is not None:
                        grad_mu_flat = grad_mu_flat / denom_norm.reshape(-1, 1)

                    gradZ_b = grad_mu_flat.T @ grad_r_list[b]

                    Z_det_b = Z_det_list[b]
                    if acquisition == "ei":
                        G_full[b] = (-gradZ_b) if Z_det_b < Z_best else np.zeros_like(gradZ_b)
                    elif acquisition == "pi":
                        sig = 1.0 / (1.0 + np.exp(-pi_k * (Z_best - Z_det_b)))
                        G_full[b] = sig * (1.0 - sig) * (-pi_k) * gradZ_b
                    elif acquisition == "ucb":
                        G_full[b] = -gradZ_b
                    else:
                        G_full[b] = (-gradZ_b) if Z_det_b < Z_best else np.zeros_like(gradZ_b)

                return G_full
            except Exception:
                return None

        def fd_grad_batch(x_batch):
            """Batched central FD gradient of acquisition."""
            x_batch = np.asarray(x_batch, float)
            n_batch, n_params_b = x_batch.shape
            G = np.zeros_like(x_batch)
            for j in range(n_params_b):
                span = hi_arr[j] - lo_arr[j]
                h = fd_eps_rel * (span if np.isfinite(span) and span > 0 else 1.0)
                x_p = x_batch.copy()
                x_m = x_batch.copy()
                x_p[:, j] = np.minimum(hi_arr[j], x_p[:, j] + h)
                x_m[:, j] = np.maximum(lo_arr[j], x_m[:, j] - h)
                fp = acquisition_value(x_p)
                fm = acquisition_value(x_m)
                denom = x_p[:, j] - x_m[:, j]
                safe = denom != 0
                gcol = np.zeros(n_batch, float)
                gcol[safe] = (fp[safe] - fm[safe]) / denom[safe]
                G[:, j] = gcol
            return G

        seed_batch = max(n_restarts, int(getattr(self, "batch_size", 1)))
        step_scale = float(getattr(self, "bo_step_scale", 0.05))
        width = hi_arr - lo_arr
        sigma = step_scale * np.maximum(width, 1e-12)
        candidate_batch = X_flat0 + rng.normal(scale=sigma, size=(seed_batch, n_params))
        candidate_batch = np.clip(candidate_batch, lo_arr, hi_arr)
        batch_vals = acquisition_value(candidate_batch)
        if np.ndim(batch_vals) == 0:
            batch_vals = np.array([batch_vals])
        seed_order = np.argsort(batch_vals)[::-1][:n_restarts]

        x_batch = candidate_batch[seed_order].copy()
        for step in range(n_steps):
            G = surrogate_grad_batch(x_batch)
            if G is None or not np.all(np.isfinite(G)):
                G = fd_grad_batch(x_batch)
            x_batch = x_batch + lr * G
            x_batch = np.clip(x_batch, lo_arr, hi_arr)

        vals = acquisition_value(x_batch)
        best_idx = int(np.argmax(vals))
        best_x = x_batch[best_idx].copy()

        if best_x is None:
            best_x = X_flat0.copy()

        X_new = self._unflatten_params(best_x)
        X_new = self._project_bounds(X_new)
        X_new_std = {
            prof: {pname: abs(val) * getattr(self._parameters, "sigma", 0.0) for pname, val in prof_dict.items()}
            for prof, prof_dict in X_new.items()
        }
        return X_new, X_new_std
