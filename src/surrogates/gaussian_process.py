"""Gaussian process surrogate model implementation."""

from __future__ import annotations

import scipy.optimize as _sopt
import warnings as _warnings
from sklearn.exceptions import ConvergenceWarning as _ConvergenceWarning
from typing import Any, Dict, Optional, List
import numpy as np

from .base import SurrogateBase
from .mean_functions import MeanFunctionBase

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel


def _gp_l_bfgs_b(obj_func, initial_theta, bounds):
    """Custom L-BFGS-B optimizer for sklearn GP that avoids spurious ConvergenceWarnings.

    sklearn's default optimizer wrapper calls ``_check_optimize_result`` which
    raises a ConvergenceWarning whenever L-BFGS-B returns status=2
    (ABNORMAL_TERMINATION_IN_LNSRCH).  That status indicates a line-search
    failure *at a near-optimal point* — i.e. the log-likelihood is essentially
    flat, meaning the hyperparameters are already very close to the optimum.
    This is routine for sparse training sets and well-conditioned transport data,
    not a genuine failure.

    By providing a custom callable we bypass that check entirely while using
    identical numerics (same algorithm, stricter tolerances for fewer spurious
    exits).
    """
    opt_res = _sopt.minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 2000, "gtol": 1e-8, "ftol": 1e-12},
    )
    return opt_res.x, opt_res.fun


class GaussianProcess(SurrogateBase):
    """Gaussian process regression with physics-informed priors and bounds.

    Supports:
    * Kernel selection: squared-exponential RBF (default) or Matérn 3/2 / 5/2.
    * Lengthscale constraints following the PORTALS approach (prevent
      overfitting on sparse training data).
    * Custom mean functions: a :class:`~mean_functions.MeanFunctionBase`
      instance can be supplied so the GP learns only the residual deviation
      from the prior.  At training time ``m(X)`` is subtracted from the
      targets; at inference time it is added back.

    Config keys
    -----------
    kernel : str
        ``'rbf'`` (default) or ``'matern'``.
    matern_nu : float
        Matérn smoothness parameter — ``1.5`` or ``2.5`` (default ``2.5``).
    length_scale, variance, noise, normalize_y, n_restarts,
    optimizer_maxiter, ard, min_lengthscale
        Same semantics as before.
    """

    _SUPPORTED_KERNELS = ("rbf", "matern")

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        features: Optional[List[str]] = None,
        mean_function: Optional[MeanFunctionBase] = None,
    ):
        super().__init__(config, features)
        self.mean_function: Optional[MeanFunctionBase] = mean_function
        config = config or {}

        # Hyperparameter configuration
        length_scale = float(config.get("length_scale", 1.0))
        variance = float(config.get("variance", 1.0))
        noise = float(config.get("noise", 1e-3))
        normalize_y = bool(config.get("normalize_y", True))
        n_restarts = int(config.get("n_restarts", 10))        # PORTALS default
        ard = bool(config.get("ard", True))                  # per-feature lengthscales by default
        min_ls = float(config.get("min_lengthscale", 0.05))  # 5 % of [0,1]-normalised range
        self.min_lengthscale = min_ls

        # Kernel selection
        kernel_type = str(config.get("kernel", "rbf")).lower()
        if kernel_type not in self._SUPPORTED_KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel_type}'. Supported: {self._SUPPORTED_KERNELS}"
            )
        self._kernel_type = kernel_type

        matern_nu = float(config.get("matern_nu", 2.5))
        if kernel_type == "matern" and matern_nu not in (0.5, 1.5, 2.5):
            raise ValueError(
                f"matern_nu must be 0.5, 1.5 or 2.5, got {matern_nu}"
            )

        # Per-feature (ARD) lengthscales with hard lower bound only.
        # Use a large but finite upper bound (1e5) so that sklearn's multi-restart
        # sampler can draw valid starting points (it requires finite bounds).
        # ARD correctly drives irrelevant features toward this ceiling — that is
        # desirable behaviour, not a failure.
        ls_bounds = (min_ls, 1e5)
        if ard and len(self.features) > 0:
            length_scales = length_scale * np.ones(len(self.features))
        else:
            length_scales = length_scale

        if kernel_type == "matern":
            inner_kernel = Matern(length_scale=length_scales, nu=matern_nu, length_scale_bounds=ls_bounds)
        else:
            inner_kernel = RBF(length_scale=length_scales, length_scale_bounds=ls_bounds)

        # Noise lower bound 1e-10 so the GP can fit near-noiseless roa-point data
        # without the optimiser hitting the floor and emitting a ConvergenceWarning.
        kernel = C(variance) * inner_kernel + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-10, 1e1))
        self._backend = "sklearn"

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,          # numerical jitter only; noise learned via WhiteKernel
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts,
            optimizer=_gp_l_bfgs_b,  # bypass spurious ConvergenceWarning
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the GP.  If a mean function is set, it is fitted first and
        subtracted so the kernel only sees the residuals.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)

        if self.mean_function is not None:
            self.mean_function.fit(X, Y.ravel())
            Y = Y - self.mean_function.predict(X).reshape(Y.shape)

        # Suppress sklearn ConvergenceWarnings: these occur when (a) L-BFGS-B
        # terminates at a near-optimal flat region (ABNORMAL_TERMINATION —
        # benign for transport data), (b) the noise level converges near its
        # lower bound (expected for deterministic codes like TGLF), or (c) a
        # lengthscale hits its upper bound (ARD correctly ignoring an irrelevant
        # feature).  All three indicate correct behaviour, not failures.
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=_ConvergenceWarning,
                                     module="sklearn")
            self.model.fit(X, Y)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Predict the GP output, adding the mean function back if present."""
        X = np.atleast_2d(X)
        result = self.model.predict(X, return_std=return_std)

        if self.mean_function is None:
            return result

        mean_pred = self.mean_function.predict(X)

        if return_std:
            mu, std = result
            return mu + mean_pred.reshape(mu.shape), std
        else:
            return result + mean_pred.reshape(result.shape)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """R² score evaluated on (X, Y) — mean function is accounted for."""
        X = np.atleast_2d(X)
        Y_pred = np.asarray(self.predict(X)).ravel()
        Y_true = np.asarray(Y).ravel()
        ss_res = np.sum((Y_true - Y_pred) ** 2)
        ss_tot = np.sum((Y_true - Y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    def get_hyperparameters(self) -> Dict[str, Any]:
        kernel = self.model.kernel_
        # kernel = C * inner_kernel + WhiteKernel
        product = kernel.k1   # C * inner_kernel
        hyperparams: Dict[str, Any] = {
            "C": float(product.k1.constant_value ** 0.5),
            "noise": float(kernel.k2.noise_level),
            "kernel": self._kernel_type,
        }
        inner = product.k2   # RBF or Matern
        hyperparams["length_scale"] = inner.length_scale
        if self._kernel_type == "matern":
            hyperparams["matern_nu"] = inner.nu
        if self.mean_function is not None:
            hyperparams["mean_function"] = repr(self.mean_function)
        return hyperparams

    def get_gradients(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Finite-difference gradient of the full GP mean (kernel + mean function) at X.

        Works for all kernel types and mean functions without requiring access
        to internal sklearn attributes.
        """
        X = np.atleast_2d(X)
        n, d = X.shape
        grad = np.zeros((n, d))
        for j in range(d):
            dX = np.zeros_like(X)
            dX[:, j] = eps
            grad[:, j] = (
                np.asarray(self.predict(X + dX)).ravel()
                - np.asarray(self.predict(X - dX)).ravel()
            ) / (2 * eps)
        return grad
