"""Gaussian process surrogate model implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .base import SurrogateBase

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


class GaussianProcess(SurrogateBase):
    """Gaussian process regression with physics-informed priors and bounds.

    Supports lengthscale constraints following PORTALS approach to prevent
    overfitting on sparse training data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, features: Optional[List[str]] = None):
        super().__init__(config, features)

        if not _SKLEARN_AVAILABLE:
            # Fallback to SimpleGPSurrogate without sklearn
            config = config or {}
            length_scale = float(config.get("length_scale", 1.0))
            variance = float(config.get("variance", 1.0))
            noise = float(config.get("noise", 1e-3))
            normalize_y = bool(config.get("normalize_y", True))

            self.model = SimpleGPSurrogate(
                length_scale=length_scale,
                variance=variance,
                noise=noise,
                normalize_y=normalize_y,
            )
            self._backend = "simple"
            return

        # Sklearn-based GP with constraints
        config = config or {}

        # Hyperparameter configuration
        length_scale = float(config.get("length_scale", 1.0))
        variance = float(config.get("variance", 1.0))
        noise = float(config.get("noise", 1e-3))
        normalize_y = bool(config.get("normalize_y", True))
        n_restarts = int(config.get("n_restarts", 5))
        optimizer_maxiter = int(config.get("optimizer_maxiter", 250))
        ard = bool(config.get("ard", False))

        # Lengthscale constraint (minimum value to prevent overfitting)
        # PORTALS uses 0.05 for normalized [0,1] inputs
        self.min_lengthscale = float(config.get("min_lengthscale", 0.05))

        # Build kernel with optional lengthscale bounds
        if ard and len(self.features) > 0:
            length_scales = length_scale * np.ones(len(self.features))
        else:
            length_scales = length_scale

        rbf_kernel = RBF(length_scale=length_scales)
        kernel = C(variance) * rbf_kernel

        self._backend = "sklearn"

        def _lbfgs_constrained(obj_func, initial_theta, bounds):
            """L-BFGS-B with lengthscale constraint enforcement."""
            # Modify bounds to enforce minimum lengthscale
            constrained_bounds = []
            for lower, upper in bounds:
                # Find lengthscale bounds and enforce minimum
                new_lower = max(lower, np.log(self.min_lengthscale))
                constrained_bounds.append((new_lower, upper))

            x_opt, f_opt, _info = fmin_l_bfgs_b(
                obj_func, initial_theta, bounds=constrained_bounds, maxiter=optimizer_maxiter
            )
            return x_opt, f_opt

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(noise),
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts,
            optimizer=_lbfgs_constrained,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.model.fit(np.atleast_2d(X), np.atleast_2d(Y))

    def predict(self, X: np.ndarray, return_std: bool = False):
        return self.model.predict(np.atleast_2d(X), return_std=return_std)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.model.score(X, Y)

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {"C": self.model.kernel_.k1.constant_value**0.5, "l_rbf": self.model.kernel_.k2.length_scale}

    def get_gradients(self, X: np.ndarray) -> np.ndarray:
        """Compute gradients of the GP mean prediction at X."""
        variance, l, alpha = self.model.kernel_.k1.constant_value, self.model.kernel_.k2.length_scale, self.model.alpha_
        X_train = self.model.X_train_

        def rbf_gradient(x1, x2, length_scale):
            diff = x1 - x2  # (n_test, n_features)
            sq_dist = np.sum(diff**2, axis=-1, keepdims=True)  # (n_test, 1)
            return -diff / (length_scale**2) * np.exp(-0.5 * sq_dist / (length_scale**2))

        # grad_k shape: (n_train, n_test, n_features)
        # alpha shape: (n_train, 1)
        # Need to sum over training points weighted by alpha
        grad_k = np.asarray([rbf_gradient(X, X_train[i], l) for i in range(len(X_train))])
        expected_gradient = np.einsum('ij,ijk->jk', alpha, grad_k)

        return expected_gradient

    def gp_mean_grad_fd(self, X, eps=1e-4):
        """Numerical gradient of GP mean prediction using finite differences."""
        X = np.atleast_2d(X)
        n, d = X.shape
        grad = np.zeros((n, d))

        for i in range(d):
            dX = np.zeros_like(X)
            dX[:, i] = eps

            pred_plus = self.model.predict(X + dX)
            pred_minus = self.model.predict(X - dX)
            mu_plus = pred_plus[0] if isinstance(pred_plus, tuple) else pred_plus
            mu_minus = pred_minus[0] if isinstance(pred_minus, tuple) else pred_minus

            grad[:, i] = (np.asarray(mu_plus).ravel() - np.asarray(mu_minus).ravel()) / (2 * eps)

        # Unscale gradient to original feature space
        if hasattr(self, 'x_scaler') and hasattr(self.x_scaler, 'scale_'):
            grad = grad / np.asarray(self.x_scaler.scale_)
        return grad


class SimpleGPSurrogate:
    """Minimal squared-exponential GP for fallback when scikit-learn isn't available.

    Single-output GP with isotropic RBF kernel.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
        normalize_y: bool = True,
    ):
        self.l = float(length_scale)
        self.s2 = float(variance)
        self.sn2 = float(noise) ** 2
        self.normalize_y = bool(normalize_y)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    @staticmethod
    def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Efficient squared Euclidean distance between rows of a (n,d) and b (m,d)
        a2 = np.sum(a * a, axis=1)[:, None]
        b2 = np.sum(b * b, axis=1)[None, :]
        return a2 + b2 - 2.0 * a @ b.T

    def _kernel(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        d2 = self._sqdist(Xa / self.l, Xb / self.l)
        return self.s2 * np.exp(-0.5 * d2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.ndim != 1:
            raise ValueError("_SimpleGP expects 1D y; use one model per output")
        self.X = X
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y) + 1e-12)
            yz = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            yz = y
        K = self._kernel(X, X)
        n = K.shape[0]
        K[np.diag_indices(n)] += self.sn2 + 1e-10
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yz))
        self._L = L
        self._alpha = alpha
        self.y = y

    def predict(self, Xs: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.X is None or self._L is None or self._alpha is None:
            raise RuntimeError("Model not fitted")
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self._kernel(self.X, Xs)
        mu_z = Ks.T @ self._alpha
        # Variance
        v = np.linalg.solve(self._L, Ks)
        kss = self.s2 * np.ones(Xs.shape[0])
        var = np.maximum(kss - np.sum(v * v, axis=0), 0.0)
        mu = self._y_mean + mu_z * self._y_std
        std = np.sqrt(var) * self._y_std if return_std else None
        return mu, std
