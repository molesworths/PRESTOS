"""Mean functions for Gaussian process regression.

Provides a base class and physics-informed implementations that encode prior
knowledge about the relationship between transport fluxes and their driving
gradients.  The mean function m(X) is subtracted from the training targets
before fitting the GP kernel, so the GP only needs to learn the *residual*
deviation from the physics prior.

Usage
-----
    from surrogates.mean_functions import create_mean_function

    mf = create_mean_function('Qe_low', feature_names=['aLne', 'aLte', ...])
    if mf is not None:
        gp = GaussianProcess(config, features=features, mean_function=mf)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Physics prior: which gradient feature drives each flux output (by prefix)
# ---------------------------------------------------------------------------

#: Maps output-variable prefix → state feature name for the linear prior.
FLUX_GRADIENT_MAP: Dict[str, str] = {
    "Qe": "aLte",  # Electron heat flux depends linearly on Te gradient
    "Qi": "aLti",  # Ion heat flux depends linearly on Ti gradient
    "Ge": "aLne",  # Particle flux depends linearly on ne gradient
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MeanFunctionBase:
    """Abstract base for GP mean functions.

    Subclasses must implement :meth:`fit` and :meth:`predict`.  The GP
    training pipeline calls ``fit(X_train, Y_train)`` once, then uses
    ``predict`` to subtract the mean from targets before fitting the kernel
    and to add it back at inference time.
    """

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Estimate mean-function parameters from training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Scaled input feature matrix (after any scaler applied by the manager).
        Y : ndarray, shape (n_samples,) or (n_samples, 1)
            Training targets (before output standardisation).
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the mean function at *X*.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        raise NotImplementedError

    def is_active(self) -> bool:
        """Return True if the mean function has a valid mapping for its output."""
        return True


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class ZeroMean(MeanFunctionBase):
    """Trivial zero mean function — equivalent to the default GP behaviour."""

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:  # noqa: D401
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class LinearParameterMean(MeanFunctionBase):
    """Affine mean function linking a flux output to its driving gradient.

    Assumes the feature matrix includes the normalized logarithmic gradient
    a/Ly as a named column (as produced by the spline parameterization).

    The prior is an ordinary-least-squares fit::

        m(X) = slope * X[:, gradient_idx] + intercept

    Supported output prefix → gradient feature mappings:

    =========  =========  ============================================
    Prefix     Feature    Physical interpretation
    =========  =========  ============================================
    ``Qe``     ``aLte``   Electron heat flux ∝ *T*\\ :sub:`e` gradient
    ``Qi``     ``aLti``   Ion heat flux ∝ *T*\\ :sub:`i` gradient
    ``Ge``     ``aLne``   Particle flux ∝ *n*\\ :sub:`e` gradient
    =========  =========  ============================================

    Parameters
    ----------
    output_key : str
        Name of the GP output variable, e.g. ``'Qe_low'`` or ``'Qi_'``.
    feature_names : list of str
        Ordered list of feature names matching the columns of the GP input
        matrix *X*.  Must be the same list used at training *and* inference.
    """

    FLUX_GRADIENT_MAP: Dict[str, str] = FLUX_GRADIENT_MAP

    def __init__(self, output_key: str, feature_names: List[str]) -> None:
        self.output_key = output_key
        self.feature_names = list(feature_names)
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self._gradient_idx: Optional[int] = None
        self._gradient_name: Optional[str] = None

        # Identify the gradient feature that drives this output
        for prefix, gradient_name in self.FLUX_GRADIENT_MAP.items():
            if output_key.startswith(prefix):
                if gradient_name in self.feature_names:
                    self._gradient_idx = self.feature_names.index(gradient_name)
                    self._gradient_name = gradient_name
                break  # stop at first matching prefix even if feature absent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True when the gradient feature for this output is present in the feature set."""
        return self._gradient_idx is not None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Estimate slope (\u2265 0) and intercept via constrained least squares.

        Positivity constraint encodes that turbulent transport is diffusive:
        flux increases with its driving gradient.
        """
        if self._gradient_idx is None:
            return  # no valid mapping — mean stays zero

        x_col = X[:, self._gradient_idx]
        y_flat = np.asarray(Y, dtype=float).ravel()

        # Demean to decouple intercept, then NNLS for slope >= 0
        x_mean, y_mean = x_col.mean(), y_flat.mean()
        slope_vec, _ = nnls((x_col - x_mean).reshape(-1, 1), y_flat - y_mean)
        self.slope = float(slope_vec[0])
        self.intercept = float(y_mean - self.slope * x_mean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate m(X) = slope * gradient_feature + intercept.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        if self._gradient_idx is None:
            return np.zeros(X.shape[0])
        return self.slope * X[:, self._gradient_idx] + self.intercept

    def __repr__(self) -> str:
        active = f"gradient='{self._gradient_name}', slope={self.slope:.4g}, intercept={self.intercept:.4g}"
        inactive = "inactive (no gradient mapping)"
        return (
            f"LinearParameterMean(output='{self.output_key}', "
            f"{active if self.is_active() else inactive})"
        )


class LinearFullMean(MeanFunctionBase):
    """Ridge-regression mean over all input features.

    Used for target channels that scale nearly linearly with all inputs.
    Avoids the PORTALS constant-kernel approximation while still capturing
    the dominant linear trend before the GP kernel corrects residuals.
    """

    def __init__(self) -> None:
        self._model = None
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        y_flat = np.asarray(Y, dtype=float).ravel()
        try:
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=1e-3)
            self._model.fit(X, y_flat)
        except ImportError:
            # Fallback: OLS via normal equations
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_flat, rcond=None)
            self._coef = coeffs[:-1]
            self._intercept = float(coeffs[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is not None:
            return self._model.predict(X)
        if self._coef is not None:
            return X @ self._coef + self._intercept
        return np.zeros(X.shape[0])

    def __repr__(self) -> str:
        fitted = self._model is not None or self._coef is not None
        return f"LinearFullMean(fitted={fitted})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_mean_function(
    output_key: str,
    feature_names: List[str],
    config: Optional[Dict[str, Any]] = None,
    is_target: bool = False,
) -> Optional[MeanFunctionBase]:
    """Construct a mean function for *output_key*, or return ``None``.

    Parameters
    ----------
    output_key : str
        Name of the GP output variable (e.g. ``'Qe_low'``, ``'Qi_'``).
    feature_names : list of str
        Ordered feature names matching the GP input matrix columns.
    config : dict, optional
        ``'type'``: ``'linear_parameter'`` (default) or ``'zero'``.
    is_target : bool
        When True, returns :class:`LinearFullMean` regardless of ``config``.
        Target channels scale nearly linearly with all inputs; a full
        ridge-regression prior outperforms the single-gradient prior here.

    Returns
    -------
    MeanFunctionBase or None
    """
    if is_target:
        return LinearFullMean()

    kind = (config or {}).get("type", "linear_parameter")

    if kind == "linear_parameter":
        mf = LinearParameterMean(output_key, feature_names)
        return mf if mf.is_active() else None

    if kind == "zero":
        return ZeroMean()

    raise ValueError(
        f"Unknown mean function type '{kind}'. "
        "Supported: 'linear_parameter', 'zero'."
    )
