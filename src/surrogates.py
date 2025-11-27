"""
Surrogate models for accelerating transport solves.

Pattern
-------
- Base class attaches options and records metadata to SolverData (solver_data.surrogate_info).
- Child models implement fit(X, Y) and predict(X) and may use PlasmaState for feature building.

Current implementations
-----------------------
- GaussianProcessModel: generalized GP with scikit-learn backend when available and a NumPy fallback.

Factory
-------
create_surrogate_model(config) to instantiate from a config dict similar to other modules.

Example config
--------------
surrogate:
  class: surrogates.GaussianProcessModel
  args:
    kernel: Matern52        # RBF | Matern32 | Matern52 | RQ
    length_scale: 1.0
    variance: 1.0
    noise: 1e-4
    n_restarts: 3
    normalize_y: true
    feature_fields: ["ne", "te", "ti"]  # optional: build X from PlasmaState
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from datetime import datetime
import jax

try:
    from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.gaussian_process.kernels import (  # type: ignore
        RBF, ConstantKernel as C,
        Matern,
        RationalQuadratic as RQ,
        WhiteKernel as White,
    )
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    _SKLEARN_AVAILABLE = False



class SurrogateManager:
    """Handles multiple surrogate models (local or global) consistent with solver structure."""

    def __init__(self, options: Dict[str, Any], **kwargs):
        self.transport_vars: List[str] = []
        self.target_vars: List[str] = []
        self.options = options
        self.roa_eval: Optional[np.ndarray] = None
        self.transport_split = ['turb','neo']
        self.mode = options.get("mode", "global").lower()
        self.model_config = dict(kwargs)
        self.features = options.get("features", None)
        self.output_list: List[str] = []
        self.models: Dict[str, Any] = {}
        self.X_train: List[Dict[str, np.ndarray]] = []
        self.Y_train: List[Dict[str, np.ndarray]] = []
        self.trained = False

    def _initialize(self, transport_vars: List[str], target_vars: List[str], roa_eval: np.ndarray, state: Any, X_params: Dict[str, Dict[str, np.ndarray]]):
        """Delayed initialization to receive vars from solver."""
        self.transport_vars = transport_vars
        self.target_vars = target_vars
        self.roa_eval = roa_eval
        self.output_list = self.transport_vars + self.target_vars
        if self.roa_eval is not None and len(self.roa_eval) > 5:
            self.mode = 'global'
        self.get_features(state, X_params)
        self.build_models()

    # ------------------------------------------------
    # Construction
    # ------------------------------------------------
    def build_models(self):
        """Instantiate surrogate models for all transport and target outputs."""
        for key in self.output_list:
            if self.mode == 'local':
                self.models[key] = [
                    self.create_surrogate_model(self.model_config) for _ in range(len(self.roa_eval))
                ]
            else:
                self.models[key] = self.create_surrogate_model(self.model_config)

    def create_surrogate_model(self, config: Dict[str, Any]) -> SurrogateBase:
        kind = config.get("type", "gaussian_process").lower()
        if kind in SURROGATE_MODELS:
            return SURROGATE_MODELS[kind](config.get("kwargs", {}))
        raise ValueError(f"Unknown surrogate type '{kind}'.")

    def get_features(self, state: Any = None, X_params: Dict[str, Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Construct surrogate input feature matrix based on mode (local/global) and plasma state.

        Parameters
        ----------
        state : PlasmaState, optional
            Object containing all derived plasma quantities (aLne, betae, etc.)
        X_params : dict, optional
            Nested dict of design parameters:
            {'ne': {'aLy1': 1, 'aLy2': 5, ...}, 'te': {...}}

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_roa_eval, n_features)
        """
        # ------------------------------------------------------------------
        # 1. Base plasma features
        # ------------------------------------------------------------------
        if not hasattr(self, 'state_features'):
            self.state_features = [ # PCA suggests the first 3 components are critical
                "aLne", "aLte", "aLti", "tite",
                "gamma_exb", "gamma_par", "betae", "nustar", "rhostar",
                "Zeff", "q", "shear", "delta", "kappa", "eps"
            ]

        # ------------------------------------------------------------------
        # 2. Add design parameter feature *names*
        # ------------------------------------------------------------------
        if not hasattr(self, 'param_features'): 
            self.param_features = []
            if X_params:
                for prof, prof_params in X_params.items():
                    # skip gradient-like entries (they're already in self.features)
                    for pname in prof_params.keys():
                        if pname.startswith("aL"):
                            continue
                        self.param_features.append(f"{prof}_{pname}")

        # ------------------------------------------------------------------
        # 3. Merge all feature names
        # ------------------------------------------------------------------
        if not hasattr(self, 'all_features'):
            self.all_features = self.state_features + self.param_features
            if self.mode == "global" and "roa" not in self.all_features:
                self.all_features.append("roa")

        # ------------------------------------------------------------------
        # 4. Populate feature *values*
        # ------------------------------------------------------------------
        n_eval = len(self.roa_eval)
        n_feat = len(self.all_features)
        X_sample = np.zeros((n_eval, n_feat), dtype=float)

        if state is None:
            return X_sample

        # Loop over radial evaluation grid
        for i, roa in enumerate(self.roa_eval):
            # Find or interpolate each plasma feature at roa
            loc_sample_state = np.array([
                np.interp(roa, state.roa, getattr(state, name))
                for name in self.state_features
            ], dtype=float)

            param_vector = np.array([X_params[prof][pname] for pname in self.param_features], dtype=float)

            # Local mode → same features per roa, but repeated for each radial location
            # Global mode → parameter + roa features are identical for all rows
            if self.mode == "local":
                row_vec = np.concatenate([loc_sample_state, param_vector])
            else:  # global
                row_vec = np.concatenate([loc_sample_state, param_vector, np.array([roa], dtype=float)])

            X_sample[i] = row_vec

        return X_sample


    def get_outputs(self, transport: dict, targets: dict) -> np.ndarray:
        """Construct output array for all transport and target variables."""
        Y_sample = np.empty((len(self.roa_eval), len(self.output_list)))
        for i, roa in enumerate(self.roa_eval):
            transport_sample = np.array([transport[name][i] for name in self.transport_vars], dtype=float)
            targets_sample = np.array([targets[name][i] for name in self.target_vars], dtype=float)
            Y_sample[i] = np.concatenate([transport_sample, targets_sample])

        return Y_sample # row for each roa_eval
    
    def add_sample(self, state, X_params, transport, targets):
        """Extract new training data from current iteration and append."""
        X_features_array = self.get_features(state, X_params)
        Y_sample_array = self.get_outputs(transport, targets)

        if any(np.array_equal(X_features_array, x_train) for x_train in self.X_train):
            return  # Skip duplicate samples

        X_sample_dict = {self.all_features[i]: X_features_array[:, i] for i in range(len(self.all_features))}
        Y_sample_dict = {self.output_list[i]: Y_sample_array[:, i] for i in range(len(self.output_list))}

        self.X_train.append(X_sample_dict)
        self.Y_train.append(Y_sample_dict)
        self.trained = False

    # ------------------------------------------------
    # Training interface
    # ------------------------------------------------

    def fit(self):
        """Train all surrogates to available training data."""
        if not self.X_train:
            return  # No data to train on

        n_samples = len(self.X_train)
        n_roa = len(self.roa_eval)
        n_features = len(self.all_features)
        n_outputs = len(self.output_list)
        scores = {key: 0 for key in self.output_list}

        # Reconstruct numpy arrays from lists of dicts
        X_all_samples = np.zeros((n_samples, n_roa, n_features))
        Y_all_samples = np.zeros((n_samples, n_roa, n_outputs))

        self.x_scaler = StandardScaler()
        self.y_scalers = {key: StandardScaler() for key in self.output_list}

        for s_idx, sample_dict in enumerate(self.X_train):
            for f_idx, feature in enumerate(self.all_features):
                X_all_samples[s_idx, :, f_idx] = sample_dict[feature]

        for s_idx, sample_dict in enumerate(self.Y_train):
            for o_idx, output in enumerate(self.output_list):
                Y_all_samples[s_idx, :, o_idx] = sample_dict[output]
        
        # Transform samples
        X_all_samples = self.x_scaler.fit_transform(X_all_samples.reshape(-1,n_features)).reshape(n_samples,n_roa,n_features)

        for j, key in enumerate(self.models.keys()):
            if self.mode == 'global':
                # Reshape to (n_samples * n_roa, n_features)
                X_fit = X_all_samples.reshape(-1, n_features)

                # Option to transform with PCA here if desired
                # if self.n_pca is not None:
                    # pca = PCA(n_components=self.n_pca)
                    # X_fit = pca.fit_transform(X_fit)

                # Reshape to (n_samples * n_roa,)
                Y_fit = self.y_scalers[key].fit_transform(Y_all_samples[:, :, j].reshape(-1,1))
                self.models[key].fit(X_fit, Y_fit)
                scores[key] = self.models[key].score(X_fit, Y_fit)
            else:  # local
                for i, model in enumerate(self.models[key]):
                    # X for i-th roa from all samples: (n_samples, n_features)
                    i_features = X_all_samples[:, i, :]

                    # Option to transform with PCA here if desired
                    # if self.n_pca is not None:
                        # pca = PCA(n_components=self.n_pca)
                        # i_features = pca.fit_transform(i_features)

                    # Y for i-th roa and j-th output from all samples: (n_samples,)
                    i_outputs = self.y_scalers[key].fit_transform(Y_all_samples[:, i, j].reshape(-1,1))
                    model.fit(i_features, i_outputs)
                    scores[key] = model.score(i_features, i_outputs)

        self.trained = True
        self.score = scores # R^2
        self.hyperparameters = {key: model.get_hyperparameters() if self.mode == 'global' else
                        [m.get_hyperparameters() for m in model]
                        for key, model in self.models.items()}

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------
    def evaluate(self, params, state: Any, train: bool = False, batched: bool = False):
        """Predict surrogate outputs for current state."""

        if not self.trained and train:
            self.fit()

        X = self.x_scaler.transform(self.get_features(state, params))
        transport = {}
        transport_std = {}
        targets = {}
        targets_std = {}

        # TODO: make eval_fn dict for call in loop below with batched inputs from Monte Carlo / Bayesian Opt
        # if batched:
        #     eval_fn = jax.vmap(self._eval_single, in_axes=(0,))
        #     eval_fn(X)
        # else:
        #     self._eval_single(X)

        for key in self.output_list:
            if self.mode == 'global':
                prediction, std = self.models[key].predict(X, return_std=True)
                # predictions is (n_roa,), std is (n_roa,)
            else:
                prediction = np.zeros(len(self.roa_eval))
                std = np.zeros(len(self.roa_eval))
                for i, model in enumerate(self.models[key]):
                    X_i = X[i, :].reshape(1, -1)
                    prediction_i, std_i = model.predict(X_i, return_std=True)
                    prediction[i] = prediction_i[0]
                    std[i] = std_i[0]
                    # predictions is (n_roa,), std is (n_roa,)
            # Select appropriate output dictionaries
            output_dict = transport if key in self.transport_vars else targets
            output_std_dict = transport_std if key in self.transport_vars else targets_std
            
            # Inverse transform and store
            output_dict[key] = self.y_scalers[key].inverse_transform(prediction if prediction.ndim == 2 else prediction.reshape(-1,1)).flatten()
            output_std_dict[key] = std * self.y_scalers[key].scale_

        # Sum components for transport variables
        for base_var in self.target_vars:
            components = [v for v in self.transport_vars if v.startswith(base_var)]
            if len(components) > 1 and all(c in transport for c in components):
                summed_value = sum(transport[c] for c in components)
                # For independent variables, variances add
                summed_std = np.sqrt(sum(transport_std[c]**2 for c in components))
                transport[base_var] = summed_value # is this correct?
                transport_std[base_var] = summed_std

        self.transport = transport
        self.transport_std = transport_std
        self.targets = targets
        self.targets_std = targets_std
        # covariance matrices are ~diagonal


# -------------------------
# Base surrogate
# -------------------------

class SurrogateBase:
    """Base class for a single surrogate model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.features = self.config.get("features", [])
        self.mode = self.config.get("mode", "global").lower()

    def fit(self, X: np.ndarray, Y: np.ndarray, solver_data: Optional[Any] = None):
        """Fit model to (X, Y)."""
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, return_std: bool = False):
        """Predict output for given inputs."""
        raise NotImplementedError

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {}

# -------------------------------------------------------------------

class GaussianProcessSurrogate(SurrogateBase):
    """Gaussian process regression surrogate."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        length_scale = config.get("length_scale", 1.0)
        variance = config.get("variance", 1.0)
        kernel = C(variance) * RBF(length_scale)
        self._backend: str = "sklearn" if _SKLEARN_AVAILABLE else "simple"
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=config.get("noise", 1e-6),normalize_y=True) if _SKLEARN_AVAILABLE else \
            SimpleGPSurrogate(noise=config.get("noise", 1e-6))

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.model.fit(np.atleast_2d(X), np.atleast_2d(Y))

    def predict(self, X: np.ndarray, return_std: bool = False):
        return self.model.predict(np.atleast_2d(X), return_std=return_std)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.model.score(X,Y)

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {"C": self.model.kernel_.k1.constant_value**0.5,"l_rbf": self.model.kernel_.k2.length_scale}


# -------------------------
# Simple NumPy GP fallback
# -------------------------


class SimpleGPSurrogate:
    """Minimal squared-exponential GP for fallback when scikit-learn isn't available.

    Single-output GP with isotropic RBF kernel.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, noise: float = 1e-6, normalize_y: bool = True):
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


class NeuralNetSurrogate(SurrogateBase):
    """Neural network surrogate model (placeholder)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Placeholder for neural network initialization
        self.model = None

    def fit(self, X: np.ndarray, Y: np.ndarray, solver_data: Optional[Any] = None) -> None:
        # Placeholder for neural network training
        pass

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Placeholder for neural network prediction
        Y_pred = np.zeros((X.shape[0], 1))  # Dummy output
        return Y_pred, None

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {"model_type": "neural_network"}


class PolynomialSurrogate(SurrogateBase):
    """Polynomial surrogate model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Placeholder for polynomial model initialization
        self.model = None

    def fit(self, X: np.ndarray, Y: np.ndarray, solver_data: Optional[Any] = None) -> None:
        # Placeholder for polynomial model training
        pass

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Placeholder for polynomial model prediction
        Y_pred = np.zeros((X.shape[0], 1))  # Dummy output
        return Y_pred, None

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {"model_type": "polynomial"}

# -------------------------
# Factory and registry
# -------------------------


SURROGATE_MODELS = {
    "gaussian_process": GaussianProcessSurrogate,
    "gp": GaussianProcessSurrogate,
    "neural_network": NeuralNetSurrogate,
    "polynomial": PolynomialSurrogate,
}


def create_surrogate_model(config: Dict[str, Any]) -> SurrogateBase:
    kind = (config.get("type", "gaussian_process") or "gaussian_process").lower()
    if kind in ("gaussian_process", "gaussian", "gp"):
        return GaussianProcessSurrogate(config.get("kwargs", {}))
    if kind in ("neural_network", "nn"):
        return NeuralNetSurrogate(config.get("kwargs", {}))
    if kind in ("polynomial",):
        return PolynomialSurrogate(config.get("kwargs", {}))
    return GaussianProcessSurrogate(config.get("kwargs", {}))
