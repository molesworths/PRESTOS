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
from scipy.optimize import fmin_l_bfgs_b

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.gaussian_process.kernels import (
        RBF, ConstantKernel as C,
        Matern,
        RationalQuadratic as RQ,
        WhiteKernel as White,
    )
    _SKLEARN_AVAILABLE = True
except Exception:
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
        self.model_config = dict(options.get("kwargs", {}))
        self.scaling = options.get("scaling", "standard") # standard | minmax | physics
        self.features = options.get("features", None)
        self.output_list: List[str] = []
        self.models: Dict[str, Any] = {}
        self.X_train: List[Dict[str, np.ndarray]] = []
        self.Y_train: List[Dict[str, np.ndarray]] = []
        self.max_train_samples = options.get("max_train_samples", 10)  # Increased default for better coverage
        self.min_train_samples = options.get("min_train_samples", 5)
        self.min_score_threshold = options.get("min_score_threshold", 0.5)  # Minimum R² to use surrogate
        self.trained = False
        self._X_train_bounds = None  # Will store (n_features, 2) array of [min, max] per feature
        self.score = 0.

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

    def _normalize_input(self, params):
        """Normalize input params to (is_batched, batched_array, input_type) format.
        
        Parameters
        ----------
        params : dict or np.ndarray
            - Dict[str, Dict[str, np.ndarray]]: single sample as nested dict
            - np.ndarray shape (n_params,): single sample flattened
            - np.ndarray shape (n_batch, n_params): batched samples
        
        Returns
        -------
        tuple
            (is_batched: bool, params_array: np.ndarray, input_type: str)
            - is_batched: True if input was batched
            - params_array: normalized to shape (n_batch, n_params) or (1, n_params)
            - input_type: 'dict', 'flat', or 'batched'
        """
        if isinstance(params, dict):
            # Dict[str, Dict[str, np.ndarray]] → flatten to (1, n_params)
            flat_params = [param_values for prof in params for param_values in params[prof].values()]
            params_array = np.array(flat_params)[np.newaxis, :]  # (1, n_params, n_roa)
            return False, params_array, 'dict'
        
        elif isinstance(params, np.ndarray):
            if params.ndim == 1:
                # (n_params,) → (1, n_params)
                return False, params[np.newaxis, :], 'flat'
            elif params.ndim == 2:
                # (n_batch, n_params) → already batched
                return True, params, 'batched'
            else:
                raise ValueError(f"Array input must be 1D or 2D, got shape {params.shape}")
        
        else:
            raise TypeError(f"params must be dict or np.ndarray, got {type(params)}")

    # ------------------------------------------------
    # Construction
    # ------------------------------------------------
    def build_models(self):
        """Instantiate surrogate models for all transport and target outputs."""
        for key in self.output_list:
            if self.mode == 'local':
                self.models[key] = [
                    self.create_surrogate_model(self.model_config,features=self.all_features) for _ in range(len(self.roa_eval))
                ]
            else:
                self.models[key] = self.create_surrogate_model(self.model_config,features=self.all_features)

    def create_surrogate_model(self, config: Dict[str, Any], features: List[str]) -> SurrogateBase:
        kind = config.get("type", "gaussian_process").lower()
        if kind in SURROGATE_MODELS:
            return SURROGATE_MODELS[kind](config, features=features)
        raise ValueError(f"Unknown surrogate type '{kind}'.")

    def get_features(self, state: Any = None, X_params = None) -> np.ndarray:
        """
        Construct surrogate input feature matrix based on mode (local/global) and plasma state.

        Parameters
        ----------
        state : PlasmaState or list of PlasmaState, optional
            Object containing all derived plasma quantities (aLne, betae, etc.)
        X_params : dict or np.ndarray, optional
            Design parameters in one of three formats:
            - Dict[str, Dict[str, np.ndarray]]: nested dict for single sample
            - np.ndarray shape (n_params, n_roa): flattened single sample
            - np.ndarray shape (n_batch, n_params, n_roa): batched samples

        Returns
        -------
        np.ndarray
            Feature matrix:
            - Single input: shape (n_roa_eval, n_features)
            - Batched input: shape (n_batch, n_roa_eval, n_features)
        """
        # ------------------------------------------------------------------
        # 1. Base plasma features
        # ------------------------------------------------------------------
        if not hasattr(self, 'state_features'):
            self.state_features = [
                "aLne", "aLte", "aLti", "tite", "betae",
                "nustar", "rhostar", # "gamma_exb", "gamma_par",
                "q", #"Zeff", "q", "shear", "delta", "kappa", "eps"
            ]

            self.state_features_scalers = {
                "aLne": lambda x: x/1.,
                "aLte": lambda x: x/1.,
                "aLti": lambda x: x/1.,
                "tite": lambda x: x/1.,
                "gamma_exb": lambda x: np.log10(x + 1e-8),
                "gamma_par": lambda x: np.log10(x + 1e-8),
                "betae": lambda x: x/0.01,
                "nustar": lambda x: x/1.,
                "rhostar": lambda x: x/0.01,
                "Zeff": lambda x: x/1.,
                "q": lambda x: x/1.,
                "shear": lambda x: x/1.,
                "delta": lambda x: x/1.,
                "kappa": lambda x: x/1.,
                "eps": lambda x: x/1.,
            }


        # ------------------------------------------------------------------
        # 2. Add design parameter feature *names* (only for dict input)
        # ------------------------------------------------------------------
        if not hasattr(self, 'param_features'): 
            self.param_features = []
            if X_params and isinstance(X_params, dict):
                for prof, prof_params in X_params.items():
                    # skip gradient-like entries (they're already in self.features)
                    for pname in prof_params.keys():
                        if pname.startswith("aL"):
                            continue
                        self.param_features.append(f"{prof}_{pname}")
            # save indices of state_features that are also design parameters
            # self.param_features may be empty if design parameters are aLy because
            # they are already included in state features, so we use aL* features, e.g., aLne, and roa_eval
            # to map back to param names, e.g., aLne_0, aLne_1, etc., from params_schema
            # if self.state_features:
            #     for feat in self.state_features:
            #         if feat not in self.param_features and feat in self.state_features:
            #             self.param_features.append(feat)

        # ------------------------------------------------------------------
        # 3. Merge all feature names
        # ------------------------------------------------------------------
        if not hasattr(self, 'all_features'):
            self.all_features = self.state_features + self.param_features
            if self.mode == "global" and "roa" not in self.all_features:
                pass #self.all_features.append("roa")

        # ------------------------------------------------------------------
        # 4. Handle batched vs single inputs
        # ------------------------------------------------------------------
        if X_params is not None:
            is_batched, params_array, input_type = self._normalize_input(X_params)
            n_batch = params_array.shape[0]
        else:
            is_batched = False
            n_batch = 1

        n_eval = len(self.roa_eval)
        n_feat = len(self.all_features)
        
        # Allocate output: (n_batch, n_roa_eval, n_features) or (n_roa_eval, n_features)
        if is_batched:
            X_samples = np.zeros((n_batch, n_eval, n_feat), dtype=float)
        else:
            X_samples = np.zeros((n_eval, n_feat), dtype=float)

        if state is None:
            return X_samples

        # ------------------------------------------------------------------
        # 5. Populate feature values
        # ------------------------------------------------------------------
        # Extract state features at all roa points (same for all batch samples)

        if is_batched:
            # use individual states for each batch sample
            state_feature_matrix = np.array([
                [np.interp(roa, state[b].roa, getattr(state[b], name)) for name in self.state_features]
                for b in range(n_batch)
                for roa in self.roa_eval
            ], dtype=float).reshape(n_batch, n_eval, len(self.state_features))  # (n_batch, n_roa_eval, n_state_features)
        else:
            state_feature_matrix = np.array([
                [np.interp(roa, state.roa, getattr(state, name)) for name in self.state_features]
                for roa in self.roa_eval
            ], dtype=float)  # (n_roa_eval, n_state_features)

        if X_params is None:
            # No params provided, return state features only
            if is_batched:
                X_samples[:, :, :len(self.state_features)] = state_feature_matrix[np.newaxis, :, :]
            else:
                X_samples[:, :len(self.state_features)] = state_feature_matrix
            return X_samples

        # Process params based on input type
        if input_type == 'dict':
            # Single dict input - extract param values
            param_vector = np.array([X_params[pfeat.split('_', 1)[0]][pfeat.split('_', 1)[1]] 
                                     for pfeat in self.param_features], dtype=float)
            # param_vector is (n_params, n_roa) or scalar per param
            
            for i, roa in enumerate(self.roa_eval):
                loc_sample_state = state_feature_matrix[i]
                # Extract param values at this roa if spatially varying
                if param_vector.ndim == 2:
                    param_vals = param_vector[:, i]
                else:
                    param_vals = param_vector
                
                if self.mode == "local":
                    row_vec = np.concatenate([loc_sample_state, param_vals])
                else:  # global
                    row_vec = np.concatenate([loc_sample_state, param_vals])
                X_samples[i] = row_vec
        
        else:
            # Array input (flat or batched) - params_array is (n_batch, n_params) where params includes all roa
            for b in range(n_batch):
                param_vector = np.array([params_array[b, pfeat.split('_', 1)[0]][pfeat.split('_', 1)[1]] 
                                     for pfeat in self.param_features], dtype=float)
                for i, roa in enumerate(self.roa_eval):
                    loc_sample_state = state_feature_matrix[b,i]
                    
                    if self.mode == "local":
                        row_vec = np.concatenate([loc_sample_state, param_vector])
                    else:  # global
                        row_vec = np.concatenate([loc_sample_state, param_vector])
                    
                    if is_batched:
                        X_samples[b, i] = row_vec
                    else:
                        X_samples[i] = row_vec

        return X_samples


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

        # Limit training set size - remove oldest samples
        if len(self.X_train) > self.max_train_samples:
            # Remove oldest 10% to avoid thrashing
            n_remove = max(1, len(self.X_train) // 10)
            self.X_train = self.X_train[n_remove:]
            self.Y_train = self.Y_train[n_remove:]
            self.X_train.pop(0)
            self.Y_train.pop(0)
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

        for s_idx, sample_dict in enumerate(self.X_train):
            for f_idx, feature in enumerate(self.all_features):
                X_all_samples[s_idx, :, f_idx] = sample_dict[feature]

        for s_idx, sample_dict in enumerate(self.Y_train):
            for o_idx, output in enumerate(self.output_list):
                Y_all_samples[s_idx, :, o_idx] = sample_dict[output]
        
        # Apply input feature scaling based on configuration
        if self.scaling == 'physics' and hasattr(self, 'state_features_scalers'):
            # Physics-based scaling using predefined scalers
            X_scaled = X_all_samples.copy()
            for f_idx, feature in enumerate(self.all_features):
                if feature in self.state_features_scalers:
                    scaler_fn = self.state_features_scalers[feature]
                    X_scaled[:, :, f_idx] = scaler_fn(X_all_samples[:, :, f_idx])
            # Store physics scaler for later use
            self.x_scaler = None
            self.physics_scaled = True
            X_all_samples = X_scaled
        else:
            # Standard statistical scaling
            self.x_scaler = StandardScaler()
            X_all_samples = self.x_scaler.fit_transform(X_all_samples.reshape(-1, n_features)).reshape(n_samples, n_roa, n_features)
            self.physics_scaled = False
        
        self.y_scalers = {key: StandardScaler() for key in self.output_list}
        
        # Compute training bounds for extrapolation detection
        X_flat = X_all_samples.reshape(-1, n_features)
        self._X_train_bounds = np.column_stack([X_flat.min(axis=0), X_flat.max(axis=0)])

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
                # Local GP fit to per-knot data: risky with few samples
                for i, model in enumerate(self.models[key]):
                    # X for i-th roa from all samples: (n_samples, n_features)
                    i_features = X_all_samples[:, i, :]

                    # Y for i-th roa and j-th output from all samples: (n_samples,)
                    i_outputs = self.y_scalers[key].fit_transform(Y_all_samples[:, i, j].reshape(-1,1))
                    
                    # Warn if too few samples for reliable GP fit
                    if n_samples < 3:
                        print(f"Warning: Local GP for '{key}' at knot {i} has only {n_samples} sample(s); fit may be unreliable.")
                    
                    try:
                        model.fit(i_features, i_outputs.ravel())
                        score = model.score(i_features, i_outputs.ravel())
                        scores[key] = max(scores[key], score)  # Use best score among knots
                    except Exception as e:
                        print(f"Error fitting local GP for '{key}' at knot {i}: {e}")
                        scores[key] = 0.0

        self.trained = True
        self.score = scores  # R^2
        
        # Warn if any surrogate has poor fit
        poor_fits = {k: v for k, v in scores.items() if v < self.min_score_threshold}
        if poor_fits:
            print(f"Warning: Low surrogate fit quality (R² < {self.min_score_threshold}): {poor_fits}")
        
        self.hyperparameters = {key: model.get_hyperparameters() if self.mode == 'global' else
                        [m.get_hyperparameters() for m in model]
                        for key, model in self.models.items()}

    def is_extrapolating(self, X_features: np.ndarray, margin: float = 0.1) -> bool:
        """Check if evaluation features are outside training bounds.
        
        Parameters
        ----------
        X_features : np.ndarray
            Transformed features, shape (n_roa, n_features) or (n_batch, n_roa, n_features)
        margin : float
            Safety margin as fraction of training range (0.1 = 10% margin)
        
        Returns
        -------
        bool
            True if any feature exceeds training bounds by margin
        """
        if self._X_train_bounds is None:
            return True  # No training data, assume extrapolation
        
        X_flat = X_features.reshape(-1, X_features.shape[-1])
        bounds = self._X_train_bounds
        ranges = bounds[:, 1] - bounds[:, 0]
        margin_vals = ranges * margin
        
        # Check if any feature is outside [min - margin, max + margin]
        is_low = np.any(X_flat < (bounds[:, 0] - margin_vals), axis=1)
        is_high = np.any(X_flat > (bounds[:, 1] + margin_vals), axis=1)
        
        return bool(np.any(is_low | is_high))
    
    def get_min_score(self) -> float:
        """Return minimum R^2 score across all surrogate models."""
        if not self.score:
            return 0.0
        return min(self.score.values())

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------
    def evaluate(self, params, state: Any, train: bool = False):
        """Predict surrogate outputs for current state.
        
        Parameters
        ----------
        params : dict or np.ndarray
            Design parameters (single or batched)
        state : PlasmaState
            Current plasma state
        train : bool
            Whether to train before evaluating
        
        Returns
        -------
        tuple
            (values, stds) where format matches input:
            - Single input: arrays of shape (n_outputs, n_roa)
            - Batched input: arrays of shape (n_batch, n_outputs, n_roa)
        """
        if not self.trained and train:
            self.fit()

        # Detect if input is batched
        is_batched = False
        if isinstance(params, np.ndarray) and params.ndim == 2:
            is_batched = True
            n_batch = params.shape[0]
        elif isinstance(params, dict):
            is_batched = False
            n_batch = 1
        else:  # 2D array
            is_batched = False
            n_batch = 1

        # Get features (handles batching internally)
        X_features = self.get_features(state, params)
        
        # Transform features using appropriate scaler
        if is_batched:
            # X_features is (n_batch, n_roa, n_features)
            n_batch, n_roa, n_feat = X_features.shape
            if getattr(self, 'physics_scaled', False) and hasattr(self, 'state_features_scalers'):
                X = X_features.copy()
                for f_idx, feature in enumerate(self.all_features):
                    if feature in self.state_features_scalers:
                        scaler_fn = self.state_features_scalers[feature]
                        X[:, :, f_idx] = scaler_fn(X_features[:, :, f_idx])
            else:
                X = self.x_scaler.transform(X_features.reshape(-1, n_feat)).reshape(n_batch, n_roa, n_feat)
        else:
            # X_features is (n_roa, n_features)
            if getattr(self, 'physics_scaled', False) and hasattr(self, 'state_features_scalers'):
                X = X_features.copy()
                for f_idx, feature in enumerate(self.all_features):
                    if feature in self.state_features_scalers:
                        scaler_fn = self.state_features_scalers[feature]
                        X[:, f_idx] = scaler_fn(X_features[:, f_idx])
            else:
                X = self.x_scaler.transform(X_features)
        
        # Check for extrapolation and warn
        # if self.is_extrapolating(X, margin=0.05):
        #     min_score = self.get_min_score()
        #     print(f"Warning: Surrogate extrapolating beyond training data (min R²={min_score:.3f}). Consider full model evaluation.")

        # Initialize storage based on batching
        if is_batched:
            transport = {key: np.zeros((n_batch, len(self.roa_eval))) for key in self.transport_vars}
            transport_std = {key: np.zeros((n_batch, len(self.roa_eval))) for key in self.transport_vars}
            targets = {key: np.zeros((n_batch, len(self.roa_eval))) for key in self.target_vars}
            targets_std = {key: np.zeros((n_batch, len(self.roa_eval))) for key in self.target_vars}
        else:
            transport = {}
            transport_std = {}
            targets = {}
            targets_std = {}

        for key in self.output_list:
            if is_batched:
                # Process each batch sample
                batch_predictions = []
                batch_stds = []
                for b in range(n_batch):
                    X_b = X[b]  # (n_roa, n_features)
                    if self.mode == 'global':
                        pred, std = self.models[key].predict(X_b, return_std=True)
                    else:
                        pred = np.zeros(len(self.roa_eval))
                        std = np.zeros(len(self.roa_eval))
                        for i, model in enumerate(self.models[key]):
                            X_i = X_b[i, :].reshape(1, -1)
                            pred_i, std_i = model.predict(X_i, return_std=True)
                            pred[i] = pred_i[0]
                            std[i] = std_i[0]
                    batch_predictions.append(pred)
                    batch_stds.append(std)
                
                prediction = np.array(batch_predictions)  # (n_batch, n_roa)
                std_array = np.array(batch_stds)  # (n_batch, n_roa)
            else:
                # Single sample prediction
                if self.mode == 'global':
                    prediction, std_array = self.models[key].predict(X, return_std=True)
                else:
                    prediction = np.zeros(len(self.roa_eval))
                    std_array = np.zeros(len(self.roa_eval))
                    for i, model in enumerate(self.models[key]):
                        X_i = X[i, :].reshape(1, -1)
                        prediction_i, std_i = model.predict(X_i, return_std=True)
                        prediction[i] = prediction_i[0]
                        std_array[i] = std_i[0]
            
            # Select appropriate output dictionaries
            output_dict = transport if key in self.transport_vars else targets
            output_std_dict = transport_std if key in self.transport_vars else targets_std
            
            # Inverse transform and store
            if is_batched:
                output_dict[key] = self.y_scalers[key].inverse_transform(prediction.reshape(-1, 1)).reshape(n_batch, -1)
                output_std_dict[key] = std_array * self.y_scalers[key].scale_
            else:
                output_dict[key] = self.y_scalers[key].inverse_transform(prediction if prediction.ndim == 2 else prediction.reshape(-1, 1)).flatten()
                output_std_dict[key] = std_array * self.y_scalers[key].scale_

        # Sum components for transport variables
        for base_var in self.target_vars:
            components = [v for v in self.transport_vars if v.startswith(base_var)]
            if len(components) > 1 and all(c in transport for c in components):
                summed_value = sum(transport[c] for c in components)
                # For independent variables, variances add
                summed_std = np.sqrt(sum(transport_std[c]**2 for c in components))
                transport[base_var] = summed_value
                transport_std[base_var] = summed_std

        self.transport = transport
        self.transport_std = transport_std
        self.targets = targets
        self.targets_std = targets_std
        # covariance matrices are ~diagonal

        # Return in format matching input
        if is_batched:
            # Stack as (n_batch, n_outputs, n_roa)
            values = np.stack([transport[k] for k in self.transport_vars+self.target_vars] + 
                             [targets[k] for k in self.target_vars], axis=1)
            stds = np.stack([transport_std[k] for k in self.transport_vars+self.target_vars] + 
                           [targets_std[k] for k in self.target_vars], axis=1)
        else:
            # Stack as (n_outputs, n_roa)
            values = np.array([transport[k] for k in self.transport_vars+self.target_vars] + 
                             [targets[k] for k in self.target_vars])
            stds = np.array([transport_std[k] for k in self.transport_vars+self.target_vars] + 
                           [targets_std[k] for k in self.target_vars])
        
        return values, stds

    def get_grads(self, params, state: Any) -> Dict[str, np.ndarray]:
        """Compute gradients of surrogate outputs with respect to inputs at current state.
        
        Parameters
        ----------
        params : dict or np.ndarray
            Design parameters (single or batched)
        state : PlasmaState or list of PlasmaState
            Current plasma state
        
        Returns
        -------
        dict
            Gradients for each output variable:
            - Single input: {key: array of shape (n_roa, n_features)}
            - Batched input: {key: array of shape (n_batch, n_roa, n_features)}
        """
        # Detect if input is batched
        is_batched = False
        if isinstance(params, np.ndarray) and params.ndim == 2:
            is_batched = True
            n_batch = params.shape[0]
        
        # Get and transform features using appropriate scaler
        X_features = self.get_features(state, params)
        
        if is_batched:
            n_batch, n_roa, n_feat = X_features.shape
            if getattr(self, 'physics_scaled', False) and hasattr(self, 'state_features_scalers'):
                X = X_features.copy()
                for f_idx, feature in enumerate(self.all_features):
                    if feature in self.state_features_scalers:
                        scaler_fn = self.state_features_scalers[feature]
                        X[:, :, f_idx] = scaler_fn(X_features[:, :, f_idx])
            else:
                X = self.x_scaler.transform(X_features.reshape(-1, n_feat)).reshape(n_batch, n_roa, n_feat)
        else:
            if getattr(self, 'physics_scaled', False) and hasattr(self, 'state_features_scalers'):
                X = X_features.copy()
                for f_idx, feature in enumerate(self.all_features):
                    if feature in self.state_features_scalers:
                        scaler_fn = self.state_features_scalers[feature]
                        X[:, f_idx] = scaler_fn(X_features[:, f_idx])
            else:
                X = self.x_scaler.transform(X_features)
        
        transport_grads = {}
        target_grads = {}
        
        for key in self.output_list:
            gradients = transport_grads if key in self.transport_vars else target_grads
            if is_batched:
                # Process each batch sample using finite differences
                batch_grads = []
                for b in range(n_batch):
                    X_b = X[b]  # (n_roa, n_features)
                    if self.mode == 'global':
                        model = self.models[key]
                        if hasattr(model, 'gp_mean_grad_fd'):
                            grad = model.gp_mean_grad_fd(X_b)
                        else:
                            # Generic finite differences on model.predict
                            eps = 1e-4
                            n_roa, n_feat = X_b.shape
                            grad = np.zeros((n_roa, n_feat))
                            for j in range(n_feat):
                                dX = np.zeros_like(X_b); dX[:, j] = eps
                                mu_plus = model.predict(X_b + dX)[0] if isinstance(model.predict(X_b + dX), tuple) else model.predict(X_b + dX)
                                mu_minus = model.predict(X_b - dX)[0] if isinstance(model.predict(X_b - dX), tuple) else model.predict(X_b - dX)
                                grad[:, j] = (np.asarray(mu_plus).ravel() - np.asarray(mu_minus).ravel()) / (2 * eps)
                            # Unscale gradient to original feature space
                            grad = grad / np.asarray(self.x_scaler.scale_)
                    else:
                        # Local models per-roa; compute FD per knot
                        n_roa, n_feat = X_b.shape
                        grad = np.zeros((n_roa, n_feat))
                        for i, model in enumerate(self.models[key]):
                            X_i = X_b[i, :].reshape(1, -1)
                            if hasattr(model, 'gp_mean_grad_fd'):
                                grad_i = model.gp_mean_grad_fd(X_i)
                                grad[i, :] = grad_i[0]
                            else:
                                eps = 1e-4
                                dX = np.zeros_like(X_i)
                                for j in range(n_feat):
                                    dX[:] = 0.0
                                    dX[:, j] = eps
                                    mu_plus = model.predict(X_i + dX)[0] if isinstance(model.predict(X_i + dX), tuple) else model.predict(X_i + dX)
                                    mu_minus = model.predict(X_i - dX)[0] if isinstance(model.predict(X_i - dX), tuple) else model.predict(X_i - dX)
                                    grad[i, j] = (float(np.asarray(mu_plus).ravel()) - float(np.asarray(mu_minus).ravel())) / (2 * eps)
                                grad[i, :] = grad[i, :] / np.asarray(self.x_scaler.scale_)
                    batch_grads.append(grad)
                gradients[key] = np.array(batch_grads)  # (n_batch, n_roa, n_features)
            else:
                # Single sample using finite differences
                if self.mode == 'global':
                    model = self.models[key]
                    if hasattr(model, 'gp_mean_grad_fd'):
                        grad = model.gp_mean_grad_fd(X)
                    else:
                        eps = 1e-4
                        n_roa, n_feat = X.shape
                        grad = np.zeros((n_roa, n_feat))
                        for j in range(n_feat):
                            dX = np.zeros_like(X); dX[:, j] = eps
                            mu_plus = model.predict(X + dX)[0] if isinstance(model.predict(X + dX), tuple) else model.predict(X + dX)
                            mu_minus = model.predict(X - dX)[0] if isinstance(model.predict(X - dX), tuple) else model.predict(X - dX)
                            grad[:, j] = (np.asarray(mu_plus).ravel() - np.asarray(mu_minus).ravel()) / (2 * eps)
                        grad = grad / np.asarray(self.x_scaler.scale_)
                else:
                    n_roa, n_feat = X.shape
                    grad = np.zeros((n_roa, n_feat))
                    for i, model in enumerate(self.models[key]):
                        X_i = X[i, :].reshape(1, -1)
                        if hasattr(model, 'gp_mean_grad_fd'):
                            grad_i = model.gp_mean_grad_fd(X_i)
                            grad[i, :] = grad_i[0]
                        else:
                            eps = 1e-4
                            dX = np.zeros_like(X_i)
                            for j in range(n_feat):
                                dX[:] = 0.0
                                dX[:, j] = eps
                                mu_plus = model.predict(X_i + dX)[0] if isinstance(model.predict(X_i + dX), tuple) else model.predict(X_i + dX)
                                mu_minus = model.predict(X_i - dX)[0] if isinstance(model.predict(X_i - dX), tuple) else model.predict(X_i - dX)
                                grad[i, j] = (float(np.asarray(mu_plus).ravel()) - float(np.asarray(mu_minus).ravel())) / (2 * eps)
                            grad[i, :] = grad[i, :] / np.asarray(self.x_scaler.scale_)
                gradients[key] = grad  # (n_roa, n_features)

        # sum components for transport variables
        for base_var in self.target_vars:
            components = [v for v in self.transport_vars if v.startswith(base_var)]
            if len(components) > 1 and all(c in transport_grads for c in components):
                if is_batched:
                    summed_grad = sum(transport_grads[c] for c in components)
                else:
                    summed_grad = sum(transport_grads[c] for c in components)
                transport_grads[base_var] = summed_grad

        return transport_grads, target_grads

# -------------------------
# Base surrogate
# -------------------------

class SurrogateBase:
    """Base class for a single surrogate model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, features: Optional[List[str]] = None):
        self.config = config or {}
        self.model = None
        self.features = features or self.config.get("features", [])
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

    def __init__(self, config: Optional[Dict[str, Any]] = None, features: Optional[List[str]] = None):
        super().__init__(config, features)
        length_scale = config.get("length_scale", 1.0)
        variance = config.get("variance", 1.0)
        noise = config.get("noise", 1e-3)
        normalize_y = bool(config.get("normalize_y", False))
        n_restarts = int(config.get("n_restarts", 0))
        optimizer_maxiter = int(config.get("optimizer_maxiter", 200))
        ard = bool(config.get("ard", False))
        kernel = C(variance) * RBF(length_scale = length_scale if not ard else length_scale*np.ones_like(self.features,dtype=float))
        self._backend: str = "sklearn" if _SKLEARN_AVAILABLE else "simple"

        if _SKLEARN_AVAILABLE:
            # Allow tuning the L-BFGS-B optimizer to avoid premature convergence warnings
            def _lbfgs(obj_func, initial_theta, bounds):
                x_opt, f_opt, _info = fmin_l_bfgs_b(
                    obj_func, initial_theta, bounds=bounds, maxiter=optimizer_maxiter
                )
                return x_opt, f_opt  # sklearn expects (theta_opt, func_min)

            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=float(noise),
                normalize_y=normalize_y,
                n_restarts_optimizer=n_restarts,
                optimizer=_lbfgs,
            )
        else:
            self.model = SimpleGPSurrogate(length_scale=length_scale, variance=variance, noise=noise, normalize_y=normalize_y)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.model.fit(np.atleast_2d(X), np.atleast_2d(Y))

    def predict(self, X: np.ndarray, return_std: bool = False):
        return self.model.predict(np.atleast_2d(X), return_std=return_std)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.model.score(X,Y)

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {"C": self.model.kernel_.k1.constant_value**0.5,"l_rbf": self.model.kernel_.k2.length_scale}
    
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
        """
        Numerical gradient of GP mean prediction using finite differences.
        """
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
