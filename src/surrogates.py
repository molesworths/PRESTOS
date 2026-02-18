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
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process.kernels import (
        RBF, ConstantKernel as C, Matern
    )
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


class SurrogateManager:
    """Manages multiple surrogate models with progressive feature enrichment.
    
    Implements two-stage transformation pipeline:
    1. Input normalization (MinMaxScaler)
    2. Output standardization (per output)
    """

    def __init__(self, options: Dict[str, Any], **kwargs):
        self.transport_vars: List[str] = []
        self.target_vars: List[str] = []
        self.options = options
        self.roa_eval: Optional[np.ndarray] = None
        self.mode = options.get("mode", "global").lower()
        self.model_config = dict(options.get("kwargs", {}))
        self.output_list: List[str] = []
        self.models: Dict[str, Any] = {}
        self.X_train: List[Dict[str, np.ndarray]] = []
        self.Y_train: List[Dict[str, np.ndarray]] = []
        self.max_train_samples = options.get("max_train_samples", 10)
        self.min_score_threshold = options.get("min_score_threshold", 0.5)
        self.min_sample_distance = options.get("min_sample_distance", 0.01)
        self.trained = False
        self._X_train_bounds = None
        self.x_scaler = None  # StandardScaler for input standardization
        self.y_scalers = {}  # Output scalers per output variable
        self.score = {}  # R² scores per output
        self.state_features: List[str] = []
        self.param_features: List[str] = []
        self.all_features: List[str] = []
        self._full_state_features = [
            "aLne", "aLte", "aLti", "vexb_shear",
            "rhostar", "nustar", "tite", "betae",
            "q", "shear", "eps", "Zeff",
        ]
        self._full_all_features: List[str] = []
        self._active_stage: Optional[str] = None

    def _initialize(self, transport_vars: List[str], target_vars: List[str], roa_eval: np.ndarray, state: Any, X_params: Dict[str, Dict[str, np.ndarray]]):
        """Delayed initialization to receive vars from solver."""
        self.transport_vars = transport_vars
        self.target_vars = target_vars
        self.roa_eval = roa_eval
        self.output_list = self.transport_vars + self.target_vars
        if self.roa_eval is not None and len(self.roa_eval) > 5:
            self.mode = 'global'
        self._ensure_param_features(X_params)
        self._refresh_full_features()
        self._set_active_features(len(self.X_train))
        self.get_features(state, X_params)
        self.build_models()
        
        # Check for warm-start from evaluation log
        if self.options.get('warm_start', False):
            self._warm_start_from_log(state, X_params)

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

    def _ensure_param_features(self, X_params) -> None:
        if self.param_features:
            return
        self.param_features = []
        if X_params and isinstance(X_params, dict):
            for prof, prof_params in X_params.items():
                for pname in prof_params.keys():
                    if not pname.startswith("aL"):
                        self.param_features.append(f"{prof}_{pname}")

    def _refresh_full_features(self) -> None:
        self._full_all_features = self._build_all_features(self._full_state_features)

    def _build_all_features(self, state_features: List[str]) -> List[str]:
        all_features = list(state_features) + list(self.param_features)

        return all_features

    def _stage_state_features(self, n_samples: int) -> List[str]:
        stage1 = ["aLne", "aLte", "aLti", "vexb_shear"]
        stage2 = stage1 + ["rhostar", "nustar", "tite", "betae"]
        if n_samples < 5:
            return stage1
        if n_samples < 10:
            return stage2
        if n_samples > 20:
            return stage2 + ["q", "shear", "eps", "Zeff"]
        
        # Later: convert to physics-informed composite features, e.g., trapped fraction, resistive drive, etc.
        return stage2

    def _set_active_features(self, n_samples: int) -> bool:
        next_state_features = self._stage_state_features(n_samples)
        if next_state_features == self.state_features:
            return False
        self.state_features = next_state_features
        self.all_features = self._build_all_features(self.state_features)
        if n_samples < 5:
            self._active_stage = "stage1"
        elif n_samples < 10:
            self._active_stage = "stage2"
        elif n_samples > 20:
            self._active_stage = "stage3"
        else:
            self._active_stage = "stage2"
        return True

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

    def get_features(self, state: Any = None, X_params = None,
                     state_features: Optional[List[str]] = None,
                     all_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Construct surrogate input feature matrix based on mode (local/global) and plasma state.

        Parameters
        ----------
        state : PlasmaState or list of PlasmaState, optional
            Object containing all derived plasma quantities (aLne, betae, etc.)
        X_params : dict or np.ndarray, optional
            Design parameters in one of three formats:
            - Dict[str, Dict[str, np.ndarray]]: nested dict for single sample
            - np.ndarray shape (n_params): flattened single sample
            - np.ndarray shape (n_batch, n_params): batched samples

        Returns
        -------
        np.ndarray
            Feature matrix:
            - Single input: shape (n_roa_eval, n_features)
            - Batched input: shape (n_batch, n_roa_eval, n_features)
        """
        self._ensure_param_features(X_params)
        if not self._full_all_features:
            self._refresh_full_features()
        if state_features is None or all_features is None:
            if not self.all_features:
                self._set_active_features(len(self.X_train))
            state_features = self.state_features
            all_features = self.all_features

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
        n_feat = len(all_features)
        
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
                [np.interp(roa, state[b].roa, getattr(state[b], name)) for name in state_features]
                for b in range(n_batch)
                for roa in self.roa_eval
            ], dtype=float).reshape(n_batch, n_eval, len(state_features))  # (n_batch, n_roa_eval, n_state_features)
        else:
            state_feature_matrix = np.array([
                [np.interp(roa, state.roa, getattr(state, name)) for name in state_features]
                for roa in self.roa_eval
            ], dtype=float)  # (n_roa_eval, n_state_features)

        if X_params is None:
            # No params provided, return state features only
            if is_batched:
                X_samples[:, :, :len(state_features)] = state_feature_matrix
            else:
                X_samples[:, :len(state_features)] = state_feature_matrix
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
                
                row_vec = np.concatenate([loc_sample_state, param_vals])
                X_samples[i] = row_vec
        
        else:
            # Array input (flat or batched) - params_array is (n_batch, n_params) where params includes all roa
            for b in range(n_batch):
                param_vector = np.array([params_array[b, pfeat.split('_', 1)[0]][pfeat.split('_', 1)[1]] 
                                     for pfeat in self.param_features], dtype=float)
                for i, roa in enumerate(self.roa_eval):
                    loc_sample_state = state_feature_matrix[b,i]
                    
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
    
    def add_sample(self, state, X_params, transport, targets, is_fd_sample: bool = False):
        """Extract new training data from current iteration and append.
        
        Parameters
        ----------
        state : PlasmaState
            Current plasma state
        X_params : dict
            Design parameters
        transport : dict
            Transport coefficients
        targets : dict
            Target values
        is_fd_sample : bool
            If True, skip adding (finite-difference perturbations pollute training data)
        """
        # Skip finite-difference samples - they cluster around single points
        # and crowd out diverse trajectory samples with limited max_train_samples
        if is_fd_sample:
            return
        
        self._ensure_param_features(X_params)
        self._refresh_full_features()
        X_features_array = self.get_features(
            state,
            X_params,
            state_features=self._full_state_features,
            all_features=self._full_all_features,
        )
        Y_sample_array = self.get_outputs(transport, targets)

        # Check diversity: skip samples too close to existing training data
        if self._is_too_similar(X_features_array):
            return  # Skip near-duplicate samples to maintain diversity

        X_sample_dict = {self._full_all_features[i]: X_features_array[:, i] for i in range(len(self._full_all_features))}
        Y_sample_dict = {self.output_list[i]: Y_sample_array[:, i] for i in range(len(self.output_list))}

        self.X_train.append(X_sample_dict)
        self.Y_train.append(Y_sample_dict)
        self.trained = False

        # Limit training set size - remove oldest samples if needed
        if len(self.X_train) > self.max_train_samples:
            # Remove oldest samples to stay at max_train_samples
            n_excess = len(self.X_train) - self.max_train_samples
            self.X_train = self.X_train[n_excess:]
            self.Y_train = self.Y_train[n_excess:]
    
    def _is_too_similar(self, X_new: np.ndarray) -> bool:
        """Check if new sample is too close to existing training data.
        
        Parameters
        ----------
        X_new : np.ndarray
            New feature array, shape (n_roa, n_features)
        
        Returns
        -------
        bool
            True if sample is redundant (too similar to existing data)
        """
        if not self.X_train or self.min_sample_distance <= 0:
            return False  # No filtering if no training data or disabled
        
        # Convert to flat feature vector for distance calculation
        X_new_flat = X_new.flatten()
        
        # Compute normalized distance to all existing samples
        feature_list = self._full_all_features if self._full_all_features else self.all_features
        for X_dict in self.X_train:
            X_old_flat = np.array([X_dict.get(feat, np.zeros_like(X_dict[self.state_features[0]])) for feat in feature_list]).flatten()
            
            # Normalize by feature ranges to make distance scale-invariant
            # Use robust scaling based on current sample range
            diff = X_new_flat - X_old_flat
            scale = np.maximum(np.abs(X_new_flat), np.abs(X_old_flat), np.ones_like(X_new_flat))
            normalized_dist = np.linalg.norm(diff / scale) / np.sqrt(len(X_new_flat))
            
            if normalized_dist < self.min_sample_distance:
                return True  # Too similar to existing sample
        
        return False  # Sufficiently different

    # ------------------------------------------------
    # Training interface
    # ------------------------------------------------

    def fit(self):
        """Train all surrogates using input normalization + output standardization."""
        if not self.X_train:
            return

        features_changed = self._set_active_features(len(self.X_train))
        if features_changed:
            self.build_models()
            self.x_scaler = None
            self.y_scalers = {}

        n_samples = len(self.X_train)
        n_roa = len(self.roa_eval)
        n_features = len(self.all_features)
        n_outputs = len(self.output_list)
        scores = {key: 0.0 for key in self.output_list}

        # Reconstruct numpy arrays from lists of dicts
        X_all_samples = np.zeros((n_samples, n_roa, n_features))
        Y_all_samples = np.zeros((n_samples, n_roa, n_outputs))

        for s_idx, sample_dict in enumerate(self.X_train):
            for f_idx, feature in enumerate(self.all_features):
                X_all_samples[s_idx, :, f_idx] = sample_dict.get(feature, 0.0)

        for s_idx, sample_dict in enumerate(self.Y_train):
            for o_idx, output in enumerate(self.output_list):
                Y_all_samples[s_idx, :, o_idx] = sample_dict[output]
        
        # Stage 1: Apply input standardization (StandardScaler)
        self.x_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X_all_samples.reshape(-1, n_features)).reshape(n_samples, n_roa, n_features)
        
        # Stage 2: Standardize outputs per variable (StandardScaler)
        self.y_scalers = {key: StandardScaler() for key in self.output_list}
        
        # Track training bounds for extrapolation detection
        X_flat = X_scaled.reshape(-1, n_features)
        self._X_train_bounds = np.column_stack([X_flat.min(axis=0), X_flat.max(axis=0)])

        # Train all models
        for j, key in enumerate(self.models.keys()):
            if self.mode == 'global':
                X_fit = X_scaled.reshape(-1, n_features)
                Y_fit = self.y_scalers[key].fit_transform(Y_all_samples[:, :, j].reshape(-1, 1))
                self.models[key].fit(X_fit, Y_fit)
                scores[key] = self.models[key].score(X_fit, Y_fit)
            else:  # local
                for i, model in enumerate(self.models[key]):
                    i_features = X_scaled[:, i, :]
                    i_outputs = self.y_scalers[key].fit_transform(Y_all_samples[:, i, j].reshape(-1, 1))
                    
                    if n_samples < 3:
                        print(f"Warning: Local GP for '{key}' at knot {i} has only {n_samples} sample(s)")
                    
                    try:
                        model.fit(i_features, i_outputs.ravel())
                        score = model.score(i_features, i_outputs.ravel())
                        scores[key] = max(scores[key], score)
                    except Exception as e:
                        print(f"Error fitting local GP for '{key}' at knot {i}: {e}")
                        scores[key] = 0.0

        self.trained = True
        self.score = scores
        
        # Warn if any surrogate has poor fit
        poor_fits = {k: v for k, v in scores.items() if v < self.min_score_threshold}
        if poor_fits:
            print(f"Warning: Low surrogate fit quality (R² < {self.min_score_threshold}): {poor_fits}")
        
        self.hyperparameters = {key: model.get_hyperparameters() if self.mode == 'global' else
                        [m.get_hyperparameters() for m in model]
                        for key, model in self.models.items()}
    
    def _warm_start_from_log(self, state: Any, X_params: Dict[str, Dict[str, np.ndarray]]):
        """Load training data from transport evaluation log for warm-start.
        
        Searches the evaluation log for entries matching the configured transport
        model and settings, then uses them to pre-train surrogates before any
        expensive evaluations in the current solver run.
        
        Supports flexible settings filtering:
        - Exact match: warm_start_model_settings={'SAT_RULE': 3}
        - Multiple values: warm_start_model_settings={'SAT_RULE': [2, 3]}
        - Partial match: warm_start_model_settings={'SAT_RULE': 3} (ignores other settings)
        - Any settings: warm_start_model_settings={} or None
        
        Parameters
        ----------
        state : PlasmaState
            Current plasma state (used to determine model class/settings)
        X_params : Dict[str, Dict[str, np.ndarray]]
            Parameter dict (used for feature extraction)
        """
        try:
            from evaluation_log import TransportEvaluationLog, get_default_log_path
            
            # Get evaluation log path
            log_path = self.options.get('evaluation_log_path') or \
                      get_default_log_path(self.options)
            
            eval_log = TransportEvaluationLog(str(log_path))
            
            # Determine transport model class and settings from options
            model_class = self.options.get('warm_start_model_class')
            model_settings = self.options.get('warm_start_model_settings')
            
            if not model_class:
                print("Warning: warm_start enabled but no warm_start_model_class specified")
                return
            
            # Flexible settings: None or {} means match ANY settings
            filter_desc = "ANY settings"
            if model_settings:
                filter_parts = []
                for key, value in model_settings.items():
                    if isinstance(value, list):
                        filter_parts.append(f"{key} in {value}")
                    else:
                        filter_parts.append(f"{key}={value}")
                filter_desc = ", ".join(filter_parts) if filter_parts else "ANY settings"
            
            print(f"\nAttempting surrogate warm-start from evaluation log...")
            print(f"  Model: {model_class}")
            print(f"  Settings filter: {filter_desc}")
            
            # Query evaluation log
            X, Y, roa = eval_log.get_for_surrogate(
                model_class=model_class,
                model_settings=model_settings,
                target_roa=self.roa_eval,
                feature_names=self.state_features,
                output_names=self.output_list,
                max_entries=self.options.get('warm_start_max_entries', 5000)
            )
            
            if X.shape[0] == 0:
                print("  No matching evaluations found in log.")
                print(f"  Tip: Try broader settings filter or check database contents")
                return
            
            print(f"  Retrieved {X.shape[0]} evaluations")
            
            # Convert to format expected by fit()
            # X shape: (n_samples, n_state_features)
            # Y shape: (n_samples, n_outputs * n_roa)
            # Need to reshape to (n_samples, n_roa, n_features) and (n_samples, n_roa, n_outputs)
            
            n_samples = X.shape[0]
            n_roa = len(self.roa_eval)
            n_outputs = len(self.output_list)
            
            # Expand X to include roa dimension (state features are same across roa)
            X_expanded = np.tile(X[:, np.newaxis, :], (1, n_roa, 1))  # (n_samples, n_roa, n_state_features)
            
            # Add parameter features if needed
            if len(self.param_features) > 0:
                # For warm-start, we don't have parameter info from log
                # Use zeros or skip warm-start if parameters are critical
                print(f"  Warning: Warm-start data lacks parameter features ({len(self.param_features)} missing)")
                param_zeros = np.zeros((n_samples, n_roa, len(self.param_features)))
                X_expanded = np.concatenate([X_expanded, param_zeros], axis=2)
            
            # Reshape Y back to (n_samples, n_roa, n_outputs)
            Y_expanded = Y.reshape(n_samples, n_roa, n_outputs)
            
            # Convert to training format (list of dicts)
            if not self._full_all_features:
                self._refresh_full_features()
            feature_index = {feat: i for i, feat in enumerate(self.state_features)}
            for s_idx in range(n_samples):
                X_dict = {}
                for feat in self._full_all_features:
                    if feat in feature_index:
                        X_dict[feat] = X_expanded[s_idx, :, feature_index[feat]]
                    elif feat in self.param_features:
                        X_dict[feat] = np.zeros(n_roa)
                    elif feat == "roa":
                        X_dict[feat] = self.roa_eval
                    else:
                        X_dict[feat] = np.zeros(n_roa)
                Y_dict = {out: Y_expanded[s_idx, :, o_idx] 
                         for o_idx, out in enumerate(self.output_list)}
                
                self.X_train.append(X_dict)
                self.Y_train.append(Y_dict)
            
            # Train surrogates on warm-start data
            print(f"  Pre-training surrogates on {len(self.X_train)} samples...")
            self.fit()
            
            print(f"  Warm-start complete. Surrogate scores: {self.score}")
            print(f"  Note: Surrogates will continue to update as solver progresses.\n")
            
        except ImportError:
            print("Warning: evaluation_log module not available for warm-start")
        except Exception as e:
            print(f"Warning: Warm-start failed: {e}")
            # Clear any partial training data
            self.X_train = []
            self.Y_train = []
            self.trained = False

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------
    def evaluate(self, params, state: Any):
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

        if is_batched:
            # X_features is (n_batch, n_roa, n_features)
            n_batch, n_roa, n_feat = X_features.shape
            # Stage 1: Apply input normalization
            if self.x_scaler is None:
                raise RuntimeError("Input scaler not initialized; call fit() first")
            X = self.x_scaler.transform(X_features.reshape(-1, n_feat)).reshape(n_batch, n_roa, n_feat)
        else:
            # X_features is (n_roa, n_features)
            # Stage 1: Apply input normalization
            if self.x_scaler is None:
                raise RuntimeError("Input scaler not initialized; call fit() first")
            X = self.x_scaler.transform(X_features)
        
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

        # Return in format matching input
        if is_batched:
            # Stack as (n_batch, n_outputs, n_roa)
            transport_vals = np.array([transport[k] for k in self.transport_vars])
            transport_std_vals = np.array([transport_std[k] for k in self.transport_vars])
            targets_vals = np.array([targets[k] for k in self.target_vars])
            targets_std_vals = np.array([targets_std[k] for k in self.target_vars])
            
            values = np.concatenate([transport_vals, targets_vals], axis=0).transpose(1, 0, 2)
            stds = np.concatenate([transport_std_vals, targets_std_vals], axis=0).transpose(1, 0, 2)
        else:
            # Stack as (n_outputs, n_roa)
            transport_vals = np.array([transport[k] for k in self.transport_vars])
            transport_std_vals = np.array([transport_std[k] for k in self.transport_vars])
            targets_vals = np.array([targets[k] for k in self.target_vars])
            targets_std_vals = np.array([targets_std[k] for k in self.target_vars])
            
            values = np.concatenate([transport_vals, targets_vals], axis=0)
            stds = np.concatenate([transport_std_vals, targets_std_vals], axis=0)
        
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
        if not self.trained:
            raise RuntimeError("Surrogate not trained. Call fit() first.")
            
        X_features = self.get_features(state, params)
        
        if is_batched:
            n_batch, n_roa, n_feat = X_features.shape
            # Apply input normalization
            X = self.x_scaler.transform(X_features.reshape(-1, n_feat)).reshape(n_batch, n_roa, n_feat)
        else:
            # Apply input normalization
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
                normalize_y=normalize_y
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


# -------------------------
# Factory and registry
# -------------------------

SURROGATE_MODELS = {
    "gaussian_process": GaussianProcess,
    "gp": GaussianProcess,
}


def create_surrogate_model(config: Dict[str, Any]) -> SurrogateBase:
    """Create a surrogate model from config dictionary."""
    kind = (config.get("type", "gaussian_process") or "gaussian_process").lower()
    if kind in ("gaussian_process", "gaussian", "gp"):
        return GaussianProcess(config.get("kwargs", {}))
    raise ValueError(f"Unknown surrogate type '{kind}'. Supported: {list(SURROGATE_MODELS.keys())}")
