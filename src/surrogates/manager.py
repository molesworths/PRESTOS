"""Surrogate manager for transport acceleration."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except Exception:
    MinMaxScaler = None
    StandardScaler = None

from .base import SurrogateBase
from .mean_functions import MeanFunctionBase, create_mean_function
from .registry import SURROGATE_MODELS


class SurrogateManager:
    """Manages multiple surrogate models with progressive feature enrichment.

    Implements two-stage transformation pipeline:
    1. Input normalization (MinMaxScaler)
    2. Output standardization (per output)
    """

    def __init__(self, options: Dict[str, Any], **kwargs):
        self.verbose = options.get("verbose", False)
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
        self.min_score_threshold = options.get("min_score_threshold", 0.5)
        self.trained = False
        self.x_scaler = None  # MinMaxScaler for input normalisation to [0, 1]
        self.y_scalers = {}  # Output scalers per output variable (StandardScaler)
        self.score = {}  # R^2 scores per output
        # Mean function options
        self.use_mean_function: bool = bool(options.get("use_mean_function", False))
        self.mean_function_config: Dict[str, Any] = dict(options.get("mean_function_config", {}))
        # gyroBohm flux option: train on gB-normalised fluxes
        # Use train_units['gB_or_real'] == 'gB' to enable; use_gB is an alias
        self.use_gB: bool = bool(options.get("use_gB", False))  # kept for config compatibility
        self.state_features: List[str] = []
        self.param_features: List[str] = []
        self.all_features: List[str] = []
        self._full_state_features = [
            "aLne", "aLte", "aLti", "vexb_shear",   # stage 1: gradients + rotation shear
            "nustar", "tite",                          # stage 2: collisionality, Ti/Te ratio
            "betae",                                   # stage 3: electromagnetic effects
        ]
        self._full_all_features: List[str] = []
        self._active_stage: Optional[str] = None

        # Unit system in which this surrogate stores training data and makes
        # predictions internally.  Configured via ``surrogate.args.train_units``
        # in run_config.  Defaults to gB-normalised fluxes (compact,
        # physics-informed).  Surrogate predictions are converted from train_units
        # → solver_output_units before being returned to the solver.
        _train_units_cfg = options.get("train_units", options.get("output_units", {}))
        self.train_units: Dict[str, str] = {
            "flux_or_flow": str(_train_units_cfg.get("flux_or_flow", "flux")),
            "gB_or_real": str(_train_units_cfg.get("gB_or_real", "gB")),
            "total_or_conduction": str(_train_units_cfg.get("total_or_conduction", "total")),
        }
        # Set by solver during _initialize() — this is the unit system the
        # solver uses for residual computation (i.e. what transport/targets return).
        # Defaults to train_units so no conversion is applied unless the solver
        # explicitly overrides it.
        self.solver_output_units: Dict[str, str] = dict(self.train_units)

    def _initialize(
        self,
        transport_vars: List[str],
        target_vars: List[str],
        roa_eval: np.ndarray,
        state: Any,
        X_params: Dict[str, Dict[str, np.ndarray]],
        solver_output_units: Optional[Dict[str, str]] = None,
    ):
        """Delayed initialization to receive vars and unit config from solver."""
        self.transport_vars = transport_vars
        self.target_vars = target_vars
        self.roa_eval = roa_eval
        self.output_list = self.transport_vars + self.target_vars
        if solver_output_units is not None:
            self.solver_output_units = dict(solver_output_units)
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
            # Dict[str, Dict[str, np.ndarray]] -> flatten to (1, n_params)
            flat_params = [param_values for prof in params for param_values in params[prof].values()]
            params_array = np.array(flat_params)[np.newaxis, :]  # (1, n_params, n_roa)
            return False, params_array, 'dict'

        if isinstance(params, np.ndarray):
            if params.ndim == 1:
                # (n_params,) -> (1, n_params)
                return False, params[np.newaxis, :], 'flat'
            if params.ndim == 2:
                # (n_batch, n_params) -> already batched
                return True, params, 'batched'
            raise ValueError(f"Array input must be 1D or 2D, got shape {params.shape}")

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
        # Thresholds aligned with PORTALS: stage transitions at 10 and 30 evaluations
        stage1 = ["aLne", "aLte", "aLti", "vexb_shear"]    # gradients + rotation shear
        stage2 = stage1 + ["nustar", "tite"]               # + collisionality, Ti/Te
        stage3 = stage2 + ["betae"]                        # + electromagnetic effects
        if n_samples < 10:
            return stage1
        if n_samples < 30:
            return stage2
        return stage3

    def _set_active_features(self, n_samples: int) -> bool:
        next_state_features = self._stage_state_features(n_samples)
        if next_state_features == self.state_features:
            return False
        self.state_features = next_state_features
        self.all_features = self._build_all_features(self.state_features)
        if n_samples < 10:
            self._active_stage = "stage1"
        elif n_samples < 30:
            self._active_stage = "stage2"
        else:
            self._active_stage = "stage3"
        return True

    # ------------------------------------------------
    # Construction
    # ------------------------------------------------
    def build_models(self):
        """Instantiate surrogate models for all transport and target outputs."""
        for key in self.output_list:
            if self.mode == 'local':
                self.models[key] = [
                    self.create_surrogate_model(
                        self.model_config,
                        features=self.all_features,
                        output_key=key,
                    )
                    for _ in range(len(self.roa_eval))
                ]
            else:
                self.models[key] = self.create_surrogate_model(
                    self.model_config,
                    features=self.all_features,
                    output_key=key,
                )

    def create_surrogate_model(
        self,
        config: Dict[str, Any],
        features: List[str],
        output_key: Optional[str] = None,
    ) -> SurrogateBase:
        kind = config.get("type", "gaussian_process").lower()
        if kind not in SURROGATE_MODELS:
            raise ValueError(f"Unknown surrogate type '{kind}'.")

        # Per-channel mean function: target channels get a full linear (ridge) mean;
        # transport channels get the physics-constrained single-gradient mean.
        mean_function: Optional[MeanFunctionBase] = None
        if self.use_mean_function and output_key is not None:
            is_target = output_key in self.target_vars
            mean_function = create_mean_function(
                output_key, features, self.mean_function_config, is_target=is_target
            )

        # GaussianProcess accepts mean_function; other models ignore it
        cls = SURROGATE_MODELS[kind]
        try:
            return cls(config, features=features, mean_function=mean_function)
        except TypeError:
            # Fallback for surrogate classes that don't accept mean_function
            return cls(config, features=features)

    def get_features(self, state: Any = None, X_params=None,
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
                    loc_sample_state = state_feature_matrix[b, i]

                    row_vec = np.concatenate([loc_sample_state, param_vector])

                    if is_batched:
                        X_samples[b, i] = row_vec
                    else:
                        X_samples[i] = row_vec

        return X_samples

    @staticmethod
    def _extract_from_transport_dict(transport_dict: dict, key: str, norm: str) -> "np.ndarray":
        """Extract a channel array from the nested transport_dict.

                Key convention:
                    - ``'{channel}'`` (total)
                    - ``'{channel}_turb'`` / ``'{channel}_neo'``
                    - ``'{channel}_exch_turb'`` / ``'{channel}_exch_neo'``

        Channel routing::

            Ge, Gi, Qe, Qi  → fluxes[norm][channel][component]
            Pe, Pi           → flows[norm][Pe|Pi]['total'][component]
            Ce, Ci           → flows[norm][Pe|Pi]['conv'][component]
            De, Di           → flows[norm][Pe|Pi]['cond'][component]
        """
        import numpy as _np
        if key.endswith('_exch_turb'):
            base, comp = key[:-10], 'turb_exch'
        elif key.endswith('_exch_neo'):
            base, comp = key[:-9], 'neo_exch'
        elif key.endswith('_turb'):
            base, comp = key[:-5], 'turb'
        elif key.endswith('_neo'):
            base, comp = key[:-4], 'neo'
        else:
            base, comp = key, 'total'

        fluxes = transport_dict.get('fluxes', {}).get(norm, {})
        flows  = transport_dict.get('flows',  {}).get(norm, {})

        if base in ('Ge', 'Gi', 'Qe', 'Qi'):
            return _np.asarray(fluxes[base][comp])
        if base in ('Pe', 'Pi'):
            return _np.asarray(flows[base]['total'][comp])
        if base == 'Ce':
            return _np.asarray(flows['Pe']['conv'][comp])
        if base == 'De':
            return _np.asarray(flows['Pe']['cond'][comp])
        if base == 'Ci':
            return _np.asarray(flows['Pi']['conv'][comp])
        if base == 'Di':
            return _np.asarray(flows['Pi']['cond'][comp])
        raise KeyError(f"Unknown transport channel '{key}' in transport_dict (norm='{norm}')")

    def get_outputs(
        self,
        transport: dict,
        targets: dict,
        transport_gB: Optional[dict] = None,
        targets_gB: Optional[dict] = None,
        transport_dict: Optional[dict] = None,
    ) -> np.ndarray:
        """Construct output array for all transport and target variables.

        Prefers ``transport_dict`` (full nested structure from TransportBase) for
        extracting turb/neo/total components.  Falls back to flat ``transport_gB``
        or ``transport`` when not available.

        Parameters
        ----------
        transport : dict
            Transport fluxes keyed by variable name (physical units, totals only).
        targets : dict
            Target values keyed by variable name (physical units).
        transport_gB : dict, optional
            Flat gB-normalised transport dict (legacy fallback).
        targets_gB : dict, optional
            Flat gB-normalised target dict.
        transport_dict : dict, optional
            Full nested transport dict ``{'fluxes': ..., 'flows': ...}`` with
            ``'gB'`` and ``'real'`` sub-dicts, each containing turb/neo/total.
            Takes precedence over ``transport_gB`` when provided.
        """
        use_gB_training = self.train_units.get('gB_or_real', 'real') == 'gB' or self.use_gB
        norm = 'gB' if use_gB_training else 'real'
        eff_targets = targets_gB if (use_gB_training and targets_gB is not None) else targets

        Y_sample = np.empty((len(self.roa_eval), len(self.output_list)))
        for i, roa in enumerate(self.roa_eval):
            if transport_dict is not None:
                transport_sample = np.array(
                    [float(self._extract_from_transport_dict(transport_dict, name, norm)[i])
                     for name in self.transport_vars],
                    dtype=float,
                )
            elif use_gB_training and transport_gB is not None:
                transport_sample = np.array([transport_gB[name][i] for name in self.transport_vars], dtype=float)
            else:
                transport_sample = np.array([transport[name][i] for name in self.transport_vars], dtype=float)
            targets_sample = np.array([eff_targets[name][i] for name in self.target_vars], dtype=float)
            Y_sample[i] = np.concatenate([transport_sample, targets_sample])

        return Y_sample  # row for each roa_eval

    def add_sample(
        self,
        state,
        X_params,
        transport: dict,
        targets: dict,
        transport_gB: Optional[dict] = None,
        targets_gB: Optional[dict] = None,
        transport_dict: Optional[dict] = None,
    ):
        """Extract new training data from current iteration and append.

        Always stores ALL stage-3 features regardless of current training stage;
        fit() uses only the stage-appropriate subset.

        Parameters
        ----------
        state : PlasmaState
        X_params : dict
        transport : dict
            Transport fluxes (physical units, totals).
        targets : dict
            Target values (physical units).
        transport_gB : dict, optional
            Flat gB-normalised transport dict (legacy fallback).
        targets_gB : dict, optional
            Flat gB-normalised target dict.
        transport_dict : dict, optional
            Full nested transport dict; takes precedence over transport_gB.
        """
        self._ensure_param_features(X_params)
        self._refresh_full_features()
        # Always extract ALL features (stage 3) for storage
        X_features_array = self.get_features(
            state,
            X_params,
            state_features=self._full_state_features,
            all_features=self._full_all_features,
        )
        Y_sample_array = self.get_outputs(transport, targets, transport_gB=transport_gB, targets_gB=targets_gB, transport_dict=transport_dict)

        # Store with ALL stage-3 features (complete feature set)
        X_sample_dict = {self._full_all_features[i]: X_features_array[:, i] for i in range(len(self._full_all_features))}
        Y_sample_dict = {self.output_list[i]: Y_sample_array[:, i] for i in range(len(self.output_list))}

        self.X_train.append(X_sample_dict)
        self.Y_train.append(Y_sample_dict)
        self.trained = False

    # ------------------------------------------------
    # Training interface
    # ------------------------------------------------

    def fit(self):
        """Train all surrogates using input normalization + output standardization.

        Data is stored with all stage-3 features. Training uses only the stage-appropriate
        subset of features based on the number of samples available (progressive complexity).
        """
        if not self.X_train:
            return

        # Determine which features to use for training based on sample count
        features_changed = self._set_active_features(len(self.X_train))
        if features_changed:
            self.build_models()
            self.x_scaler = None
            self.y_scalers = {}

        n_samples = len(self.X_train)
        n_roa = len(self.roa_eval)
        # Use stage-appropriate features for training (subset of all stored features)
        n_features = len(self.all_features)
        n_outputs = len(self.output_list)
        scores = {key: 0.0 for key in self.output_list}

        # Reconstruct numpy arrays from training data (stored with ALL stage-3 features)
        # Then extract only the stage-appropriate subset for training
        X_all_samples = np.zeros((n_samples, n_roa, n_features))
        Y_all_samples = np.zeros((n_samples, n_roa, n_outputs))

        # Extract stage-appropriate features from full training data
        for s_idx, sample_dict in enumerate(self.X_train):
            for f_idx, feature in enumerate(self.all_features):
                # Training data stored with _full_all_features, extract stage-appropriate features
                if feature in sample_dict:
                    X_all_samples[s_idx, :, f_idx] = sample_dict[feature]
                else:
                    # This should not happen if add_sample always stores all features
                    # But keep as fallback for robustness
                    X_all_samples[s_idx, :, f_idx] = 0.0

        for s_idx, sample_dict in enumerate(self.Y_train):
            for o_idx, output in enumerate(self.output_list):
                Y_all_samples[s_idx, :, o_idx] = sample_dict[output]

        # Stage 1: Normalise inputs to [0, 1] per feature (calibrates lengthscale priors)
        self.x_scaler = MinMaxScaler()
        X_scaled = self.x_scaler.fit_transform(X_all_samples.reshape(-1, n_features)).reshape(n_samples, n_roa, n_features)

        # Stage 2: Standardize outputs per variable (StandardScaler)
        self.y_scalers = {key: StandardScaler() for key in self.output_list}

        # Track training bounds for extrapolation detection (available post-fit via x_scaler)
        X_flat = X_scaled.reshape(-1, n_features)

        # Train all models using stage-appropriate features
        for j, key in enumerate(self.models.keys()):
            if self.mode == 'global':
                X_fit = X_scaled.reshape(-1, n_features)
                Y_fit = self.y_scalers[key].fit_transform(Y_all_samples[:, :, j].reshape(-1, 1))
                self.models[key].fit(X_fit, Y_fit)
                scores[key] = self.models[key].score(X_fit, Y_fit)
            else:  # local
                # Fit a single scaler on all roa-point data for this channel so that
                # evaluate() can use one inverse_transform for all roa positions.
                self.y_scalers[key].fit(Y_all_samples[:, :, j].reshape(-1, 1))

                for i, model in enumerate(self.models[key]):
                    i_features = X_scaled[:, i, :]
                    i_outputs = self.y_scalers[key].transform(Y_all_samples[:, i, j].reshape(-1, 1))

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
            print(f"Warning: Low surrogate fit quality (R^2 < {self.min_score_threshold}): {poor_fits}")

        self.hyperparameters = {
            key: model.get_hyperparameters() if self.mode == 'global' else [m.get_hyperparameters() for m in model]
            for key, model in self.models.items()
        }

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

            print("\nAttempting surrogate warm-start from evaluation log...")
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
                print("  Tip: Try broader settings filter or check database contents")
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

            # Convert to training format (list of dicts) - always store ALL stage-3 features
            if not self._full_all_features:
                self._refresh_full_features()

            # Build feature index from the full state features passed in X
            # X contains features from get_for_surrogate call which returns state_features
            feature_index = {feat: i for i, feat in enumerate(self.state_features if hasattr(self, 'state_features') and self.state_features else self._full_state_features)}

            for s_idx in range(n_samples):
                X_dict = {}
                # Always store all stage-3 features
                for feat_idx, feat in enumerate(self._full_all_features):
                    if feat in feature_index and feature_index[feat] < X_expanded.shape[2]:
                        # Feature was in the retrieved data
                        X_dict[feat] = X_expanded[s_idx, :, feature_index[feat]]
                    elif feat in self.param_features:
                        # Parameter feature - use zeros as fallback
                        X_dict[feat] = np.zeros(n_roa)
                    elif feat == "roa":
                        X_dict[feat] = self.roa_eval
                    else:
                        # State feature not in retrieved data - use zeros as fallback
                        X_dict[feat] = np.zeros(n_roa)
                Y_dict = {out: Y_expanded[s_idx, :, o_idx]
                         for o_idx, out in enumerate(self.output_list)}

                self.X_train.append(X_dict)
                self.Y_train.append(Y_dict)

            # Train surrogates on warm-start data
            print(f"  Pre-training surrogates on {len(self.X_train)} samples...")
            self.fit()

            print(f"  Warm-start complete. Surrogate scores: {self.score}")
            print("  Note: Surrogates will continue to update as solver progresses.\n")

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

    def _convert_surr_to_solver_units(
        self,
        transport: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        state: Any,
    ) -> tuple:
        """Convert surrogate predictions from ``train_units`` to ``solver_output_units``.

        If both unit configs are identical, the dicts are returned unchanged.
        Conversion is performed via ``tools.plasma.convert_output_units``.

        Returns
        -------
        tuple
            (transport_converted, targets_converted)
        """
        from tools import plasma as _plasma

        if self.train_units == self.solver_output_units:
            return transport, targets

        roa_points = self.roa_eval
        from_units = self.train_units
        to_units = self.solver_output_units

        # Assemble a combined particle-flux dict for conduction conversions.
        Gamma_combining = {}
        for key in list(transport.keys()) + list(targets.keys()):
            base = _plasma._get_base_channel(key)
            if base in _plasma._PARTICLE_BASE_CHANNELS:
                spec = 'e' if 'e' in base.lower() else 'i'
                # Convert to real first if in gB
                val = np.asarray(transport.get(key, targets.get(key, np.zeros_like(roa_points))), dtype=float)
                if from_units.get('gB_or_real', 'real') == 'gB':
                    g_gb = np.interp(roa_points, state.roa, state.g_gb)
                    val = _plasma.particle_flux_gB_to_real(val, g_gb)
                Gamma_combining[f'G{spec}'] = val

        def _convert_dict(d):
            out = {}
            for ch, val in d.items():
                base = _plasma._get_base_channel(ch)
                spec = 'e' if 'e' in base.lower() else 'i'
                Gamma = Gamma_combining.get(f'G{spec}')
                out[ch] = _plasma.convert_output_units(
                    value=np.asarray(val, dtype=float),
                    channel=ch,
                    from_units=from_units,
                    to_units=to_units,
                    state=state,
                    roa_points=roa_points,
                    Gamma_1e19=Gamma,
                )
            return out

        return _convert_dict(transport), _convert_dict(targets)

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

        # Convert from surrogate training units → solver's residual units
        # (no-op when both unit configs are identical)
        if not is_batched:
            transport, targets = self._convert_surr_to_solver_units(transport, targets, state)

        return transport, transport_std, targets, targets_std

    # ------------------------------------------------
    # Differentiation
    # ------------------------------------------------

    @staticmethod
    def _model_grad_fd(
        model: Any,
        X: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Finite-difference gradient of model.predict mean w.r.t. normalised inputs X.

        Works for any model with a `predict(X)` → array or (array, std) interface.
        Returns shape (n_roa, n_features) matching X.
        """
        if hasattr(model, 'get_gradients'):
            return model.get_gradients(X)
        n, d = X.shape
        grad = np.zeros((n, d))
        for j in range(d):
            dX = np.zeros_like(X)
            dX[:, j] = eps
            p = model.predict(X + dX)
            m = model.predict(X - dX)
            grad[:, j] = (
                np.asarray(p[0] if isinstance(p, tuple) else p).ravel()
                - np.asarray(m[0] if isinstance(m, tuple) else m).ravel()
            ) / (2 * eps)
        return grad

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
                batch_grads = []
                for b in range(n_batch):
                    X_b = X[b]  # (n_roa, n_features)
                    if self.mode == 'global':
                        g = self._model_grad_fd(self.models[key], X_b)
                    else:
                        g = np.zeros_like(X_b)
                        for i, model in enumerate(self.models[key]):
                            g[i] = self._model_grad_fd(model, X_b[i:i+1])[0]
                    batch_grads.append(g)
                gradients[key] = np.array(batch_grads)  # (n_batch, n_roa, n_features)
            else:
                if self.mode == 'global':
                    gradients[key] = self._model_grad_fd(self.models[key], X)
                else:
                    g = np.zeros_like(X)
                    for i, model in enumerate(self.models[key]):
                        g[i] = self._model_grad_fd(model, X[i:i+1])[0]
                    gradients[key] = g

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
