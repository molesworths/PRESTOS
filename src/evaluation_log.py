"""Transport model evaluation logging system.

This module provides a centralized database for storing and retrieving transport
model evaluations. Useful for:
- Surrogate warm-start (pre-training from previous runs)
- Cross-user knowledge sharing
- Caching expensive evaluations
- Reproducibility tracking

The log uses SQLite for structured, queryable storage that can be:
- Version controlled (small database files)
- Shared via network filesystem
- Queried for matching conditions
"""

import sqlite3
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import asdict


class TransportEvaluationLog:
    """Database for transport model evaluations across runs and users.
    
    Features:
    - Automatic deduplication via content hashing
    - Flexible schema supporting arbitrary model settings
    - Fast queries for surrogate warm-start data
    - Thread-safe for concurrent access
    
    Database Schema:
    ----------------
    evaluations:
        id              INTEGER PRIMARY KEY
        timestamp       TEXT (ISO format)
        model_class     TEXT (e.g., 'transport.Tglf')
        model_settings  TEXT (JSON blob of settings dict)
        roa             REAL (normalized minor radius)
        state_features  TEXT (JSON blob of dimensionless plasma parameters)
        outputs         TEXT (JSON blob of model outputs)
        evaluation_hash TEXT (unique hash for deduplication)
        user            TEXT (username or identifier)
        machine         TEXT (hostname)
        
    Indexes on: model_class, roa, evaluation_hash for fast queries
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize evaluation log database.
        
        Parameters
        ----------
        db_path : str
            Path to SQLite database file. Can be:
            - Relative: stored in working directory
            - Absolute: stored at specified location
            - Shared network path: accessible to multiple users
            
        Examples
        --------
        Local database (version controlled):
        >>> log = TransportEvaluationLog("./logs/transport.db")
        
        Shared database (network filesystem):
        >>> log = TransportEvaluationLog("/shared/prestos/transport_eval.db")
        """
        if db_path is None:
            db_path = str(_default_log_path())
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_database()
    
    def _init_database(self):
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_class TEXT NOT NULL,
                    model_settings TEXT NOT NULL,
                    roa REAL NOT NULL,
                    state_features TEXT NOT NULL,
                    outputs TEXT NOT NULL,
                    evaluation_hash TEXT NOT NULL,
                    user TEXT,
                    machine TEXT,
                    git_commit TEXT,
                    notes TEXT
                )
            """)
            
            # Create indexes for fast queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_class 
                ON evaluations(model_class)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_roa 
                ON evaluations(roa)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hash 
                ON evaluations(evaluation_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON evaluations(timestamp)
            """)
            conn.commit()
    
    def add_evaluation(self,
                      model_class: str,
                      model_settings: Dict[str, Any],
                      roa: float,
                      state_features: Dict[str, float],
                      outputs: Dict[str, Any],
                      user: Optional[str] = None,
                      machine: Optional[str] = None,
                      git_commit: Optional[str] = None,
                      notes: Optional[str] = None,
                      skip_duplicates: bool = True) -> int:
        """Add a transport evaluation to the database.
        
        Parameters
        ----------
        model_class : str
            Fully qualified class name (e.g., 'transport.Tglf')
        model_settings : Dict[str, Any]
            Model configuration (e.g., {'SAT_RULE': 3, 'USE_BPER': True})
        roa : float
            Normalized minor radius location
        state_features : Dict[str, float]
            Dimensionless plasma parameters (e.g., from SurrogateManager.state_features)
            Typically: {'Ti_Te': ..., 'aLTi': ..., 'aLTe': ..., etc.}
        outputs : Dict[str, Any]
            Model outputs (e.g., {'Pe_turb': 1.23, 'Pi_turb': 0.45, ...})
        user : str, optional
            Username (defaults to $USER environment variable)
        machine : str, optional
            Hostname (defaults to system hostname)
        git_commit : str, optional
            Git commit hash for reproducibility
        notes : str, optional
            Free-form notes about this evaluation
        skip_duplicates : bool, default True
            If True, skip insertion if identical evaluation exists
            
        Returns
        -------
        int
            Row ID of inserted evaluation (-1 if skipped duplicate)
        """
        import os
        import socket
        
        # Reject low-quality data before hashing or insert
        if self._has_invalid_values(state_features) or self._has_invalid_values(outputs):
            return -1

        # Generate content hash for deduplication
        eval_hash = self._compute_hash(model_class, model_settings, roa, state_features)
        
        # Check for duplicates
        if skip_duplicates:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM evaluations WHERE evaluation_hash = ?",
                    (eval_hash,)
                )
                existing = cursor.fetchone()
                if existing:
                    return -1  # Skip duplicate
        
        # Prepare data
        timestamp = datetime.now(timezone.utc).isoformat()
        user = user or os.environ.get('USER', 'unknown')
        machine = machine or socket.gethostname()
        
        # Serialize JSON fields
        settings_json = json.dumps(model_settings, sort_keys=True)
        features_json = json.dumps(self._serialize_dict(state_features), sort_keys=True)
        outputs_json = json.dumps(self._serialize_dict(outputs), sort_keys=True)
        
        # Insert into database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO evaluations 
                (timestamp, model_class, model_settings, roa, state_features, 
                 outputs, evaluation_hash, user, machine, git_commit, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, model_class, settings_json, roa, features_json,
                  outputs_json, eval_hash, user, machine, git_commit, notes))
            conn.commit()
            return cursor.lastrowid
    
    def query_evaluations(self,
                         model_class: Optional[str] = None,
                         model_settings: Optional[Dict[str, Any]] = None,
                         roa_range: Optional[Tuple[float, float]] = None,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query evaluations matching specified criteria.
        
        Parameters
        ----------
        model_class : str, optional
            Filter by model class (exact match)
        model_settings : Dict[str, Any], optional
            Filter by settings. For each specified key-value pair:
            - Scalar value: must match exactly
            - List value: record must match ANY value in list
            - Unspecified keys: no filtering on those settings
        roa_range : Tuple[float, float], optional
            Filter by roa range [min, max] inclusive
        limit : int, optional
            Maximum number of results to return
            
        Returns
        -------
        List[Dict[str, Any]]
            List of evaluation records with deserialized JSON fields
            
        Examples
        --------
        # Exact match on SAT_RULE
        >>> log.query_evaluations(
        ...     model_class='transport.Tglf',
        ...     model_settings={'SAT_RULE': 3},
        ...     roa_range=(0.85, 0.95)
        ... )
        
        # Match SAT_RULE 2 OR 3
        >>> log.query_evaluations(
        ...     model_class='transport.Tglf',
        ...     model_settings={'SAT_RULE': [2, 3]}
        ... )
        
        # Any Tglf evaluation (no settings filter)
        >>> log.query_evaluations(model_class='transport.Tglf')
        """
        query = "SELECT * FROM evaluations WHERE 1=1"
        params = []
        
        # Build query
        if model_class:
            query += " AND model_class = ?"
            params.append(model_class)
        
        if roa_range:
            query += " AND roa BETWEEN ? AND ?"
            params.extend(roa_range)
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        # Execute query
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        # Deserialize and filter by settings if needed
        results = []
        for row in rows:
            record = dict(row)
            record['model_settings'] = json.loads(record['model_settings'])
            record['state_features'] = json.loads(record['state_features'])
            record['outputs'] = json.loads(record['outputs'])
            
            # Filter by model_settings if specified
            if model_settings:
                matches = True
                for key, required_value in model_settings.items():
                    record_value = record['model_settings'].get(key)
                    
                    # Support list values: match if record_value in list
                    if isinstance(required_value, list):
                        if record_value not in required_value:
                            matches = False
                            break
                    # Scalar value: exact match required
                    else:
                        if record_value != required_value:
                            matches = False
                            break
                
                if not matches:
                    continue
            
            results.append(record)
        
        return results
    
    def get_for_surrogate(self,
                         model_class: str,
                         model_settings: Optional[Dict[str, Any]] = None,
                         target_roa: np.ndarray = None,
                         feature_names: List[str] = None,
                         output_names: List[str] = None,
                         max_entries: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve and interpolate evaluations for surrogate warm-start.
        
        This method:
        1. Queries evaluations matching model_class and settings
        2. Groups by unique state_features (different radial locations)
        3. Interpolates outputs to target_roa grid
        4. Returns matrices suitable for surrogate training
        
        Supports flexible settings filtering:
        - None or {}: Match ANY settings for the model_class
        - {'SAT_RULE': 3}: Match only SAT_RULE=3 (other settings any)
        - {'SAT_RULE': [2, 3]}: Match SAT_RULE 2 OR 3
        - {'SAT_RULE': 3, 'USE_BPER': True}: Both must match
        
        Parameters
        ----------
        model_class : str
            Transport model class name
        model_settings : Optional[Dict[str, Any]], default None
            Model settings to match. None or {} matches any settings.
            Can use list values for OR matching: {'SAT_RULE': [2, 3]}
        target_roa : np.ndarray
            Target roa grid for interpolation
        feature_names : List[str]
            Ordered list of feature names to extract from state_features
        output_names : List[str]
            Ordered list of output names to extract
        max_entries : int, default 10000
            Maximum number of evaluations to retrieve
            
        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            State features matrix
        Y : np.ndarray, shape (n_samples, n_outputs * n_roa)
            Interpolated outputs matrix (flattened across roa)
        roa : np.ndarray
            Target roa grid used for interpolation
            
        Examples
        --------
        # Match any SAT_RULE setting
        >>> X, Y, roa = log.get_for_surrogate(
        ...     model_class='transport.Tglf',
        ...     model_settings={},
        ...     target_roa=np.array([0.88, 0.91, 0.94, 0.97]),
        ...     feature_names=['Ti_Te', 'aLTi', 'aLTe', 'aLne'],
        ...     output_names=['Pe_turb', 'Pi_turb']
        ... )
        
        # Match SAT_RULE 2 or 3
        >>> X, Y, roa = log.get_for_surrogate(
        ...     model_class='transport.Tglf',
        ...     model_settings={'SAT_RULE': [2, 3]},
        ...     target_roa=np.array([0.88, 0.91, 0.94, 0.97]),
        ...     feature_names=['Ti_Te', 'aLTi', 'aLTe', 'aLne'],
        ...     output_names=['Pe_turb', 'Pi_turb']
        ... )
        >>> print(f"Retrieved {X.shape[0]} samples for warm-start")
        """
        # Query all matching evaluations
        evals = self.query_evaluations(
            model_class=model_class,
            model_settings=model_settings,
            limit=max_entries
        )
        
        if not evals:
            # No data available, return empty arrays
            return np.array([]).reshape(0, len(feature_names)), \
                   np.array([]).reshape(0, len(output_names) * len(target_roa)), \
                   target_roa
        
        # Group evaluations by unique state (for radial profiles)
        from collections import defaultdict
        profiles = defaultdict(lambda: {'roa': [], 'outputs': defaultdict(list)})
        
        for ev in evals:
            # Create unique key from state features
            feature_key = tuple(ev['state_features'].get(f, 0.0) for f in feature_names)
            
            profiles[feature_key]['roa'].append(ev['roa'])
            for out_name in output_names:
                profiles[feature_key]['outputs'][out_name].append(
                    ev['outputs'].get(out_name, 0.0)
                )
        
        # Build training matrices
        X_list = []
        Y_list = []
        
        for feature_key, profile_data in profiles.items():
            # State features (same for all roa in this profile)
            X_list.append(list(feature_key))
            
            # Interpolate outputs to target_roa
            roa_data = np.array(profile_data['roa'])
            
            y_interp = []
            for out_name in output_names:
                output_data = np.array(profile_data['outputs'][out_name])
                
                # Sort by roa for interpolation
                sort_idx = np.argsort(roa_data)
                roa_sorted = roa_data[sort_idx]
                output_sorted = output_data[sort_idx]
                
                # Interpolate to target grid
                y_target = np.interp(target_roa, roa_sorted, output_sorted)
                y_interp.extend(y_target)
            
            Y_list.append(y_interp)
        
        X = np.array(X_list)
        Y = np.array(Y_list)
        
        return X, Y, target_roa
    
    def _compute_hash(self, model_class: str, model_settings: Dict[str, Any],
                     roa: float, state_features: Dict[str, float]) -> str:
        """Compute unique hash for evaluation deduplication."""
        # Create canonical representation
        canonical = {
            'model_class': model_class,
            'model_settings': model_settings,
            'roa': round(roa, 6),  # Round to avoid floating point issues
            'state_features': {k: round(v, 6) for k, v in state_features.items()}
        }
        
        # Hash JSON representation
        json_str = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _has_invalid_values(self, data: Any) -> bool:
        """Return True if data contains NaN, inf, or 0.0 values."""
        if data is None:
            return True

        if isinstance(data, dict):
            return any(self._has_invalid_values(v) for v in data.values())

        if isinstance(data, (list, tuple, set)):
            return any(self._has_invalid_values(v) for v in data)

        if isinstance(data, np.ndarray):
            if data.size == 0:
                return True
            if np.issubdtype(data.dtype, np.floating):
                if not np.isfinite(data).all():
                    return True
                return np.any(data == 0.0)
            if np.issubdtype(data.dtype, np.integer):
                return np.any(data == 0)
            if np.issubdtype(data.dtype, np.bool_):
                return False
            return any(self._has_invalid_values(v) for v in data.tolist())

        if isinstance(data, (np.floating, float)):
            if not np.isfinite(data):
                return True
            return data == 0.0

        if isinstance(data, (np.integer, int)) and not isinstance(data, bool):
            return data == 0

        return False
    
    def _serialize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize dict for JSON storage, converting numpy types."""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics including total entries, unique models, roa coverage, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
            
            # Unique model classes
            models = conn.execute(
                "SELECT DISTINCT model_class FROM evaluations"
            ).fetchall()
            
            # Roa range
            roa_stats = conn.execute(
                "SELECT MIN(roa), MAX(roa), AVG(roa) FROM evaluations"
            ).fetchone()
            
            # Entries per model
            model_counts = {}
            for (model,) in models:
                count = conn.execute(
                    "SELECT COUNT(*) FROM evaluations WHERE model_class = ?",
                    (model,)
                ).fetchone()[0]
                model_counts[model] = count
        
        return {
            'total_evaluations': total,
            'unique_models': len(models),
            'model_counts': model_counts,
            'roa_min': roa_stats[0],
            'roa_max': roa_stats[1],
            'roa_mean': roa_stats[2],
            'database_path': str(self.db_path),
            'database_size_mb': self.db_path.stat().st_size / 1024**2 if self.db_path.exists() else 0,
        }
    
    def export_to_hdf5(self, output_path: str):
        """Export database to HDF5 format for large-scale analysis.
        
        Parameters
        ----------
        output_path : str
            Path to output HDF5 file
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 export. Install with: pip install h5py")
        
        evals = self.query_evaluations()
        
        with h5py.File(output_path, 'w') as f:
            # Create groups
            f.attrs['n_evaluations'] = len(evals)
            f.attrs['export_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            for i, ev in enumerate(evals):
                grp = f.create_group(f"eval_{i:06d}")
                grp.attrs['model_class'] = ev['model_class']
                grp.attrs['roa'] = ev['roa']
                grp.attrs['timestamp'] = ev['timestamp']
                
                # Store settings, features, outputs as JSON
                grp.create_dataset('model_settings', data=json.dumps(ev['model_settings']))
                grp.create_dataset('state_features', data=json.dumps(ev['state_features']))
                grp.create_dataset('outputs', data=json.dumps(ev['outputs']))
        
        print(f"Exported {len(evals)} evaluations to {output_path}")


def get_default_log_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Get default path for transport evaluation log.
    
    Checks (in order):
    1. config['evaluation_log']['path'] if provided
    2. Environment variable PRESTOS_EVAL_LOG
    3. Shared location /shared/prestos/transport_evaluations.db if writable
    4. Repo default src/tools/transport_evaluations.db
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dict that may contain evaluation_log settings
        
    Returns
    -------
    Path
        Path to evaluation log database
    """
    import os
    
    # Check config
    if config and 'evaluation_log' in config:
        log_config = config['evaluation_log']
        if 'path' in log_config:
            return Path(log_config['path']).expanduser().resolve()
    
    # Check environment variable
    if 'PRESTOS_EVAL_LOG' in os.environ:
        return Path(os.environ['PRESTOS_EVAL_LOG']).expanduser().resolve()
    
    # Try shared location
    shared_path = Path("/shared/prestos/transport_evaluations.db")
    if shared_path.parent.exists() and os.access(shared_path.parent, os.W_OK):
        return shared_path
    
    # Fallback to repo default
    return _default_log_path()


def _default_log_path() -> Path:
    """Default evaluation log path inside the repo."""
    return (Path(__file__).resolve().parent / "tools" / "transport_evaluations.db").resolve()
