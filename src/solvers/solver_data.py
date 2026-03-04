from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SolverData:
    iterations: List[int] = field(default_factory=list)
    X: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    X_std: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    R: List[Optional[np.ndarray]] = field(default_factory=list)
    R_std: List[Optional[np.ndarray]] = field(default_factory=list)
    R_dict: List[Optional[Dict[str, np.ndarray]]] = field(default_factory=list)
    Z: List[Optional[float]] = field(default_factory=list)
    Z_std: List[Optional[float]] = field(default_factory=list)
    Y: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_std: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_target: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Y_target_std: List[Dict[str, np.ndarray]] = field(default_factory=list)
    used_surrogate: List[bool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        i: int,
        X: Dict[str, Dict[str, float]],
        X_std: Dict[str, Dict[str, float]],
        R: Optional[np.ndarray],
        R_std: Optional[np.ndarray],
        R_dict: Optional[Dict[str, np.ndarray]],
        Z: Optional[float],
        Z_std: Optional[float],
        Y: Dict[str, np.ndarray],
        Y_std: Dict[str, np.ndarray],
        Y_target: Dict[str, np.ndarray],
        Y_target_std: Dict[str, np.ndarray],
        used_surr: bool,
    ) -> None:
        self.iterations.append(int(i))
        self.X.append(copy.deepcopy(X))
        self.X_std.append({k: copy.deepcopy(v) for k, v in (X_std or {}).items()})
        self.R.append(None if R is None else np.asarray(R).copy())
        self.R_std.append(None if R_std is None else np.asarray(R_std).copy())
        self.R_dict.append({k: np.asarray(v).copy() for k, v in (R_dict or {}).items()} if R_dict is not None else {})
        self.Z.append(None if Z is None else float(Z))
        self.Z_std.append(None if Z_std is None else float(Z_std))
        self.Y.append({k: np.asarray(v).copy() for k, v in (Y or {}).items()})
        self.Y_std.append({k: np.asarray(v).copy() for k, v in (Y_std or {}).items()})
        self.Y_target.append({k: np.asarray(v).copy() for k, v in (Y_target or {}).items()})
        self.Y_target_std.append({k: np.asarray(v).copy() for k, v in (Y_target_std or {}).items()})
        self.used_surrogate.append(bool(used_surr))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert solver history to readable DataFrame format."""
        rows = []
        for i, iter_num in enumerate(self.iterations):
            row = {
                "iter": iter_num,
                "Z": self.Z[i],
                "Z_std": self.Z_std[i],
                "used_surrogate": self.used_surrogate[i],
            }

            if i < len(self.X) and self.X[i]:
                for prof, params in self.X[i].items():
                    for pname, pval in params.items():
                        row[f"X_{prof}_{pname}"] = pval
                        row[f"X_std_{prof}_{pname}"] = self.X_std[i].get(prof, {}).get(pname, 0.0)

            if i < len(self.R) and self.R[i] is not None:
                for j, rval in enumerate(self.R[i]):
                    row[f"R_{j}"] = rval
                    row[f"R_std_{j}"] = self.R_std[i][j]

            if i < len(self.R_dict) and self.R_dict[i]:
                for block, arr in self.R_dict[i].items():
                    arr_flat = np.asarray(arr).flatten()
                    for j, val in enumerate(arr_flat):
                        row[f"R_{block}_{j}"] = val

            if i < len(self.Y) and self.Y[i]:
                for var, arr in self.Y[i].items():
                    arr_flat = np.asarray(arr).flatten()
                    for j, val in enumerate(arr_flat):
                        row[f"model_{var}_{j}"] = val
                        row[f"model_{var}_std_{j}"] = self.Y_std[i].get(var, np.zeros_like(arr_flat))[j]

            if i < len(self.Y_target) and self.Y_target[i]:
                for var, arr in self.Y_target[i].items():
                    arr_flat = np.asarray(arr).flatten()
                    for j, val in enumerate(arr_flat):
                        row[f"target_{var}_{j}"] = val
                        row[f"target_{var}_std_{j}"] = self.Y_target_std[i].get(var, np.zeros_like(arr_flat))[j]

            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save solver history to CSV with flattened structure."""
        self.metadata["last_saved"] = datetime.now(timezone.utc).isoformat()
        df = self.to_dataframe()
        df.to_csv(path, index=False)
