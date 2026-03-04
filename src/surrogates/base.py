"""Base classes for surrogate models."""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np


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
