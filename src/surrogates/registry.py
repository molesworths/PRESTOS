"""Surrogate model registry and factory."""

from __future__ import annotations

from typing import Any, Dict

from .base import SurrogateBase
from .gaussian_process import GaussianProcess


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
