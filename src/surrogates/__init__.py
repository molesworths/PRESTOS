"""Surrogate model package."""

from .base import SurrogateBase
from .gaussian_process import GaussianProcess, SimpleGPSurrogate
from .manager import SurrogateManager
from .registry import SURROGATE_MODELS, create_surrogate_model

# Legacy aliases for older config/class paths
GaussianProcessSurrogate = GaussianProcess
GaussianProcessModel = GaussianProcess

__all__ = [
    "SurrogateBase",
    "GaussianProcess",
    "SimpleGPSurrogate",
    "SurrogateManager",
    "SURROGATE_MODELS",
    "create_surrogate_model",
    "GaussianProcessSurrogate",
    "GaussianProcessModel",
]
