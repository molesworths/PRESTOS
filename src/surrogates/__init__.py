"""Surrogate model package."""

from .base import SurrogateBase
from .gaussian_process import GaussianProcess
from .manager import SurrogateManager
from .mean_functions import (
    MeanFunctionBase,
    ZeroMean,
    LinearParameterMean,
    FLUX_GRADIENT_MAP,
    create_mean_function,
)
from .registry import SURROGATE_MODELS, create_surrogate_model

# Legacy aliases for older config/class paths
GaussianProcessSurrogate = GaussianProcess
GaussianProcessModel = GaussianProcess

__all__ = [
    "SurrogateBase",
    "GaussianProcess",
    "SurrogateManager",
    "MeanFunctionBase",
    "ZeroMean",
    "LinearParameterMean",
    "FLUX_GRADIENT_MAP",
    "create_mean_function",
    "SURROGATE_MODELS",
    "create_surrogate_model",
    "GaussianProcessSurrogate",
    "GaussianProcessModel",
]
