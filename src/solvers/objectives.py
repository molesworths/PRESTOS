from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class ObjectiveFunction:
    def __init__(self, scale: bool = False):
        self.scale = scale

    def __call__(self, residual: np.ndarray) -> float:
        raise NotImplementedError

    def _scale(self, value: float, residual: np.ndarray) -> float:
        return value / residual.size if self.scale and residual.size else value

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        eps = 1e-6 * (np.maximum(1.0, np.abs(r)))
        grad = np.zeros_like(r)
        base = float(self.__call__(r))
        for i in range(r.size):
            rp = r.copy()
            rm = r.copy()
            rp[i] += eps[i]
            rm[i] -= eps[i]
            gp = float(self.__call__(rp))
            gm = float(self.__call__(rm))
            grad[i] = (gp - gm) / (2.0 * eps[i])
        varg = float(grad.T @ (C_R @ grad))
        return varg


class SumSquares(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = float(np.sum(r * r))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        grad = 2.0 * r
        if self.scale and r.size:
            grad = grad / r.size
        return float(grad.T @ (C_R @ grad))

    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        grad = 2.0 * r
        if self.scale and r.size:
            grad = grad / r.size
        return grad


class MeanSquares(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = float(np.mean(r * r))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = (2.0 * r) / float(N)
        if self.scale:
            grad = grad / N
        return float(grad.T @ (C_R @ grad))

    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = (2.0 * r) / float(N)
        if self.scale:
            grad = grad / N
        return grad


class MeanAbsolute(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = float(np.mean(np.abs(r)))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = np.sign(r) / float(N)
        if self.scale:
            grad = grad / N
        return float(grad.T @ (C_R @ grad))

    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        grad = np.sign(r) / float(N)
        if self.scale:
            grad = grad / N
        return grad


class RootMeanSquareError(ObjectiveFunction):
    def __call__(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        val = np.sqrt(float(np.mean(r * r)))
        return self._scale(val, r)

    def variance(self, residual: np.ndarray, C_R: Optional[np.ndarray]) -> float:
        if C_R is None:
            return 0.0
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        rmse = np.sqrt(float(np.mean(r * r))) + 1e-12
        grad = (r / (rmse * N))
        if self.scale:
            grad = grad / N
        return float(grad.T @ (C_R @ grad))

    def gradient(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, float)
        N = r.size if r.size else 1
        rmse = np.sqrt(float(np.mean(r * r))) + 1e-12
        grad = (r / (rmse * N))
        if self.scale:
            grad = grad / N
        return grad


OBJECTIVE_FUNCTIONS = {
    "sse": SumSquares,
    "sum_squares": SumSquares,
    "mse": MeanSquares,
    "mean_squares": MeanSquares,
    "mae": MeanAbsolute,
    "mean_absolute": MeanAbsolute,
    "rmse": RootMeanSquareError,
    "root_mean_square_error": RootMeanSquareError,
}


def create_objective_function(cfg: Any) -> ObjectiveFunction:
    if isinstance(cfg, str):
        cls = OBJECTIVE_FUNCTIONS.get(cfg.lower())
        if cls is None:
            raise ValueError(f"Unknown objective '{cfg}'")
        return cls()
    if isinstance(cfg, dict):
        key = str(cfg.get("type", "mse")).lower()
        scale = bool(cfg.get("scale", True))
        cls = OBJECTIVE_FUNCTIONS.get(key)
        if cls is None:
            raise ValueError(f"Unknown objective '{key}'")
        return cls(scale=scale)
    return MeanSquares()
