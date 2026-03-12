from __future__ import annotations

from typing import Any

from .bayesian_opt import Bayesian
from .relax import Relax
from .root import RootSolver as Root
from .solver_base import SolverBase


SOLVER_MODELS = {
    "relax": Relax,
    "root": Root,
    "bayesian_opt": Bayesian,
}


def create_solver(config: Any) -> SolverBase:
    if isinstance(config, str):
        key = config.lower()
        kwargs = {}
    else:
        key = str((config or {}).get("type", "relax")).lower()
        kwargs = (config or {}).get("kwargs", {})
    cls = SOLVER_MODELS.get(key)
    if cls is None:
        raise ValueError(f"Unknown solver type '{key}'. Available: {list(SOLVER_MODELS)}")
    return cls(kwargs)
