from __future__ import annotations

from typing import Any

from .bayesian_opt import BayesianOptSolver
from .ivp import IvpSolver
from .relax import RelaxSolver
from .solver_base import SolverBase


SOLVER_MODELS = {
    "relax": RelaxSolver,
    "bayesian_opt": BayesianOptSolver,
    "ivp": IvpSolver,
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
