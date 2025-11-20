"""
MinT Standalone Solver

A modular, lightweight transport solver for rapid development and testing.
"""

__version__ = "0.1.0"

# Top-level workflow entrypoint
from .workflow import run_workflow

# input interfaces
from .interfaces import gacode

# State and parameterization
from .state import PlasmaState
from .parameterizations import (
    create_parameter_model,
    ParameterModel,
    SplineParameterModel,
    MTanhParameterModel,
    GaussianRBFParameterModel,
)

# Transport and targets
from .transport import (
    create_transport_model,
    TransportBase,
    TransportModel,
    FingerprintsModel,
    FixedTransport,
)
from .targets import (
    TargetModel,
    AnalyticTargetModel,
    create_target_model,
)

# Neutrals
from .neutrals import (
    NeutralModel,
    KineticNeutralModel,
    DiffusiveNeutralModel,
)

# Solvers and surrogates
from .solvers import (
    create_solver,
    SolverBase,
    RelaxSolver,
    FiniteDifferenceSolver,
    BayesianOptSolver,
    IVPSolver,
    ObjectiveFunction,
)
from .surrogates import (
    SurrogateManager,
    create_surrogate_model,
    GaussianProcessSurrogate,
    NeuralNetSurrogate,
)

from .analysis import *

# Tools namespace (plasma, calc, io, geometry)
from . import tools

__all__ = [
    # workflow
    'run_workflow',
    # interfaces
    'gacode',
    # state & params
    'PlasmaState',
    'create_parameter_model', 'ParameterModel', 'SplineParameterModel', 'MTanhParameterModel', 'GaussianRBFParameterModel',
    # transport & targets
    'create_transport_model', 'TransportBase', 'TransportModel', 'FingerprintsModel', 'FixedTransport',
    'TargetModel', 'AnalyticTargetModel', 'create_target_model',
    # neutrals
    'NeutralModel', 'KineticNeutralModel', 'DiffusiveNeutralModel',
    # solvers
    'create_solver', 'SolverBase', 'RelaxSolver', 'FiniteDifferenceSolver', 'BayesianOptSolver', 'IVPSolver', 'ObjectiveFunction',
    # surrogates
    'SurrogateManager', 'create_surrogate_model', 'GaussianProcessSurrogate', 'NeuralNetSurrogate',
    # tools package
    'tools',
]
