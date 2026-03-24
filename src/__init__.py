"""
PRESTOS Standalone Solver

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
    ParameterBase,
    Spline,
    Mtanh,
    Gaussian,
    LogSpline,
    LogSlopeSpline,
)

# Transport and targets
from .transport import (
    create_transport_model,
    TransportBase,
    Fingerprints,
    CH_fingerprints,
    Cgyro,
    Tglf,
    Qlgyro,
    Fixed,
    Analytic,
)
from .targets import (
    TargetModel,
    Analytic,
    create_target_model,
)

# Neutrals
from .neutrals import (
    NeutralModel,
    Kinetic,
    Diffusive,
)

# Solvers and surrogates
from .solvers import (
    create_solver,
    SolverBase,
    Bayesian,
    RootSolver,
    ObjectiveFunction,
)
from .surrogates import (
    SurrogateManager,
    SurrogateBase,
    create_surrogate_model,
    GaussianProcess,
    GaussianProcessSurrogate,
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
    'create_parameter_model', 'ParameterBase', 'Spline', 'Mtanh', 'Gaussian', 'LogSpline', 'LogSlopeSpline',
    # transport & targets
    'create_transport_model', 'TransportBase', 'Fingerprints', 'CH_fingerprints', 'Cgyro', 'Tglf', 'Qlgyro', 'Fixed', 'Analytic',
    'TargetModel', 'Analytic', 'create_target_model',
    # neutrals
    'NeutralModel', 'Kinetic', 'Diffusive',
    # solvers
    'create_solver', 'SolverBase', 'Relax', 'Bayesian', 'RootSolver', 'ObjectiveFunction',
    # surrogates
    'SurrogateManager', 'SurrogateBase', 'create_surrogate_model', 'GaussianProcess', 'GaussianProcessSurrogate',
    # tools package
    'tools',
]
