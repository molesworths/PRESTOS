"""Physics helper function namespace.

Groups lightweight numerical and plasma utilities under a unified import.

Usage::

    from MinT import tools
    tools.aLy(...)
    tools.c_s(...)

Only a curated subset is exported via __all__ for external users; the full
module contents remain accessible through attribute access.
"""

# Gradient & integration utilities
from .calc import (
    aLy,
    produce_log_gradient,
    integrate_log_gradient,
    produce_gradient_lin,
    integrate_gradient_lin,
    flux_surface_average,
    integrateFS,
    integrated_flux,
    calculateVolumeAverage,
    volume_integrate,
    interpolate,
)

# Plasma physics utilities
from .plasma import (
    c_s,
    rho_s,
    omega_c,
    rho_star,
    nu_star,
    betae,
    vthermal,
    magnetic_shear,
    convective_flux,
    calculate_pressure,
    p_prime,
    calculate_content,
    gyrobohm_units,
    loglam,
    nue,
    xnue,
    debye,
    energy_exchange,
    calculate_collisionalities,
    coll_Angioni07,
    calculate_coulomb_logarithm,
    synchrotron,
    calculate_debye_length,
    plasma_frequency,
    Bunit,
    construct_vtor_from_mach,
    get_Zeff,
)

# I/O helpers
from .io import (
    isfloat,
    isint,
    isnum,
    islistarray,
    isAnyNan,
    clipstr,
    get_root,
)

# Geometry utilities
from .geometry import (
    calculateGeometricFactors,
    volp_surf_Miller_vectorized,
    xsec_area_RZ,
)

__all__ = [
    # calc
    'aLy', 'produce_log_gradient', 'integrate_log_gradient', 'produce_gradient_lin', 'integrate_gradient_lin',
    'flux_surface_average', 'integrateFS', 'integrated_flux', 'calculateVolumeAverage', 'volume_integrate', 'interpolate',
    # plasma core
    'c_s', 'rho_s', 'omega_c', 'rho_star', 'nu_star', 'betae', 'vthermal', 'magnetic_shear',
    'convective_flux', 'calculate_pressure', 'p_prime', 'calculate_content', 'gyrobohm_units',
    'loglam', 'nue', 'xnue', 'debye', 'energy_exchange', 'calculate_collisionalities', 'coll_Angioni07',
    'calculate_coulomb_logarithm', 'synchrotron', 'calculate_debye_length', 'plasma_frequency', 'Bunit',
    'construct_vtor_from_mach', 'get_Zeff',
    # io
    'isfloat', 'isint', 'isnum', 'islistarray', 'isAnyNan', 'clipstr', 'get_root',
    # geometry
    'calculateGeometricFactors', 'volp_surf_Miller_vectorized', 'xsec_area_RZ',
]
