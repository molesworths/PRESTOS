"""
Lightweight implementations of plasma physics formulas, inspired by
MITIM's PLASMAtools.py.


Should convert to PlasmaPy instead.
"""

import numpy as np
from scipy.constants import e, m_p, mu_0, k, m_e, u, epsilon_0 as eps0
from . import calc

# --- Constants ---
PI = np.pi
E_J = e  # Electron charge in Joules
U_KG = m_p  # Atomic mass unit in kg (using proton mass as reference)
MU0 = mu_0
factor_convection = 3/2
mp_over_me = m_p / m_e

def c_s(Te_keV: float, m_ref_u: float) -> float:
    """
    Calculate the ion sound speed.

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    m_ref_u : float
        Reference ion mass in atomic mass units.

    Returns
    -------
    float
        Ion sound speed in m/s.
    """
    Te_J = Te_keV * 1e3 * E_J
    mi_kg = m_ref_u * U_KG
    return np.sqrt(Te_J / mi_kg)

def rho_s(Te_keV: float, mi_u: float, B_T: float) -> float:
    """
    Calculate the ion sound gyroradius.

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    mi_u : float
        Ion mass in atomic mass units.
    B_T : float
        Toroidal magnetic field in Tesla.

    Returns
    -------
    float
        Ion sound gyroradius in meters.
    """
    cs = c_s(Te_keV, mi_u)
    mi_kg = mi_u * U_KG
    omega_ci = E_J * abs(B_T) / mi_kg
    return cs / omega_ci

def omega_c(m_u: float, Z: float, B_T: float) -> float:
    """
    Calculate the cyclotron frequency for any charged particle.

    Parameters
    ----------
    m_u : float
        Particle mass in atomic mass units.
    Z : float
        Particle charge in units of elementary charge (e.g., +1 for protons, -1 for electrons).
    B_T : float
        Magnetic field in Tesla.

    Returns
    -------
    float
        Cyclotron frequency in rad/s.
    """
    m_kg = m_u * U_KG
    return abs(Z) * E_J * B_T / m_kg

def rho_star(Te_keV: float, mi_u: float, a_m: float, B_T: float) -> float:
    """
    Calculate the normalized ion sound gyroradius (rho*).

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    mi_u : float
        Ion mass in atomic mass units.
    a_m : float
        Minor radius in meters.
    B_T : float
        Toroidal magnetic field in Tesla.

    Returns
    -------
    float
        Normalized ion sound gyroradius (dimensionless).
    """
    rho_s_m = rho_s(Te_keV, mi_u, B_T)
    return rho_s_m / a_m

def nu_star(Te_keV: float, ne_19: float, a_m: float, mi_u: float) -> float:
    """
    Calculate the normalized electron collision frequency (nu*).

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    ne_19 : float
        Electron density in 1e19 m^-3.
    a_m : float
        Minor radius in meters.
    mi_u : float
        Ion mass in atomic mass units.

    Returns
    -------
    float
        Normalized electron collision frequency (dimensionless).
    """
    nu_e = nue(Te_keV, ne_19/10)
    cs = c_s(Te_keV, mi_u)
    return nu_e / (cs / a_m)

def betae(Te_keV: float, ne_19: float, B_T: float) -> float:
    """
    Calculate electron beta.

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    ne_19 : float
        Electron density in 1e19 m^-3.
    B_T : float
        Toroidal magnetic field in Tesla.

    Returns
    -------
    float
        Electron beta.
    """
    pe = ne_19 * 1e19 * Te_keV * 1e3 * E_J
    p_mag = B_T**2 / (2 * MU0)
    return pe / p_mag

def loglam(Te_keV: float, ne_19: float) -> float:
    """
    Calculate the Coulomb logarithm for electron-ion collisions.

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    ne_20 : float
        Electron density in 1e20 m^-3.

    Returns
    -------
    float
        Coulomb logarithm.
    """
    ne_cm3 = ne_19 * 1e13
    Te_eV = Te_keV * 1e3
    return 24.0 - np.log(np.sqrt(ne_cm3) / Te_eV)

# def nue(Te_keV: float, ne_19: float) -> float:
#     """
#     Calculate the electron collision frequency.

#     Parameters
#     ----------
#     Te_keV : float
#         Electron temperature in keV.
#     ne_19 : float
#         Electron density in 1e19 m^-3.

#     Returns
#     -------
#     float
#         Electron collision frequency in s^-1.
#     """
#     loglambda = loglam(Te_keV, ne_19)
#     return (
#         2.91e-6 * (ne_19 * 1e13) * loglambda * (Te_keV * 1e3) ** (-1.5)
#     ) / 1e6  # Convert from cm3 to m3

def vthermal(T_keV: float, m_u: float, particle_type: str = 'ion') -> float:
    """
    Calculate the thermal velocity for ions or electrons.

    Parameters
    ----------
    T_keV : float
        Temperature in keV.
    m_u : float
        Mass in atomic mass units (for ions) or ignored (for electrons).
    particle_type : str, optional
        Type of particle: 'ion' or 'electron'. Default is 'ion'.

    Returns
    -------
    float
        Thermal velocity in m/s.
    """
    T_J = T_keV * 1e3 * E_J
    
    if particle_type.lower() == 'electron':
        m_kg = m_e
    elif particle_type.lower() == 'ion':
        m_kg = m_u * U_KG
    else:
        raise ValueError("particle_type must be 'ion' or 'electron'")
    
    return np.sqrt(2 * T_J / m_kg)

def drift_frequencies(Te_keV: float, mi_u: float, B_T: float) -> tuple:
    """
    Calculate the ion and electron diamagnetic drift frequencies.

    omega_De = grad(pe)...

    Parameters 
    ----------
    Te_keV : float
        Electron temperature in keV.
    mi_u : float
        Ion mass in atomic mass units.
    B_T : float
        Toroidal magnetic field in Tesla.

    Returns
    -------
    tuple
        (omega_star_i, omega_star_e) in rad/s.
        
    """
    cs = c_s(Te_keV, mi_u)
    rho_s_m = rho_s(Te_keV, mi_u, B_T)
    omega_star_i = cs / rho_s_m
    omega_star_e = (mp_over_me) * omega_star_i
    return omega_star_i, omega_star_e


def magnetic_shear(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Calculate the magnetic shear, s = (r/q) * dq/dr.

    Parameters
    ----------
    q : np.ndarray
        Safety factor profile.
    r : np.ndarray
        Radial coordinate profile.

    Returns
    -------
    np.ndarray
        Magnetic shear profile.
    """
    dqdr = np.gradient(q, r, edge_order=2)
    # Avoid division by zero at the magnetic axis
    q_safe = np.where(np.abs(q) > 1e-6, q, 1e-6)
    return (r / q_safe) * dqdr

def convective_flux(Te, Gamma_e):
    # keV and 1E20 m^-3/s (or /m^2)

    Te_J = Te * 1e3 * E_J
    Gamma = Gamma_e * 1e20

    Qe_conv = factor_convection * Te_J * Gamma * 1e-6  # MW (or /m^2)

    return Qe_conv

def calculate_pressure(Te, Ti, ne, ni):
    """
    T in keV
    n in 1E20m^-3

    p in MPa

    Te, ne: shape (N_roa,)
    Ti, ni: shape (N_roa, N_species)
    Output will have shape (N_roa,)

    Notes:
            - It assumes all thermal ions.
            - It only works if the vectors contain the entire plasma (i.e. roa[-1]=1.0), otherwise it will miss that contribution.
    """

    # Electron pressure (MPa)
    pe = (Te * 1e3 * E_J) * (ne * 1e20) * 1e-6  # shape (N_roa,)

    # Ion pressure (MPa)
    pi = np.sum((Ti * 1e3 * E_J) * (ni * 1e20) * 1e-6, axis=1)  # shape (N_roa,)

    # Total pressure
    p = pe + pi  # shape (N_roa,)

    return p, pe, pi

def p_prime(te, ne, aLte, aLne, ti, ni, aLti, aLni, a, B_unit, q, r):
    
    # pressure gradient
    dpdr = ne*te*(aLte + aLne)
    for i in range(ni.shape[-1]):
        dpdr += ni[...,i]*ti*(aLti + aLni)

    dpdr = -1/a * 1e3 * E_J * 1e20 * dpdr

    # p_prime is q*a^2/r/B^2 * dpdr
    p_prime = 1E-7 * q * a**2 / r / B_unit**2 * dpdr

    # First one would be nan because of the division by r, correct that
    p_prime[0] = 0

    return p_prime

def calculate_content(rmin, Te, Ti, ne, ni, dVdr):
    """
    T in keV
    n in 1E20m^-3
    V in m^3
    rmin in m

    Provides:
            Number of electrons (in 1E20)
            Number of ions (in 1E20)
            Electron Energy (MJ)
            Ion Energy (MJ)

    vectors have dimensions of (iteration, radius) or (iteration,ion,radius)
    Output will have dimensions of (iteration)

    Notes:
            - It assumes all thermal ions.
            - It only works if the vectors contain the entire plasma (i.e. roa[-1]=1.0), otherwise it will miss that contribution.

    """

    p, pe, pi = calculate_pressure(Te, Ti, ne, ni)

    We, Wi, Ne, Ni = [], [], [], []
    for it in range(rmin.shape[0]):
        # Number of electrons
        Ne.append(
            calc.integrateFS(ne[it, :], rmin[it, :], dVdr[it, :])[-1]
        )  # Number of particles total

        # Number of ions
        ni0 = np.zeros(Te.shape[1])
        for i in range(ni.shape[1]):
            ni0 += ni[it, i, :]
        Ni.append(
            calc.integrateFS(ni0, rmin[it, :], dVdr[it, :])[-1]
        )  # Number of particles total

        # Electron stored energy
        Wx = 3 / 2 * pe[it, :]
        We.append(calc.integrateFS(Wx, rmin[it, :], dVdr[it, :])[-1])  # MJ
        Wx = 3 / 2 * pi[it, :]
        Wi.append(calc.integrateFS(Wx, rmin[it, :], dVdr[it, :])[-1])  # MJ

    We = np.array(We)
    Wi = np.array(Wi)
    Ne = np.array(Ne)
    Ni = np.array(Ni)

    return We, Wi, Ne, Ni


def gyrobohm_units(Te_keV, ne_20, mref_u, Bunit, a):
    """
    I send Bunit because of X to XB transformations happen outside
    """

    precomputed_factor = 0.0160218  # e_J *1E17 = 0.0160218

    commonFactor = ne_20 * (rho_s(Te_keV, mref_u, Bunit) / a) ** 2

    # Particle flux
    Ggb = commonFactor * c_s(Te_keV, mref_u)  # in 1E20/s/m^2
    # Heat flux
    Qgb = Ggb * Te_keV * precomputed_factor  # in MW/m^2
    # Momentum flux
    Pgb = commonFactor * Te_keV * precomputed_factor * a * 1e6  # J/m^2
    # Exchange source
    Sgb = Qgb / a

    # Particle flux when using convective flux
    Qgb_convection = Qgb * factor_convection

    return Qgb, Ggb, Pgb, Sgb, Qgb_convection

# --------------------------------------------------------------------------------------------------------------------------------
# Collisions
# --------------------------------------------------------------------------------------------------------------------------------


def nue(Te_keV, ne_20):
    # From tgyro_profile_functions.f90

    precomputed_factor = 12216.801897044845  # (4*pi*e**4 / (2*2**0.5*me_g**0.5)) / k**1.5 / (1E3)**1.5  * 1E20*1E-6

    nu = precomputed_factor * loglam(Te_keV, ne_20) * ne_20 / Te_keV**1.5

    return nu  # in 1/s


def xnue(Te_keV, ne_20, a_m, mref_u):
    # From tgyro_tglf_map.f90

    xnu = nue(Te_keV, ne_20) / (c_s(Te_keV, mref_u) / a_m)

    return xnu

def debye(Te_keV, ne_20, mi_u, B_T):
    # From tgyro_tglf_map.f90:
    #       tglf_debye_in = 7.43e2*sqrt(te(i_r)/(ne(i_r)))/abs(rho_s(i_r))
    #       debye length/rhos   te in ev, rho_s in cm ne in 10^13/cm^3
    
    precomputed_factor = 2.349572301505106e-05  # 7.43e2 * (1000/1E14)**0.5/(1E2)

    db = precomputed_factor * (Te_keV/ne_20)**0.5 / rho_s(Te_keV, mi_u, B_T)

    return db

def energy_exchange(ne,Te,Ti,nu_exch):
    """
    Compute energy exchange source S_exch per radial point for single ion species.

    Parameters
    ----------
        ne : (N_roa,) 1e19/m^3
        Te : (N_roa,) keV
        Ti : (N_roa,) keV
        nu_exch : (N_roa,) 1/s

    Returns
        s_exch : (N_roa, 1)
    """
    s_exch = 1.5*ne*1e19*nu_exch*(Te - Ti)*1e3*e  # W/m^3 (N_roa,)

    return s_exch*1e-6  # MW/m^3 (N_roa,)


def calculate_collisionalities(
    ne, Te, Zeff, R, q, epsilon, ne_avol, Te_avol, R0, Zeff_avol, mi_u=1.0
):  #TODO: Needs fixing
    """
    Notes:
            ne in m^-3
            Te in keV
            R is the flux surface major radius
    """

    # Factor to multiply collisionalities (PRF OneNotes)
    factorIsotopeCs = mi_u**0.5

    # Ion-electron collision frequency
    wpe = plasma_frequency(ne)  # in s^-1
    L, _ = calculate_coulomb_logarithm(Te, ne*1e19)
    nu_ei = (
        Zeff
        * wpe**2
        * E_J**2
        * m_e * 1e-3
        * 1e-3**0.5
        * L
        / (4 * np.pi * eps0 * (2 * Te * 1e3 * E_J) ** 1.5)
    )

    # self.nu_ei = 1.0 / ( 1.09E16 * (self.Te**1.5) / ( self.nmain*1E20*self.Lambda_e ) )

    # Effective dimensionless collisionality (As in Angioni 2005 PoP)
    nu_eff = Zeff * (ne * 1e-20) * R * Te ** (-2) * factorIsotopeCs

    # Normalized collisionality (As in Conway 2006 NF)
    nu_star = 0.0118 * q * R * Zeff * (ne * 1e-20) / (Te**2 * epsilon**1.5)

    # Dimensionless collisionality (as in GACODE, and Angioni 2005 PoP)
    nu_norm = 0.89 * epsilon * nu_eff * factorIsotopeCs

    # Vol av from Angioni NF 2007
    nu_eff_Angioni = coll_Angioni07(ne_avol * 1e-19, Te_avol, R0, Zeff=Zeff_avol)

    return nu_ei, nu_eff, nu_star, nu_norm, nu_eff_Angioni


def coll_Angioni07(ne19, TekeV, Rgeo, Zeff=2.0):
    """
    As explained in Martin07, Angioni assumed Zeff=2.0 for the entire database, that's why it has a factor of 0.2
    """

    return 0.1 * Zeff * ne19 * Rgeo * TekeV ** (-2)


def calculate_coulomb_logarithm(Te, ne, Z=None, ni=None, Ti=None):  #TODO: FIX
    """
    Notes:
            Te in keV
            ne in m^-3
    """

    # Freidberg p. 194
    # L = 4.9e7 * Te**1.5 / (ne * 1e-20) ** 0.5
    # logL = np.log(L)

    # Lee = 30.0 - np.log( ne**0.5 * (Te*1E-3)**-1.5 )
    # Lc = 24.0 - np.log( np.sqrt( ne/Te ) )
    # Coulomb logarithm (Fitzpatrick)
    # self.LambdaCoul = 6.6 - 0.5*np.log(self.ne)+1.5*np.log(self.Te*1E3)

    # logLe = 31.3 - np.log( ne**0.5 * (Te*1E3)**-1 )
    # if Z is not None:   logLi = 30.0 - np.log( Z**3*ni**0.5 * (Ti*1E3)**-1 )
    # else:               logLi = logLe

    # # NRL Plasma formulary
    # logLe  = 23.5 - np.log( ne**0.5 * (Te*1E3)**-(5/4) ) - ( 1E-5 + ( np.log(Te*1E3)-2 )^2 /16 )**0.5
    # logLei = 24.0 - np.log( ne**0.5 * (Te*1E3)**-(1) )
    # #logLi  = 24.0 - np.log( ne**0.5 * (Te*1E3)**-(1) )

    # P.A. Schneider thesis 2012
    Ti = Te
    logLii = 17.3 - 0.5 * np.log((ne * 1e-20) * Ti ** -(3 / 2))  # Valid for Te<20keV
    logLei = 15.2 - 0.5 * np.log((ne * 1e-20) * Te ** -(1))  # Valid for Te>10eV

    return logLei, logLii


def synchrotron(Te_keV, ne20, B_ref, aspect_rat, r_min, r_coeff=0.8):
    # From TGYRO

    c1 = 22.6049  # 1/( k*1E3/(me_g*c_light**2) )**0.5
    c2 = 4.1533  # c1**2.5 * 20/pi*  e**3.5*me_g*1E16/(me_g*c_light)**3 *(4*pi)**0.5
    f = c2 * ((1.0 - r_coeff) * (1 + c1 / aspect_rat / Te_keV**0.5) / r_min) ** 0.5
    qsync = f * B_ref**2.5 * Te_keV**2.5 * ne20**0.5

    return qsync * 1e-7  # from erg


def calculate_debye_length(Te, ne):
    """
    Notes:
            ne in m^-3
            Te in keV
    """

    lD = 2.35e-5 * np.sqrt(
        Te / (ne * 1e-20)
    )  # np.sqrt(self.Eps0*self.Te_J/(self.e_J**2*self.ne*1E20))

    return lD


def plasma_frequency(ne):
    """
    Notes:
            ne in m^-3
    """

    wpe = (ne * E_J**2) / (m_e * 1e-6 * eps0) ** 0.5

    return wpe  # rad/s

def Bunit(phi_wb, rmin):

    B = np.gradient(phi_wb / (2 * PI), 0.5 * rmin**2, edge_order=2)
    return B

def construct_vtor_from_mach(Mach, Ti_keV, mi_u):
    vTi = (2 * (Ti_keV * 1e3 * E_J) / (mi_u * u)) ** 0.5  # m/s
    Vtor = Mach * vTi  # m/s

    return Vtor

def get_Zeff(ne, ni,species) -> np.ndarray:
    """
    Compute effective charge Zeff(r) = sum_i n_i Z_i^2 / n_e.

    Returns
    -------
    np.ndarray
        Zeff profile on the fine grid.
    """
    if not species or ne is None:
        return np.ones_like(ni[:,0])  # Default Zeff=1 if no species or ne provided
    Z2 = np.array([sp["Z"]**2 for sp in species])  # [n_species]
    num = np.dot(ni, Z2)
    ne_safe = np.maximum(ne, 1e-12)
    return num / ne_safe


# ---------------------------------------------------------------
# Flux / flow unit conversions
#
# Canonical unit reference (matching flux_flow_ref.txt):
#   Heat flux  (Qe, Qi)       : real [MW/m²]  | gB [dimensionless]
#   Particle flux (Ge, Gi)    : real [1e19/m²/s] | gB [dimensionless]
#   Heat flow  (Pe, Pi)       : real [MW]      | gB [dimensionless]
#   Convective flow (Ce, Ci)  : real [MW]      | gB [dimensionless]
#   Conductive flow (De, Di)  : real [MW]      | gB [dimensionless]
#
# gB normalization factors (from state.process()):
#   state.q_gb  [MW/m²]       – heat-flux gyroBohm unit
#   state.g_gb  [1e20/m²/s]   – particle-flux gyroBohm unit
#   state.surfArea [m²]       – flux-surface area profile
#
# For flows: gB_norm = q_gb * surfArea  [MW]
# For particle flows: gB_norm = g_gb * 10 * surfArea  [1e19/s]
# ---------------------------------------------------------------

# Physical conversion constant: keV * 1e19 particles/s → MW
# 1 keV = 1.602e-16 MJ; 1e19 particles * 1.602e-16 MJ/keV = 1.602e-3 MW per (keV × 1e19/s)
_keV_1e19_to_MW: float = 1e3 * E_J * 1e19 * 1e-6  # ≈ 1.602e-3


def heat_flux_real_to_gB(flux_mwpm2: np.ndarray, q_gb_mwpm2: np.ndarray) -> np.ndarray:
    """Normalize heat flux [MW/m²] to dimensionless gyroBohm units.

    Parameters
    ----------
    flux_mwpm2 : np.ndarray
        Heat flux in physical units [MW/m²].
    q_gb_mwpm2 : np.ndarray
        gyroBohm heat-flux normalization [MW/m²] (``state.q_gb``).

    Returns
    -------
    np.ndarray
        Dimensionless gyroBohm-normalized heat flux.
    """
    return flux_mwpm2 / np.maximum(q_gb_mwpm2, 1e-30)


def heat_flux_gB_to_real(flux_gB: np.ndarray, q_gb_mwpm2: np.ndarray) -> np.ndarray:
    """Convert dimensionless gyroBohm heat flux to physical units [MW/m²].

    Parameters
    ----------
    flux_gB : np.ndarray
        Dimensionless gyroBohm-normalized heat flux.
    q_gb_mwpm2 : np.ndarray
        gyroBohm heat-flux normalization [MW/m²] (``state.q_gb``).

    Returns
    -------
    np.ndarray
        Heat flux in physical units [MW/m²].
    """
    return flux_gB * q_gb_mwpm2


def particle_flux_real_to_gB(Gamma_1e19: np.ndarray, g_gb_1e20: np.ndarray) -> np.ndarray:
    """Normalize particle flux [1e19/m²/s] to dimensionless gyroBohm units.

    ``g_gb`` is stored in units of 1e20/m²/s; multiply by 10 to match the
    1e19 scale of ``Gamma``.

    Parameters
    ----------
    Gamma_1e19 : np.ndarray
        Particle flux [1e19/m²/s].
    g_gb_1e20 : np.ndarray
        gyroBohm particle-flux normalization [1e20/m²/s] (``state.g_gb``).

    Returns
    -------
    np.ndarray
        Dimensionless gyroBohm-normalized particle flux.
    """
    return Gamma_1e19 / np.maximum(g_gb_1e20 * 10.0, 1e-30)


def particle_flux_gB_to_real(Gamma_gB: np.ndarray, g_gb_1e20: np.ndarray) -> np.ndarray:
    """Convert dimensionless gyroBohm particle flux to physical units [1e19/m²/s].

    Parameters
    ----------
    Gamma_gB : np.ndarray
        Dimensionless gyroBohm-normalized particle flux.
    g_gb_1e20 : np.ndarray
        gyroBohm particle-flux normalization [1e20/m²/s] (``state.g_gb``).

    Returns
    -------
    np.ndarray
        Particle flux in physical units [1e19/m²/s].
    """
    return Gamma_gB * g_gb_1e20 * 10.0


def heat_flux_to_flow(flux_mwpm2: np.ndarray, surfArea_m2: np.ndarray) -> np.ndarray:
    """Integrate heat flux over a flux surface to obtain power flow [MW].

    Parameters
    ----------
    flux_mwpm2 : np.ndarray
        Heat flux [MW/m²].
    surfArea_m2 : np.ndarray
        Flux-surface area [m²] (``state.surfArea``).

    Returns
    -------
    np.ndarray
        Power flow [MW].
    """
    return flux_mwpm2 * surfArea_m2


def heat_flow_to_flux(flow_mw: np.ndarray, surfArea_m2: np.ndarray) -> np.ndarray:
    """Convert power flow [MW] to heat flux [MW/m²].

    Parameters
    ----------
    flow_mw : np.ndarray
        Power flow [MW].
    surfArea_m2 : np.ndarray
        Flux-surface area [m²] (``state.surfArea``).

    Returns
    -------
    np.ndarray
        Heat flux [MW/m²].
    """
    return flow_mw / np.maximum(surfArea_m2, 1e-30)


def get_convective_flow(
    T_keV: np.ndarray, Gamma_1e19: np.ndarray, surfArea_m2: np.ndarray
) -> np.ndarray:
    """Compute the convective component of the heat flow [MW].

    ``P_conv = (3/2) · T [keV] · Γ [1e19/m²/s] · A [m²]``

    Parameters
    ----------
    T_keV : np.ndarray
        Species temperature [keV].
    Gamma_1e19 : np.ndarray
        Species particle flux [1e19/m²/s].
    surfArea_m2 : np.ndarray
        Flux-surface area [m²].

    Returns
    -------
    np.ndarray
        Convective heat flow [MW].
    """
    return 1.5 * _keV_1e19_to_MW * T_keV * Gamma_1e19 * surfArea_m2


def get_conductive_flow(
    Q_total_mwpm2: np.ndarray,
    T_keV: np.ndarray,
    Gamma_1e19: np.ndarray,
    surfArea_m2: np.ndarray,
) -> np.ndarray:
    """Compute the conductive component of the heat flow [MW].

    ``P_cond = Q_total · A  −  (3/2) · T · Γ · A``

    Parameters
    ----------
    Q_total_mwpm2 : np.ndarray
        Total heat flux [MW/m²].
    T_keV : np.ndarray
        Species temperature [keV].
    Gamma_1e19 : np.ndarray
        Species particle flux [1e19/m²/s].
    surfArea_m2 : np.ndarray
        Flux-surface area [m²].

    Returns
    -------
    np.ndarray
        Conductive heat flow [MW].
    """
    P_total = heat_flux_to_flow(Q_total_mwpm2, surfArea_m2)
    P_conv = get_convective_flow(T_keV, Gamma_1e19, surfArea_m2)
    return P_total - P_conv


# Channel classification helpers
_HEAT_BASE_CHANNELS = {'Qe', 'Qi', 'Pe', 'Pi', 'Ce', 'Ci', 'De', 'Di'}
_PARTICLE_BASE_CHANNELS = {'Ge', 'Gi'}
_INHERENT_FLOW_CHANNELS = {'Pe', 'Pi', 'Ce', 'Ci', 'De', 'Di'}
_CONV_CHANNELS = {'Ce', 'Ci'}
_COND_CHANNELS = {'De', 'Di'}


def _get_base_channel(channel: str) -> str:
    """Strip trailing qualifiers (e.g. '_turb', '_neo') to get the base name."""
    for base in list(_HEAT_BASE_CHANNELS) + list(_PARTICLE_BASE_CHANNELS):
        if channel == base or channel.startswith(base + '_'):
            return base
    return channel[:2] if len(channel) >= 2 else channel


def _interp_state_quantity(state, attr: str, roa_points: np.ndarray) -> np.ndarray:
    """Interpolate a PlasmaState profile to *roa_points*."""
    return np.interp(roa_points, state.roa, getattr(state, attr))


def convert_output_units(
    value: np.ndarray,
    channel: str,
    from_units: dict,
    to_units: dict,
    state,
    roa_points: np.ndarray,
    Gamma_1e19: np.ndarray = None,
) -> np.ndarray:
    """Convert a flux/flow array between unit representations.

    Handles all combinations of:
      - ``gB_or_real``: 'gB' (dimensionless) ↔ 'real' (physical units)
      - ``flux_or_flow``: 'flux' (per m²) ↔ 'flow' (surface-integrated, MW)
      - ``total_or_conduction``: 'total' ↔ 'conduction' (heat-flow channels only;
        requires *Gamma_1e19* for the convective subtraction)

    The conversion path is always performed in three ordered steps:

    1. gB → real (or real → gB)
    2. flux → flow (or flow → flux) via ``state.surfArea``
    3. total → conduction (or conduction → total) via ``Gamma_1e19``

    Step 3 is only meaningful for heat channels when ``flux_or_flow == 'flow'``.

    Parameters
    ----------
    value : np.ndarray
        Input values at *roa_points*.
    channel : str
        Physical channel identifier, e.g. ``'Qe'``, ``'Qi'``, ``'Ge'``, ``'Gi'``,
        ``'Pe'``, ``'Pi'``.  Trailing qualifiers such as ``'_turb'`` are stripped.
    from_units : dict
        Source unit specification with keys:
        ``flux_or_flow`` ('flux'|'flow'), ``gB_or_real`` ('gB'|'real'),
        ``total_or_conduction`` ('total'|'conduction').
    to_units : dict
        Target unit specification (same keys as *from_units*).
    state : PlasmaState
        Provides ``q_gb``, ``g_gb``, ``surfArea``, ``te``, ``ti`` for interpolation.
    roa_points : np.ndarray
        Radial evaluation points (r/a) at which *value* is given.
    Gamma_1e19 : np.ndarray, optional
        Particle flux [1e19/m²/s] at *roa_points*, required when converting
        between ``total`` and ``conduction`` heat-flow representations.

    Returns
    -------
    np.ndarray
        Converted values at *roa_points*.

    Notes
    -----
    Particle channels (Ge, Gi) do not have a total/conduction decomposition;
    the ``total_or_conduction`` option is silently ignored for them.
    """
    value = np.asarray(value, dtype=float)
    base = _get_base_channel(channel)
    is_particle = base in _PARTICLE_BASE_CHANNELS
    is_heat = base in _HEAT_BASE_CHANNELS

    from_flow = from_units.get('flux_or_flow', 'flux') == 'flow'
    to_flow = to_units.get('flux_or_flow', 'flux') == 'flow'
    from_gB = from_units.get('gB_or_real', 'real') == 'gB'
    to_gB = to_units.get('gB_or_real', 'real') == 'gB'
    from_cond = from_units.get('total_or_conduction', 'total') == 'conduction'
    to_cond = to_units.get('total_or_conduction', 'total') == 'conduction'

    # Interpolate state quantities to roa_points once
    q_gb = _interp_state_quantity(state, 'q_gb', roa_points)       # [MW/m²]
    g_gb = _interp_state_quantity(state, 'g_gb', roa_points)       # [1e20/m²/s]
    surfArea = _interp_state_quantity(state, 'surfArea', roa_points)  # [m²]

    # --- Step 1: Convert gB ↔ real in flux space ---
    # Work entirely in real-units flux space as the intermediate representation.
    # First, if from_cond, convert to total to undo the convective subtraction.
    if from_cond and is_heat and from_flow and Gamma_1e19 is not None:
        # from_cond, from_flow → value is conductive flow [MW or gB-flow]
        # Recover total flow by adding convective: need T at roa_points
        spec = 'te' if 'e' in base.lower() else 'ti'
        T = _interp_state_quantity(state, spec, roa_points)
        if from_gB:
            P_gB_norm = q_gb * surfArea
            value_real_flow = value * P_gB_norm
        else:
            value_real_flow = value  # total [MW]
        P_conv = get_convective_flow(T, Gamma_1e19, surfArea)
        value_real_flow = value_real_flow + P_conv  # now total flow [MW]
        # Convert total flow → real flux
        v = heat_flow_to_flux(value_real_flow, surfArea)
        from_gB = False   # already in real flux
        from_flow = False
        from_cond = False
    elif from_gB:
        # Step 1a: gB → real (flux level)
        if is_particle:
            if from_flow:
                # gB particle flow → real particle flow [1e19/s]; defer to step 2
                G_gB_norm = g_gb * 10.0 * surfArea  # [1e19/s]
                v = value * G_gB_norm
                from_gB = False
                # v is now real particle flow; from_flow still True → handled below
            else:
                v = particle_flux_gB_to_real(value, g_gb)
                from_gB = False
        else:  # heat
            if from_flow:
                P_gB_norm = q_gb * surfArea  # [MW]
                v = value * P_gB_norm
                from_gB = False
            else:
                v = heat_flux_gB_to_real(value, q_gb)
                from_gB = False
    else:
        v = value.copy()

    # --- Step 2: Convert flux ↔ flow (real units) ---
    if from_flow and not to_flow:
        # flow → flux
        if is_particle:
            v = v / np.maximum(g_gb * 10.0 * surfArea, 1e-30)  # [1e19/m²/s]
        else:
            v = heat_flow_to_flux(v, surfArea)
    elif not from_flow and to_flow:
        # flux → flow
        if is_particle:
            v = v * surfArea  # [1e19/s]
        else:
            v = heat_flux_to_flow(v, surfArea)

    # --- Step 3: total ↔ conduction (heat flows only) ---
    if is_heat and Gamma_1e19 is not None:
        spec = 'te' if 'e' in base.lower() else 'ti'
        T = _interp_state_quantity(state, spec, roa_points)
        if to_flow:  # operate in flow space
            P_conv = get_convective_flow(T, Gamma_1e19, surfArea)
            if not from_cond and to_cond:
                # total → conduction
                v = v - P_conv
            elif from_cond and not to_cond:
                # conduction → total (already handled above; this path shouldn't fire)
                v = v + P_conv
        else:  # flux space: convert to conduction flux = Q - 3/2*T*Gamma/A... not standard
            pass  # conduction in flux space is unconventional; skip

    # --- Step 4: Convert real → gB if requested ---
    if to_gB:
        if to_flow:
            if is_particle:
                G_gB_norm = g_gb * 10.0 * surfArea
                v = v / np.maximum(G_gB_norm, 1e-30)
            else:
                P_gB_norm = q_gb * surfArea
                v = v / np.maximum(P_gB_norm, 1e-30)
        else:
            if is_particle:
                v = particle_flux_real_to_gB(v, g_gb)
            else:
                v = heat_flux_real_to_gB(v, q_gb)

    return v


def build_flux_flow_dict(
    state,
    roa_points: np.ndarray,
    Qe_turb_gB: np.ndarray,
    Qi_turb_gB: np.ndarray,
    Ge_turb_gB: np.ndarray,
    Gi_turb_gB: np.ndarray,
    Qe_neo_gB: np.ndarray = None,
    Qi_neo_gB: np.ndarray = None,
    Ge_neo_gB: np.ndarray = None,
    Gi_neo_gB: np.ndarray = None,
) -> dict:
    """Build the canonical nested flux/flow dictionary at *roa_points*.

    Mirrors the structure in ``flux_flow_ref.txt`` with nested keys
    ``['gB'|'real'] → ['Qe'|'Qi'|'Ge'|'Gi'] → ['turb'|'neo'|'total']``
    for fluxes, and
    ``['gB'|'real'] → ['Pe'|'Pi'] → ['conv'|'cond'|'total']``
    for flows.

    Heat-flux gB normalization  : ``state.q_gb``  [MW/m²]
    Particle-flux gB normalization: ``state.g_gb`` [1e20/m²/s] (×10 for 1e19)
    Surface area                : ``state.surfArea`` [m²]

    Parameters
    ----------
    state : PlasmaState
        Source of normalization factors and temperature profiles.
    roa_points : np.ndarray
        Radial evaluation grid (r/a).
    Qe_turb_gB, Qi_turb_gB : np.ndarray
        Turbulent electron/ion heat fluxes [dimensionless gB].
    Ge_turb_gB, Gi_turb_gB : np.ndarray
        Turbulent electron/ion particle fluxes [dimensionless gB].
    Qe_neo_gB, Qi_neo_gB : np.ndarray, optional
        Neoclassical heat fluxes [dimensionless gB].  Defaults to zeros.
    Ge_neo_gB, Gi_neo_gB : np.ndarray, optional
        Neoclassical particle fluxes [dimensionless gB].  Defaults to zeros.

    Returns
    -------
    dict
        Nested dict with 'fluxes' and 'flows' sub-dicts.
    """
    zeros = np.zeros_like(roa_points)
    Qe_neo_gB = zeros if Qe_neo_gB is None else Qe_neo_gB
    Qi_neo_gB = zeros if Qi_neo_gB is None else Qi_neo_gB
    Ge_neo_gB = zeros if Ge_neo_gB is None else Ge_neo_gB
    Gi_neo_gB = zeros if Gi_neo_gB is None else Gi_neo_gB

    q_gb = _interp_state_quantity(state, 'q_gb', roa_points)
    g_gb = _interp_state_quantity(state, 'g_gb', roa_points)
    surfArea = _interp_state_quantity(state, 'surfArea', roa_points)
    Te = _interp_state_quantity(state, 'te', roa_points)
    Ti = _interp_state_quantity(state, 'ti', roa_points)

    # ---- gB fluxes ----
    fluxes_gB = {
        'Ge': {'turb': Ge_turb_gB, 'neo': Ge_neo_gB, 'total': Ge_turb_gB + Ge_neo_gB},
        'Gi': {'turb': Gi_turb_gB, 'neo': Gi_neo_gB, 'total': Gi_turb_gB + Gi_neo_gB},
        'Qe': {'turb': Qe_turb_gB, 'neo': Qe_neo_gB, 'total': Qe_turb_gB + Qe_neo_gB},
        'Qi': {'turb': Qi_turb_gB, 'neo': Qi_neo_gB, 'total': Qi_turb_gB + Qi_neo_gB},
    }

    # ---- real fluxes ----
    fluxes_real = {}
    for key, gB_dict in fluxes_gB.items():
        is_part = key in _PARTICLE_BASE_CHANNELS
        fluxes_real[key] = {}
        for comp, val_gB in gB_dict.items():
            if is_part:
                fluxes_real[key][comp] = particle_flux_gB_to_real(val_gB, g_gb)
            else:
                fluxes_real[key][comp] = heat_flux_gB_to_real(val_gB, q_gb)

    # ---- real flows ----
    flows_real = {}
    for spec, (G_key, Q_key, T_spec) in {'e': ('Ge', 'Qe', Te), 'i': ('Gi', 'Qi', Ti)}.items():
        P_key = f'P{spec}'
        flows_real[P_key] = {}
        for comp in ('turb', 'neo', 'total'):
            Gamma = fluxes_real[G_key][comp]
            P_total = heat_flux_to_flow(fluxes_real[Q_key][comp], surfArea)
            P_conv = get_convective_flow(T_spec, Gamma, surfArea)
            P_cond = P_total - P_conv
            flows_real[P_key][comp] = {'conv': P_conv, 'cond': P_cond, 'total': P_total}

    # ---- gB flows ----
    P_gB_norm = q_gb * surfArea  # [MW] reference flow
    flows_gB = {}
    for P_key, real_dict in flows_real.items():
        flows_gB[P_key] = {}
        for comp, level_dict in real_dict.items():
            flows_gB[P_key][comp] = {
                level: val / np.maximum(P_gB_norm, 1e-30)
                for level, val in level_dict.items()
            }

    return {
        'fluxes': {'gB': fluxes_gB, 'real': fluxes_real},
        'flows': {'gB': flows_gB, 'real': flows_real},
    }
