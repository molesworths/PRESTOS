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
    nu_e = nue(Te_keV, ne_19)
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

def nue(Te_keV: float, ne_19: float) -> float:
    """
    Calculate the electron collision frequency.

    Parameters
    ----------
    Te_keV : float
        Electron temperature in keV.
    ne_19 : float
        Electron density in 1e19 m^-3.

    Returns
    -------
    float
        Electron collision frequency in s^-1.
    """
    loglambda = loglam(Te_keV, ne_19)
    return (
        2.91e-6 * (ne_19 * 1e13) * loglambda * (Te_keV * 1e3) ** (-1.5)
    ) / 1e6  # Convert from cm3 to m3

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
        dpdr += ni[...,i]*ti*(aLti + aLni[i][:])

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


# def nue(Te_keV, ne_20):
#     # From tgyro_profile_functions.f90

#     precomputed_factor = 12216.801897044845  # (4*pi*e**4 / (2*2**0.5*me_g**0.5)) / k**1.5 / (1E3)**1.5  * 1E20*1E-6

#     nu = precomputed_factor * loglam(Te_keV, ne_20) * ne_20 / Te_keV**1.5

#     return nu  # in 1/s


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
