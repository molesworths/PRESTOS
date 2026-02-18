"""
Lightweight implementations of gradient and integration routines,
inspired by MITIM's CALCtools.py.
"""

import numpy as np
import scipy as sp

def aLy(r,y):
    """
    Produces the normalized logarithmic gradient: a/Ly = -a * d(ln(y))/dr.
    d(ln(y))/dr = 1/y * dy/dr.
    If r is the minor radius (rmin), this returns a/Ly.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    p : np.ndarray
        Profile data.

    Returns
    -------
    np.ndarray
        Normalized logarithmic gradient of the profile.
    """
        
    return -r[-1]*produce_log_gradient(r,y)

def produce_log_gradient(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Produces the normalized logarithmic gradient: 1/Lp = -1/p * dp/dr.
    If r is the minor radius (rmin), this is a/Lp.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    p : np.ndarray
        Profile data.

    Returns
    -------
    np.ndarray
        Normalized logarithmic gradient of the profile.
    """
    # Use a safe version of p to avoid division by zero or log of zero
    p_safe = np.maximum(p, 1e-9)
    log_p = np.log(p_safe)
    
    # Use numpy's gradient function for robust derivative calculation
    # switch to fine grid
    x_fine = np.linspace(r[0], r[-1], num=10*len(r))
    log_p_fine = np.interp(x_fine, r, log_p)
    r_fine = x_fine
    grad_log_p = np.gradient(log_p_fine, r_fine, edge_order=2)
    # switch back to coarse grid
    grad_log_p = np.interp(r, r_fine, grad_log_p)
    
    return grad_log_p

def integrate_log_gradient(
    x: np.ndarray, z: np.ndarray, z0_bound: float
) -> np.ndarray:
    """
    Integrates a normalized logarithmic gradient to reconstruct a profile.
    
    z = -1/T * dT/dx, so T(x) = T_sep * exp(-integral(z dx))

    Parameters
    ----------
    x : np.ndarray
        Radial coordinate.
    z : np.ndarray
        Normalized logarithmic gradient profile.
    z0_bound : float
        Boundary condition value at the last point of x.

    Returns
    -------
    np.ndarray
        Reconstructed profile.
    """
    # Integrate z dx using cumulative trapezoidal integration
    integral_z = np.concatenate(([0], np.cumsum(0.5 * (z[:-1] + z[1:]) * np.diff(x))))
    
    # The profile is T(x) = C * exp(-integral(z dx))
    # We find C such that T(x_end) = z0_bound
    # C * exp(-integral_z_end) = z0_bound => C = z0_bound * exp(integral_z_end)
    C = z0_bound * np.exp(integral_z[-1])
    
    profile = C * np.exp(-integral_z)
    
    return profile

def produce_gradient_lin(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Produces the linear gradient: dp/dr.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    p : np.ndarray
        Profile data.

    Returns
    -------
    np.ndarray
        Linear gradient of the profile.
    """
    return np.gradient(p, r, edge_order=1)

def integrate_gradient_lin(
    x: np.ndarray, z: np.ndarray, z0_bound: float
) -> np.ndarray:
    """
    Integrates a linear gradient to reconstruct a profile.
    
    z = -dT/dx, so T(x) = T_sep - integral(z dx) from x to x_sep

    Parameters
    ----------
    x : np.ndarray
        Radial coordinate.
    z : np.ndarray
        Linear gradient profile.
    z0_bound : float
        Boundary condition value at the last point of x.

    Returns
    -------
    np.ndarray
        Reconstructed profile.
    """
    # Integrate z dx
    integral_z = np.concatenate(([0], np.cumsum(0.5 * (z[:-1] + z[1:]) * np.diff(x))))
    
    # The profile is T(x) = C - integral(z dx)
    # We find C such that T(x_end) = z0_bound
    # C - integral_z_end = z0_bound => C = z0_bound + integral_z_end
    C = z0_bound + integral_z[-1]
    
    profile = C - integral_z
    
    return profile

def flux_surface_average(
    r: np.ndarray, quantity: np.ndarray, dVdr: np.ndarray
) -> float:
    """
    Calculates the volume average of a quantity.
    <Q> = (1/V) * integral(Q dV) = (1/V) * integral(Q * (dV/dr) dr)

    Parameters
    ----------
    r : np.ndarray
        Radial coordinate.
    quantity : np.ndarray
        The quantity to be averaged, defined on the radial grid r.
    dVdr : np.ndarray
        The derivative of the volume with respect to the radial coordinate r.

    Returns
    -------
    float
        Volume-averaged value of the quantity.
    """
    integrand = quantity * dVdr
    vol_integrated_quantity = np.trapz(integrand, r)
    
    total_volume = np.trapz(dVdr, r)
    
    if total_volume == 0:
        return 0.0
        
    return vol_integrated_quantity / total_volume

def integrateFS(P, r, dVdr):
    """
    Based on the idea that dVdr = dV/dr, whatever r is

    Ptot = int_V P*dV = int_r P*V'*dr

    """

    I = sp.integrate.cumulative_trapezoid(
        P * dVdr, r, initial=0.0)

    return I

def integrated_flux(P, r, dVdr, area):
    """
    Compute cumulative flux [#/m^2/s]
    from volumetric source density [#/m^3/s].
    """
    # integrate volumetric source over enclosed volume
    I = sp.integrate.cumulative_trapezoid(P * dVdr, r, initial=0.0)
    # divide by local surface area to get flux [#/m^2/s]
    G = I / area
    return G

def calculateVolumeAverage(rmin, var, dVdr):
    W, vals = [], []
    for it in range(rmin.shape[0]):
        W.append(integrateFS(var[it, :], rmin[it, :], dVdr[it, :])[-1])
        vals.append(
            integrateFS(np.ones(rmin.shape[1]), rmin[it, :], dVdr[it, :])[-1]
        )

    W = np.array(W) / np.array(vals)

    return W

def volume_integrate(r, var, dVdr):
    """
    If var in MW/m^3, this gives as output the MW total value
    """

    return integrateFS(var, r, dVdr)

def interpolate(x, xp, yp):
    from scipy.interpolate import CubicSpline
    s = CubicSpline(xp, yp)  # ,bc_type='clamped')

    return s(x)
