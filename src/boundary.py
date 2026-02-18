"""
Boundary condition models for LCFS (separatrix).

Implements two-point model and other boundary physics
based on MINTboundary.LCFShandler.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import least_squares, fsolve
from scipy.integrate import cumulative_trapezoid
from scipy.constants import e, m_p, mu_0, k
from tools import plasma, calc

class BoundaryConditions:

    def __init__(self, options, **kwargs):
        self.sigma = options.get('sigma_bc', 0.0)


    def get_boundary_conditions(self,state,targets):
        # Initial guess from LCFS values
        self.ne = state.ne[-1]*1e19  # 1e19/m^3
        self.ni = state.ni[-1, 0]*1e19 if state.ni.ndim == 2 else state.ni[-1]*1e19
        self.te = state.te[-1]*1e3
        self.ti = state.ti[-1, 0]*1e3 if state.ti.ndim == 2 else state.ti[-1]*1e3  # eV
        self.pe = state.pe[-1] # pressure
        self.pi = state.pi[-1, 0] if state.pi.ndim == 2 else state.pi[-1]
        self.Pe = state.Pe[-1] # MW
        self.Pi = state.Pi[-1] # MW
        self.alpha_t = state.alphat[-1]
        self.rhostar = state.rhostar[-1]

        # Constants
        self.me_over_mi = 1/state.mi_over_me
        self.Zeff = state.Zeff[-1]
        self.keV_to_J = state.keV_to_J
        self.R0 = state.R[-1]
        self.a = state.a
        self.BT = abs(state.B_T[-1])
        self.Bp = abs(state.B_p[-1])
        self.Lpar = state.Lpar
        self.mi_ref = state.mi_ref


class FixedInitial(BoundaryConditions):

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)
        self.fixed_values = None

    def get_boundary_conditions(self, state, targets) -> None:
        if self.fixed_values is None:
            ni_lcfs = state.ni[-1, 0] if getattr(state.ni, "ndim", 1) == 2 else state.ni[-1]
            ti_lcfs = state.ti[-1, 0] if getattr(state.ti, "ndim", 1) == 2 else state.ti[-1]
            self.fixed_values = {
                "ne": float(state.ne[-1]),
                "te": float(state.te[-1]),
                "ti": float(ti_lcfs),
                "ni": float(ni_lcfs),
                "aLne": float(state.aLne[-1]),
                "aLte": float(state.aLte[-1]),
                "aLti": float(state.aLti[-1]),
                "aLni": float(state.aLni[-1]),
            }
        self.bc_dict = {key: [val, 1] for key, val in self.fixed_values.items()}

class TwoFluidTwoPoint_PeretSSF(BoundaryConditions):
    """
    Two-fluid two-point model with Peret 2025 SSF flux decay length model.
    """
    def __init__(self, options, **kwargs):
        self.options = options

    def get_boundary_conditions(self, state, targets) -> None:

        super().get_boundary_conditions(state, targets)
        self.solve_lcfs_iterative()


    def solve_peret_SSF(self,
        G0=1.5, alpha_s=0.3, f_Delta=1.0, Lambda=3.0
        ):
        """
        Solve for lambda_p, lambda_n, and lambda_T using the Peret 2025 SSF flux decay length model.

        Parameters:
            state (PlasmaState): Plasma state object
            G0 (float): Curvature drive coefficient (default: 2.0)
            alpha_s (float): Shear suppression coefficient (default: 0.25)
            f_Delta (float): Blob width scaling factor (default: 1.0)
            Lambda (float): Potential sheath drop (default: 3.0)

        Returns:
            k_pe (m), k_ne (m), k_Te (m) where k ~ a/lambda
            Gamma_e (1e20/m^2), q_e (MW/m^2)
        """

        # Constants
        me_over_mi = self.me_over_mi
        ti = self.ti
        te = self.te
        Te_J = self.te*e
        ne = self.ne

        # Sheath power exhaust factor
        #gamma_0 = (c_s / R0) * np.sqrt(rho_s / R0)
        gamma_0 = 2.5*ti/te - 0.5*np.log10(2*np.pi*me_over_mi*(1+ti/te))
        gamma = max(1+1e-6,2*gamma_0/3)

        # Get plasma parameters from state
        rho_s = self.rho_s
        c_s = self.c_s
        R0 = self.R0
        Lpar = self.Lpar
        a = self.a

        # instability curvature drive due to magnetic topology
        g = G0*rho_s/R0

        # Solve nonlinear equations (24) and (11)
        def equations(vars):
            lambda_pe, lambda_n, lambda_T = vars

            # Avoid divide-by-zero
            if lambda_pe <= 0 or lambda_n <= 0 or lambda_T <= 0:
                return [1e10, 1e10, 1e10]

            beta = f_Delta*(1/Lambda + lambda_pe/lambda_T)
            #gamma_ExB = Lambda*beta*rho_s**2/lambda_pe**2 # eqn 21
            #tau = 0.43*np.sqrt(lambda_pe/(rho_s*g)) # characteristic turbulent time
            #alpha_ExB = -tau*gamma_ExB # eqn 22
            alpha_ExB = -0.43*beta*Lambda*(rho_s/lambda_pe)**1.5/g**0.5

            # Eqn 24
            eq1 = (lambda_pe / rho_s) - 3.9*g**(3/11)*(2*rho_s/Lpar)**(-6/11)*\
                gamma**(-4/11)/(1+(alpha_s+alpha_ExB)**2)**(9/11)
            
            # Eqn 18

            eq2 = lambda_n/lambda_pe - gamma**0.5

            # Eqn 11: lambda_T definition
            eq3 = (1 / lambda_pe - (1 / lambda_T + 1 / lambda_n))*rho_s

            return np.nan_to_num(np.array([eq1, eq2, eq3]),nan=1e6,posinf=1e6,neginf=-1e6)

        # Initial guesses
        lambda_pe0,lambda_n0 = 0.01,0.02
        lambda_T0 = 1 / (1 / lambda_pe0 - 1 / lambda_n0)

        # Solve
        sol = least_squares(equations, [lambda_pe0, lambda_n0, lambda_T0], bounds=(1e-6,0.2), max_nfev=int(1e2))
        # fsolve(equations, [lambda_pe0, lambda_n0, lambda_T0])
        if sol.success is False:
            raise Exception("Peret SSF solver did not converge")
        lambda_pe, lambda_n, lambda_T = sol.x

        beta = f_Delta*(1/Lambda + lambda_pe/lambda_T)
        alpha_ExB = -0.43*beta*Lambda*(rho_s/lambda_pe)**1.5/g**0.5

        # Perpendicular turbulent fluxes, Eqn 7-8
        q0 = 41*g**0.75*(2*rho_s/Lpar)**-0.5*(rho_s/lambda_pe)**(7/4)/\
            (1+(alpha_s+alpha_ExB)**2)**(9/4)
        Gamma_e = q0*(lambda_pe/lambda_n)*ne*c_s # eqn 7
        q_e = q0*(ne*Te_J)*c_s # eqn 8, W/m^2
        lambda_qe = (2/7)*lambda_T*gamma**0.5/(gamma**0.5-1) # Spitzer-Harm, m

        return a/lambda_pe, a/lambda_n, a/lambda_T, Gamma_e, q_e,lambda_qe

    
    def solve_two_fluid_two_point(self, lambda_q: float, fmom=0.5,
            ne_target=4., Te_target=0.005, Ti_target=None):
        '''
        Inputs:
            state (PlasmaState): Plasma state object
            lambda_q (float): LCFS upstream heat decay length (m)
            fmom (float): LCFS upstream momentum loss factor
            ne_target (float): divertor electron density (1e19/m^3)
            Te_target (float): divertor electron temperature (keV)
            Ti_target (float): divertor ion temperature (keV)
            DoD (float): Degree of detachment
            Gamma_sheath (float): total sheath transmission factor

        Returns:
            ne_u (float): LCFS upstream electron density at detachment (1e19/m^3)
            Te_u (float): LCFS upstream electron temperature at detachment (keV)
            Ti_u (float): LCFS upstream ion temperature at detachment (keV)

        Note: Not yet considering volumetric (radiative) or cross-field convective losses in along flux tubes SOL
        '''

        # Constants
        if Ti_target is None: Ti_target = Te_target
        me_over_mi = self.me_over_mi

        ne_target,Te_target,Ti_target = 1e19*ne_target, Te_target*1e3, Ti_target*1e3 # remove normalizations
        kappa_0e = 2600/(0.672 + 0.076*self.Zeff**0.5 + 0.252*self.Zeff) # from Eich-Manz SepOS
        kappa_0i = kappa_0e * (me_over_mi)**0.5
        Apar_sol = 4 * np.pi * self.R0 * lambda_q * (self.Bp / self.BT) # Stangeby 5.51

        # Get heat fluxes
        qpar_e, qpar_i = self.Pe*1e6/Apar_sol, self.Pi*1e6/Apar_sol

        Te_u = max(Te_target,(Te_target**3.5 + 3.5 * fmom * qpar_e * self.Lpar / kappa_0e)**(2/7))
        Ti_u = max(Te_u,(Ti_target**3.5 + 3.5 * fmom * qpar_i * self.Lpar / kappa_0i)**(2/7))
        ne_u = (2 * ne_target * (Te_target + Ti_target))/(fmom*(Te_u+Ti_u/self.Zeff))
        # Assuming momentum losses (fmom) ~ convective/volumetric heat losses of ~50%

        return (ne_u, Te_u, Ti_u, qpar_e, qpar_i) 
                # 1/m^3, eV, eV, W/m^2, W/m^2


    def solve_lcfs_iterative(self, max_iter=100, tol=1e-3, damping=0.5):
        """
        Iteratively solve for LCFS (ne_lcfs, Te_lcfs, Ti_lcfs) by alternating between
        Peret SSF and TFTP models until convergence.
        
        Parameters:
            state (PlasmaState): Plasma state object
            max_iter (int): Maximum number of iterations
            tol (float): Relative tolerance for convergence
            damping (float): Damping factor for updates (0 < damping <= 1)
        
        Returns:
            None (stores results in state.BC.*)
        """
        
        # Initial guess from LCFS values

        #print(f"Initial guess: ne={self.ne:.2f}, Te={self.te:.3f}, Ti={self.ti:.3f}")
        converged = False

        for iteration in range(max_iter):
            # Store old values for convergence check
            ne_old, ni_old, Te_old, Ti_old = self.ne, self.ni, self.te, self.ti

            # update terms
            self.rho_s = plasma.rho_s(self.te*1e-3, self.mi_ref, self.BT)
            self.c_s = plasma.c_s(self.te*1e-3, self.mi_ref)

            # Step 1: Use Peret SSF to get lambda_q and scale lengths
            try:
                aLpe, aLne, aLTe, Gamma_peret, q_e_peret, lambda_q = self.solve_peret_SSF()
            except Exception as e:
                print(f"Peret SSF failed at iteration {iteration}: {e}")
                break
                
            # Step 2: Use TFTP with lambda_q to get updated LCFS values
            try:
                ne_tftp, Te_tftp, Ti_tftp, qpar_e_tftp, qpar_i_tftp = self.solve_two_fluid_two_point(lambda_q, fmom=0.5)
            except Exception as e:
                print(f"TFTP failed at iteration {iteration}: {e}")
                break
            
            # Step 3: Apply damping to updates
            self.ne = (1 - damping) * self.ne + damping * ne_tftp
            self.te = (1 - damping) * self.te + damping * Te_tftp
            self.ti = (1 - damping) * self.ti + damping * Ti_tftp
            self.ni = self.ne/self.Zeff

            # Check for convergence
            rel_change_ne = abs((self.ne - ne_old) / ne_old)
            rel_change_Te = abs((self.te - Te_old) / Te_old)
            rel_change_Ti = abs((self.ti - Ti_old) / Ti_old)

            max_change = max(rel_change_ne, rel_change_Te, rel_change_Ti)

            # print(f"Iter {iteration+1}: ne={self.ne:.2f}, Te={self.te:.3f}, Ti={self.ti:.3f}, "
            #     f"max_change={max_change:.1e}")
            
            if max_change < tol:
                #print(f"Converged after {iteration+1} iterations")
                #print(f"Iter {iteration+1}: ne={1e-19*self.ne:.2f}, Te={1e-3*self.te:.3f}, Ti={1e-3*self.ti:.3f}, "
                #    f"max_change={max_change:.1e}")
                converged = True
                break
            if iteration == max_iter-1:
                print(f"Did not converge after {max_iter} iterations")

        
        # Final calculation of scale lengths (note: solve_peret_SSF reads from state)
        _, self.aLne, self.aLTe, _, _, _ = self.solve_peret_SSF()
        self.aLTi = (self.te/self.ti)*self.aLTe  # TODO: implement proper ion scale length

        # Store results. (value, location in roa or rho)
        self.bc_dict = {y: [val, 1] for y, val in zip(
            ['ne', 'te', 'ti', 'ni'], [self.ne*1e-19, self.te*1e-3, self.ti*1e-3, self.ni*1e-19])}
        self.bc_dict.update({aLy: [aLy_val, 1] for aLy, aLy_val in zip(
            ['aLne', 'aLte', 'aLti', 'aLni'], [self.aLne, self.aLTe, self.aLTi, self.aLne])})
        self.converged = converged

class Tftp_SepOS(BoundaryConditions):
    """
    LCFS (Last Closed Flux Surface) boundary condition handler.
    
    Implements Two-Fluid Two-Point model with Eich-Manz SepOS scale lengths.
    
    Based on MINTboundary.LCFShandler
    """
    
    def __init__(self, state):
        """
        Initialize boundary conditions from plasma state.
        
        Parameters
        ----------
        state : PlasmaState
            Plasma state with profiles
        """

    def get_boundary_conditions(self, state, targets) -> None:

        super().get_boundary_conditions(state, targets)
        self.solve_lcfs_iterative()


    def get_SepOS_aLy(self) -> None:
        """
        Calculate gradient scale lengths at separatrix.
        
        Uses Eich-Manz SepOS scale lengths
        
        Parameters
        ----------
        none
        
        Returns
        -------
        aLy 
        """

                # Constants
        me_over_mi = self.me_over_mi
        ti = self.ti
        te = self.te
        Te_J = self.te*e
        ne = self.ne

        # Sheath power exhaust factor
        #gamma_0 = (c_s / R0) * np.sqrt(rho_s / R0)
        gamma_0 = 2.5*ti/te - 0.5*np.log10(2*np.pi*me_over_mi*(1+ti/te))
        gamma = max(1+1e-6,2*gamma_0/3)

        # Get plasma parameters from state
        rho_s = self.rho_s
        c_s = self.c_s
        R0 = self.R0
        Lpar = self.Lpar
        a = self.a

        # Eich-Manz SepOS scaling
        # TODO Double check these and the alpha_t definition in PlasmaState
        aLne = a*max(0.0, 1.0 / (2.9 * (1 + 10.4 * self.alpha_t**2.5) *
                                   self.rhostar * self.BT / self.Bp))
        aLte = a*max(0.0, 1.0 / (2.1 * (1 + 2.1 * self.alpha_t**1.7) *
                                   self.rhostar * self.BT / self.Bp))
        
        lambda_q = (2/7)*self.te*1e-3*gamma**0.5/(gamma**0.5-1)*aLte*self.a

        return aLne, aLte, lambda_q

    
    def solve_two_fluid_two_point(self, lambda_q: float, fmom=0.5,
            ne_target=4., Te_target=0.005, Ti_target=None):
        '''
        Inputs:
            state (PlasmaState): Plasma state object
            lambda_q (float): LCFS upstream heat decay length (m)
            fmom (float): LCFS upstream momentum loss factor
            ne_target (float): divertor electron density (1e19/m^3)
            Te_target (float): divertor electron temperature (keV)
            Ti_target (float): divertor ion temperature (keV)
            DoD (float): Degree of detachment
            Gamma_sheath (float): total sheath transmission factor

        Returns:
            ne_u (float): LCFS upstream electron density at detachment (1e19/m^3)
            Te_u (float): LCFS upstream electron temperature at detachment (keV)
            Ti_u (float): LCFS upstream ion temperature at detachment (keV)

        Note: Not yet considering volumetric (radiative) or cross-field convective losses in along flux tubes SOL
        '''

        # Constants
        if Ti_target is None: Ti_target = Te_target
        me_over_mi = self.me_over_mi

        ne_target,Te_target,Ti_target = 1e19*ne_target, Te_target*1e3, Ti_target*1e3 # remove normalizations
        kappa_0e = 2600/(0.672 + 0.076*self.Zeff**0.5 + 0.252*self.Zeff) # from Eich-Manz SepOS
        kappa_0i = kappa_0e * (me_over_mi)**0.5
        Apar_sol = 4 * np.pi * self.R0 * lambda_q * (self.Bp / self.BT) # Stangeby 5.51

        # Get heat fluxes
        qpar_e, qpar_i = self.Pe*1e6/Apar_sol, self.Pi*1e6/Apar_sol

        Te_u = max(Te_target,(Te_target**3.5 + 3.5 * fmom * qpar_e * self.Lpar / kappa_0e)**(2/7))
        Ti_u = max(Te_u,(Ti_target**3.5 + 3.5 * fmom * qpar_i * self.Lpar / kappa_0i)**(2/7))
        ne_u = (2 * ne_target * (Te_target + Ti_target))/(fmom*(Te_u+Ti_u/self.Zeff))
        # Assuming momentum losses (fmom) ~ convective/volumetric heat losses of ~50%

        return (ne_u, Te_u, Ti_u, qpar_e, qpar_i) 
                # 1/m^3, eV, eV, W/m^2, W/m^2


    def solve_lcfs_iterative(self, max_iter=100, tol=1e-3, damping=0.5):
        """
        Iteratively solve for LCFS (ne_lcfs, Te_lcfs, Ti_lcfs) by alternating between
        Eich-Manz SepOS and TFTP models until convergence.
        
        Parameters:
            state (PlasmaState): Plasma state object
            max_iter (int): Maximum number of iterations
            tol (float): Relative tolerance for convergence
            damping (float): Damping factor for updates (0 < damping <= 1)
        
        Returns:
            None (stores results in state.BC.*)
        """
        
        # Initial guess from LCFS values

        #print(f"Initial guess: ne={self.ne:.2f}, Te={self.te:.3f}, Ti={self.ti:.3f}")
        converged = False

        for iteration in range(max_iter):
            # Store old values for convergence check
            ne_old, ni_old, Te_old, Ti_old = self.ne, self.ni, self.te, self.ti

            # update terms
            self.rho_s = plasma.rho_s(self.te*1e-3, self.mi_ref, self.BT)
            self.c_s = plasma.c_s(self.te*1e-3, self.mi_ref)

            # Step 1: Use Peret SSF to get lambda_q and scale lengths
            try:
                aLne, aLTe, lambda_q = self.get_SepOS_aLy()
            except Exception as e:
                print(f"Peret SSF failed at iteration {iteration}: {e}")
                break
                
            # Step 2: Use TFTP with lambda_q to get updated LCFS values
            try:
                ne_tftp, Te_tftp, Ti_tftp, qpar_e_tftp, qpar_i_tftp = self.solve_two_fluid_two_point(lambda_q, fmom=0.5)
            except Exception as e:
                print(f"TFTP failed at iteration {iteration}: {e}")
                break
            
            # Step 3: Apply damping to updates
            self.ne = (1 - damping) * self.ne + damping * ne_tftp
            self.te = (1 - damping) * self.te + damping * Te_tftp
            self.ti = (1 - damping) * self.ti + damping * Ti_tftp
            self.ni = self.ne/self.Zeff

            # Check for convergence
            rel_change_ne = abs((self.ne - ne_old) / ne_old)
            rel_change_Te = abs((self.te - Te_old) / Te_old)
            rel_change_Ti = abs((self.ti - Ti_old) / Ti_old)

            max_change = max(rel_change_ne, rel_change_Te, rel_change_Ti)

            # print(f"Iter {iteration+1}: ne={self.ne:.2f}, Te={self.te:.3f}, Ti={self.ti:.3f}, "
            #     f"max_change={max_change:.1e}")
            
            if max_change < tol:
                #print(f"Converged after {iteration+1} iterations")
                #print(f"Iter {iteration+1}: ne={1e-19*self.ne:.2f}, Te={1e-3*self.te:.3f}, Ti={1e-3*self.ti:.3f}, "
                #    f"max_change={max_change:.1e}")
                converged = True
                break
            if iteration == max_iter-1:
                print(f"Did not converge after {max_iter} iterations")

        
        # Final calculation of scale lengths (note: solve_peret_SSF reads from state)
        _, self.aLne, self.aLTe, _, _, _ = self.solve_peret_SSF()
        self.aLTi = (self.te/self.ti)*self.aLTe  # TODO: implement proper ion scale length

        # Store results. (value, location in roa or rho)
        self.bc_dict = {y: [val, 1] for y, val in zip(
            ['ne', 'te', 'ti', 'ni'], [self.ne*1e-19, self.te*1e-3, self.ti*1e-3, self.ni*1e-19])}
        self.bc_dict.update({aLy: [aLy_val, 1] for aLy, aLy_val in zip(
            ['aLne', 'aLte', 'aLti', 'aLni'], [self.aLne, self.aLTe, self.aLTi, self.aLne])})
        self.converged = converged


BOUNDARY_CONDITION_MODELS = {
    'TwoFluid_EichManz': Tftp_SepOS,
    'TwoFluidTwoPoint_PeretSSF': TwoFluidTwoPoint_PeretSSF,
    'FixedInitial': FixedInitial,
}