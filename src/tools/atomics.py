"""
Aurora-compatible atomic data evaluation using polynomial fits.

Provides drop-in replacement for Aurora API when Aurora is unavailable:
  - atomic.get_cs_balance_terms(...) -> ionization, recombination, CX rates
  - atomic.get_frac_abundances(...) -> fractional abundances (placeholder)
  - radiation.compute_rad(...) -> radiation (placeholder)

Uses Chebyshev polynomial fits for fast atomic rate evaluation.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Optional

# Global cache of loaded species data
_species_cache: Dict[str, 'AtomicDataInterpolator'] = {}

class AtomicDataInterpolator:
    """
    Mimics Aurora API for atomic rates using polynomial evaluation.
    
    Provides drop-in replacement functions:
      - get_cs_balance_terms(...) -> ionization, recombination, CX rates
      - get_frac_abundances(...) -> fZ (placeholder, compute from rates)
      - compute_rad(...) -> radiation dict (placeholder, use Aurora)
    
    Uses pre-fitted Chebyshev polynomials for fast atomic rate evaluation.
    """
    
    def __init__(self, species_name: str, data_dir: str = 'atomic_data'):
        self.species_name = species_name
        self.data_dir = data_dir
        
        # Load polynomial coefficients for evaluation
        self.coefficients: Dict = {}
        self.scalers: Dict = {}
        
        self._load_polynomial_data()
    
    def _load_polynomial_data(self, master_file: str = 'atomic_rate_polynomials.csv'):
        """Load polynomial coefficients for atomic rate evaluation."""
        # Prefer PRESTOS_ROOT if set, otherwise infer repo root from this file (../../)
        prestos_root = os.environ.get('PRESTOS_ROOT')
        if not prestos_root:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            prestos_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
        master_filepath = os.path.join(prestos_root, 'src', 'tools', master_file)
        if not os.path.exists(master_filepath):
            return  # No polynomial data available
        
        try:
            master_df = pd.read_csv(master_filepath)
            species_data = master_df[master_df['species'] == self.species_name]
            
            if species_data.empty:
                return
            
            # Load scalers
            first_row = species_data.iloc[0]
            scaler_cols = ['ne_min', 'ne_max', 'Te_min', 'Te_max', 'Ti_min', 'Ti_max']
            if all(col in first_row.index for col in scaler_cols):
                self.scalers = {col: first_row[col] for col in scaler_cols}
            
            self.coefficients = {'radiation': {}}
            
            # Load coefficients
            for _, row in species_data.iterrows():
                process = row['process']
                charge_state_key = int(row['charge_state'])
                degree = int(row['degree'])
                n_features = int(row['n_features'])
                intercept = row['intercept']
                coeff_cols = [col for col in row.index if col.startswith('coeff_')]
                coefficients = [row[col] for col in coeff_cols if not pd.isna(row[col])]
                component = process.replace('radiation_', '') if process.startswith('radiation_') else None

                container = self.coefficients['radiation'].setdefault(component,{}) if component else self.coefficients.setdefault(process, {})

                entry_dict = {
                    'degree': degree,
                    'n_features': n_features,
                    'coefficients': np.array(coefficients),
                    'intercept': intercept,
                }
                container[charge_state_key] = entry_dict
            
        except Exception:
            # Loading failed; keep coefficients empty so caller can handle fallback
            pass
    
    # -------------------- Aurora-compatible API --------------------
    
    def get_cs_balance_terms(self, ne_cm3: np.ndarray, Te_eV: np.ndarray, Ti_eV: np.ndarray, 
                             include_cx: bool = True) -> Tuple:

        """Mimic aurora.atomic.get_cs_balance_terms.
        
        Parameters
        ----------
        ne_cm3 : array
            Electron density [cm^-3]
        Te_eV : array
            Electron temperature [eV]
        Ti_eV : array
            Ion temperature [eV]
        include_cx : bool
            Include charge exchange rates
        
        Returns
        -------
        tuple
            (None, S_rates, R_rates, CX_rates) matching Aurora API
            S_rates: ionization (N, Zmax+1)
            R_rates: recombination (N, Zmax+1)
            CX_rates: charge exchange (N, Zmax+1)
        """

        # Convert units to CSV convention
        ne = ne_cm3 * 1e-19 * 1e6  # cm^-3 -> 1e19/m^3
        Te = Te_eV / 1000.0  # eV -> keV
        Ti = Ti_eV / 1000.0  # eV -> keV
        
        ne = np.atleast_1d(ne)
        Te = np.atleast_1d(Te)
        Ti = np.atleast_1d(Ti)
        
        # Determine max charge state from polynomial coefficients
        n_charge = max([max(self.coefficients.get('ionization', {}).keys(), default=0) + 1, 1])
        
        n_samples = ne.size
        S_rates = np.zeros((n_samples, n_charge))
        R_rates = np.zeros((n_samples, n_charge))
        CX_rates = np.zeros((n_samples, n_charge)) if include_cx else None
        
        # Use polynomial evaluation
        for z in range(n_charge):
            S_rates[:, z] = self._poly_rate(ne, Te, Ti, 'ionization', z)
            R_rates[:, z] = self._poly_rate(ne, Te, Ti, 'recombination', z)
            if include_cx:
                CX_rates[:, z] = self._poly_rate(ne, Te, Ti, 'charge_exchange', z)
        
        return None, S_rates, R_rates, CX_rates
    
    def get_frac_abundances(self, ne_cm3: np.ndarray, Te_eV: np.ndarray, Ti_eV: np.ndarray,
                           n0_by_ne: Optional[np.ndarray] = None, plot: bool = False) -> Tuple:
        """Mimic aurora.atomic.get_frac_abundances.
        
        Parameters
        ----------
        ne_cm3 : array
            Electron density [cm^-3]
        Te_eV : array
            Electron temperature [eV]
        Ti_eV : array
            Ion temperature [eV]
        n0_by_ne : array, optional
            Neutral fraction (unused in CSV lookup)
        plot : bool
            Whether to plot (ignored)
        
        Returns
        -------
        tuple
            (fZ, None) where fZ has shape (N, Zmax+1)
        """
        ne = ne_cm3 * 1e-19 * 1e6  # cm^-3 -> 1e19/m^3
        Te = Te_eV / 1000.0  # eV -> keV
        Ti = Ti_eV / 1000.0  # eV -> keV
        
        ne = np.atleast_1d(ne)
        Te = np.atleast_1d(Te)
        Ti = np.atleast_1d(Ti)
        
        # Determine charge states from coefficients
        if 'fZ' in self.coefficients:
            n_charge = len(self.coefficients['fZ'])
        else:
            n_charge = max([max(self.coefficients.get('ionization', {}).keys(), default=0) + 1, 1])
        
        n_samples = ne.size
        fZ = np.zeros((n_samples, n_charge))
        
        # Use polynomial evaluation if available
        if 'fZ' in self.coefficients:
            for z in range(n_charge):
                if z in self.coefficients['fZ']:
                    fZ[:, z] = self._poly_fZ(ne, Te, Ti, z)
            
            # Normalize to ensure sum = 1
            fZ = fZ / (fZ.sum(axis=1, keepdims=True) + 1e-12)
        else:
            # Fallback: all neutrals
            fZ[:, 0] = 1.0
        
        return None, fZ
    
    def compute_rad(self, ne_cm3: np.ndarray, Te_eV: np.ndarray, Ti_eV: Optional[np.ndarray] = None,
                   prad_flag: bool = False, thermal_cx_rad_flag: bool = False) -> Dict[str, np.ndarray]:
        """Mimic aurora.radiation.compute_rad.
        
        Parameters
        ----------
        ne_cm3 : array
            Electron density [cm^-3]
        Te_eV : array
            Electron temperature [eV]
        Ti_eV : array, optional
            Ion temperature [eV]
        prad_flag : bool
            Include power radiation (always True for CSV data)
        thermal_cx_rad_flag : bool
            Include CX radiation (always True for CSV data)
        
        Returns
        -------
        dict
            Radiation components with arrays of shape (n_charge_states, n_spatial):
            {'tot', 'line_rad', 'cont_rad', 'brems', 'thermal_cx_cont_rad', 'synchrotron'}
        """
        ne = ne_cm3 * 1e-19 * 1e6  # cm^-3 -> 1e19/m^3
        Te = Te_eV / 1000.0  # eV -> keV
        Ti = Ti_eV / 1000.0 if Ti_eV is not None else Te
        
        ne = np.atleast_1d(ne)
        Te = np.atleast_1d(Te)
        Ti = np.atleast_1d(Ti)
        
        n_spatial = ne.size
        result = {}
        
        # Use polynomial evaluation if radiation coefficients available
        if 'radiation' in self.coefficients:
            rad_coeffs = self.coefficients['radiation']
            
            # Map internal names to Aurora output keys
            component_map = {
                'tot': 'tot',
                'line': 'line_rad',
                'cont': 'cont_rad',
                'brems': 'brems',
                'cx': 'thermal_cx_cont_rad'
            }
            
            for internal_name, output_key in component_map.items():
                if internal_name in rad_coeffs:
                    component_data = rad_coeffs[internal_name]
                    
                    # Check if this is total radiation (scalar per spatial point)
                    if component_data.get(-1,None):
                        # Total radiation: (n_spatial,) -> need to broadcast to (n_charge, n_spatial)
                        total_rad = self._poly_radiation(ne, Te, Ti, internal_name, charge_state=None)
                        # For Aurora compatibility, return as (1, n_spatial) or distribute equally
                        result[output_key] = total_rad[None, :]  # (1, n_spatial)
                    else:
                        # Per-charge-state radiation
                        n_charge = len(component_data)
                        rad_array = np.zeros((n_charge, n_spatial))
                        for z in range(n_charge):
                            if z in component_data:
                                rad_array[z, :] = self._poly_radiation(ne, Te, Ti, internal_name, charge_state=z)
                        result[output_key] = rad_array
        
        # If no radiation coefficients, return zeros
        if not result:
            n_charge = max([max(self.coefficients.get('ionization', {}).keys(), default=0) + 1, 1])
            result = {
                'tot': np.zeros((n_charge, n_spatial)),
                'line_rad': np.zeros((n_charge, n_spatial)),
                'cont_rad': np.zeros((n_charge, n_spatial)),
                'brems': np.zeros((n_charge, n_spatial)),
                'thermal_cx_cont_rad': np.zeros((n_charge, n_spatial)),
            }
        
        return result
    
    def _poly_rate(self, ne, Te, Ti, process, charge_state):
        """Evaluate polynomial for a specific rate.
        
        Parameters
        ----------
        ne : array
            Electron density [1e19/m^3]
        Te : array
            Electron temperature [keV]
        Ti : array
            Ion temperature [keV]
        process : str
            'ionization', 'recombination', or 'charge_exchange'
        charge_state : int
            Charge state index
        
        Returns
        -------
        rate : array
            Rate coefficient [m^3/s] or similar
        """
        if process not in self.coefficients or charge_state not in self.coefficients[process]:
            return np.zeros_like(ne)
        
        # Normalize to [-1, 1] for Chebyshev
        ne_norm = 2 * (np.log10(ne) - self.scalers['ne_min']) / (self.scalers['ne_max'] - self.scalers['ne_min']) - 1
        Te_norm = 2 * (np.log10(Te) - self.scalers['Te_min']) / (self.scalers['Te_max'] - self.scalers['Te_min']) - 1
        Ti_norm = 2 * (np.log10(Ti) - self.scalers['Ti_min']) / (self.scalers['Ti_max'] - self.scalers['Ti_min']) - 1
        
        coeff_info = self.coefficients[process][charge_state]
        n_features = coeff_info['n_features']
        
        if n_features == 3:  # Charge exchange (ne, Te, Ti)
            X_norm = np.column_stack([ne_norm, Te_norm, Ti_norm])
        else:  # Ionization/recombination (ne, Te)
            X_norm = np.column_stack([ne_norm, Te_norm])
        
        degree = coeff_info['degree']
        X_poly = self._chebyshev_features(X_norm, degree)
        
        # Predict
        log_rate = (X_poly @ coeff_info['coefficients'] + coeff_info['intercept'])
        rate = 10**log_rate
        
        return rate
    
    def _poly_fZ(self, ne, Te, Ti, charge_state):
        """Evaluate polynomial for fractional abundance.
        
        Parameters
        ----------
        ne : array
            Electron density [1e19/m^3]
        Te : array
            Electron temperature [keV]
        Ti : array
            Ion temperature [keV]
        charge_state : int
            Charge state index
        
        Returns
        -------
        fZ : array
            Fractional abundance [0, 1]
        """
        if 'fZ' not in self.coefficients or charge_state not in self.coefficients['fZ']:
            return np.zeros_like(ne)
        
        # Normalize to [-1, 1] for Chebyshev
        ne_norm = 2 * (np.log10(ne) - self.scalers['ne_min']) / (self.scalers['ne_max'] - self.scalers['ne_min']) - 1
        Te_norm = 2 * (np.log10(Te) - self.scalers['Te_min']) / (self.scalers['Te_max'] - self.scalers['Te_min']) - 1
        Ti_norm = 2 * (np.log10(Ti) - self.scalers['Ti_min']) / (self.scalers['Ti_max'] - self.scalers['Ti_min']) - 1
        
        coeff_info = self.coefficients['fZ'][charge_state]
        
        # fZ uses 3D features (ne, Te, Ti)
        X_norm = np.column_stack([ne_norm, Te_norm, Ti_norm])
        
        degree = coeff_info['degree']
        X_poly = self._chebyshev_features(X_norm, degree)
        
        # Predict in logit space
        logit_fZ = (X_poly @ coeff_info['coefficients'] + coeff_info['intercept'])
        
        # Transform back to [0, 1] using inverse logit (sigmoid)
        fZ = 1 / (1 + np.exp(-logit_fZ))
        
        return fZ
    
    def _poly_radiation(self, ne, Te, Ti, component, charge_state):
        """Evaluate polynomial for radiation component.
        
        Parameters
        ----------
        ne : array
            Electron density [1e19/m^3]
        Te : array
            Electron temperature [keV]
        Ti : array
            Ion temperature [keV]
        component : str
            Radiation component ('tot', 'line', 'cont', 'brems', 'cx')
        charge_state : int or None
            Charge state index (None for total)
        
        Returns
        -------
        rad : array
            Radiation power density [W/cm^3]
        """
        if 'radiation' not in self.coefficients or component not in self.coefficients['radiation']:
            return np.zeros_like(ne)
        
        component_data = self.coefficients['radiation'][component]
        
        # Check if this is total (no charge state) or per-charge-state
        if charge_state is None:
            if not isinstance(component_data, dict) or 'coefficients' not in component_data:
                return np.zeros_like(ne)
            coeff_info = component_data
        else:
            if charge_state not in component_data:
                return np.zeros_like(ne)
            coeff_info = component_data[charge_state]
        
        # Normalize to [-1, 1] for Chebyshev
        ne_norm = 2 * (np.log10(ne) - self.scalers['ne_min']) / (self.scalers['ne_max'] - self.scalers['ne_min']) - 1
        Te_norm = 2 * (np.log10(Te) - self.scalers['Te_min']) / (self.scalers['Te_max'] - self.scalers['Te_min']) - 1
        Ti_norm = 2 * (np.log10(Ti) - self.scalers['Ti_min']) / (self.scalers['Ti_max'] - self.scalers['Ti_min']) - 1
        
        # Radiation uses 3D features (ne, Te, Ti)
        X_norm = np.column_stack([ne_norm, Te_norm, Ti_norm])
        
        degree = coeff_info['degree']
        X_poly = self._chebyshev_features(X_norm, degree)
        
        # Predict in log space
        log_rad = (X_poly @ coeff_info['coefficients'] + coeff_info['intercept'])
        rad = 10**log_rad
        
        return rad
    
    def _chebyshev_features(self, X, degree):
        """Generate Chebyshev polynomial features up to given degree."""
        from numpy.polynomial.chebyshev import chebval
        
        n_samples, n_features = X.shape
        features = [np.ones(n_samples)]  # Constant term
        
        # First-order terms
        for i in range(n_features):
            features.append(X[:, i])
        
        # Higher-order terms
        for d in range(2, degree + 1):
            for i in range(n_features):
                # Chebyshev polynomials of the first kind
                features.append(chebval(X[:, i], [0] * d + [1]))
            
            # Cross terms (for degree >= 2)
            if d == 2:
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        features.append(X[:, i] * X[:, j])
        
        return np.column_stack(features)


# ============================================================================
# Module-level Aurora-compatible API
# ============================================================================

def get_atom_data(species: str, aurora_files, data_dir: str = 'atomic_data'):
    """Mimic aurora.get_atom_data by returning cached interpolator.
    
    Parameters
    ----------
    species : str
        Species name (e.g., 'D', 'He', 'Ar')
    files : list, optional
        File types to load (unused, for API compatibility)
    data_dir : str
        Directory containing CSV files
    
    Returns
    -------
    AtomicDataInterpolator
        Cached interpolator object
    """
    # Create or refresh cached interpolator
    interp = _species_cache.get(species)
    if interp is None:
        interp = AtomicDataInterpolator(species, data_dir)
        _species_cache[species] = interp
    else:
        # Ensure polynomial data is loaded when coefficients are missing
        if not getattr(interp, 'coefficients', None):
            interp._load_polynomial_data()
    return interp


class atomic:
    """Namespace mimicking aurora.atomic module."""
    
    @staticmethod
    def get_cs_balance_terms(atom_data, ne_cm3, Te_eV, Ti_eV, include_cx=True):
        """Wrapper for atom_data.get_cs_balance_terms."""
        return atom_data.get_cs_balance_terms(ne_cm3, Te_eV, Ti_eV, include_cx)
    
    @staticmethod
    def get_frac_abundances(atom_data, ne_cm3, Te_eV, Ti_eV, n0_by_ne=None, plot=False):
        """Wrapper for atom_data.get_frac_abundances."""
        return atom_data.get_frac_abundances(ne_cm3, Te_eV, Ti_eV, n0_by_ne, plot)


class radiation:
    """Namespace mimicking aurora.radiation module."""
    
    @staticmethod
    def compute_rad(species, nZ, ne_cm3, Te_eV, n0=None, Ti=None, 
                   adas_files_sub=None, prad_flag=False, sxr_flag=False,
                   thermal_cx_rad_flag=False, spectral_brem_flag=False):
        """Compute radiation using interpolated data.
        
        Parameters
        ----------
        species : str
            Species name
        nZ : array
            Charge state densities (unused in CSV interpolation mode)
        ne_cm3 : array
            Electron density [cm^-3]
        Te_eV : array
            Electron temperature [eV]
        n0 : array, optional
            Neutral density (unused)
        Ti : array, optional
            Ion temperature [eV]
        prad_flag : bool
            Include power radiation
        thermal_cx_rad_flag : bool
            Include thermal CX radiation
        
        Returns
        -------
        dict
            Radiation components with arrays of shape (n_charge_states, n_spatial)
        """
        if species not in _species_cache:
            _species_cache[species] = AtomicDataInterpolator(species)
        
        atom_data = _species_cache[species]
        Ti_eV = Ti if Ti is not None else Te_eV
        return atom_data.compute_rad(ne_cm3, Te_eV, Ti_eV, prad_flag, thermal_cx_rad_flag)


# Example usage
# if __name__ == "__main__":
    
#     interp = AtomicDataInterpolator('D', data_dir='atomic_data')
    
#     # Test conditions (Aurora units)
#     ne_cm3 = np.array([5e13, 2e14])  # cm^-3
#     Te_eV = np.array([50.0, 100.0])  # eV
#     Ti_eV = np.array([80.0, 150.0])  # eV
    
#     print("Testing Aurora-compatible API:\n")
    
#     # Test get_cs_balance_terms
#     _, S, R, CX = interp.get_cs_balance_terms(ne_cm3, Te_eV, Ti_eV, include_cx=True)
#     print(f"Shape of S (ionization): {S.shape}")  # (n_spatial, n_charge)
#     print(f"Ionization rates (z=0): {S[:, 0]}")
#     print(f"Recombination rates (z=1->0): {R[:, 0]}")  # R[:, z] is rate for z+1 -> z
#     print(f"Charge exchange rates (z=0): {CX[:, 0]}")
    
#     # Test get_frac_abundances
#     _, fZ = interp.get_frac_abundances(ne_cm3, Te_eV, Ti_eV)
#     print(f"\nShape of fZ: {fZ.shape}")  # (n_spatial, n_charge+1) includes neutrals
#     print(f"Fractional abundances (z=0): {fZ[:, 0]}")
#     if fZ.shape[1] > 1:
#         print(f"Fractional abundances (z=1): {fZ[:, 1]}")
#     print(f"Sum of fractions: {fZ.sum(axis=1)}")
    
#     # Test compute_rad
#     rad = interp.compute_rad(ne_cm3, Te_eV, Ti_eV, prad_flag=True, thermal_cx_rad_flag=True)
#     print(f"\nRadiation components available: {list(rad.keys())}")
#     if 'tot' in rad:
#         print(f"Shape of rad['tot']: {rad['tot'].shape}")  # (n_charge, n_spatial)
#         print(f"Total radiation sum: {rad['tot'].sum():.3e} W/cm^3")
#     if 'line_rad' in rad and rad['line_rad'].size > 0:
#         print(f"Line radiation per charge state: {rad['line_rad'][:, 0]}")
    
#     print("\nTesting module-level Aurora-compatible API:\n")
    
#     # Test module-level wrappers
#     atom_data = get_atom_data('D', data_dir='atomic_data')
#     _, S2, R2, CX2 = atomic.get_cs_balance_terms(atom_data, ne_cm3, Te_eV, Ti_eV)
#     print(f"Module-level ionization rates match: {np.allclose(S, S2)}")
    
#     _, fZ2 = atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, Ti_eV)
#     print(f"Module-level fZ match: {np.allclose(fZ, fZ2)}")
