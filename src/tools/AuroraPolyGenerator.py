import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import pandas as pd
import aurora
import os

# Species dictionary mapping
SPECIES_DICT = {
    'H': 1, 'D': 1, 'T': 1,
    'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Kr': 36, 'Mo': 42, 'Xe': 54, 'W': 74
}

class AuroraPolyFitter:
        """
        Fits ionization, recombination, and charge exchange rates as polynomials.
        Also generates and saves separate CSVs for:
            - raw rates (ionization, recombination, CX) via aurora.atomic.get_cs_balance_terms
            - fractional abundances fZ via aurora.atomic.get_frac_abundances
            - radiation components via aurora.radiation.compute_rad(..., prad_flag=True, thermal_cx_rad_flag=True)

        These outputs align with usage in targets.AnalyticTargetModel and neutrals.NeutralModel.
        """
    
        def __init__(self, species_name, max_degree=3, output_dir='atomic_data'):
                self.species_name = species_name
                self.max_degree = max_degree
                self.Z_atom = SPECIES_DICT[species_name]
                self.coefficients = {}  # {process: {charge_state: {coeffs...}}}
                self.scalers = {}
                self.fit_quality = {}
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
                self.training_data = {}
                self._raw_data = {}
        
        def generate_training_data(self, ne_range=(0.1, 10.), Te_range=(0.01, 1.0), 
                                Ti_range=(0.01, 1.0), n_samples=2000):
            """
            Generate training data for all atomic processes.
            """
            print(f"Generating atomic rate training data for {self.species_name}...")
            
            # Create random samples in log space for better coverage
            np.random.seed(42)  # Reproducible
            log_ne = np.random.uniform(np.log10(ne_range[0]), np.log10(ne_range[1]), n_samples)
            log_Te = np.random.uniform(np.log10(Te_range[0]), np.log10(Te_range[1]), n_samples)
            log_Ti = np.random.uniform(np.log10(Ti_range[0]), np.log10(Ti_range[1]), n_samples)
            
            ne_samples = 10**log_ne  # 1e19/m^-3
            Te_samples = 10**log_Te  # keV
            Ti_samples = 10**log_Ti  # keV
            
            # Get atomic data (consistent with neutrals/targets usage)
            atom_data = aurora.get_atom_data(self.species_name, ['acd', 'scd', 'ccd'])
            
            # Convert for Aurora
            ne_cm3 = ne_samples * 1e19 * 1e-6  # Convert to cm^-3
            Te_eV = Te_samples * 1000.0
            Ti_eV = Ti_samples * 1000.0
            
            # Calculate all rates using Aurora
            # Reactivity coefficients (rates) – not multiplied by density
            out = aurora.atomic.get_cs_balance_terms(
                atom_data, ne_cm3=ne_cm3, Te_eV=Te_eV, Ti_eV=Ti_eV, include_cx=True
            )
            S_coeff, R_coeff, CX_coeff = out[1], out[2], out[3]

            # Fractional abundances fZ (neutrals.py uses get_frac_abundances)
            # Provide n0_by_ne to avoid singularities; fraction ~1e-2 is a benign choice
            n0_by_ne = abs((3+np.random.default_rng().normal(size=ne_cm3.size))*1e-2)
            _, fZ = aurora.atomic.get_frac_abundances(
                atom_data, ne_cm3=ne_cm3, Te_eV=Te_eV, Ti_eV=Ti_eV,
                n0_by_ne=n0_by_ne, plot=False
            )
            if fZ.ndim == 1:
                fZ = fZ[:, None]  # Ensure 2D

            # Build charge state densities from fZ for radiation calculation
            # Assume uniform ion density across samples for training data generation
            ni_cm3 = ne_cm3.copy()  # Approximate: total ion density ≈ electron density
            n0_cm3 = n0_by_ne * ne_cm3
            # nZ_cm3 shape: (n_samples, n_charge_states) including neutrals
            nZ_cm3 = fZ * (n0_cm3[:, None] + ni_cm3[:, None])

            # Radiation components (targets.AnalyticTargetModel references compute_rad)
            # Returns dict with arrays of shape (n_charge_states, n_samples)
            rad = aurora.radiation.compute_rad(
                self.species_name,
                (nZ_cm3.T)[None,:],
                ne_cm3[None,:],
                Te_eV[None,:],
                n0=n0_cm3[None,:],
                Ti=Ti_eV[None,:],
                prad_flag=True,
                thermal_cx_rad_flag=True,
            ) # radiation has a leading time dimension
            
            # Store training data (rates = reactivity coefficients, not multiplied by density)
            # Radiation arrays have shape (n_charge_states, n_samples)
            n_charge = fZ.shape[1]
            self.training_data = {
                'ne': ne_cm3,      # 1e19/m^-3
                'Te': Te_eV,      # eV
                'Ti': Ti_eV,      # eV
                'S_rates': S_coeff,    # Ionization reactivities [1/s]
                'R_rates': R_coeff,    # Recombination reactivities [1/s]
                'CX_rates': CX_coeff,  # Charge exchange reactivities [1/s]
                'fZ': fZ,              # fractional abundances (N, Zmax+1)
                'rad': {
                    'tot': rad.get('tot', np.zeros((1,n_samples)))[0,:],
                    'line': rad.get('line_rad', np.zeros((1,n_charge, n_samples)))[0,:,:],
                    'cont': rad.get('cont_rad', np.zeros((1,n_charge, n_samples)))[0,:,:],
                    'brems': rad.get('brems', np.zeros((1,n_charge, n_samples)))[0,:,:],
                    'cx': rad.get('thermal_cx_cont_rad', np.zeros((1,n_charge, n_samples)))[0,:,:],
                }, # Radiation components [W/cm^3]
            }

            # Keep raw data for saving separate CSVs
            self._raw_data = {
                'ne': ne_samples,      # 1e19/m^-3
                'Te': Te_samples,      # keV
                'Ti': Ti_samples,      # keV
                'S_rates': S_coeff,
                'R_rates': R_coeff,
                'CX_rates': CX_coeff,
                'fZ': fZ,
                'rad': self.training_data['rad'],
            }
            
            print(f"Generated {n_samples} training points")
            return self.training_data
        
        def fit_polynomials(self, processes=['ionization', 'recombination', 'charge_exchange'], 
                           fit_fZ=True, fit_radiation=True, alpha=1e-6):
            """
            Fit Chebyshev polynomials to atomic rates, fractional abundances, and radiation.
            
            Parameters
            ----------
            processes : list
                Atomic processes to fit: 'ionization', 'recombination', 'charge_exchange'
            fit_fZ : bool
                Whether to fit fractional abundances
            fit_radiation : bool
                Whether to fit radiation components
            alpha : float
                Ridge regression regularization parameter
            """
            print("Fitting polynomial models...")
            
            ne = self.training_data['ne']
            Te = self.training_data['Te']
            Ti = self.training_data['Ti']
            
            # Transform to normalized coordinates [-1, 1] for Chebyshev
            ne_norm = 2 * (np.log10(ne) - np.log10(ne).min()) / (np.log10(ne).max() - np.log10(ne).min()) - 1
            Te_norm = 2 * (np.log10(Te) - np.log10(Te).min()) / (np.log10(Te).max() - np.log10(Te).min()) - 1
            Ti_norm = 2 * (np.log10(Ti) - np.log10(Ti).min()) / (np.log10(Ti).max() - np.log10(Ti).min()) - 1
            
            # Store normalization parameters (same for all processes)
            self.scalers = {
                'ne_min': np.log10(ne).min(),
                'ne_max': np.log10(ne).max(),
                'Te_min': np.log10(Te).min(),
                'Te_max': np.log10(Te).max(),
                'Ti_min': np.log10(Ti).min(),
                'Ti_max': np.log10(Ti).max()
            }
            
            # Map process names to data
            process_data_map = {
                'ionization': self.training_data['S_rates'],
                'recombination': self.training_data['R_rates'],
                'charge_exchange': self.training_data['CX_rates']
            }
            
            # For ionization and recombination, we can use 2D features (ne, Te)
            # For charge exchange, we use 3D features (ne, Te, Ti)
            X_2d = np.column_stack([ne_norm, Te_norm])
            X_3d = np.column_stack([ne_norm, Te_norm, Ti_norm])
            
            for process in processes:
                if process not in process_data_map:
                    continue
                    
                print(f"\nFitting {process} rates...")
                rate_data = process_data_map[process]
                
                # Choose feature matrix based on process
                if process == 'charge_exchange':
                    X_norm = X_3d
                else:
                    X_norm = X_2d
                
                self.coefficients[process] = {}
                self.fit_quality[process] = {}
                
                for charge_state in range(self.Z_atom + 1):
                    if charge_state < rate_data.shape[1]:
                        y = np.log10(rate_data[:, charge_state] + 1e-30)  # Avoid log(0)
                        
                        # Try different polynomial degrees and select best
                        best_score = -np.inf
                        best_model = None
                        best_degree = 1
                        
                        for degree in range(1, self.max_degree + 1):
                            try:
                                # Create Chebyshev polynomial features
                                X_poly = self._chebyshev_features(X_norm, degree)
                                
                                # Fit with Ridge regression for stability
                                model = Ridge(alpha=alpha)
                                model.fit(X_poly, y)
                                
                                # Evaluate
                                y_pred = model.predict(X_poly)
                                score = r2_score(y, y_pred)
                                
                                if score > best_score:
                                    best_score = score
                                    best_model = model
                                    best_degree = degree
                                    
                            except Exception as e:
                                print(f"Warning: Failed to fit degree {degree} for {process} charge {charge_state}: {e}")
                                continue
                        
                        # Store best model
                        if best_model is not None:
                            self.coefficients[process][charge_state] = {
                                'degree': best_degree,
                                'coefficients': best_model.coef_,
                                'intercept': best_model.intercept_,
                                'n_features': X_norm.shape[1]  # 2 for ion/rec, 3 for CX
                            }
                            
                            # Calculate fit quality metrics
                            X_poly_best = self._chebyshev_features(X_norm, best_degree)
                            y_pred_best = best_model.predict(X_poly_best)
                            
                            self.fit_quality[process][charge_state] = {
                                'r2_score': best_score,
                                'mae_percent': mean_absolute_percentage_error(10**y, 10**y_pred_best),
                                'max_error_percent': np.max(np.abs(10**y - 10**y_pred_best)/10**y),
                                'degree_used': best_degree
                            }
                            
                            print(f"  {process} charge {charge_state}: R² = {best_score:.4f}, "
                                f"MAE (percent) = {self.fit_quality[process][charge_state]['mae_percent']:.4f}, "
                                f"degree = {best_degree}")
        
            # Fit fractional abundances (fZ)
            if fit_fZ and 'fZ' in self.training_data:
                print("\nFitting fractional abundance (fZ) polynomials...")
                fZ_data = self.training_data['fZ']
                
                self.coefficients['fZ'] = {}
                self.fit_quality['fZ'] = {}
                
                # Use 3D features (ne, Te, Ti) for fZ
                X_3d = np.column_stack([ne_norm, Te_norm, Ti_norm])
                
                for charge_state in range(fZ_data.shape[1]):
                    # Use logit transform for bounded [0,1] data
                    y_raw = fZ_data[:, charge_state]
                    # Clip to avoid log(0)
                    y_clipped = np.clip(y_raw, 1e-10, 1 - 1e-10)
                    y = np.log(y_clipped / (1 - y_clipped))  # logit transform
                    
                    best_score = -np.inf
                    best_model = None
                    best_degree = 1
                    
                    for degree in range(1, self.max_degree + 1):
                        try:
                            X_poly = self._chebyshev_features(X_3d, degree)
                            model = Ridge(alpha=alpha)
                            model.fit(X_poly, y)
                            
                            y_pred = model.predict(X_poly)
                            score = r2_score(y, y_pred)
                            
                            if score > best_score:
                                best_score = score
                                best_model = model
                                best_degree = degree
                        except Exception as e:
                            continue
                    
                    if best_model is not None:
                        self.coefficients['fZ'][charge_state] = {
                            'degree': best_degree,
                            'coefficients': best_model.coef_,
                            'intercept': best_model.intercept_,
                            'n_features': 3,
                            'transform': 'logit'  # Mark that we used logit transform
                        }
                        
                        X_poly_best = self._chebyshev_features(X_3d, best_degree)
                        y_pred_best = best_model.predict(X_poly_best)
                        # Transform back to [0,1]
                        y_pred_prob = 1 / (1 + np.exp(-y_pred_best))
                        y_prob = 1 / (1 + np.exp(-y))
                        
                        self.fit_quality['fZ'][charge_state] = {
                            'r2_score': best_score,
                            'mae': np.mean(np.abs(y_prob - y_pred_prob)),
                            'max_error': np.max(np.abs(y_prob - y_pred_prob)),
                            'degree_used': best_degree
                        }
                        
                        print(f"  fZ charge {charge_state}: R² = {best_score:.4f}, "
                            f"MAE = {self.fit_quality['fZ'][charge_state]['mae']:.4e}, "
                            f"degree = {best_degree}")
            
            # Fit radiation components
            if fit_radiation and 'rad' in self.training_data:
                print("\nFitting radiation component polynomials...")
                rad_data = self.training_data['rad']
                
                self.coefficients['radiation'] = {}
                self.fit_quality['radiation'] = {}
                
                # Use 3D features (ne, Te, Ti)
                X_3d = np.column_stack([ne_norm, Te_norm, Ti_norm])
                
                # Radiation components to fit
                rad_components = ['tot', 'line', 'cont', 'brems', 'cx']
                
                for component in rad_components:
                    if component not in rad_data:
                        continue
                    
                    component_data = rad_data[component]
                    
                    # Handle different shapes
                    if component == 'tot':
                        # Total radiation: 1D array (n_samples,)
                        y = np.log10(np.abs(component_data) + 1e-30)
                        
                        best_score = -np.inf
                        best_model = None
                        best_degree = 1
                        
                        for degree in range(1, self.max_degree + 1):
                            try:
                                X_poly = self._chebyshev_features(X_3d, degree)
                                model = Ridge(alpha=alpha)
                                model.fit(X_poly, y)
                                
                                y_pred = model.predict(X_poly)
                                score = r2_score(y, y_pred)
                                
                                if score > best_score:
                                    best_score = score
                                    best_model = model
                                    best_degree = degree
                            except Exception:
                                continue
                        
                        if best_model is not None:
                            self.coefficients['radiation'][component] = {
                                'degree': best_degree,
                                'coefficients': best_model.coef_,
                                'intercept': best_model.intercept_,
                                'n_features': 3,
                                'charge_state': None  # Total is not per-charge
                            }
                            
                            X_poly_best = self._chebyshev_features(X_3d, best_degree)
                            y_pred_best = best_model.predict(X_poly_best)
                            
                            self.fit_quality['radiation'][component] = {
                                'r2_score': best_score,
                                'mae_percent': mean_absolute_percentage_error(10**y, 10**y_pred_best),
                                'degree_used': best_degree
                            }
                            
                            print(f"  rad_{component}: R² = {best_score:.4f}, degree = {best_degree}")
                    
                    else:
                        # Per-charge-state radiation: 2D array (n_charge, n_samples)
                        n_charge = component_data.shape[0]
                        
                        if component not in self.coefficients['radiation']:
                            self.coefficients['radiation'][component] = {}
                        if component not in self.fit_quality['radiation']:
                            self.fit_quality['radiation'][component] = {}
                        
                        for z in range(n_charge):
                            y = np.log10(np.abs(component_data[z, :]) + 1e-30)
                            
                            best_score = -np.inf
                            best_model = None
                            best_degree = 1
                            
                            for degree in range(1, self.max_degree + 1):
                                try:
                                    X_poly = self._chebyshev_features(X_3d, degree)
                                    model = Ridge(alpha=alpha)
                                    model.fit(X_poly, y)
                                    
                                    y_pred = model.predict(X_poly)
                                    score = r2_score(y, y_pred)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_model = model
                                        best_degree = degree
                                except Exception:
                                    continue
                            
                            if best_model is not None:
                                self.coefficients['radiation'][component][z] = {
                                    'degree': best_degree,
                                    'coefficients': best_model.coef_,
                                    'intercept': best_model.intercept_,
                                    'n_features': 3,
                                    'charge_state': z
                                }
                                
                                X_poly_best = self._chebyshev_features(X_3d, best_degree)
                                y_pred_best = best_model.predict(X_poly_best)
                                
                                self.fit_quality['radiation'][component][z] = {
                                    'r2_score': best_score,
                                    'mae_percent': mean_absolute_percentage_error(10**y, 10**y_pred_best),
                                    'degree_used': best_degree
                                }
                                
                                print(f"  rad_{component}_z{z}: R² = {best_score:.4f}, degree = {best_degree}")
        
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

        # ---------------------- Saving separate CSVs ----------------------
        def save_raw_rates(self) -> str:
            """Save ionization/recombination/CX reactivities to CSV per species."""
            path = os.path.join(self.output_dir, f"{self.species_name}_rates.csv")
            rows = []
            ne = self._raw_data['ne']; Te = self._raw_data['Te']; Ti = self._raw_data['Ti']
            S = self._raw_data['S_rates']; R = self._raw_data['R_rates']; CX = self._raw_data['CX_rates']
            for i in range(ne.size):
                row = {'ne_1e19_m3': ne[i], 'Te_keV': Te[i], 'Ti_keV': Ti[i]}
                for z in range(S.shape[1]):
                    row[f'S_z{z}'] = S[i, z]
                    row[f'R_z{z}'] = R[i, z]
                    row[f'CX_z{z}'] = CX[i, z]
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            return path

        def save_raw_fZ(self) -> str:
            """Save fractional abundances fZ to CSV per species."""
            path = os.path.join(self.output_dir, f"{self.species_name}_fZ.csv")
            rows = []
            ne = self._raw_data['ne']; Te = self._raw_data['Te']; Ti = self._raw_data['Ti']
            fZ = self._raw_data['fZ']
            for i in range(ne.size):
                row = {'ne_1e19_m3': ne[i], 'Te_keV': Te[i], 'Ti_keV': Ti[i]}
                for z in range(fZ.shape[1]):
                    row[f'fZ_z{z}'] = fZ[i, z]
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            return path

        def save_raw_radiation(self) -> str:
            """Save radiation components to CSV per species.
            Each charge state gets its own columns: rad_tot_z0, rad_tot_z1, etc.
            """
            path = os.path.join(self.output_dir, f"{self.species_name}_radiation.csv")
            rad = self._raw_data['rad']
            rows = []
            ne = self._raw_data['ne']; Te = self._raw_data['Te']; Ti = self._raw_data['Ti']
            
            # Get number of charge states from radiation data shape
            rad_tot = rad.get('tot', rad.get('line', None))
            rad_line = rad.get('line', None)
            if rad_tot is None:
                return path  # No radiation data to save
            
            n_charge = rad_line.shape[0]  # (n_charge, n_samples)
            n_samples = rad_tot.shape[0]
            
            for i in range(n_samples):
                row = {'ne_1e19_m3': ne[i], 'Te_keV': Te[i], 'Ti_keV': Ti[i]}
                row[f'rad_tot'] = rad.get('tot', np.zeros((n_samples)))[i]
                for z in range(n_charge):
                    row[f'rad_line_z{z}'] = rad.get('line', np.zeros((n_charge, n_samples)))[z, i]
                    row[f'rad_cont_z{z}'] = rad.get('cont', np.zeros((n_charge, n_samples)))[z, i]
                    row[f'rad_brems_z{z}'] = rad.get('brems', np.zeros((n_charge, n_samples)))[z, i]
                    row[f'rad_cx_z{z}'] = rad.get('cx', np.zeros((n_charge, n_samples)))[z, i]
                rows.append(row)
            
            pd.DataFrame(rows).to_csv(path, index=False)
            return path
        
        def save_coefficients(self, output_dir='atomic_data', master_file='atomic_rate_polynomials.csv'):
            """Save all polynomial coefficients to a single master CSV file with scalers included."""
            os.makedirs(output_dir, exist_ok=True)
            
            master_filepath = os.path.join(output_dir, master_file)
            
            # Load existing data if file exists
            if os.path.exists(master_filepath):
                existing_df = pd.read_csv(master_filepath)
                print(f"Loading existing data from {master_filepath}")
            else:
                existing_df = pd.DataFrame()
            
            # First pass: determine maximum number of coefficients needed
            max_coeffs = 0
            for process, process_coeffs in self.coefficients.items():
                if process == 'radiation':
                    for component, component_data in process_coeffs.items():
                        if isinstance(component_data, dict) and 'coefficients' in component_data:
                            # Total radiation (not per-charge)
                            coeff_info = component_data
                            max_coeffs = max(max_coeffs, len(coeff_info['coefficients']))
                        else:
                            # Per-charge-state radiation
                            for charge_state, coeff_info in component_data.items():
                                if not isinstance(coeff_info, dict) or 'coefficients' not in coeff_info:
                                    continue
                                max_coeffs = max(max_coeffs, len(coeff_info['coefficients']))
                    continue
                else:
                    for charge_state, coeff_info in process_coeffs.items():
                        max_coeffs = max(max_coeffs, len(coeff_info['coefficients']))
            
            # Also check existing data for max coefficients
            if not existing_df.empty:
                existing_coeff_cols = [col for col in existing_df.columns if col.startswith('coeff_')]
                if existing_coeff_cols:
                    existing_max = max([int(col.split('_')[1]) for col in existing_coeff_cols]) + 1
                    max_coeffs = max(max_coeffs, existing_max)
            
            print(f"Maximum coefficients needed: {max_coeffs}")
            
            # Prepare new data with organized column order
            new_data = []
            
            # Add coefficients for all processes (rates, fZ, radiation)
            for process, process_coeffs in self.coefficients.items():
                # Handle nested structure for radiation (component -> charge_state)
                if process == 'radiation':
                    for component, component_data in process_coeffs.items():
                        if isinstance(component_data, dict) and 'coefficients' in component_data:
                            # Total radiation (not per-charge)
                            coeff_info = component_data
                            row = {
                                'species': self.species_name,
                                'process': f'radiation_{component}',
                                'initial_charge': -1,
                                'final_charge': -1,
                                'charge_state': -1,
                                'degree': coeff_info['degree'],
                                'n_features': coeff_info['n_features'],
                                'ne_min': self.scalers['ne_min'],
                                'ne_max': self.scalers['ne_max'],
                                'Te_min': self.scalers['Te_min'],
                                'Te_max': self.scalers['Te_max'],
                                'Ti_min': self.scalers['Ti_min'],
                                'Ti_max': self.scalers['Ti_max'],
                                'intercept': coeff_info['intercept']
                            }
                            for i in range(max_coeffs):
                                if i < len(coeff_info['coefficients']):
                                    row[f'coeff_{i}'] = coeff_info['coefficients'][i]
                                else:
                                    row[f'coeff_{i}'] = np.nan
                            
                            if component in self.fit_quality.get('radiation', {}):
                                quality = self.fit_quality['radiation'][component]
                                row['r2_score'] = quality.get('r2_score', np.nan)
                                row['mae_percent'] = quality.get('mae_percent', np.nan)
                                row['max_error_percent'] = np.nan
                            else:
                                row['r2_score'] = np.nan
                                row['mae_percent'] = np.nan
                                row['max_error_percent'] = np.nan
                            
                            new_data.append(row)
                        else:
                            # Per-charge-state radiation
                            for charge_state, coeff_info in component_data.items():
                                if not isinstance(coeff_info, dict) or 'coefficients' not in coeff_info:
                                    continue
                                row = {
                                    'species': self.species_name,
                                    'process': f'radiation_{component}',
                                    'initial_charge': charge_state,
                                    'final_charge': charge_state,
                                    'charge_state': charge_state,
                                    'degree': coeff_info['degree'],
                                    'n_features': coeff_info['n_features'],
                                    'ne_min': self.scalers['ne_min'],
                                    'ne_max': self.scalers['ne_max'],
                                    'Te_min': self.scalers['Te_min'],
                                    'Te_max': self.scalers['Te_max'],
                                    'Ti_min': self.scalers['Ti_min'],
                                    'Ti_max': self.scalers['Ti_max'],
                                    'intercept': coeff_info['intercept']
                                }
                                for i in range(max_coeffs):
                                    if i < len(coeff_info['coefficients']):
                                        row[f'coeff_{i}'] = coeff_info['coefficients'][i]
                                    else:
                                        row[f'coeff_{i}'] = np.nan
                                
                                if component in self.fit_quality.get('radiation', {}) and charge_state in self.fit_quality['radiation'][component]:
                                    quality = self.fit_quality['radiation'][component][charge_state]
                                    row['r2_score'] = quality.get('r2_score', np.nan)
                                    row['mae_percent'] = quality.get('mae_percent', np.nan)
                                    row['max_error_percent'] = np.nan
                                else:
                                    row['r2_score'] = np.nan
                                    row['mae_percent'] = np.nan
                                    row['max_error_percent'] = np.nan
                                
                                new_data.append(row)
                    continue
                
                # Handle fZ with special transform flag
                if process == 'fZ':
                    for charge_state, coeff_info in process_coeffs.items():
                        row = {
                            'species': self.species_name,
                            'process': 'fZ',
                            'initial_charge': charge_state,
                            'final_charge': charge_state,
                            'charge_state': charge_state,
                            'degree': coeff_info['degree'],
                            'n_features': coeff_info['n_features'],
                            'ne_min': self.scalers['ne_min'],
                            'ne_max': self.scalers['ne_max'],
                            'Te_min': self.scalers['Te_min'],
                            'Te_max': self.scalers['Te_max'],
                            'Ti_min': self.scalers['Ti_min'],
                            'Ti_max': self.scalers['Ti_max'],
                            'intercept': coeff_info['intercept']
                        }
                        for i in range(max_coeffs):
                            if i < len(coeff_info['coefficients']):
                                row[f'coeff_{i}'] = coeff_info['coefficients'][i]
                            else:
                                row[f'coeff_{i}'] = np.nan
                        
                        if charge_state in self.fit_quality.get('fZ', {}):
                            quality = self.fit_quality['fZ'][charge_state]
                            row['r2_score'] = quality.get('r2_score', np.nan)
                            row['mae_percent'] = quality.get('mae', np.nan) * 100  # Convert to percent
                            row['max_error_percent'] = quality.get('max_error', np.nan) * 100
                        else:
                            row['r2_score'] = np.nan
                            row['mae_percent'] = np.nan
                            row['max_error_percent'] = np.nan
                        
                        new_data.append(row)
                    continue
                
                # Handle regular atomic rates
                for charge_state, coeff_info in process_coeffs.items():
                    # Create descriptive information
                    if process == 'ionization':
                        initial_charge = charge_state
                        final_charge = charge_state + 1
                    elif process == 'recombination':
                        initial_charge = charge_state + 1
                        final_charge = charge_state
                    else:  # charge_exchange
                        initial_charge = charge_state
                        final_charge = charge_state
                    
                    # Start with metadata columns including scalers
                    row = {
                        'species': self.species_name,
                        'process': process,
                        'initial_charge': initial_charge,
                        'final_charge': final_charge,
                        'charge_state': charge_state,
                        'degree': coeff_info['degree'],
                        'n_features': coeff_info['n_features'],
                        
                        # Add scaler parameters here
                        'ne_min': self.scalers['ne_min'],
                        'ne_max': self.scalers['ne_max'],
                        'Te_min': self.scalers['Te_min'],
                        'Te_max': self.scalers['Te_max'],
                        'Ti_min': self.scalers['Ti_min'],
                        'Ti_max': self.scalers['Ti_max'],
                        
                        'intercept': coeff_info['intercept']
                    }
                    
                    # Add ALL coefficient columns (fill with NaN if not present)
                    for i in range(max_coeffs):
                        if i < len(coeff_info['coefficients']):
                            row[f'coeff_{i}'] = coeff_info['coefficients'][i]
                        else:
                            row[f'coeff_{i}'] = np.nan
                    
                    # Add fit quality metrics at the end
                    if process in self.fit_quality and charge_state in self.fit_quality[process]:
                        quality = self.fit_quality[process][charge_state]
                        row['r2_score'] = quality['r2_score']
                        row['mae_percent'] = quality['mae_percent']
                        row['max_error_percent'] = quality['max_error_percent']
                    else:
                        row['r2_score'] = np.nan
                        row['mae_percent'] = np.nan
                        row['max_error_percent'] = np.nan
                    
                    new_data.append(row)
            
            new_df = pd.DataFrame(new_data)
            
            # Define the desired column order
            metadata_cols = ['species', 'process', 'initial_charge', 'final_charge', 'charge_state', 
                            'degree', 'n_features']
            scaler_cols = ['ne_min', 'ne_max', 'Te_min', 'Te_max', 'Ti_min', 'Ti_max']
            other_metadata_cols = ['intercept']
            coeff_cols = [f'coeff_{i}' for i in range(max_coeffs)]
            quality_cols = ['r2_score', 'mae_percent', 'max_error_percent']
            
            desired_order = metadata_cols + scaler_cols + other_metadata_cols + coeff_cols + quality_cols
            
            # Reorder new dataframe columns
            new_df = new_df.reindex(columns=desired_order)
            
            # Handle existing data
            if not existing_df.empty:
                # Remove existing entries for this species to avoid duplicates
                existing_df = existing_df[existing_df['species'] != self.species_name]
                
                # Ensure existing dataframe has all necessary columns
                for col in desired_order:
                    if col not in existing_df.columns:
                        existing_df[col] = np.nan
                
                # Reorder existing dataframe columns to match
                existing_df = existing_df.reindex(columns=desired_order)
            
            # Combine and save
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Sort for better organization
            combined_df = combined_df.sort_values(['species', 'process', 'initial_charge'])
            combined_df.to_csv(master_filepath, index=False)
            
            print(f"Saved polynomial data to {master_filepath}")
            print(f"Column order: {list(combined_df.columns)}")
            
            return master_filepath

# Example usage
if __name__ == "__main__":
    
    # Fit atomic rate polynomials for multiple species
    species_list = ['H','D','He','Li','C','N','Ne','Ar']
    
    for species in species_list:
        print(f"\n{'='*50}")
        print(f"Processing {species}")
        print(f"{'='*50}")
        
        fitter = AuroraPolyFitter(species, max_degree=2, output_dir='atomic_data')
        training_data = fitter.generate_training_data(n_samples=3000)
        # Save separate raw datasets for interpolation fallback
        print("Saving raw Aurora datasets...")
        print(f"Rates -> {fitter.save_raw_rates()}")
        print(f"fZ    -> {fitter.save_raw_fZ()}")
        print(f"Rad   -> {fitter.save_raw_radiation()}")
        # Fit polynomials for rates, fZ, and radiation
        fitter.fit_polynomials(processes=['ionization', 'recombination', 'charge_exchange'],
                              fit_fZ=True, fit_radiation=True)
        fitter.save_coefficients()