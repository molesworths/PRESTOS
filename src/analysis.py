"""Reporting and plotting utilities for MinT solver runs.

Usage:
    python -m analysis.reporting --config analysis/plot_config.yaml

Figures produced depend on analysis_level in config:
  minimal  -> objective, initial vs final profiles, final targets
  standard -> minimal + residual channels, surrogate PCA
  full     -> standard + surrogate sensitivities

Assumptions:
  - solver_history.csv produced by SolverData.save contains columns:
      iter, Z|J (objective), R (residual array), X (parameter dict), Y (model dict), Y_target (target dict), used_surrogate (bool)
  - Dict-like columns stored as string representations evaluable by ast.literal_eval.
  - setup.yaml available if surrogate sensitivity requires reconstructing modules.

Surrogate sensitivity (full) builds a simple dataset from iterations (flattened parameters) and fits PCA.
If original SurrogateManager training artifacts are not saved, we approximate using history.
"""
from __future__ import annotations
import os, json, yaml, ast, argparse, importlib, pickle
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val

@dataclass
class HistoryData:
    df: pd.DataFrame
    iterations: np.ndarray
    objective: np.ndarray
    used_surrogate: np.ndarray
    X_list: List[Dict[str, Dict[str, float]]]
    R_list: List[np.ndarray]
    Y_list: List[Dict[str, np.ndarray]]
    Y_target_list: List[Dict[str, np.ndarray]]


def load_history(path: str) -> HistoryData:
    df = pd.read_csv(path)
    # Normalize objective column name
    obj_col = 'Z' if 'Z' in df.columns else ('J' if 'J' in df.columns else None)
    if obj_col is None:
        raise ValueError('Objective column (Z or J) not found in history CSV.')
    iterations = df['iter'].to_numpy()
    objective = df[obj_col].to_numpy()
    used_surrogate = df.get('used_surrogate', pd.Series([False]*len(df))).astype(bool).to_numpy()

    X_list = []
    R_list = []
    Y_list = []
    Y_target_list = []
    for _, row in df.iterrows():
        X_list.append(_safe_eval(row.get('X', '{}')))
        R_raw = _safe_eval(row.get('R', '[]'))
        R_list.append(np.asarray(R_raw, dtype=float) if isinstance(R_raw, (list, tuple, np.ndarray)) else np.array([]))
        Y_list.append(_safe_eval(row.get('Y', '{}')))
        Y_target_list.append(_safe_eval(row.get('Y_target', '{}')))

    return df


# ---------------------------------------------------------------------------
# Module reconstruction from solver checkpoint
# ---------------------------------------------------------------------------

def rebuild_modules_from_checkpoint(checkpoint_path: str, strict: bool = False) -> Dict[str, Any]:
    """Rebuild solver modules (solver, state, parameters, transport, targets, boundary,
    neutrals, surrogate) from a checkpoint produced by SolverBase.save.

    Parameters
    ----------
    checkpoint_path : str
        Path to pickle file created by `SolverBase.save` (e.g., solver_checkpoint.pkl).
    strict : bool, default False
        When True, raise exceptions on reconstruction failures. When False, continue and
        record errors.

    Returns
    -------
    modules : Dict[str, Any]
        Dictionary keyed by module role name with reconstructed objects. An 'errors' key
        may be present containing any reconstruction issues.

    Notes
    -----
    The checkpoint stores a lightweight spec:
      { name: { "class_path": "package.module.Class", "attributes": { ... } } }
    Only public (non "_" prefixed) attributes are stored. Internal runtime state, caches,
    or properties that depend on external resources may need manual reinitialization.

    Reconstruction strategy:
      1. Import class by path.
      2. Attempt to instantiate with no args. If that fails, allocate via __new__ without __init__.
      3. Populate public attributes from stored spec (shallow assignment).
      4. Collect errors but proceed unless strict=True.
    """
    checkpoint_path = os.path.expanduser(os.path.expandvars(str(checkpoint_path)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as fh:
        module_specs = pickle.load(fh)

    modules: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for name, spec in module_specs.items():
        if name == 'timestamp':
            continue
        try:
            class_path = spec.get('class_path')
            if not class_path:
                raise ValueError("Missing class_path in spec")
            mod_name, cls_name = class_path.rsplit('.', 1)
            cls = getattr(importlib.import_module(mod_name), cls_name)
            # Try normal instantiation first
            try:
                obj = cls()
            except Exception:
                # Fallback: allocate without calling __init__
                obj = cls.__new__(cls)
            attrs = spec.get('attributes', {})
            for attr, stored in attrs.items():
                if attr.startswith('__'):
                    continue
                # stored expected as (value, type_name)
                if isinstance(stored, tuple) and len(stored) == 2 and isinstance(stored[1], str):
                    raw_val, type_name = stored
                else:
                    raw_val, type_name = stored, None

                try:
                    if type_name == 'ndarray':
                        val = np.asarray(raw_val)
                    elif type_name == 'list':
                        val = list(raw_val)
                    elif type_name == 'tuple':
                        val = tuple(raw_val)
                    elif type_name == 'set':
                        val = set(raw_val)
                    elif type_name == 'dict':
                        val = dict(raw_val)
                    elif type_name == 'int':
                        val = int(raw_val)
                    elif type_name == 'float':
                        val = float(raw_val)
                    elif type_name == 'bool':
                        # Accept "True"/"False" strings or numeric 0/1
                        if isinstance(raw_val, str):
                            val = raw_val.lower() == 'true'
                        else:
                            val = bool(raw_val)
                    elif type_name == 'str':
                        val = str(raw_val)
                    else:
                        # Fallback: keep raw_val
                        val = raw_val
                    setattr(obj, attr, val)
                except Exception:
                    errors[f"{name}.{attr}"] = f"Failed to set attribute (type {type_name})"
            modules[name] = obj
        except Exception as e:
            errors[name] = str(e)
            if strict:
                raise

    # convert lists to arrays in state

    if errors:
        modules['errors'] = errors
    return modules

# --------------------- Module helpers ---------------------

def _update_from_params(modules, X: np.ndarray):
    X = _unflatten_params(modules['solver'], X, modules['solver'].schema) if isinstance(X, (np.ndarray, list)) else X
    if not hasattr(modules['parameters'],'spline'):
        pass
    modules['state'].update(X, modules['parameters'])
    modules['neutrals'].solve(modules['state'])
    _ = modules['targets'].evaluate(modules['state'])
    modules['boundary'].get_boundary_conditions(modules['state'], modules['targets'])
    modules['parameters'].update(X, modules['boundary'].bc_dict, modules['solver'].roa_eval)
    modules['state'].update(X, modules['parameters'])
    modules['neutrals'].solve(modules['state'])
    _ = modules['targets'].evaluate(modules['state'])
    
def _flatten_params(self, X_dict: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Convert dict-of-dicts parameters to flat array with schema.
    
    Returns
    -------
    X_flat : np.ndarray
        Flattened parameter vector
    schema : List[Tuple[str, str]]
        List of (profile, param_name) tuples defining the order
    """
    if not hasattr(self, 'schema'):
        schema = []
        values = []
        for prof in sorted(X_dict.keys()):
            param_dict = X_dict[prof]
            for pname in sorted(param_dict.keys()):
                schema.append((prof, pname))
                values.append(float(param_dict[pname]))
        self.schema = schema
    else:
        schema = self.schema
        values = []
        for prof, pname in schema:
            values.append(float(X_dict[prof][pname]))

    return np.array(values, dtype=float), schema

def _unflatten_params(self, X_flat: np.ndarray, schema: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    """Convert flat array back to dict-of-dicts using schema.
    
    Parameters
    ----------
    X_flat : np.ndarray
        Flattened parameter vector
    schema : List[Tuple[str, str]]
        List of (profile, param_name) tuples defining the order
        
    Returns
    -------
    X_dict : Dict[str, Dict[str, float]]
        Reconstructed dict-of-dicts parameters
    """
    X_dict = {}
    for i, (prof, pname) in enumerate(schema):
        if prof not in X_dict:
            X_dict[prof] = {}
        X_dict[prof][pname] = float(X_flat[i])
    return X_dict

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _init_style(style_cfg: Dict[str, Any]):
    plt.rcParams.update({
        'figure.dpi': style_cfg.get('dpi', 120),
        'font.size': 10 * style_cfg.get('font_scale', 1.0),
        'axes.titlesize': 11 * style_cfg.get('font_scale', 1.0),
        'axes.labelsize': 10 * style_cfg.get('font_scale', 1.0),
        'legend.fontsize': 9 * style_cfg.get('font_scale', 1.0),
    })

def _subplots_adjust(fig, cfg: Dict[str, Any]):
    borders = cfg.get('borders', {})
    fig.subplots_adjust(
        left=borders.get('left', 0.1),
        right=borders.get('right', 0.92),
        top=borders.get('top', 0.9),
        bottom=borders.get('bottom', 0.1),
        wspace=cfg.get('wspace', 0.25),
        hspace=cfg.get('hspace', 0.25)
    )


def plot_objective(hist_df: pd.DataFrame, cfg: Dict[str, Any], outdir: Path):
    fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_small', (6,4))))
    _subplots_adjust(fig, cfg['style'])

    ax.scatter(hist_df['iter'][~hist_df['used_surrogate']], hist_df['Z'][~hist_df['used_surrogate']], label='model')
    ax.scatter(hist_df['iter'][hist_df['used_surrogate']], hist_df['Z'][hist_df['used_surrogate']], label='surrogate')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    ax.set_title('Objective vs Iteration')
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir/'objective')


def plot_residual_channels(hist_df: pd.DataFrame, modules, cfg: Dict[str, Any], outdir: Path):
    # Decide number of channels to visualize
    n_channels = len(modules['solver'].R)
    x = modules['solver'].roa_eval
    n_x = len(x)
    target_vars = modules['solver'].target_vars

    fig, ax = plt.subplots(1,len(target_vars),figsize=tuple(cfg['style'].get('figsize_medium', (5,7))))
    if len(target_vars) == 1:
        ax = [ax]
    _subplots_adjust(fig, cfg['style'])

    for ix,key in enumerate(target_vars):
        for j in range(len(x)):
            ax[ix].plot(hist_df['iter'], hist_df[f"R_{j+ix*n_x}"], marker='o', ms=3, label=f'{key}, r/a={round(x[j],2)}')
        ax[ix].set_xlabel('Iteration')
        ax[ix].set_title(f'{key} Residuals')
        ax[ix].legend(fontsize=6)

    _save(fig, outdir/'residual_channels')


def plot_profiles(modules, hist_df: pd.DataFrame, cfg: Dict[str, Any], outdir: Path):

    # Extract initial and final profiles from parameters module
    Xi = hist_df[[i for i in hist_df.columns if i[:2]=='X_' and 'std' not in i]].iloc[0].values
    Xf = hist_df[[i for i in hist_df.columns if i[:2]=='X_' and 'std' not in i]].iloc[-1].values
    Xf_std = hist_df[[i for i in hist_df.columns if i[:2]=='X_' and 'std' in i]].iloc[-1].values
    X_ = [Xi,Xf]

    pred_prof = modules['solver'].predicted_profiles
    domain = modules['solver'].domain
    x_eval = modules['solver'].roa_eval
    x = modules['state'].roa
    plot_x = np.union1d(x, x_eval)
    eval_ix = np.searchsorted(plot_x, x_eval)

    # Plot initial vs final for each predicted profile
    fig, ax = plt.subplots(1,len(pred_prof),figsize=tuple(cfg['style'].get('figsize_wide', (10,5))))
    if len(pred_prof) == 1:
        ax = [ax]
    _subplots_adjust(fig, cfg['style'])

    for ix, X in enumerate(X_):
        _update_from_params(modules, X)
        if ix == 0:
            label = 'initial'
            style = 'o--'
            X_std = None
        else:
            label = 'final'
            style = 's-'
            X_std = Xf_std

        for prof, ax_i in zip(pred_prof, ax):
            profile_vals = getattr(modules['state'], prof)
            plot_vals = np.interp(plot_x, x, profile_vals)
            ax_i.plot(plot_x, plot_vals, style, markevery=eval_ix, label=label)
            if X_std is not None:
                _update_from_params(modules, X-X_std)
                profile_vals_lb = getattr(modules['state'], prof)
                _update_from_params(modules, X+X_std)
                profile_vals_ub = getattr(modules['state'], prof)
                ax_i.fill_between(x, profile_vals_lb, profile_vals_ub, color='C1', alpha=0.2, label=f'1 std')
                _update_from_params(modules,X) # restore state
            ax_i.set_xlabel('roa')
            ax_i.set_title(f'{prof}')
            ax_i.legend(fontsize=8)
            ax_i.set_xlim(domain[0],domain[1])

    fig.suptitle('Profile Parameters: Initial vs Final')
    fig.tight_layout()
    _save(fig, outdir/f'profiles')


def plot_power_flows(hist_df: pd.DataFrame, modules, cfg: Dict[str, Any], outdir: Path):

    domain = modules['solver'].domain
    x = modules['solver'].roa_eval[:-1]
    Yf = hist_df[[i for i in hist_df.columns if 'model_' in i and 'std' not in i]].iloc[-1][:-1]
    Yf_std = hist_df[[i for i in hist_df.columns if 'model_' in i and 'std' in i]].iloc[-1][:-1]
    Ytf = hist_df[[i for i in hist_df.columns if 'target_' in i and 'std' not in i]].iloc[-1][:-1]
    Ytf_std = hist_df[[i for i in hist_df.columns if 'target_' in i and 'std' in i]].iloc[-1][:-1]

    transport_components = list(set([str.split(i,'_')[-1] for i in modules['solver'].transport_vars]))
    target_vars = modules['solver'].target_vars

    # Color code for transport components
    colors = plt.get_cmap('tab10', len(transport_components))

    # Plot final power flows for each predicted profile
    fig, ax = plt.subplots(1,len(target_vars),figsize=tuple(cfg['style'].get('figsize_wide', (10,5))))
    if len(target_vars) == 1:
        ax = [ax]
    _subplots_adjust(fig, cfg['style'])

    for ix,key in enumerate(target_vars):
        axi = ax[ix]
        y = np.asarray([Yf['model_'+key+'_'+str(i)] for i in range(len(x))], dtype=float)
        y_std = np.asarray([Yf_std['model_'+key+'_std_'+str(i)] for i in range(len(x))], dtype=float)
        yt = np.asarray([Ytf['target_'+key+'_'+str(i)] for i in range(len(x))], dtype=float)
        yt_std = np.asarray([Ytf_std['target_'+key+'_std_'+str(i)] for i in range(len(x))], dtype=float)
        axi.plot(x, yt, '-', label=f"target {key}", marker='o', linewidth=2)
        axi.plot(x, y, '--', label=f"model {key}", marker='s', linewidth=2)
        axi.fill_between(x, y-y_std, y+y_std, color='C1', alpha=0.2, label='model 1 std')

        for c in transport_components:
            yc = np.asarray([Yf['model_'+key+'_'+c+'_'+str(i)] for i in range(len(x))], dtype=float)
            axi.plot(x, yc, ':', label=f"model {c}", color=colors(transport_components.index(c)))

        axi.set_xlim(domain[0],domain[1])
        axi.set_xlabel('r/a')
        axi.set_title(key)
        axi.legend(fontsize=8)
        axi.grid(alpha=0.3)

    fig.suptitle('Final Power Flows')
    fig.tight_layout()
    _save(fig, outdir/f'power_flows')

# ---------------------------------------------------------------------------
# Surrogate sensitivity & PCA using history only
# ---------------------------------------------------------------------------

def _flatten_param_dict(X_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat = {}
    for prof, params in (X_dict or {}).items():
        for pname, val in (params or {}).items():
            flat[f"{prof}:{pname}"] = float(val)
    return flat


def surrogate_heatmap(modules, cfg: Dict[str, Any], outdir: Path):
    """Generate surrogate feature importance heat map(s).

    Importance metric: inverse RBF length-scales (1 / l_rbf) extracted from
    modules['surrogate'].hyperparameters per output.

    Global mode: single heat map of shape (n_features x n_outputs).
    Local mode: one heat map per evaluation point (len(roa_eval)), each using
    l_rbf lists indexed by evaluation point.

    Returns
    -------
    top_features : List[str]
        Names of most important features aggregated across outputs (and space).
    importance_dict : Dict[str, np.ndarray]
        For global mode: {'global': importance_matrix}.
        For local mode: {'local': [importance_matrix_at_i for i]}
    """
    surrogate = modules.get('surrogate')
    if surrogate is None:
        return [], {}

    features = surrogate.all_features  # length n_features
    outputs = surrogate.output_list    # length n_outputs
    mode = getattr(surrogate, 'mode', 'global')
    hyper = getattr(surrogate, 'hyperparameters', {})
    models = getattr(surrogate, 'models', {})
    n_eval = len(getattr(modules['solver'], 'roa_eval', []))
    n_features = len(features)
    n_outputs = len(outputs)

    # Reconstruct models if needed
    if not models or all(m is None for m in models.values()):
        from src import surrogates  # Adjust import path as needed
        reconstructed_models = {}
        for out in outputs:
            try:
                # Try to get class name from existing model string representation
                model_str = str(models.get(out, ''))
                # Extract class name from string like "<surrogates.GaussianProcessSurrogate object at 0x...>"
                if '<' in model_str and 'object at' in model_str:
                    class_path = model_str.split('<')[1].split(' object')[0]
                # class_path should be like "surrogates.GaussianProcessSurrogate"
                if '.' in class_path:
                    module_part, class_name = class_path.rsplit('.', 1)
                else:
                    class_name = class_path
                
                # Get the class from the surrogates module
                model_class = getattr(surrogates, class_name, None)
                if model_class == 'surrogates.GaussianProcessSurrogate':
                    # Build config dict from hyperparameters
                    config = {}
                    if 'C' in hyper[out]:
                        config['variance'] = hyper[out]['C']
                    if 'l_rbf' in hyper[out]:
                        config['length'] = hyper[out]['l_rbf']
                    
                    # Instantiate with config
                    model_instance = model_class(config=config)
                    reconstructed_models[out] = model_instance
            except Exception as e:
                print(f"Failed to reconstruct model for {out}: {e}")
        if reconstructed_models:
            models = reconstructed_models

    # Build importance matrices
    importances, top_features = feature_importance(modules, cfg, outdir)

    # Plotting
    cmap = plt.get_cmap(cfg.get('style', {}).get('heatmap_cmap', 'viridis'))
    fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_medium', (7,5))))
    _subplots_adjust(fig, cfg['style'])

    output_labels = [f"target_{out}" if out in surrogate.target_vars else f"model_{out}" for out in outputs]
    
    im = ax.imshow(importances, aspect='auto', cmap=cmap)
    ax.set_yticks(np.arange(n_outputs))
    ax.set_yticklabels(output_labels, fontsize=8)
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    ax.set_title('Surrogate Feature Importance')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, outdir/'surrogate_heatmap')

    print(f"Top features: {top_features}")

    return top_features


def feature_importance(modules, cfg: Dict[str, Any], outdir: Path) -> List[str]:

    """Compute top features hierarchically:
    1) Try permutation_importance on rebuilt surrogate models
    2) Else mutual_info_regression on samples
    3) Else Pearson correlation on samples

    Normalizes per-output importances to sum to 1.
    Handles global vs local based on surrogate.mode and shapes of X/Y.

    currently limited to gaussian process regressors for rebuilding.
    """

    surrogate = modules.get('surrogate')
    if surrogate is None:
        return

    # Attempt to rebuild models (if missing) using hyperparameters
    models = getattr(surrogate, 'models', {}) or {}
    outputs = getattr(surrogate, 'output_list', [])
    features = getattr(surrogate, 'all_features', [])
    mode = getattr(surrogate, 'mode', 'global')
    hyper = getattr(surrogate, 'hyperparameters', {})

    # Data arrays; accept either *_train or *_samples naming
    X = getattr(surrogate, 'X_train', None)
    Y = getattr(surrogate, 'Y_train', None)

    n_samples = len(X)
    n_roa = len(surrogate.roa_eval)
    n_features = len(surrogate.all_features)
    n_outputs = len(surrogate.output_list)

    # Reconstruct numpy arrays from lists of dicts
    X_all_samples = np.zeros((n_samples, n_roa, n_features))
    Y_all_samples = np.zeros((n_samples, n_roa, n_outputs))

    for s_idx, sample_dict in enumerate(surrogate.X_train):
        for f_idx, feature in enumerate(surrogate.all_features):
            X_all_samples[s_idx, :, f_idx] = sample_dict[feature]

    for s_idx, sample_dict in enumerate(surrogate.Y_train):
        for o_idx, output in enumerate(surrogate.output_list):
            Y_all_samples[s_idx, :, o_idx] = sample_dict[output]
    
    # Flatten for fitting: reshape to (n_samples * n_roa, n_features) and (n_samples * n_roa, n_outputs)
    X_flat = X_all_samples.reshape(-1, n_features)
    Y_flat = Y_all_samples.reshape(-1, n_outputs)

    # Try permutation importance using rebuilt models
    # Final importance shape: (n_outputs, n_features)
    importance = None
    # try:
    #     from sklearn.gaussian_process import GaussianProcessRegressor
    #     rebuilt = {}
    #     for j, out in enumerate(outputs):
    #         gpr = GaussianProcessRegressor()
    #         rebuilt[out] = gpr
        
    #     importance = np.zeros((n_outputs, n_features))
    #     for j, out in enumerate(outputs):
    #         y_j = Y_flat[:, j]
    #         # Fit and compute permutation importance
    #         rebuilt[out].fit(X_flat, y_j)
    #         pi = permutation_importance(rebuilt[out], X_flat, y_j, n_repeats=5, random_state=0)
    #         importance[j, :] = pi.importances_mean
    # except Exception:
    #     pass

    # If permutation importance fails, try mutual information
    if importance is None:
        try:
            importance = np.zeros((n_outputs, n_features))
            for j in range(n_outputs):
                y_j = Y_flat[:, j]
                mi = mutual_info_regression(X_flat, y_j, random_state=0)
                importance[j, :] = mi
        except Exception:
            importance = None

    # Fallback to Pearson correlation
    if importance is None:
        importance = np.zeros((n_outputs, n_features))
        for j in range(n_outputs):
            y_j = Y_flat[:, j]
            for i in range(n_features):
                try:
                    r, _ = pearsonr(X_flat[:, i], y_j)
                    importance[j, i] = abs(r)
                except Exception:
                    importance[j, i] = 0.0

    # Normalizer per output (max to 1)
    importance = importance / np.maximum(importance.max(axis=1, keepdims=True), 1e-8)

    # Normalize per output (each row sums to 1)
    #for j in range(n_outputs):
        # s = importance[j, :].sum()
        # if s > 0:
        #     importance[j, :] = importance[j, :] / s

    # Select top features based on mean importance across outputs
    n_select = cfg.get('sensitivity', {}).get('n_features', 10)
    if n_select == 'all' or n_select is None:
        top_idx = np.arange(n_features)
    else:
        n_select = int(min(n_features, n_select))
        feature_scores = importance.mean(axis=0)  # (n_features,)
        top_idx = np.argsort(-feature_scores)[:n_select]
    top_features = [features[i] for i in top_idx]

    return importance, top_features

def surrogate_sensitivity(modules, top_features: List[str], cfg: Dict[str, Any], outdir: Path):
    """Scatter plots of selected most-important features vs surrogate outputs.

    Colors distinguish evaluation (roa) index.
    Data source: modules['surrogate'].X_train, Y_train arrays of shape
      X: (n_samples, n_eval, n_features)
      Y: (n_samples, n_eval, n_outputs)
    """
    surrogate = modules.get('surrogate')
    if surrogate is None:
        return
    
    roa_eval = surrogate.roa_eval
    # Data arrays; accept either *_train or *_samples naming
    X = getattr(surrogate, 'X_train', None)
    Y = getattr(surrogate, 'Y_train', None)

    n_samples = len(X)
    n_roa = len(surrogate.roa_eval)
    n_features = len(surrogate.all_features)
    n_outputs = len(surrogate.output_list)

    # Reconstruct numpy arrays from lists of dicts
    X_all_samples = np.zeros((n_samples, n_roa, n_features))
    Y_all_samples = np.zeros((n_samples, n_roa, n_outputs))

    for s_idx, sample_dict in enumerate(surrogate.X_train):
        for f_idx, feature in enumerate(surrogate.all_features):
            X_all_samples[s_idx, :, f_idx] = sample_dict[feature]

    for s_idx, sample_dict in enumerate(surrogate.Y_train):
        for o_idx, output in enumerate(surrogate.output_list):
            Y_all_samples[s_idx, :, o_idx] = sample_dict[output]
    
    # Flatten for fitting: reshape to (n_samples * n_roa, n_features) and (n_samples * n_roa, n_outputs)
    #X_flat = X_all_samples.reshape(-1, n_features)
    #Y_flat = Y_all_samples.reshape(-1, n_outputs)

    feat_to_idx = {feat: i for i, feat in enumerate(surrogate.all_features)}
    output_names = surrogate.output_list

    # Create subplots grid: rows = outputs, cols = selected features
    fig, axes = plt.subplots(n_outputs, len(top_features), figsize=tuple(cfg['style'].get('figsize_large', (12,8))))
    axes = np.atleast_2d(axes)
    _subplots_adjust(fig, cfg['style'])

    # Color by evaluation index
    cmap = plt.get_cmap(cfg.get('style', {}).get('scatter_cmap', 'turbo'))
    colors = [cmap(i / max(1, n_roa - 1)) for i in range(n_roa)]

    for r, out_name in enumerate(output_names):
        out_idx = r
        out_vals = Y_all_samples[:, :, out_idx]  # shape (n_samples, n_eval)
        for c, feat_name in enumerate(top_features):
            fi = feat_to_idx[feat_name]
            # Extract feature values across samples and eval points -> (n_samples*n_eval,)
            feat_vals = X_all_samples[:, :, fi]  # shape (n_samples, n_eval)
            ax = axes[r, c]
            # Scatter each eval index separately for color coding
            for eval_i in range(n_roa):
                ax.scatter(feat_vals[:, eval_i], out_vals[:, eval_i],
                           color=colors[eval_i], alpha=0.6, s=12, label=(f'r/a = {roa_eval[eval_i]:.2f}' if r==0 and c==0 else None))
            # Y labels and ticks only on leftmost column
            if c == 0:
                ylabel = f'target_{out_name}' if out_name in surrogate.target_vars else f'model_{out_name}'
                ax.set_ylabel(ylabel)
                ax.locator_params(axis='y', nbins=2)
            else:
                ax.set_yticklabels([])
            
            # X labels and ticks only on bottom row
            if r == n_outputs - 1:
                ax.set_xlabel(feat_name)
                ax.locator_params(axis='x', nbins=2)
            else:
                ax.set_xticklabels([])
            
            ax.grid(alpha=0.3)
    # Single legend for eval indices
    handles = [plt.Line2D([0],[0], marker='o', linestyle='None', color=colors[i], label=f'r/a = {roa_eval[i]:.2f}') for i in range(n_roa)]
    fig.legend(handles=handles, loc='upper right', fontsize=8)
    fig.suptitle('Surrogate Sensitivity Scatter Plots')
    fig.tight_layout()
    _save(fig, outdir/'surrogate_sensitivity')

# ---------------------------------------------------------------------------
# Saving utility
# ---------------------------------------------------------------------------

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png']:
        fig.savefig(f"{path}.{ext}", bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_report(config_path: str, work_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    history_path = Path(work_path) / cfg.get('solver_history', 'solver_history.csv')
    hist_df = load_history(history_path)
    modules = rebuild_modules_from_checkpoint(
        str(Path(work_path) / cfg.get('solver_checkpoint', 'solver_checkpoint.pkl')),
        strict=False
    )
    _init_style(cfg.get('style', {}))
    outdir = Path(work_path) / cfg.get('output_dir', 'analysis_outputs')
    level = cfg.get('analysis_level', 'standard').lower()

    # Level gating
    plots_cfg = cfg.get('plots', {})

    def enabled(name: str):
        if name == 'surrogate_sensitivity' and level != 'full':
            return False
        if name == 'surrogate_pca' and level not in ('standard','full'):
            return False
        if level == 'minimal' and name not in ('objective','profiles','power_flows'):
            return False
        return plots_cfg.get(name, True)

    #try:
    if enabled('objective'):    plot_objective(hist_df, cfg, outdir)
    if enabled('residual_channels'):    plot_residual_channels(hist_df, modules, cfg, outdir)
    if enabled('profiles'):   plot_profiles(modules, hist_df, cfg, outdir)
    if enabled('power_flows'):    plot_power_flows(hist_df, modules, cfg, outdir)
    if enabled('surrogate_heatmap'):
        top_features = surrogate_heatmap(modules, cfg, outdir)
    else:
        top_features = []
    if enabled('surrogate_sensitivity'):
        surrogate_sensitivity(modules, top_features, cfg, outdir)
    #except Exception as e:
       # print(f"Error during plotting: {e}")

    print(f"Report complete. Outputs in {outdir}")


def main():
    parser = argparse.ArgumentParser(description='Generate PRESTOS solver analysis report.')
    parser.add_argument('--config', '-c', default='analysis/plot_config.yaml', help='Path to plot config YAML.')
    parser.add_argument('--workdir', '-w', default='.', help='Working directory containing files to analyze.')
    args = parser.parse_args()
    run_report(args.config,args.workdir)

if __name__ == '__main__':
    main()
