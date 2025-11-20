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
import os, json, yaml, ast, argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.decomposition import PCA

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

    return HistoryData(df, iterations, objective, used_surrogate, X_list, R_list, Y_list, Y_target_list)

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


def plot_objective(hist: HistoryData, cfg: Dict[str, Any], outdir: Path):
    fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_small', (6,4))))
    ax.plot(hist.iterations[~hist.used_surrogate], hist.objective[~hist.used_surrogate], 'o-', label='model')
    ax.plot(hist.iterations[hist.used_surrogate], hist.objective[hist.used_surrogate], 's-', label='surrogate')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    ax.set_title('Objective vs Iteration')
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir/'objective')


def plot_residual_channels(hist: HistoryData, cfg: Dict[str, Any], outdir: Path, max_channels: int = 8):
    # Build channel matrix: residual arrays concatenated; we extract per-variable segments by length heuristic.
    # Simplify: treat each residual position as channel if small; else split into equal-sized blocks.
    R_nonempty = [r for r in hist.R_list if r.size > 0]
    if not R_nonempty:
        return
    n = R_nonempty[0].size
    # Decide number of channels to visualize
    n_channels = min(max_channels, n)
    fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_medium', (7,5))))
    for ch in range(n_channels):
        series = [r[ch] if ch < r.size else np.nan for r in hist.R_list]
        ax.plot(hist.iterations, series, marker='o', ms=3, label=f'R[{ch}]')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual component (normed)')
    ax.set_title('Residual Channels (subset)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, outdir/'residual_channels')


def plot_profiles_initial_final(hist: HistoryData, cfg: Dict[str, Any], outdir: Path, domain: Tuple[float,float]|None=None):
    if not hist.X_list:
        return
    X0 = hist.X_list[0]
    Xf = hist.X_list[-1]
    # For each profile key in final dict, if value is array-like plot initial vs final
    for prof, final_params in Xf.items():
        # Collect parameter scalar values; shade by +/- 5% as placeholder (std not stored per param)
        init_params = X0.get(prof, {})
        param_names = list(final_params.keys())
        fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_small', (6,4))))
        x_idx = np.arange(len(param_names))
        fin_vals = [final_params[p] for p in param_names]
        init_vals = [init_params.get(p, np.nan) for p in param_names]
        ax.plot(x_idx, init_vals, 'o--', label='initial')
        ax.plot(x_idx, fin_vals, 's-', label='final')
        # Shade bounds if available via +/- 5% proxy
        std = 0.05 * np.abs(fin_vals)
        ax.fill_between(x_idx, np.array(fin_vals)-std, np.array(fin_vals)+std, color='C1', alpha=0.2, label='final ±5%')
        ax.set_xticks(x_idx)
        ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Parameter value')
        ax.set_title(f'Profile parameters: {prof}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, outdir/f'profile_{prof}')


def plot_final_targets(hist: HistoryData, cfg: Dict[str, Any], outdir: Path):
    if not hist.Y_list or not hist.Y_target_list:
        return
    Yf = hist.Y_list[-1]
    Ytf = hist.Y_target_list[-1]
    keys = [k for k in Ytf.keys() if k in Yf]
    if not keys:
        return
    # Assume all arrays same length (roa grid length)
    for key in keys:
        y = np.asarray(Yf[key], dtype=float)
        yt = np.asarray(Ytf[key], dtype=float)
        x = np.linspace(0,1,len(y))
        fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_small', (6,4))))
        ax.plot(x, yt, '-', label=f"target {key}")
        ax.plot(x, y, '--', label=f"model {key}")
        std = 0.05 * np.abs(y)  # placeholder uncertainty
        ax.fill_between(x, y-std, y+std, color='C1', alpha=0.2, label='model ±5%')
        ax.set_xlabel('roa (normalized)')
        ax.set_ylabel(key)
        ax.set_title(f'Final model vs target: {key}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, outdir/f'target_{key}')

# ---------------------------------------------------------------------------
# Surrogate sensitivity & PCA using history only
# ---------------------------------------------------------------------------

def _flatten_param_dict(X_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat = {}
    for prof, params in (X_dict or {}).items():
        for pname, val in (params or {}).items():
            flat[f"{prof}:{pname}"] = float(val)
    return flat


def surrogate_pca(hist: HistoryData, cfg: Dict[str, Any], outdir: Path):
    flat_rows = [_flatten_param_dict(x) for x in hist.X_list]
    all_keys = sorted({k for row in flat_rows for k in row.keys()})
    Xmat = np.array([[row.get(k, np.nan) for k in all_keys] for row in flat_rows], dtype=float)
    # Drop columns with any NaN
    mask = ~np.isnan(Xmat).any(axis=0)
    Xclean = Xmat[:, mask]
    feat_names = [all_keys[i] for i, m in enumerate(mask) if m]
    if Xclean.shape[1] < 2:
        return
    pca = PCA(n_components=min(cfg['pca'].get('n_components',5), Xclean.shape[1]))
    comps = pca.fit_transform(Xclean)
    fig, axes = plt.subplots(1,2, figsize=tuple(cfg['style'].get('figsize_wide', (10,5))))
    axes[0].plot(hist.iterations, comps[:,0], 'o-')
    axes[0].set_title('PCA Component 1 vs iter')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('PC1')
    # Feature loadings
    loadings = pca.components_[0]
    top_idx = np.argsort(np.abs(loadings))[-min(15,len(loadings)):][::-1]
    axes[1].bar(np.arange(len(top_idx)), loadings[top_idx])
    axes[1].set_xticks(np.arange(len(top_idx)))
    axes[1].set_xticklabels([feat_names[i] for i in top_idx], rotation=45, ha='right', fontsize=8)
    axes[1].set_title('PC1 Loadings (top)')
    fig.tight_layout()
    _save(fig, outdir/'surrogate_pca')


def surrogate_sensitivity(hist: HistoryData, cfg: Dict[str, Any], outdir: Path):
    # Approximate sensitivity: numeric gradient of objective wrt flattened parameters around final iterate.
    if not hist.X_list:
        return
    Xf_flat = _flatten_param_dict(hist.X_list[-1])
    params = list(Xf_flat.keys())
    base_obj = hist.objective[-1]
    perturb_frac = cfg['sensitivity'].get('perturb_fraction', 0.05)
    grads = []
    for p in params:
        val = Xf_flat[p]
        dval = perturb_frac * (abs(val) if val != 0 else 1.0)
        # Find closest previous iteration where parameter differs by > dval/2
        series = [ _flatten_param_dict(x).get(p, np.nan) for x in hist.X_list ]
        idx = np.argmax(np.abs(np.array(series) - val) > dval/2)
        if idx == 0 and len(series) > 1:
            idx = len(series)-2
        prev_obj = hist.objective[idx] if idx < len(hist.objective) else base_obj
        grad = (base_obj - prev_obj) / ( (val - series[idx]) if (val - series[idx]) !=0 else dval )
        grads.append(grad)
    # Plot top sensitivities
    grads = np.array(grads)
    top_idx = np.argsort(np.abs(grads))[-min(cfg['sensitivity'].get('max_features',12), len(grads)) :][::-1]
    fig, ax = plt.subplots(figsize=tuple(cfg['style'].get('figsize_medium', (7,5))))
    ax.bar(np.arange(len(top_idx)), grads[top_idx])
    ax.set_xticks(np.arange(len(top_idx)))
    ax.set_xticklabels([params[i] for i in top_idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Approx. dObjective/dParam')
    ax.set_title('Surrogate/Objective Sensitivity (heuristic)')
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

def run_report(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    history_path = cfg.get('solver_history', 'solver_history.csv')
    hist = load_history(history_path)
    _init_style(cfg.get('style', {}))
    outdir = Path(cfg.get('output_dir', 'analysis_outputs'))
    level = cfg.get('analysis_level', 'standard').lower()

    # Level gating
    plots_cfg = cfg.get('plots', {})

    def enabled(name: str):
        if name == 'surrogate_sensitivity' and level != 'full':
            return False
        if name == 'surrogate_pca' and level not in ('standard','full'):
            return False
        if level == 'minimal' and name not in ('objective','profiles_initial_final','targets_final'):
            return False
        return plots_cfg.get(name, True)

    if enabled('objective'):            plot_objective(hist, cfg, outdir)
    if enabled('residual_channels'):    plot_residual_channels(hist, cfg, outdir)
    if enabled('profiles_initial_final'): plot_profiles_initial_final(hist, cfg, outdir)
    if enabled('targets_final'):        plot_final_targets(hist, cfg, outdir)
    if enabled('surrogate_pca'):        surrogate_pca(hist, cfg, outdir)
    if enabled('surrogate_sensitivity'): surrogate_sensitivity(hist, cfg, outdir)

    print(f"Report complete. Outputs in {outdir}")


def main():
    parser = argparse.ArgumentParser(description='Generate MinT solver analysis report.')
    parser.add_argument('--config', '-c', default='analysis/plot_config.yaml', help='Path to plot config YAML.')
    args = parser.parse_args()
    run_report(args.config)

if __name__ == '__main__':
    main()
