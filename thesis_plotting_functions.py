"""
Thesis Plotting Functions
This module contains all plotting functions for generating thesis figures from W&B data.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from typing import List, Dict, Optional
from matplotlib.ticker import ScalarFormatter

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_run_data(data_dir: str, metadata: Dict) -> List[pd.DataFrame]:
    """Load all run data from CSV files.

    Supports metadata saved as either a dict mapping run_id->meta or a list of meta dicts.
    Falls back to glob matching when run_id is unavailable.
    """
    runs_data: List[pd.DataFrame] = []

    def _canonicalize_env(env_value: str) -> str:
        if not env_value:
            return 'unknown'
        # Normalize Safe Point Goal variants
        if 'point_resetting_goal' in env_value or 'safe_point_goal' in env_value:
            return 'safe_point_goal'
        return env_value

    def _canonicalize_alg(alg_value: str) -> str:
        if not alg_value:
            return 'unknown'
        if alg_value in ('ppo-cost', 'ppoc'):
            return 'ppo_cost'
        return alg_value

    def _load_one(history_file: str, run_id: str, run_meta: Dict):
        try:
            if not os.path.exists(history_file):
                return
            df = pd.read_csv(history_file)
            if df.empty:
                return
            # Add metadata to dataframe
            df['run_id'] = run_id
            df['run_name'] = run_meta.get('name') or run_meta.get('run_name') or 'unknown'
            cfg = run_meta.get('config', {}) or {}
            raw_env = cfg.get('env_name', cfg.get('env', 'unknown'))
            raw_alg = cfg.get('alg', 'unknown')
            df['env'] = _canonicalize_env(raw_env)
            df['alg'] = _canonicalize_alg(raw_alg)
            df['seed'] = cfg.get('seed', 0)
            df['safety_bound'] = cfg.get('safety_bound', None)
            runs_data.append(df)
        except Exception as e:
            print(f"Error loading {history_file}: {e}")

    # Case 1: metadata is a dict of run_id -> run_meta
    if isinstance(metadata, dict):
        for run_id, run_meta in metadata.items():
            name = run_meta.get('name') or run_meta.get('run_name')
            if not name:
                # Skip if no name; try next
                continue
            history_file = os.path.join(data_dir, f"{name}_{run_id}_history.csv")
            _load_one(history_file, run_id, run_meta)

    # Case 2: metadata is a list of run_meta dicts
    elif isinstance(metadata, list):
        for run_meta in metadata:
            run_id = run_meta.get('id') or run_meta.get('run_id') or ''
            name = run_meta.get('name') or run_meta.get('run_name') or ''

            history_file = ''
            if run_id and name:
                candidate = os.path.join(data_dir, f"{name}_{run_id}_history.csv")
                if os.path.exists(candidate):
                    history_file = candidate
            # Fallback: glob by name if exact id-based path not found
            if not history_file and name:
                matches = sorted(glob(os.path.join(data_dir, f"{name}_*_history.csv")))
                if len(matches) == 1:
                    history_file = matches[0]
                    # Try to parse run_id from filename: {name}_{rid}_history.csv
                    base = os.path.basename(history_file)
                    try:
                        rid = base[len(name) + 1 : -len('_history.csv')]
                        if rid:
                            run_id = rid
                    except Exception:
                        pass
                elif len(matches) > 1:
                    # Pick the newest by mtime
                    history_file = max(matches, key=os.path.getmtime)
                    base = os.path.basename(history_file)
                    try:
                        rid = base[len(name) + 1 : -len('_history.csv')]
                        if rid:
                            run_id = rid
                    except Exception:
                        pass

            if history_file:
                _load_one(history_file, run_id or 'unknown', run_meta)
            else:
                # No matching CSV found; skip silently
                continue

    else:
        print("Warning: Unrecognized metadata format; expected dict or list.")

    print(f"Loaded {len(runs_data)} runs with history data")
    return runs_data


def create_learning_curves(runs_data: List[pd.DataFrame], env_name: str, 
                          safety_bound: float = 0.2, save_path: Optional[str] = None):
    """Create learning curves comparing PPO and PPO-Lagrange."""
    
    # Filter runs for this environment
    env_runs = [df for df in runs_data if not df.empty and df['env'].iloc[0] == env_name]
    
    if not env_runs:
        print(f"No data found for environment: {env_name}")
        return None
    
    # Separate by algorithm
    ppo_runs = [df for df in env_runs if df['alg'].iloc[0] == 'ppo']
    # Prefer specified bound; if absent for this env, fall back to the most common bound available
    all_ppol_runs = [df for df in env_runs if df['alg'].iloc[0] == 'ppo_lagrange']
    ppol_runs = [df for df in all_ppol_runs if df['safety_bound'].iloc[0] == safety_bound]
    fallback_bound = safety_bound
    if not ppol_runs and all_ppol_runs:
        # Choose the mode of available bounds
        available_bounds = [float(df['safety_bound'].iloc[0]) for df in all_ppol_runs]
        vals, counts = np.unique(available_bounds, return_counts=True)
        fallback_bound = float(vals[np.argmax(counts)])
        ppol_runs = [df for df in all_ppol_runs if float(df['safety_bound'].iloc[0]) == fallback_bound]
    
    print(f"Environment: {env_name}")
    print(f"  PPO runs: {len(ppo_runs)} (seeds: {sorted(set(df['seed'].iloc[0] for df in ppo_runs))})")
    print(f"  PPO-Lagrange (bound={fallback_bound}) runs: {len(ppol_runs)} (seeds: {sorted(set(df['seed'].iloc[0] for df in ppol_runs))})")
    
    # Create figure with subplots (side-by-side panels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    
    # Colors and markers for algorithms (color-blind friendly defaults)
    colors = {'ppo': 'blue', 'ppo_lagrange': 'orange'}
    markers = {'ppo': 'o', 'ppo_lagrange': 's'}
    band_alpha = 0.12
    line_kw = dict(linewidth=2.2, markersize=5, zorder=3)
    
    # Helper function to aggregate data across seeds
    def aggregate_runs(runs, metric):
        if not runs:
            return None, None, None, 0, False

        # For each run, select only rows where the metric is present (drop NaNs)
        per_run_series = []
        per_run_steps = []
        for df in runs:
            if metric not in df.columns:
                continue
            series = df[metric]
            series = series[~pd.isna(series)]
            if len(series) == 0:
                continue
            per_run_series.append(series.values)
            # Collect corresponding _step values if available
            if '_step' in df.columns:
                step_series = df.loc[series.index, '_step'].values
                per_run_steps.append(step_series)

        if not per_run_series:
            return None, None, None, 0, False

        # Align by evaluation index across seeds
        min_len = min(len(s) for s in per_run_series)
        aligned = np.stack([s[:min_len] for s in per_run_series], axis=0)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)
        n = aligned.shape[0]
        ci95 = 1.96 * (std / np.sqrt(max(n, 1)))
        # Prefer actual training steps when available; otherwise use eval index
        if per_run_steps and len(per_run_steps) >= 1:
            aligned_steps = np.stack([s[:min_len] for s in per_run_steps], axis=0)
            steps = np.nanmean(aligned_steps, axis=0) / 1e6  # convert to millions for readability
            used_steps = True
        else:
            steps = np.arange(min_len)
            used_steps = False

        return steps, mean, ci95, n, used_steps
    
    # Plot reward
    ax1 = axes[0]
    
    line_handles = []
    legend_labels = []

    # PPO
    if ppo_runs:
        steps, mean, ci, n, used_steps_x = aggregate_runs(ppo_runs, 'eval/episode_reward')
        if steps is not None:
            ppo_line, = ax1.plot(steps, mean, color=colors['ppo'], marker=markers['ppo'], **line_kw)
            ax1.fill_between(steps, mean - ci, mean + ci, alpha=band_alpha, color=colors['ppo'], zorder=2)
            line_handles.append(ppo_line)
            legend_labels.append("PPO")
            ppo_reward_last = (steps[-1], float(mean[-1]))
    
    # PPO-Lagrange
    if ppol_runs:
        steps, mean, ci, n, used_steps_x2 = aggregate_runs(ppol_runs, 'eval/episode_reward')
        if steps is not None:
            ppol_line, = ax1.plot(steps, mean, color=colors['ppo_lagrange'], marker=markers['ppo_lagrange'], **line_kw)
            ax1.fill_between(steps, mean - ci, mean + ci, alpha=band_alpha, color=colors['ppo_lagrange'], zorder=2)
            line_handles.append(ppol_line)
            legend_labels.append(f"PPO-Lagrange (bound={fallback_bound})")
            ppol_reward_last = (steps[-1], float(mean[-1]))

    # PPO-Cost excluded
    
    ax1.set_ylabel('Episode Reward')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot cost
    ax2 = axes[1]
    
    # Determine cost metric name
    cost_metrics = ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']
    cost_metric = None
    for metric in cost_metrics:
        if ppo_runs and metric in ppo_runs[0].columns:
            cost_metric = metric
            break
        if ppol_runs and metric in ppol_runs[0].columns:
            cost_metric = metric
            break
        # PPO-Cost excluded
    
    if cost_metric:
        # PPO
        x_label_is_steps = False
        if ppo_runs:
            steps, mean, ci, n, used_steps_cost1 = aggregate_runs(ppo_runs, cost_metric)
            if steps is not None:
                ax2.plot(steps, mean, color=colors['ppo'], marker=markers['ppo'], **line_kw)
                ax2.fill_between(steps, mean - ci, mean + ci, alpha=band_alpha, color=colors['ppo'], zorder=2)
                x_label_is_steps = x_label_is_steps or used_steps_cost1
                ppo_cost_last = (steps[-1], float(mean[-1]))
        
        # PPO-Lagrange
        if ppol_runs:
            steps, mean, ci, n, used_steps_cost2 = aggregate_runs(ppol_runs, cost_metric)
            if steps is not None:
                ax2.plot(steps, mean, color=colors['ppo_lagrange'], marker=markers['ppo_lagrange'], **line_kw)
                ax2.fill_between(steps, mean - ci, mean + ci, alpha=band_alpha, color=colors['ppo_lagrange'], zorder=2)
                x_label_is_steps = x_label_is_steps or used_steps_cost2
                ppol_cost_last = (steps[-1], float(mean[-1]))

        # PPO-Cost excluded
        
        # Add safety bound line scaled by episode length
        if ppol_runs and steps is not None:
            episode_length = 2000 if env_name == 'safe_point_goal' else 1000
            ax2.axhline(y=fallback_bound * episode_length, color='red', linestyle='--', 
                       alpha=0.5, label=f"Safety bound (= {fallback_bound} × L, L={episode_length})")
    
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')
    # X labels: show on both panels for readability when side-by-side
    x_label_text = 'Training steps (millions)' if 'x_label_is_steps' in locals() and x_label_is_steps else 'Evaluation index'
    ax1.set_xlabel(x_label_text)
    ax2.set_xlabel(x_label_text)
    ax2.set_ylabel('Episode Cost')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Title
    env_display_name = env_name.replace('_', ' ').title()
    if 'point' in env_name.lower():
        env_display_name = 'Point Goal Navigation'
    elif 'ant' in env_name.lower():
        env_display_name = 'Ant Velocity'
    elif 'humanoid' in env_name.lower():
        env_display_name = 'Humanoid Height'
    
    fig.suptitle(f'{env_display_name}: Learning Curves', fontsize=14, fontweight='bold')

    # Single shared legend placed below the plots (centered)
    if line_handles:
        fig.legend(line_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.03),
                   ncol=2, frameon=False, title='Mean ± 95% CI')
        plt.subplots_adjust(top=0.90, bottom=0.14, right=0.97)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def create_safety_performance_tradeoff(runs_data: List[pd.DataFrame], save_path: Optional[str] = None):
    """Create safety-performance tradeoff plot showing Pareto frontier."""
    
    # Collect final performance for each algorithm and environment
    results = []
    
    for df in runs_data:
        if df.empty:
            continue
        
        env = df['env'].iloc[0]
        alg = df['alg'].iloc[0]
        seed = df['seed'].iloc[0]
        bound = df['safety_bound'].iloc[0]
        
        # Get final performance (last row)
        final_row = df.iloc[-1]
        
        # Find reward and cost columns
        reward_col = None
        cost_col = None
        
        for col in ['eval/episode_reward', 'episode_reward', 'episode/reward']:
            if col in df.columns:
                reward_col = col
                break
        
        for col in ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost', 
                   'episode_cost', 'episode/cost']:
            if col in df.columns:
                cost_col = col
                break
        
        if reward_col and cost_col:
            results.append({
                'env': env,
                'alg': alg,
                'seed': seed,
                'bound': bound,
                'final_reward': final_row[reward_col],
                'final_cost': final_row[cost_col]
            })
    
    results_df = pd.DataFrame(results)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    environments = [
        ('ant_velocity_constrained', 'Ant Velocity'),
        ('humanoid_height_constrained', 'Humanoid Height'),
        ('safe_point_goal', 'Point Goal')
    ]
    
    for idx, (env, title) in enumerate(environments):
        ax = axes[idx]
        env_data = results_df[results_df['env'] == env]
        
        if env_data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue
        
        # Plot PPO (unconstrained baseline)
        ppo_data = env_data[env_data['alg'] == 'ppo']
        if not ppo_data.empty:
            ppo_mean = ppo_data.groupby('alg')[['final_cost', 'final_reward']].mean()
            ppo_std = ppo_data.groupby('alg')[['final_cost', 'final_reward']].std()
            ax.errorbar(ppo_mean['final_cost'].values[0], ppo_mean['final_reward'].values[0],
                       xerr=ppo_std['final_cost'].values[0] if len(ppo_std) > 0 else 0,
                       yerr=ppo_std['final_reward'].values[0] if len(ppo_std) > 0 else 0,
                       marker='o', markersize=10, label='PPO (unconstrained)',
                       color='blue', capsize=5)

        # PPO-Cost excluded
        
        # Plot PPO-Lagrange with different bounds
        ppol_data = env_data[env_data['alg'] == 'ppo_lagrange']
        if not ppol_data.empty:
            # Group by bound and calculate mean/std
            bounds = sorted(ppol_data['bound'].unique())
            
            # Create colormap for bounds
            cmap = plt.cm.YlOrRd
            colors = [cmap(i/len(bounds)) for i in range(len(bounds))]
            
            for i, bound in enumerate(bounds):
                bound_data = ppol_data[ppol_data['bound'] == bound]
                if not bound_data.empty:
                    mean_cost = bound_data['final_cost'].mean()
                    std_cost = bound_data['final_cost'].std()
                    mean_reward = bound_data['final_reward'].mean()
                    std_reward = bound_data['final_reward'].std()
                    
                    ax.errorbar(mean_cost, mean_reward,
                               xerr=std_cost, yerr=std_reward,
                               marker='s', markersize=8,
                               label=f'PPO-L (b={bound})',
                               color=colors[i], capsize=3)
            
            # Draw Pareto frontier
            if len(bounds) > 1:
                # Sort by cost for line drawing
                frontier_points = []
                for bound in bounds:
                    bound_data = ppol_data[ppol_data['bound'] == bound]
                    if not bound_data.empty:
                        frontier_points.append((
                            bound_data['final_cost'].mean(),
                            bound_data['final_reward'].mean()
                        ))
                frontier_points.sort(key=lambda x: x[0])
                
                if len(frontier_points) > 1:
                    xs, ys = zip(*frontier_points)
                    ax.plot(xs, ys, 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Episode Cost')
        ax.set_ylabel('Episode Reward')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--')
        # Move legends to a single shared legend below
        ax.legend_.remove() if ax.get_legend() else None
    
    fig.suptitle('Safety-Performance Tradeoff', fontsize=16, fontweight='bold')
    # Shared legend at the bottom center
    handles, labels = [], []
    for ax in axes:
        handles_local, labels_local = ax.get_legend_handles_labels()
        handles += handles_local
        labels += labels_local
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
        plt.subplots_adjust(bottom=0.14, top=0.88)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def create_lambda_evolution(runs_data: List[pd.DataFrame], env_name: str, save_path: Optional[str] = None):
    """Create lambda evolution plot for PPO-Lagrange."""
    
    # Filter for PPO-Lagrange runs in this environment
    ppol_runs = [df for df in runs_data 
                 if not df.empty 
                 and df['env'].iloc[0] == env_name 
                 and df['alg'].iloc[0] == 'ppo_lagrange']
    
    if not ppol_runs:
        print(f"No PPO-Lagrange data found for {env_name}")
        return None
    
    # Group by safety bound
    bounds = sorted(set(float(df['safety_bound'].iloc[0]) for df in ppol_runs))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colormap for different bounds
    cmap = plt.cm.viridis
    colors = [cmap(i/len(bounds)) for i in range(len(bounds))]
    
    used_steps_axis = False
    band_alpha = 0.15
    
    for i, bound in enumerate(bounds):
        bound_runs = [df for df in ppol_runs if float(df['safety_bound'].iloc[0]) == bound]
        if not bound_runs:
            continue
        
        # Find lambda column
        lambda_col = None
        for col in ['training/lambda_lagr', 'lambda_lagr', 'lambda']:
            if col in bound_runs[0].columns:
                lambda_col = col
                break
        if not lambda_col:
            continue
        
        # Align across seeds and drop NaNs
        per_run_lambda = []
        per_run_steps = []
        for df in bound_runs:
            series = df[lambda_col]
            series = series[~pd.isna(series)]
            if len(series) == 0:
                continue
            per_run_lambda.append(series.values)
            if '_step' in df.columns:
                per_run_steps.append(df.loc[series.index, '_step'].values)
        if not per_run_lambda:
            continue
        min_len = min(len(s) for s in per_run_lambda)
        aligned = np.stack([s[:min_len] for s in per_run_lambda], axis=0)
        mean_lambda = np.nanmean(aligned, axis=0)
        std_lambda = np.nanstd(aligned, axis=0)
        if per_run_steps:
            aligned_steps = np.stack([s[:min_len] for s in per_run_steps], axis=0)
            steps = np.nanmean(aligned_steps, axis=0) / 1e6
            used_steps_axis = True
        else:
            steps = np.arange(min_len)
        
        # Plot
        ax.plot(steps, mean_lambda, label=f'Bound = {bound}', color=colors[i], linewidth=2.2)
        ax.fill_between(steps, mean_lambda - std_lambda, mean_lambda + std_lambda, alpha=band_alpha, color=colors[i])
    
    # X axis formatting
    ax.set_xlabel('Training steps (millions)' if used_steps_axis else 'Evaluation index')
    ax.set_ylabel('Lagrange Multiplier (λ)')
    
    # Format title
    env_display_name = env_name.replace('_', ' ').title()
    if 'point' in env_name.lower():
        env_display_name = 'Point Goal Navigation'
    elif 'ant' in env_name.lower():
        env_display_name = 'Ant Velocity'
    elif 'humanoid' in env_name.lower():
        env_display_name = 'Humanoid Height'
    
    ax.set_title(f'{env_display_name}: Lagrange Multiplier Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(bottom=0.20)
    
    # Format x-axis
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def create_reward_over_steps_by_bound(runs_data: List[pd.DataFrame], env_name: str,
                                      save_path: Optional[str] = None,
                                      metric_source: str = 'eval') -> Optional[plt.Figure]:
    """Plot reward over steps for PPO-Lagrange with separate lines per safety bound.

    Aggregates across seeds per bound, showing mean ± 95% CI.
    """
    # Filter PPO-Lagrange runs for the environment
    ppol_runs = [df for df in runs_data
                 if not df.empty and df['env'].iloc[0] == env_name and df['alg'].iloc[0] == 'ppo_lagrange']
    if not ppol_runs:
        print(f"No PPO-Lagrange data found for {env_name}")
        return None

    # Group runs by safety bound
    bounds = sorted(set(float(df['safety_bound'].iloc[0]) for df in ppol_runs))

    fig, ax = plt.subplots(figsize=(10, 6))
    # High-contrast Okabe–Ito palette (colorblind-friendly)
    okabe_ito = [
        '#000000',  # black
        '#E69F00',  # orange
        '#56B4E9',  # sky blue
        '#009E73',  # bluish green
        '#F0E442',  # yellow
        '#0072B2',  # blue
        '#D55E00',  # vermillion
        '#CC79A7',  # reddish purple
    ]
    colors = [okabe_ito[i % len(okabe_ito)] for i in range(len(bounds))]
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    linestyles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 2, 1, 2)), (0, (1, 1)), (0, (4, 2, 1, 2))]
    band_alpha = 0.10

    def aggregate_bound_runs(bound_runs: List[pd.DataFrame], metric: str):
        per_run_values = []
        per_run_steps = []
        for df in bound_runs:
            if metric not in df.columns:
                continue
            series = df[metric]
            series = series[~pd.isna(series)]
            if len(series) == 0:
                continue
            per_run_values.append(series.values)
            if '_step' in df.columns:
                per_run_steps.append(df.loc[series.index, '_step'].values)
        if not per_run_values:
            return None, None
        min_len = min(len(s) for s in per_run_values)
        aligned = np.stack([s[:min_len] for s in per_run_values], axis=0)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)
        n = aligned.shape[0]
        ci95 = 1.96 * (std / np.sqrt(max(n, 1)))
        if per_run_steps:
            aligned_steps = np.stack([s[:min_len] for s in per_run_steps], axis=0)
            steps = np.nanmean(aligned_steps, axis=0) / 1e6
        else:
            steps = np.arange(min_len)
        return steps, (mean, ci95)

    for i, bound in enumerate(bounds):
        bound_runs = [df for df in ppol_runs if float(df['safety_bound'].iloc[0]) == bound]
        # Choose reward metric based on source
        reward_metric_candidates_eval = ['eval/episode_reward']
        reward_metric_candidates_episode = ['episode/reward', 'episode_reward']
        metric = None
        if metric_source == 'eval':
            for m in reward_metric_candidates_eval:
                if m in bound_runs[0].columns:
                    metric = m
                    break
        else:
            for m in reward_metric_candidates_episode:
                if m in bound_runs[0].columns:
                    metric = m
                    break
        if metric is None:
            # Fallback: try the other source list
            for m in (reward_metric_candidates_eval + reward_metric_candidates_episode):
                if m in bound_runs[0].columns:
                    metric = m
                    break
        steps, stats = aggregate_bound_runs(bound_runs, metric) if metric else (None, None)
        if steps is None:
            continue
        mean, ci95 = stats
        ax.plot(
            steps,
            mean,
            color=colors[i],
            linewidth=2.6,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5,
            label=f"b={bound}",
            zorder=3,
        )
        ax.fill_between(steps, mean - ci95, mean + ci95, color=colors[i], alpha=band_alpha, zorder=2)

    ax.set_xlabel('Training steps (millions)')
    ax.set_ylabel('Episode Reward')
    env_display_name = env_name.replace('_', ' ').title()
    if 'point' in env_name.lower():
        env_display_name = 'Point Goal Navigation'
    elif 'ant' in env_name.lower():
        env_display_name = 'Ant Velocity'
    elif 'humanoid' in env_name.lower():
        env_display_name = 'Humanoid Height'
    title_suffix = ' (Eval)' if metric_source == 'eval' else ' (Training episodes)'
    ax.set_title(f'{env_display_name}: Reward over Steps by Safety Bound{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.subplots_adjust(bottom=0.18)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def create_cost_over_steps_by_bound(runs_data: List[pd.DataFrame], env_name: str,
                                    save_path: Optional[str] = None,
                                    metric_source: str = 'eval') -> Optional[plt.Figure]:
    """Plot episode cost over steps for PPO-Lagrange with separate lines per safety bound.

    Aggregates across seeds per bound, showing mean ± 95% CI.
    """
    # Filter PPO-Lagrange runs for the environment
    ppol_runs = [df for df in runs_data
                 if not df.empty and df['env'].iloc[0] == env_name and df['alg'].iloc[0] == 'ppo_lagrange']
    if not ppol_runs:
        print(f"No PPO-Lagrange data found for {env_name}")
        return None

    # Determine cost metric name based on source by checking first available run
    cost_metrics_eval = ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']
    cost_metrics_episode = ['episode/cost', 'episode_cost']
    candidates = cost_metrics_eval if metric_source == 'eval' else cost_metrics_episode
    cost_metric = None
    for m in candidates:
        if m in ppol_runs[0].columns:
            cost_metric = m
            break
    if cost_metric is None:
        # Fallback: search all known
        for m in (cost_metrics_eval + cost_metrics_episode):
            if m in ppol_runs[0].columns:
                cost_metric = m
                break
    if not cost_metric:
        print("No cost metric found in PPO-Lagrange runs; skipping cost plot")
        return None

    # Group runs by safety bound
    bounds = sorted(set(float(df['safety_bound'].iloc[0]) for df in ppol_runs))

    fig, ax = plt.subplots(figsize=(10, 6))
    # High-contrast Okabe–Ito palette (colorblind-friendly)
    okabe_ito = [
        '#000000',  # black
        '#E69F00',  # orange
        '#56B4E9',  # sky blue
        '#009E73',  # bluish green
        '#F0E442',  # yellow
        '#0072B2',  # blue
        '#D55E00',  # vermillion
        '#CC79A7',  # reddish purple
    ]
    colors = [okabe_ito[i % len(okabe_ito)] for i in range(len(bounds))]
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    linestyles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 2, 1, 2)), (0, (1, 1)), (0, (4, 2, 1, 2))]
    band_alpha = 0.10

    def aggregate_bound_runs(bound_runs: List[pd.DataFrame], metric: str):
        per_run_values = []
        per_run_steps = []
        for df in bound_runs:
            if metric not in df.columns:
                continue
            series = df[metric]
            series = series[~pd.isna(series)]
            if len(series) == 0:
                continue
            per_run_values.append(series.values)
            if '_step' in df.columns:
                per_run_steps.append(df.loc[series.index, '_step'].values)
        if not per_run_values:
            return None, None
        min_len = min(len(s) for s in per_run_values)
        aligned = np.stack([s[:min_len] for s in per_run_values], axis=0)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)
        n = aligned.shape[0]
        ci95 = 1.96 * (std / np.sqrt(max(n, 1)))
        if per_run_steps:
            aligned_steps = np.stack([s[:min_len] for s in per_run_steps], axis=0)
            steps = np.nanmean(aligned_steps, axis=0) / 1e6
        else:
            steps = np.arange(min_len)
        return steps, (mean, ci95)

    for i, bound in enumerate(bounds):
        bound_runs = [df for df in ppol_runs if float(df['safety_bound'].iloc[0]) == bound]
        steps, stats = aggregate_bound_runs(bound_runs, cost_metric)
        if steps is None:
            continue
        mean, ci95 = stats
        ax.plot(
            steps,
            mean,
            color=colors[i],
            linewidth=2.6,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5,
            label=f"b={bound}",
            zorder=3,
        )
        ax.fill_between(steps, mean - ci95, mean + ci95, color=colors[i], alpha=band_alpha, zorder=2)

    ax.set_xlabel('Training steps (millions)')
    ax.set_ylabel('Episode Cost')
    env_display_name = env_name.replace('_', ' ').title()
    if 'point' in env_name.lower():
        env_display_name = 'Point Goal Navigation'
    elif 'ant' in env_name.lower():
        env_display_name = 'Ant Velocity'
    elif 'humanoid' in env_name.lower():
        env_display_name = 'Humanoid Height'
    title_suffix = ' (Eval)' if metric_source == 'eval' else ' (Training episodes)'
    ax.set_title(f'{env_display_name}: Cost over Steps by Safety Bound{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.subplots_adjust(bottom=0.18)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig
def create_final_performance_comparison(runs_data: List[pd.DataFrame], env_name: str, 
                                       safety_bound: float = 0.2, save_path: Optional[str] = None):
    """Create bar chart comparing final performance of PPO vs PPO-Lagrange."""
    
    # Filter runs
    env_runs = [df for df in runs_data if not df.empty and df['env'].iloc[0] == env_name]
    
    if not env_runs:
        print(f"No data found for {env_name}")
        return None
    
    # Separate by algorithm
    ppo_runs = [df for df in env_runs if df['alg'].iloc[0] == 'ppo']
    # PPO-Cost excluded
    all_ppol_runs = [df for df in env_runs if df['alg'].iloc[0] == 'ppo_lagrange']
    ppol_runs = [df for df in all_ppol_runs if df['safety_bound'].iloc[0] == safety_bound]
    # Fallback to available bound if requested one doesn't exist (e.g., ant/humanoid only have 0.2)
    fallback_bound = safety_bound
    if not ppol_runs and all_ppol_runs:
        available_bounds = [float(df['safety_bound'].iloc[0]) for df in all_ppol_runs]
        vals, counts = np.unique(available_bounds, return_counts=True)
        fallback_bound = float(vals[np.argmax(counts)])
        ppol_runs = [df for df in all_ppol_runs if float(df['safety_bound'].iloc[0]) == fallback_bound]
    
    # Collect final metrics (only include present algorithms)
    metrics: Dict[str, Dict[str, list]] = {}
    if ppo_runs:
        metrics['PPO'] = {'rewards': [], 'costs': []}
    # PPO-Cost excluded
    if ppol_runs:
        metrics[f'PPO-L ({fallback_bound})'] = {'rewards': [], 'costs': []}
    
    if ppo_runs and 'PPO' in metrics:
        for df in ppo_runs:
            final_row = df.iloc[-1]
            for col in ['eval/episode_reward', 'episode_reward', 'episode/reward']:
                if col in df.columns:
                    metrics['PPO']['rewards'].append(final_row[col])
                    break
            for col in ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']:
                if col in df.columns:
                    metrics['PPO']['costs'].append(final_row[col])
                    break
    
    if ppol_runs and f'PPO-L ({fallback_bound})' in metrics:
        for df in ppol_runs:
            final_row = df.iloc[-1]
            for col in ['eval/episode_reward', 'episode_reward', 'episode/reward']:
                if col in df.columns:
                    metrics[f'PPO-L ({fallback_bound})']['rewards'].append(final_row[col])
                    break
            for col in ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']:
                if col in df.columns:
                    metrics[f'PPO-L ({fallback_bound})']['costs'].append(final_row[col])
                    break

    # PPO-Cost excluded
    
    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rewards
    ax1 = axes[0]
    algs = list(metrics.keys())
    rewards_mean = [np.mean(metrics[alg]['rewards']) if metrics[alg]['rewards'] else 0 for alg in algs]
    # 95% CI instead of SD
    rewards_std = [np.std(metrics[alg]['rewards']) if metrics[alg]['rewards'] else 0 for alg in algs]
    rewards_n = [len(metrics[alg]['rewards']) for alg in algs]
    rewards_ci = [1.96 * (s / np.sqrt(n)) if n > 0 else 0 for s, n in zip(rewards_std, rewards_n)]

    # Choose colors matching other plots
    color_map = {'PPO': 'blue'}
    bar_colors = [color_map.get(a, 'orange') for a in algs]
    ax1.bar(algs, rewards_mean, yerr=rewards_ci, capsize=5, 
            color=bar_colors, alpha=0.7)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Final Reward Comparison')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Costs
    ax2 = axes[1]
    costs_mean = [np.mean(metrics[alg]['costs']) if metrics[alg]['costs'] else 0 for alg in algs]
    # 95% CI instead of SD
    costs_std = [np.std(metrics[alg]['costs']) if metrics[alg]['costs'] else 0 for alg in algs]
    costs_n = [len(metrics[alg]['costs']) for alg in algs]
    costs_ci = [1.96 * (s / np.sqrt(n)) if n > 0 else 0 for s, n in zip(costs_std, costs_n)]

    ax2.bar(algs, costs_mean, yerr=costs_ci, capsize=5,
            color=bar_colors, alpha=0.7)
    
    # Add safety bound line scaled by episode length
    episode_length = 2000 if env_name == 'safe_point_goal' else 1000
    bound_for_line = fallback_bound if ppol_runs else safety_bound
    ax2.axhline(y=bound_for_line * episode_length, color='red', linestyle='--', 
               alpha=0.5, label=f'Safety Bound ({bound_for_line}/step)')
    
    ax2.set_ylabel('Episode Cost')
    ax2.set_title('Final Cost Comparison')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.subplots_adjust(bottom=0.20)
    
    # Overall title
    env_display_name = env_name.replace('_', ' ').title()
    if 'point' in env_name.lower():
        env_display_name = 'Point Goal Navigation'
    elif 'ant' in env_name.lower():
        env_display_name = 'Ant Velocity'
    elif 'humanoid' in env_name.lower():
        env_display_name = 'Humanoid Height'
    
    fig.suptitle(f'{env_display_name}: Final Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def _format_pm(mean_value: float, ci95_value: float) -> str:
    """Format a mean ± 95% CI string with one decimal."""
    if np.isnan(mean_value) or np.isnan(ci95_value):
        return "-"
    return f"{mean_value:.1f} ± {ci95_value:.1f}"


def create_bound_sweep_table(
    runs_data: List[pd.DataFrame],
    env_name: str = 'safe_point_goal',
    episode_length: int = 2000,
    metric_source: str = 'eval',
    save_csv_path: Optional[str] = None,
    save_tex_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a safety-bound sweep summary for PPO-Lagrange on a given environment.

    Columns:
    - Safety Bound
    - Seeds
    - Final Reward (mean ± 95% CI)
    - Mean Episode Cost (mean ± 95% CI)
    - Constraint Violation Rate (%)  [fraction of evaluation checkpoints with per-step mean cost > bound]
    """

    # Filter PPO-Lagrange runs for the environment
    ppol_runs = [
        df for df in runs_data
        if not df.empty
        and df['env'].iloc[0] == env_name
        and df['alg'].iloc[0] == 'ppo_lagrange'
    ]

    if not ppol_runs:
        df_empty = pd.DataFrame(columns=[
            'Safety Bound', 'Seeds', 'Final Reward', 'Mean Episode Cost', 'Constraint Violation Rate (%)'
        ])
        if save_csv_path:
            df_empty.to_csv(save_csv_path, index=False)
        if save_tex_path:
            with open(save_tex_path, 'w') as f:
                f.write('% No PPO-Lagrange runs found for bound sweep table\n')
        return df_empty

    # Determine bounds present
    try:
        bounds = sorted(set(float(df['safety_bound'].iloc[0]) for df in ppol_runs))
    except Exception:
        bounds = sorted(set(df['safety_bound'].iloc[0] for df in ppol_runs))

    # Metric selection
    if metric_source == 'eval':
        reward_candidates = ['eval/episode_reward']
        cost_candidates = ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']
    else:
        reward_candidates = ['episode/reward', 'episode_reward']
        cost_candidates = ['episode/cost', 'episode_cost']

    rows = []
    for bound in bounds:
        bound_runs = [df for df in ppol_runs if float(df['safety_bound'].iloc[0]) == float(bound)]

        final_rewards: List[float] = []
        final_costs: List[float] = []
        violation_fracs: List[float] = []

        for df in bound_runs:
            # Reward series
            reward_col = next((c for c in reward_candidates if c in df.columns), None)
            cost_col = next((c for c in cost_candidates if c in df.columns), None)
            if cost_col is None and metric_source == 'eval':
                # As a last resort, try episode cost columns if eval not present
                cost_col = next((c for c in ['episode/cost', 'episode_cost'] if c in df.columns), None)

            if reward_col is None or cost_col is None:
                continue

            reward_series = df[reward_col].dropna()
            cost_series = df[cost_col].dropna()
            if reward_series.empty or cost_series.empty:
                continue

            final_rewards.append(float(reward_series.iloc[-1]))
            final_costs.append(float(cost_series.iloc[-1]))

            # Violation fraction across evaluation checkpoints
            per_step_cost = cost_series.astype(float) / float(episode_length)
            if len(per_step_cost) > 0:
                violation_frac = float((per_step_cost > float(bound)).mean())
                violation_fracs.append(violation_frac)

        seeds = len(final_rewards)
        if seeds == 0:
            continue

        reward_mean = float(np.mean(final_rewards))
        reward_ci = 1.96 * float(np.std(final_rewards, ddof=0)) / np.sqrt(seeds)
        cost_mean = float(np.mean(final_costs))
        cost_ci = 1.96 * float(np.std(final_costs, ddof=0)) / np.sqrt(seeds)
        violation_rate = float(np.nanmean(violation_fracs)) * 100.0 if violation_fracs else np.nan

        # Display formatting
        bound_label = f"{bound:.2f}" if float(bound) != 0.0 else "0.0 (No constraint)"

        rows.append({
            'Safety Bound': bound_label,
            'Seeds': seeds,
            'Final Reward': _format_pm(reward_mean, reward_ci),
            'Mean Episode Cost': _format_pm(cost_mean, cost_ci),
            'Constraint Violation Rate (%)': f"{violation_rate:.1f}" if not np.isnan(violation_rate) else "-",
        })

    table_df = pd.DataFrame(rows)

    # Save CSV
    if save_csv_path:
        table_df.to_csv(save_csv_path, index=False)

    # Save LaTeX table
    if save_tex_path:
        lines: List[str] = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{@{}lccc@{}}")
        lines.append("\\toprule")
        lines.append("Safety Bound & Final Reward & Mean Episode Cost & Constraint Violation Rate (\\%) \\\\")
        lines.append("\\midrule")
        for _, r in table_df.iterrows():
            lines.append(f"{r['Safety Bound']} & {r['Final Reward']} & {r['Mean Episode Cost']} & {r['Constraint Violation Rate (%)']} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Constraint satisfaction analysis for different safety bounds on Point-Goal. Violation rate computed as the fraction of evaluation checkpoints with per-step mean episode cost $> d$. Values show mean final performance (mean $\\pm$ 95\\% CI) across available seeds per bound.}")
        lines.append("\\label{tab:constraint_satisfaction}")
        lines.append("\\end{table}")
        with open(save_tex_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

    return table_df

def create_summary_table(runs_data: List[pd.DataFrame], safety_bound: float = 0.2) -> pd.DataFrame:
    """Create a summary table of all experimental results."""
    
    environments = [
        'ant_velocity_constrained',
        'humanoid_height_constrained',
        'safe_point_goal',
    ]
    
    summary = []
    
    for env in environments:
        env_runs = [df for df in runs_data if not df.empty and df['env'].iloc[0] == env]
        
        # PPO
        ppo_runs = [df for df in env_runs if df['alg'].iloc[0] == 'ppo']
        if ppo_runs:
            rewards = []
            costs = []
            for df in ppo_runs:
                final_row = df.iloc[-1]
                for col in ['eval/episode_reward', 'episode_reward']:
                    if col in df.columns:
                        rewards.append(final_row[col])
                        break
                for col in ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']:
                    if col in df.columns:
                        costs.append(final_row[col])
                        break
            
            if rewards and costs:
                summary.append({
                    'Environment': env,
                    'Algorithm': 'PPO',
                    'Safety Bound': 'N/A',
                    'Seeds': len(ppo_runs),
                    'Mean Reward': f"{np.mean(rewards):.1f} ± {np.std(rewards):.1f}",
                    'Mean Cost': f"{np.mean(costs):.1f} ± {np.std(costs):.1f}"
                })
        
        # PPO-Lagrange (with specified bound)
        ppol_runs = [df for df in env_runs 
                     if df['alg'].iloc[0] == 'ppo_lagrange' 
                     and df['safety_bound'].iloc[0] == safety_bound]
        if ppol_runs:
            rewards = []
            costs = []
            for df in ppol_runs:
                final_row = df.iloc[-1]
                for col in ['eval/episode_reward', 'episode_reward']:
                    if col in df.columns:
                        rewards.append(final_row[col])
                        break
                for col in ['eval/episode_cost', 'eval/episode_velocity_cost', 'eval/episode_height_cost']:
                    if col in df.columns:
                        costs.append(final_row[col])
                        break
            
            if rewards and costs:
                summary.append({
                    'Environment': env,
                    'Algorithm': 'PPO-Lagrange',
                    'Safety Bound': safety_bound,
                    'Seeds': len(ppol_runs),
                    'Mean Reward': f"{np.mean(rewards):.1f} ± {np.std(rewards):.1f}",
                    'Mean Cost': f"{np.mean(costs):.1f} ± {np.std(costs):.1f}"
                })
    
    summary_df = pd.DataFrame(summary)
    return summary_df


# Main execution function
def generate_all_plots(data_dir: str = 'wandb_data_20250903_094505_deduped',
                      output_dir: str = 'plots/thesis_figures',
                      comparison_bound: float = 0.05):
    """Generate all plots for thesis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    with open(os.path.join(data_dir, 'all_runs_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata for {len(metadata)} runs")
    
    # Load all run data
    runs_data = load_run_data(data_dir, metadata)
    
    environments = [
        'ant_velocity_constrained',
        'humanoid_height_constrained',
        'safe_point_goal',
    ]
    
    # Generate learning curves
    print("\n" + "="*60)
    print("Generating Learning Curves...")
    print("="*60)
    for env in environments:
        save_path = os.path.join(output_dir, f"{env}_learning_curves.png")
        create_learning_curves(runs_data, env, safety_bound=comparison_bound, save_path=save_path)
        print()
    
    # Generate safety-performance tradeoff
    print("\n" + "="*60)
    print("Generating Reward-over-Steps by Safety Bound (PPO-Lagrange)...")
    print("="*60)
    # For baselines section, we only need this for Point-Goal
    save_path = os.path.join(output_dir, "safe_point_goal_reward_by_bound.png")
    create_reward_over_steps_by_bound(runs_data, env_name='safe_point_goal', save_path=save_path)
    
    # Also generate Cost-over-Steps by Safety Bound for Point-Goal
    print("\n" + "="*60)
    print("Generating Cost-over-Steps by Safety Bound (PPO-Lagrange)...")
    print("="*60)
    save_path = os.path.join(output_dir, "safe_point_goal_cost_by_bound.png")
    create_cost_over_steps_by_bound(runs_data, env_name='safe_point_goal', save_path=save_path)
    
    # Episode-based variants (training episodes)
    print("\n" + "="*60)
    print("Generating Episode-based Reward-over-Steps by Safety Bound (PPO-Lagrange)...")
    print("="*60)
    save_path = os.path.join(output_dir, "safe_point_goal_reward_by_bound_episode.png")
    create_reward_over_steps_by_bound(runs_data, env_name='safe_point_goal', save_path=save_path, metric_source='episode')
    
    print("\n" + "="*60)
    print("Generating Episode-based Cost-over-Steps by Safety Bound (PPO-Lagrange)...")
    print("="*60)
    save_path = os.path.join(output_dir, "safe_point_goal_cost_by_bound_episode.png")
    create_cost_over_steps_by_bound(runs_data, env_name='safe_point_goal', save_path=save_path, metric_source='episode')
    
    # Generate lambda evolution plots
    print("\n" + "="*60)
    print("Generating Lambda Evolution Plots...")
    print("="*60)
    for env in environments:
        save_path = os.path.join(output_dir, f"{env}_lambda_evolution.png")
        create_lambda_evolution(runs_data, env, save_path=save_path)
        print()
    
    # Generate final performance comparisons
    print("\n" + "="*60)
    print("Generating Final Performance Comparisons...")
    print("="*60)
    for env in environments:
        save_path = os.path.join(output_dir, f"{env}_final_performance.png")
        create_final_performance_comparison(runs_data, env, safety_bound=comparison_bound, save_path=save_path)
        print()
    
    # Create summary table
    print("\n" + "="*60)
    print("Creating Summary Table...")
    print("="*60)
    summary_table = create_summary_table(runs_data, safety_bound=comparison_bound)
    print(f"\nExperimental Results Summary (PPO vs PPO-Lagrange with bound={comparison_bound})")
    print("="*80)
    print(summary_table.to_string(index=False))
    
    # Save summary to CSV
    summary_table.to_csv(os.path.join(output_dir, 'experiment_summary.csv'), index=False)
    print(f"\nSaved summary to {os.path.join(output_dir, 'experiment_summary.csv')}")

    # Generate safety-bound sweep table (Point-Goal)
    print("\n" + "="*60)
    print("Generating Safety-Bound Sweep Table (PPO-Lagrange, Point-Goal)...")
    print("="*60)
    bound_csv = os.path.join(output_dir, 'safety_bound_summary.csv')
    bound_tex = os.path.join(output_dir, 'bound_sweep_table.tex')
    bound_df = create_bound_sweep_table(
        runs_data,
        env_name='safe_point_goal',
        episode_length=2000,
        metric_source='eval',
        save_csv_path=bound_csv,
        save_tex_path=bound_tex,
    )
    if not bound_df.empty:
        print(bound_df.to_string(index=False))
        print(f"\nSaved bound sweep CSV to {bound_csv}")
        print(f"Saved bound sweep LaTeX to {bound_tex}")
    
    print("\n" + "="*60)
    print(f"All plots generated and saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    generate_all_plots()
