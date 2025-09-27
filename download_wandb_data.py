#!/usr/bin/env python3
"""
Script to download full training data from W&B for creating plots.
Supports deduplication of runs with identical configurations.
"""

import os
import wandb
import pandas as pd
import json
import argparse
from datetime import datetime

def get_run_signature(run):
    """
    Generate a unique signature for a run based on its configuration.
    This helps identify duplicate runs with the same setup.
    """
    config = run.config
    key_params = [
        config.get('env'),
        config.get('alg'),
        config.get('seed'),
        config.get('safety_bound'),
        config.get('num_timesteps'),
        config.get('learning_rate'),
        config.get('entropy_cost')
    ]
    # Convert to string and handle None values
    signature = '_'.join(str(x) if x is not None else 'None' for x in key_params)
    return signature


def deduplicate_runs(runs, strategy='latest'):
    """
    Remove duplicate runs based on configuration.
    Strategies: 'latest' (keep most recent), 'longest' (keep run with most steps),
               'first' (keep first completed run)
    """
    run_groups = {}
    duplicates_found = 0

    # Group runs by signature
    for run in runs:
        signature = get_run_signature(run)

        if signature not in run_groups:
            run_groups[signature] = []
        run_groups[signature].append(run)

    # Select best run from each group
    deduplicated_runs = []

    for signature, group_runs in run_groups.items():
        if len(group_runs) == 1:
            deduplicated_runs.append(group_runs[0])
        else:
            duplicates_found += len(group_runs) - 1

            if strategy == 'latest':
                # Keep the most recently created run
                selected_run = max(group_runs, key=lambda r: r.created_at)
            elif strategy == 'longest':
                # Keep the run with most training steps
                selected_run = max(group_runs, key=lambda r: len(r.history()) if hasattr(r, 'history') else 0)
            elif strategy == 'first':
                # Keep the first completed run
                completed_runs = [r for r in group_runs if r.state == 'finished']
                if completed_runs:
                    selected_run = min(completed_runs, key=lambda r: r.created_at)
                else:
                    selected_run = group_runs[0]
            else:
                selected_run = group_runs[0]

            deduplicated_runs.append(selected_run)

            print(f"  Deduplicated {len(group_runs)} runs with signature '{signature}' -> kept: {selected_run.name}")

    return deduplicated_runs, duplicates_found


def download_wandb_runs(project_name="safe-brax-experimental-results",
                       entity="m-boustani-eindhoven-university-of-technology",
                       deduplicate=True, dedup_strategy='latest',
                       filter_envs=None, filter_algs=None):
    """
    Download all run data from a W&B project with optional deduplication.

    Args:
        deduplicate: Whether to remove duplicate runs with same config
        dedup_strategy: Strategy for selecting which duplicate to keep
                       ('latest', 'longest', 'first')
    """
    # Initialize W&B API
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{entity}/{project_name}")

    # Optional filtering by env/alg (match on config and common name variants)
    def canon_env(v: str) -> str:
        v = (v or '').lower()
        if 'safe_point_goal' in v or 'point_resetting_goal' in v or 'point goal' in v:
            return 'safe_point_goal'
        return v

    def canon_alg(v: str) -> str:
        v = (v or '').lower()
        if v in ('ppol', 'ppo-lagrange', 'ppo_lagrange_v2', 'ppo_lagrange_v3'):
            return 'ppo_lagrange'
        if v in ('ppoc', 'ppo-cost', 'ppo_cost'):
            return 'ppo_cost'
        return v

    if filter_envs or filter_algs:
        f_envs = set(canon_env(e) for e in (filter_envs or []))
        f_algs = set(canon_alg(a) for a in (filter_algs or []))
        filtered = []
        for run in runs:
            cfg = run.config or {}
            env_val = canon_env(str(cfg.get('env_name', cfg.get('env', ''))))
            alg_val = canon_alg(str(cfg.get('alg', '')))
            ok_env = (not f_envs) or (env_val in f_envs)
            ok_alg = (not f_algs) or (alg_val in f_algs)
            if ok_env and ok_alg:
                filtered.append(run)
        print(f"Filtered runs: {len(filtered)} / {len(runs)} matched envs={list(f_envs) or 'ANY'} algs={list(f_algs) or 'ANY'}")
        runs = filtered

    print(f"Found {len(runs)} runs in project {project_name}")

    # Deduplicate if requested
    if deduplicate:
        runs, duplicates_removed = deduplicate_runs(runs, strategy=dedup_strategy)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate runs, keeping {len(runs)} unique configurations")
        else:
            print("No duplicate runs found")

    # Create output directory
    output_dir = f"wandb_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if deduplicate:
        output_dir += "_deduped"
    os.makedirs(output_dir, exist_ok=True)

    all_runs_data = []

    for i, run in enumerate(runs):
        print(f"\nProcessing run {i+1}/{len(runs)}: {run.name}")
        
        # Get run metadata
        run_data = {
            'name': run.name,
            'id': run.id,
            'state': run.state,
            'config': dict(run.config),
            'summary': dict(run.summary),
            'tags': run.tags,
            'created_at': run.created_at,
        }
        
        # Download full history
        try:
            history = run.history()
            if not history.empty:
                # Save individual run history
                history_file = os.path.join(output_dir, f"{run.name}_{run.id}_history.csv")
                history.to_csv(history_file, index=False)
                print(f"  Saved history: {history_file} ({len(history)} steps)")
                
                run_data['history_file'] = history_file
                run_data['num_steps'] = len(history)
            else:
                print(f"  No history data available")
                run_data['history_file'] = None
                run_data['num_steps'] = 0
                
        except Exception as e:
            print(f"  Error downloading history: {e}")
            run_data['history_file'] = None
            run_data['num_steps'] = 0
        
        all_runs_data.append(run_data)
    
    # Save metadata for all runs
    metadata_file = os.path.join(output_dir, "all_runs_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(all_runs_data, f, indent=2, default=str)
    
    print(f"\nDownload complete!")
    print(f"Data saved to: {output_dir}/")
    print(f"Metadata: {metadata_file}")
    
    return output_dir, all_runs_data


def summarize_runs(runs_data):
    """
    Print a summary of the downloaded runs.
    """
    print("\nRun Summary:")
    print("-" * 80)
    
    for run in runs_data:
        env = run['config'].get('env', 'unknown')
        alg = run['config'].get('alg', 'unknown')
        seed = run['config'].get('seed', 'unknown')
        
        # Get final metrics
        final_reward = run['summary'].get('eval/episode_reward', 'N/A')
        final_cost = run['summary'].get('eval/episode_cost', 'N/A')
        
        print(f"\n{run['name']}:")
        print(f"  Environment: {env}")
        print(f"  Algorithm: {alg}")
        print(f"  Seed: {seed}")
        print(f"  Final Reward: {final_reward}")
        print(f"  Final Cost: {final_cost}")
        print(f"  Steps: {run['num_steps']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download W&B data with optional deduplication')
    parser.add_argument('--no-deduplicate', action='store_true',
                       help='Download all runs without deduplication')
    parser.add_argument('--dedup-strategy', choices=['latest', 'longest', 'first'],
                       default='latest',
                       help='Strategy for selecting which duplicate to keep')
    parser.add_argument('--project', default='safe-brax-experimental-results',
                       help='W&B project name')
    parser.add_argument('--entity', default='m-boustani-eindhoven-university-of-technology',
                       help='W&B entity/username')
    parser.add_argument('--filter-env', dest='filter_envs', nargs='*', default=None,
                        help='Optional env filters (e.g., safe_point_goal)')
    parser.add_argument('--filter-alg', dest='filter_algs', nargs='*', default=None,
                        help='Optional alg filters (e.g., ppo_lagrange)')

    args = parser.parse_args()

    print(f"Downloading W&B data from {args.entity}/{args.project}...")
    if not args.no_deduplicate:
        print(f"Deduplication enabled (strategy: {args.dedup_strategy})")
    else:
        print("Deduplication disabled - downloading all runs")

    # Check if user is logged in
    try:
        wandb.login()
    except:
        print("Please run 'wandb login' first or set WANDB_API_KEY environment variable")
        exit(1)

    # Download data
    output_dir, runs_data = download_wandb_runs(
        project_name=args.project,
        entity=args.entity,
        deduplicate=not args.no_deduplicate,
        dedup_strategy=args.dedup_strategy,
        filter_envs=args.filter_envs,
        filter_algs=args.filter_algs
    )

    # Show summary
    summarize_runs(runs_data)

    print(f"\nYou can now use the data in '{output_dir}/' to create plots!")
    if not args.no_deduplicate:
        print("Note: Duplicate runs were automatically removed based on configuration.")
