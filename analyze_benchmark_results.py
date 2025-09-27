#!/usr/bin/env python3
"""
Analyze and plot benchmark comparison results for thesis Section 5.4.
Generates all the figures and tables needed for the computational efficiency comparison.
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    csv_path = results_dir / "benchmark_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Parse milestone_times JSON column
    df['milestone_times'] = df['milestone_times'].apply(json.loads)
    
    return df


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot throughput (SPS) comparison across different numbers of environments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by framework and num_envs
    grouped = df.groupby(['framework', 'num_envs'])['steps_per_second'].agg(['mean', 'std', 'sem'])
    
    # Plot 1: Absolute throughput
    for framework in df['framework'].unique():
        data = grouped.loc[framework]
        x = data.index
        y = data['mean']
        yerr = data['sem'] * 1.96  # 95% CI
        
        ax1.errorbar(x, y, yerr=yerr, marker='o', label=framework, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('Steps Per Second (SPS)')
    ax1.set_title('Training Throughput Comparison')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Speedup relative to single environment
    for framework in df['framework'].unique():
        data = grouped.loc[framework]
        baseline = data.loc[1, 'mean'] if 1 in data.index else data.iloc[0]['mean']
        
        x = data.index
        speedup = data['mean'] / baseline
        
        ax2.plot(x, speedup, marker='o', label=framework, linewidth=2, markersize=8)
        
    # Add ideal scaling line
    x_ideal = np.array(sorted(df['num_envs'].unique()))
    ax2.plot(x_ideal, x_ideal, 'k--', alpha=0.5, label='Ideal scaling')
    
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('Speedup (relative to 1 environment)')
    ax2.set_title('Scaling Efficiency')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png')
    plt.savefig(output_dir / 'throughput_comparison.pdf')
    print(f"✓ Saved throughput comparison plot")


def plot_memory_usage(df: pd.DataFrame, output_dir: Path):
    """Plot memory usage comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # GPU Memory
    grouped_gpu = df.groupby(['framework', 'num_envs'])['gpu_memory_mb'].agg(['mean', 'std', 'sem'])
    
    for framework in df['framework'].unique():
        if framework in grouped_gpu.index.get_level_values(0):
            data = grouped_gpu.loc[framework]
            x = data.index
            y = data['mean']
            yerr = data['sem'] * 1.96  # 95% CI
            
            ax1.errorbar(x, y, yerr=yerr, marker='s', label=framework, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('GPU Memory Usage (MB)')
    ax1.set_title('GPU Memory Consumption')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # CPU Memory
    grouped_cpu = df.groupby(['framework', 'num_envs'])['cpu_memory_mb'].agg(['mean', 'std', 'sem'])
    
    for framework in df['framework'].unique():
        if framework in grouped_cpu.index.get_level_values(0):
            data = grouped_cpu.loc[framework]
            x = data.index
            y = data['mean']
            yerr = data['sem'] * 1.96  # 95% CI
            
            ax2.errorbar(x, y, yerr=yerr, marker='^', label=framework, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('CPU Memory Usage (MB)')
    ax2.set_title('CPU Memory Consumption')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png')
    plt.savefig(output_dir / 'memory_usage.pdf')
    print(f"✓ Saved memory usage plot")


def plot_milestone_times(df: pd.DataFrame, output_dir: Path):
    """Plot time to reach reward milestones."""
    # Extract milestone data
    milestone_data = []
    for _, row in df.iterrows():
        for milestone, time_val in row['milestone_times'].items():
            milestone_data.append({
                'framework': row['framework'],
                'num_envs': row['num_envs'],
                'seed': row['seed'],
                'milestone': float(milestone),
                'time': time_val
            })
    
    if not milestone_data:
        print("⚠️ No milestone data found")
        return
    
    milestone_df = pd.DataFrame(milestone_data)
    
    # Plot for each milestone
    milestones = sorted(milestone_df['milestone'].unique())
    fig, axes = plt.subplots(1, len(milestones), figsize=(4*len(milestones), 4))
    
    if len(milestones) == 1:
        axes = [axes]
    
    for ax, milestone in zip(axes, milestones):
        milestone_subset = milestone_df[milestone_df['milestone'] == milestone]
        grouped = milestone_subset.groupby(['framework', 'num_envs'])['time'].agg(['mean', 'std', 'sem'])
        
        for framework in milestone_subset['framework'].unique():
            if framework in grouped.index.get_level_values(0):
                data = grouped.loc[framework]
                x = data.index
                y = data['mean']
                yerr = data['sem'] * 1.96  # 95% CI
                
                ax.errorbar(x, y, yerr=yerr, marker='o', label=framework, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Parallel Environments')
        ax.set_ylabel('Time to Milestone (seconds)')
        ax.set_title(f'Time to Reach Reward = {milestone}')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'milestone_times.png')
    plt.savefig(output_dir / 'milestone_times.pdf')
    print(f"✓ Saved milestone times plot")


def plot_jit_overhead(df: pd.DataFrame, output_dir: Path):
    """Plot JIT compilation overhead for SafeBrax."""
    safebrax_df = df[df['framework'] == 'SafeBrax']
    
    if safebrax_df.empty:
        print("⚠️ No SafeBrax data found for JIT analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # JIT time vs num_envs
    grouped = safebrax_df.groupby('num_envs')['jit_compilation_time'].agg(['mean', 'std', 'sem'])
    
    x = grouped.index
    y = grouped['mean']
    yerr = grouped['sem'] * 1.96  # 95% CI
    
    ax1.errorbar(x, y, yerr=yerr, marker='o', color='steelblue', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('JIT Compilation Time (seconds)')
    ax1.set_title('JAX JIT Compilation Overhead')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    
    # JIT time as percentage of total training time
    safebrax_df['jit_percentage'] = (safebrax_df['jit_compilation_time'] / 
                                     (safebrax_df['warmup_time'] + safebrax_df['measure_time'])) * 100
    
    grouped_pct = safebrax_df.groupby('num_envs')['jit_percentage'].agg(['mean', 'std', 'sem'])
    
    x = grouped_pct.index
    y = grouped_pct['mean']
    yerr = grouped_pct['sem'] * 1.96  # 95% CI
    
    ax2.bar(range(len(x)), y, yerr=yerr, color='coral', alpha=0.7, capsize=5)
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels(x, rotation=45)
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('JIT Time as % of Total Training')
    ax2.set_title('Relative JIT Overhead')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'jit_overhead.png')
    plt.savefig(output_dir / 'jit_overhead.pdf')
    print(f"✓ Saved JIT overhead plot")


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table summarizing key metrics."""
    # Select representative num_envs values
    representative_envs = [1, 16, 128, 1024]
    subset = df[df['num_envs'].isin(representative_envs)]
    
    # Aggregate metrics
    summary = subset.groupby(['framework', 'num_envs']).agg({
        'steps_per_second': ['mean', 'std'],
        'gpu_memory_mb': ['mean', 'std'],
        'final_reward': ['mean', 'std'],
        'jit_compilation_time': 'mean'  # SafeBrax only
    })
    
    # Create LaTeX table
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Computational efficiency comparison on Point-Goal environment}",
        r"\label{tab:efficiency_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Framework & Envs & SPS ($\times 10^3$) & GPU Mem (MB) & JIT (s) \\",
        r"\midrule"
    ]
    
    for framework in df['framework'].unique():
        first_row = True
        for num_envs in representative_envs:
            try:
                data = summary.loc[(framework, num_envs)]
                sps_mean = data[('steps_per_second', 'mean')] / 1000
                sps_std = data[('steps_per_second', 'std')] / 1000
                gpu_mean = data[('gpu_memory_mb', 'mean')]
                gpu_std = data[('gpu_memory_mb', 'std')]
                
                if framework == 'SafeBrax':
                    jit_time = data[('jit_compilation_time', 'mean')]
                    jit_str = f"{jit_time:.1f}"
                else:
                    jit_str = "--"
                
                if first_row:
                    fw_name = framework
                    first_row = False
                else:
                    fw_name = ""
                
                latex_lines.append(
                    f"{fw_name} & {num_envs} & "
                    f"{sps_mean:.1f} $\\pm$ {sps_std:.1f} & "
                    f"{gpu_mean:.0f} $\\pm$ {gpu_std:.0f} & "
                    f"{jit_str} \\\\"
                )
            except KeyError:
                continue
        
        if framework != df['framework'].unique()[-1]:
            latex_lines.append(r"\midrule")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    # Save LaTeX table
    latex_path = output_dir / 'efficiency_comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✓ Saved LaTeX table to {latex_path}")
    
    # Also save as CSV for reference
    summary.to_csv(output_dir / 'efficiency_summary.csv')


def plot_combined_efficiency(df: pd.DataFrame, output_dir: Path):
    """Create combined figure for thesis showing key efficiency metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Throughput comparison
    ax = axes[0, 0]
    grouped = df.groupby(['framework', 'num_envs'])['steps_per_second'].agg(['mean', 'sem'])
    
    colors = {'SafeBrax': 'steelblue', 'Safety-Gymnasium': 'coral'}
    markers = {'SafeBrax': 'o', 'Safety-Gymnasium': 's'}
    
    for framework in df['framework'].unique():
        if framework in grouped.index.get_level_values(0):
            data = grouped.loc[framework]
            x = data.index
            y = data['mean'] / 1000  # Convert to thousands
            yerr = data['sem'] * 1.96 / 1000  # 95% CI
            
            ax.errorbar(x, y, yerr=yerr, marker=markers[framework], 
                       label=framework, linewidth=2, markersize=8,
                       color=colors.get(framework, None))
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Throughput (×10³ SPS)')
    ax.set_title('(a) Training Throughput')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Scaling efficiency
    ax = axes[0, 1]
    for framework in df['framework'].unique():
        if framework in grouped.index.get_level_values(0):
            data = grouped.loc[framework]
            baseline = data.loc[1, 'mean'] if 1 in data.index else data.iloc[0]['mean']
            
            x = data.index
            speedup = data['mean'] / baseline
            
            ax.plot(x, speedup, marker=markers[framework], label=framework, 
                   linewidth=2, markersize=8, color=colors.get(framework, None))
    
    # Add ideal scaling line
    x_ideal = np.array(sorted(df['num_envs'].unique()))
    ax.plot(x_ideal, x_ideal, 'k--', alpha=0.5, label='Ideal scaling', linewidth=1)
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('(b) Scaling Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Memory usage
    ax = axes[1, 0]
    grouped_mem = df.groupby(['framework', 'num_envs'])['gpu_memory_mb'].agg(['mean', 'sem'])
    
    for framework in df['framework'].unique():
        if framework in grouped_mem.index.get_level_values(0):
            data = grouped_mem.loc[framework]
            x = data.index
            y = data['mean']
            yerr = data['sem'] * 1.96  # 95% CI
            
            ax.errorbar(x, y, yerr=yerr, marker=markers[framework], 
                       label=framework, linewidth=2, markersize=8,
                       color=colors.get(framework, None))
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('(c) Memory Consumption')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Time to convergence (first milestone)
    ax = axes[1, 1]
    
    # Extract first milestone data
    first_milestone = None
    milestone_data = []
    for _, row in df.iterrows():
        if row['milestone_times']:
            milestones = list(row['milestone_times'].keys())
            if milestones:
                if first_milestone is None:
                    first_milestone = min(float(m) for m in milestones)
                
                if str(first_milestone) in row['milestone_times']:
                    milestone_data.append({
                        'framework': row['framework'],
                        'num_envs': row['num_envs'],
                        'time': row['milestone_times'][str(first_milestone)]
                    })
    
    if milestone_data:
        milestone_df = pd.DataFrame(milestone_data)
        grouped_milestone = milestone_df.groupby(['framework', 'num_envs'])['time'].agg(['mean', 'sem'])
        
        for framework in milestone_df['framework'].unique():
            if framework in grouped_milestone.index.get_level_values(0):
                data = grouped_milestone.loc[framework]
                x = data.index
                y = data['mean']
                yerr = data['sem'] * 1.96  # 95% CI
                
                ax.errorbar(x, y, yerr=yerr, marker=markers[framework], 
                           label=framework, linewidth=2, markersize=8,
                           color=colors.get(framework, None))
        
        ax.set_xlabel('Number of Parallel Environments')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'(d) Time to Reward = {first_milestone}')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('SafeBrax vs Safety-Gymnasium: Computational Efficiency Comparison', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_efficiency_comparison.png')
    plt.savefig(output_dir / 'combined_efficiency_comparison.pdf')
    print(f"✓ Saved combined efficiency comparison plot")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate plots for thesis"
    )
    parser.add_argument('--results-dir', type=str, required=True,
                      help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for plots (default: results-dir/analysis)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📊 Analyzing Benchmark Results")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load results
    df = load_results(results_dir)
    print(f"\n✓ Loaded {len(df)} benchmark results")
    print(f"  Frameworks: {df['framework'].unique()}")
    print(f"  Num envs tested: {sorted(df['num_envs'].unique())}")
    print(f"  Seeds per config: {df.groupby(['framework', 'num_envs']).size().mean():.1f}")
    
    # Generate plots
    print("\n📈 Generating plots...")
    
    plot_throughput_comparison(df, output_dir)
    plot_memory_usage(df, output_dir)
    plot_milestone_times(df, output_dir)
    plot_jit_overhead(df, output_dir)
    plot_combined_efficiency(df, output_dir)
    
    # Generate summary table
    print("\n📄 Generating summary table...")
    generate_summary_table(df, output_dir)
    
    # Print key statistics
    print("\n📊 Key Statistics:")
    print("-" * 40)
    
    for framework in df['framework'].unique():
        fw_df = df[df['framework'] == framework]
        max_sps = fw_df['steps_per_second'].max()
        avg_sps = fw_df['steps_per_second'].mean()
        max_envs = fw_df.loc[fw_df['steps_per_second'].idxmax(), 'num_envs']
        
        print(f"\n{framework}:")
        print(f"  Peak throughput: {max_sps:,.0f} SPS (at {max_envs} envs)")
        print(f"  Average throughput: {avg_sps:,.0f} SPS")
        
        if framework == 'SafeBrax':
            avg_jit = fw_df['jit_compilation_time'].mean()
            print(f"  Average JIT time: {avg_jit:.1f}s")
    
    # Calculate speedup
    safebrax_avg = df[df['framework'] == 'SafeBrax']['steps_per_second'].mean()
    safety_avg = df[df['framework'] == 'Safety-Gymnasium']['steps_per_second'].mean()
    
    if safety_avg > 0:
        speedup = safebrax_avg / safety_avg
        print(f"\n🚀 SafeBrax average speedup: {speedup:.1f}x")
    
    print("\n✅ Analysis complete!")
    print(f"📁 All results saved to: {output_dir}")


if __name__ == "__main__":
    main()


