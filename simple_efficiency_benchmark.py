#!/usr/bin/env python3
"""
Simplified efficiency benchmark for thesis Section 5.4.
Focuses on the core metrics: throughput, memory, and scaling.
"""

import time
import os
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# GPU and performance optimizations
def setup_gpu_environment():
    """Setup GPU environment for optimal performance."""
    # Check for GPU availability
    try:
        if subprocess.run(['nvidia-smi'], capture_output=True).returncode != 0:
            print("⚠️ Warning: Cannot communicate with GPU. Running on CPU.")
    except FileNotFoundError:
        print("⚠️ Warning: nvidia-smi not found. Running on CPU.")
    
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
        xla_flags += ' --xla_gpu_triton_gemm_any=True'
        os.environ['XLA_FLAGS'] = xla_flags
        print("✓ XLA Triton GEMM optimization enabled (~30% speedup)")
    
    print(f"✓ XLA flags configured: {os.environ.get('XLA_FLAGS', '')}")

# Apply optimizations before importing JAX
setup_gpu_environment()

# System monitoring
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available - GPU metrics will be skipped")

# JAX/SafeBrax (import after setting environment)
import jax
import jax.numpy as jnp
from brax import envs

# Verify JAX backend
print(f"✓ JAX backend: {jax.default_backend()}")
print(f"✓ JAX devices: {jax.devices()}")


def measure_safebrax_throughput(num_envs: int, num_steps: int = 1_000_000) -> Dict:
    """Measure SafeBrax throughput using random actions (no training).

    Args:
        num_envs: Number of parallel batched environments.
        num_steps: TOTAL environment steps across all envs (matches PPO semantics).
    """
    print(f"\n📊 SafeBrax benchmark (num_envs={num_envs})...")

    # Create a batched environment
    try:
        env = envs.create(
            env_name='safe_point_goal',
            batch_size=num_envs,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
        )
    except Exception:
        # Fallback in case create is unavailable: use get_environment and assume it supports batching via batch_size
        env = envs.get_environment('safe_point_goal')

    # Monitor resources before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        gpu_mem_before = gpus[0].memoryUsed if gpus else 0
    else:
        gpu_mem_before = 0

    # RNG and initial state
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Build a chunked JIT-compiled random-action rollout to avoid huge compilations
    def make_rollout(length: int):
        def rollout(rng_key, init_state):
            def body(carry, _):
                rng_inner, s = carry
                rng_inner, key = jax.random.split(rng_inner)
                actions = jax.random.uniform(
                    key,
                    shape=(num_envs, env.action_size),
                    minval=-1.0,
                    maxval=1.0,
                )
                s = env.step(s, actions)
                return (rng_inner, s), None

            (rng_out, s_out), _ = jax.lax.scan(body, (rng_key, init_state), xs=None, length=length)
            return rng_out, s_out

        return jax.jit(rollout)

    # Translate total env-steps to per-env steps (ceil) to match PPO semantics
    steps_per_env = (num_steps + max(1, num_envs) - 1) // max(1, num_envs)

    # Chunking setup based on per-env steps to avoid huge compilations
    chunk_len = 4096 if steps_per_env >= 4096 else steps_per_env
    rollout_chunk = make_rollout(chunk_len)
    num_full = steps_per_env // chunk_len
    remainder = steps_per_env - num_full * chunk_len

    # Precisely measure JIT compile time vs execution time
    jit_time = 0.0
    exec_time = 0.0

    # Compile chunk kernel once
    t0 = time.time()
    compiled_chunk = rollout_chunk.lower(rng, state).compile()
    jit_time += time.time() - t0

    # Execute chunk kernel num_full times
    t1 = time.time()
    for _ in range(num_full):
        rng, state = compiled_chunk(rng, state)
    exec_time += time.time() - t1

    # Compile and execute remainder kernel if needed
    if remainder:
        rollout_rem = make_rollout(remainder)
        t2 = time.time()
        compiled_rem = rollout_rem.lower(rng, state).compile()
        jit_time += time.time() - t2
        t3 = time.time()
        rng, state = compiled_rem(rng, state)
        exec_time += time.time() - t3

    # Ensure all computations finish before final timing
    _ = jnp.sum(state.obs).block_until_ready()

    # Monitor resources after
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before

    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        gpu_mem_after = gpus[0].memoryUsed if gpus else 0
        gpu_mem_used = gpu_mem_after - gpu_mem_before
    else:
        gpu_mem_used = 0

    # Steps per second counts all env steps; use execution time only (exclude JIT)
    total_env_steps = steps_per_env * max(1, num_envs)
    sps = total_env_steps / exec_time if exec_time > 0 else float('inf')

    print(f"  ✓ SPS: {sps:,.0f}")
    print(f"  ✓ Memory: CPU={mem_used:.0f}MB, GPU={gpu_mem_used:.0f}MB")
    print(f"  ✓ Time: {exec_time:.1f}s (JIT {jit_time:.1f}s)")

    return {
        'framework': 'SafeBrax',
        'num_envs': num_envs,
        'steps_per_second': sps,
        'cpu_memory_mb': mem_used,
        'gpu_memory_mb': gpu_mem_used,
        'total_time': exec_time,
        'jit_time': jit_time,
        'num_steps': num_steps,
    }


def measure_safety_gymnasium_throughput(num_envs: int, num_steps: int = 100_000) -> Dict:
    """Measure Safety-Gymnasium throughput (simplified using step timing)."""
    print(f"\n📊 Safety-Gymnasium benchmark (num_envs={num_envs})...")
    
    try:
        import safety_gymnasium
        from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium
    except ImportError:
        print("  ⚠️ Safety-Gymnasium not available")
        return {}
    
    # Create environment
    env = safety_gymnasium.make('SafetyPointGoal1-v0')
    env = SafetyGymnasium2Gymnasium(env)
    
    # Monitor resources before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        gpu_mem_before = gpus[0].memoryUsed if gpus else 0
    else:
        gpu_mem_before = 0
    
    # Measure stepping throughput
    obs, _ = env.reset(seed=42)
    
    # Warmup
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Measure
    start_time = time.time()
    steps_done = 0
    
    while steps_done < num_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        steps_done += 1
    
    total_time = time.time() - start_time
    sps = steps_done / total_time
    
    # Monitor resources after
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        gpu_mem_after = gpus[0].memoryUsed if gpus else 0
        gpu_mem_used = gpu_mem_after - gpu_mem_before
    else:
        gpu_mem_used = 0
    
    env.close()
    
    print(f"  ✓ SPS: {sps:,.0f}")
    print(f"  ✓ Memory: CPU={mem_used:.0f}MB, GPU={gpu_mem_used:.0f}MB")
    print(f"  ✓ Time: {total_time:.1f}s")
    
    return {
        'framework': 'Safety-Gymnasium',
        'num_envs': num_envs,  # Note: single env for now
        'steps_per_second': sps,
        'cpu_memory_mb': mem_used,
        'gpu_memory_mb': gpu_mem_used,
        'total_time': total_time,
        'jit_time': 0,
        'num_steps': num_steps
    }


def plot_results(results: List[Dict], output_dir: Path):
    """Generate comparison plots."""
    if not results:
        print("No results to plot")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    
    # Prepare data
    safebrax_results = [r for r in results if r['framework'] == 'SafeBrax']
    safety_results = [r for r in results if r['framework'] == 'Safety-Gymnasium']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Throughput comparison
    ax = axes[0]
    if safebrax_results:
        x = [r['num_envs'] for r in safebrax_results]
        y = [r['steps_per_second'] for r in safebrax_results]
        ax.plot(x, y, 'o-', label='SafeBrax', linewidth=2, markersize=8)
    
    if safety_results:
        x = [r['num_envs'] for r in safety_results]
        y = [r['steps_per_second'] for r in safety_results]
        ax.plot(x, y, 's-', label='Safety-Gymnasium', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Steps Per Second')
    ax.set_title('Training Throughput')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Scaling efficiency
    ax = axes[1]
    if safebrax_results:
        baseline = safebrax_results[0]['steps_per_second']
        x = [r['num_envs'] for r in safebrax_results]
        y = [r['steps_per_second'] / baseline for r in safebrax_results]
        ax.plot(x, y, 'o-', label='SafeBrax', linewidth=2, markersize=8)
    
    # Ideal scaling line
    if safebrax_results:
        x_ideal = [r['num_envs'] for r in safebrax_results]
        ax.plot(x_ideal, x_ideal, 'k--', alpha=0.5, label='Ideal scaling')
    
    ax.set_xlabel('Number of Parallel Environments')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Scaling Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # (Removed memory usage subplot due to unreliable GPU stats on this setup)
    
    plt.suptitle('SafeBrax vs Safety-Gymnasium: Efficiency Comparison', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'efficiency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Plot saved to: {output_path}")
    
    # Also save as PDF for thesis
    output_path_pdf = output_dir / 'efficiency_comparison.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"📄 PDF saved to: {output_path_pdf}")


def generate_latex_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table for thesis."""
    if not results:
        return
    
    # Group results
    safebrax_results = [r for r in results if r['framework'] == 'SafeBrax']
    safety_results = [r for r in results if r['framework'] == 'Safety-Gymnasium']
    
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Computational efficiency comparison: SafeBrax vs Safety-Gymnasium}",
        r"\label{tab:efficiency_comparison}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Framework & Envs & SPS & GPU Mem (MB) & JIT (s) \\",
        r"\midrule"
    ]
    
    # Add SafeBrax results
    for r in safebrax_results:
        latex_lines.append(
            f"SafeBrax & {r['num_envs']} & "
            f"{r['steps_per_second']:,.0f} & "
            f"{r['gpu_memory_mb']:.0f} & "
            f"{r['jit_time']:.1f} \\\\"
        )
    
    if safebrax_results and safety_results:
        latex_lines.append(r"\midrule")
    
    # Add Safety-Gymnasium results
    for r in safety_results:
        latex_lines.append(
            f"Safety-Gymnasium & {r['num_envs']} & "
            f"{r['steps_per_second']:,.0f} & "
            f"{r['gpu_memory_mb']:.0f} & "
            f"-- \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    # Save table
    table_path = output_dir / 'efficiency_table.tex'
    with open(table_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"📄 LaTeX table saved to: {table_path}")


def main():
    """Run simplified efficiency benchmark."""
    print("=" * 60)
    print("🏁 Simplified Efficiency Benchmark")
    print("=" * 60)
    
    # Configuration
    num_envs_list = [1, 4, 16, 64, 256, 1024, 2048]
    output_dir = Path(f"efficiency_results_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🔢 Testing with num_envs: {num_envs_list}")
    
    results = []
    csv_path = output_dir / 'benchmark_results.csv'
    
    # Run SafeBrax benchmarks
    print("\n" + "=" * 40)
    print("Running SafeBrax benchmarks...")
    print("=" * 40)
    
    for num_envs in num_envs_list:
        try:
            result = measure_safebrax_throughput(num_envs, num_steps=500_000)
            results.append(result)
            
            # Save to CSV
            file_exists = csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)
        except Exception as e:
            print(f"  ❌ Failed for num_envs={num_envs}: {e}")
    
    # Run Safety-Gymnasium benchmark (single env for comparison)
    print("\n" + "=" * 40)
    print("Running Safety-Gymnasium benchmark...")
    print("=" * 40)
    
    try:
        result = measure_safety_gymnasium_throughput(1, num_steps=50_000)
        if result:
            results.append(result)
            
            # Save to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                writer.writerow(result)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Generate plots and tables
    print("\n" + "=" * 40)
    print("Generating plots and tables...")
    print("=" * 40)
    
    plot_results(results, output_dir)
    generate_latex_table(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Benchmark Summary")
    print("=" * 60)
    
    if results:
        safebrax_results = [r for r in results if r['framework'] == 'SafeBrax']
        safety_results = [r for r in results if r['framework'] == 'Safety-Gymnasium']
        
        if safebrax_results:
            max_sps = max(r['steps_per_second'] for r in safebrax_results)
            print("\nSafeBrax:")
            print(f"  Peak throughput: {max_sps:,.0f} SPS")
            
            # Find best num_envs
            best = max(safebrax_results, key=lambda r: r['steps_per_second'])
            print(f"  Best config: {best['num_envs']} envs")
            
            # Average JIT time
            avg_jit = np.mean([r['jit_time'] for r in safebrax_results])
            print(f"  Avg JIT time: {avg_jit:.1f}s")
        
        if safety_results:
            sps = safety_results[0]['steps_per_second']
            print("\nSafety-Gymnasium:")
            print(f"  Throughput: {sps:,.0f} SPS")
        
        # Calculate speedup
        if safebrax_results and safety_results:
            # Compare single env performance
            safebrax_single = next((r for r in safebrax_results if r['num_envs'] == 1), None)
            safety_single = safety_results[0]
            
            if safebrax_single:
                speedup = safebrax_single['steps_per_second'] / safety_single['steps_per_second']
                print(f"\n🚀 SafeBrax speedup (1 env): {speedup:.1f}x")
            
            # Peak speedup
            peak_speedup = max_sps / safety_single['steps_per_second']
            print(f"🚀 SafeBrax peak speedup: {peak_speedup:.1f}x")
    
    print("\n✅ Benchmark complete!")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
