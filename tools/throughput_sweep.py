#!/usr/bin/env python3
import argparse
import csv
import os
import time
from typing import Dict, Any, List

from brax import envs
from brax.training.agents import ppo
from brax.training.agents import ppo_lagrange_v2 as ppo_lagr


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sweep_throughput(
    env_name: str,
    alg: str,
    out_dir: str,
    num_envs_list: List[int],
    # short run params
    warmup_timesteps: int,
    measure_timesteps: int,
    episode_length: int,
    action_repeat: int,
    unroll_length: int,
    batch_size: int,
    num_minibatches: int,
    num_updates_per_batch: int,
    learning_rate: float,
    entropy_cost: float,
    discounting: float,
    reward_scaling: float,
    gae_lambda: float,
    clipping_epsilon: float,
    normalize_observations: bool,
    seed: int,
    # lagrange
    safety_bound: float | None,
    lagrangian_coef_rate: float | None,
    initial_lambda_lagr: float | None,
):
    _ensure_dir(out_dir)
    env = envs.get_environment(env_name)
    # Create organized directory structure for throughput results
    throughput_dir = os.path.join(out_dir, "throughput", env_name.replace("_", "-"))
    _ensure_dir(throughput_dir)
    
    # Create descriptive filename with algorithm and key parameters
    filename_parts = ["throughput", env_name, alg]
    if safety_bound is not None:
        filename_parts.append(f"bound{safety_bound}")
    filename_parts.append(f"seed{seed}")
    
    csv_path = os.path.join(throughput_dir, f"{'_'.join(filename_parts)}.csv")

    def run_short(num_envs: int) -> float:
        sps_values: List[float] = []

        def progress(_, metrics: Dict[str, Any]):
            if "training/sps" in metrics:
                sps_values.append(float(metrics["training/sps"]))

        common = dict(
            environment=env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            unroll_length=unroll_length,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            learning_rate=learning_rate,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_observations=normalize_observations,
            seed=seed,
            num_evals=2,
            num_eval_envs=min(128, num_envs),
            deterministic_eval=True,
            log_training_metrics=True,
            training_metrics_steps=batch_size * unroll_length * num_minibatches * action_repeat,
            progress_fn=progress,
        )

        # Warmup pass (JIT compile etc.)
        if alg == "ppo":
            ppo.train(num_timesteps=warmup_timesteps, num_envs=num_envs, **common)
        else:
            ppo_lagr.train(
                num_timesteps=warmup_timesteps,
                num_envs=num_envs,
                safety_bound=safety_bound if safety_bound is not None else 0.0,
                lagrangian_coef_rate=lagrangian_coef_rate if lagrangian_coef_rate is not None else 0.01,
                initial_lambda_lagr=initial_lambda_lagr if initial_lambda_lagr is not None else 0.0,
                **common,
            )

        # Measure pass
        sps_values.clear()
        if alg == "ppo":
            ppo.train(num_timesteps=measure_timesteps, num_envs=num_envs, **common)
        else:
            ppo_lagr.train(
                num_timesteps=measure_timesteps,
                num_envs=num_envs,
                safety_bound=safety_bound if safety_bound is not None else 0.0,
                lagrangian_coef_rate=lagrangian_coef_rate if lagrangian_coef_rate is not None else 0.01,
                initial_lambda_lagr=initial_lambda_lagr if initial_lambda_lagr is not None else 0.0,
                **common,
            )

        return sum(sps_values) / max(1, len(sps_values))

    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["num_envs", "sps_mean", "alg", "env", "timestamp"],
        )
        if is_new:
            writer.writeheader()
        for nenv in num_envs_list:
            sps_mean = run_short(nenv)
            writer.writerow(
                {
                    "num_envs": nenv,
                    "sps_mean": sps_mean,
                    "alg": alg,
                    "env": env_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Sweep throughput over num_envs and log CSV.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--alg", required=True, choices=["ppo", "ppo_lagrange"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_envs_list", required=True, help="Comma-separated num_envs list, e.g., 64,128,256,512,1024")
    parser.add_argument("--warmup_timesteps", type=int, default=200_000)
    parser.add_argument("--measure_timesteps", type=int, default=400_000)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--unroll_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--num_updates_per_batch", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_cost", type=float, default=1e-3)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward_scaling", type=float, default=1.0)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clipping_epsilon", type=float, default=0.3)
    parser.add_argument("--normalize_observations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    # Lagrange
    parser.add_argument("--safety_bound", type=float, default=0.0)
    parser.add_argument("--lagrangian_coef_rate", type=float, default=0.01)
    parser.add_argument("--initial_lambda_lagr", type=float, default=0.0)

    args = parser.parse_args()
    num_envs_list = [int(x) for x in args.num_envs_list.split(",") if x]

    sweep_throughput(
        env_name=args.env,
        alg=args.alg,
        out_dir=args.out_dir,
        num_envs_list=num_envs_list,
        warmup_timesteps=args.warmup_timesteps,
        measure_timesteps=args.measure_timesteps,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=args.reward_scaling,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        normalize_observations=bool(args.normalize_observations),
        seed=args.seed,
        safety_bound=args.safety_bound,
        lagrangian_coef_rate=args.lagrangian_coef_rate,
        initial_lambda_lagr=args.initial_lambda_lagr,
    )


if __name__ == "__main__":
    main()




