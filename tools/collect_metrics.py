#!/usr/bin/env python3
import argparse
import csv
import os
import time
from typing import Dict, Any

from brax import envs
from brax.training.agents import ppo
from brax.training.agents import ppo_lagrange_v2 as ppo_lagr
from brax.training.agents.ppo import ppo_cost as ppo_cost


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _split_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    categorized = {
        "train": {},
        "eval": {},
        "other": {},
    }
    for k, v in metrics.items():
        if k.startswith("training/"):
            categorized["train"][k[len("training/"):]] = v
        elif k.startswith("eval/"):
            categorized["eval"][k[len("eval/"):]] = v
        else:
            categorized["other"][k] = v
    return categorized


def _append_row(csv_path: str, row: Dict[str, Any]) -> None:
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp"] + sorted(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow({"timestamp": _timestamp(), **row})


def run_experiment(
    env_name: str,
    alg: str,
    out_dir: str,
    num_timesteps: int,
    # env + rollout config
    episode_length: int,
    num_envs: int,
    action_repeat: int,
    unroll_length: int,
    batch_size: int,
    num_minibatches: int,
    num_updates_per_batch: int,
    # optimization
    learning_rate: float,
    entropy_cost: float,
    discounting: float,
    reward_scaling: float,
    gae_lambda: float,
    clipping_epsilon: float,
    normalize_observations: bool,
    # eval + logging
    num_evals: int,
    num_eval_envs: int,
    deterministic_eval: bool,
    training_metrics_steps: int | None,
    seed: int,
    # ppo-lagrange only
    safety_bound: float | None,
    lagrangian_coef_rate: float | None,
    initial_lambda_lagr: float | None,
    # ppo-cost only
    cost_weight: float | None,
):
    _ensure_dir(out_dir)
    env = envs.get_environment(env_name)

    run_tag = f"{env_name}_{alg}"
    train_csv = os.path.join(out_dir, f"{run_tag}_train_metrics.csv")
    eval_csv = os.path.join(out_dir, f"{run_tag}_eval_metrics.csv")

    def progress(step: int, metrics: Dict[str, Any]):
        categorized = _split_metrics(metrics)
        train_row = {"env_steps": step}
        eval_row = {"env_steps": step}

        # Training metrics of interest
        # reward curve: mean_episode_return
        if "mean_episode_return" in categorized["train"]:
            train_row["mean_episode_return"] = float(categorized["train"]["mean_episode_return"])  # noqa: E501
        # cost/violation
        for k in ["mean_cost", "cost_violation", "violation_rate"]:
            if k in categorized["train"]:
                train_row[k] = float(categorized["train"][k])
        # lambda (ppo-lagr)
        if "lambda_lagr" in categorized["train"]:
            train_row["lambda_lagr"] = float(categorized["train"]["lambda_lagr"])  # noqa: E501
        # SPS and walltime
        if "sps" in categorized["train"]:
            train_row["sps"] = float(categorized["train"]["sps"])  # steps/sec
        if "walltime" in categorized["train"]:
            train_row["walltime"] = float(categorized["train"]["walltime"])  # seconds

        # Add any other scalar training metrics (stable order)
        for k, v in sorted(categorized["train"].items()):
            if k not in train_row and isinstance(v, (int, float)):
                train_row[k] = float(v)

        if len(train_row) > 1:
            _append_row(train_csv, train_row)

        # Eval metrics
        if categorized["eval"]:
            for k, v in sorted(categorized["eval"].items()):
                if isinstance(v, (int, float)):
                    eval_row[k] = float(v)
            if len(eval_row) > 1:
                _append_row(eval_csv, eval_row)

    common_kwargs = dict(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        action_repeat=action_repeat,
        num_envs=num_envs,
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
        num_evals=num_evals,
        num_eval_envs=num_eval_envs,
        deterministic_eval=deterministic_eval,
        log_training_metrics=True,
        training_metrics_steps=training_metrics_steps,
        progress_fn=progress,
    )

    if alg == "ppo":
        ppo.train(**common_kwargs)
    elif alg in ("ppo_cost", "ppo-cost"):
        if cost_weight is None:
            cost_weight = 1.0
        ppo_cost.train_ppo_cost(
            **common_kwargs,
            cost_weight=cost_weight,
        )
    elif alg in ("ppo_lagrange", "ppo-lagrange", "ppol"):
        if safety_bound is None:
            raise ValueError("safety_bound must be provided for PPO-Lagrange")
        if lagrangian_coef_rate is None:
            raise ValueError("lagrangian_coef_rate must be provided for PPO-Lagrange")
        if initial_lambda_lagr is None:
            initial_lambda_lagr = 0.0
        ppo_lagr.train(
            **common_kwargs,
            safety_bound=safety_bound,
            lagrangian_coef_rate=lagrangian_coef_rate,
            initial_lambda_lagr=initial_lambda_lagr,
        )
    else:
        raise ValueError(f"Unknown alg: {alg}")


def main():
    parser = argparse.ArgumentParser(description="Collect train/eval metrics to CSV for PPO/PPO-Lagrange.")
    parser.add_argument("--env", required=True, help="Environment name (see brax.envs)")
    parser.add_argument("--alg", required=True, choices=["ppo", "ppo_lagrange", "ppo_cost"], help="Algorithm")
    parser.add_argument("--out_dir", required=True, help="Output directory for CSV logs")
    parser.add_argument("--num_timesteps", type=int, default=3_000_000)
    # env + rollout
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--unroll_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--num_updates_per_batch", type=int, default=4)
    # optimization
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_cost", type=float, default=1e-3)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward_scaling", type=float, default=1.0)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clipping_epsilon", type=float, default=0.3)
    parser.add_argument("--normalize_observations", type=int, default=1)
    # eval + logging
    parser.add_argument("--num_evals", type=int, default=5)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--deterministic_eval", type=int, default=0)
    parser.add_argument("--training_metrics_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # ppo-lagrange-only
    parser.add_argument("--safety_bound", type=float, default=None)
    parser.add_argument("--lagrangian_coef_rate", type=float, default=0.01)
    parser.add_argument("--initial_lambda_lagr", type=float, default=0.0)
    # ppo-cost-only
    parser.add_argument("--cost_weight", type=float, default=1.0)

    args = parser.parse_args()

    # If logging period not set, default to one rollout's worth of env steps
    if args.training_metrics_steps is None:
        args.training_metrics_steps = args.batch_size * args.unroll_length * args.num_minibatches * args.action_repeat

    run_experiment(
        env_name=args.env,
        alg=args.alg,
        out_dir=args.out_dir,
        num_timesteps=args.num_timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
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
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        deterministic_eval=bool(args.deterministic_eval),
        training_metrics_steps=args.training_metrics_steps,
        seed=args.seed,
        safety_bound=args.safety_bound,
        lagrangian_coef_rate=args.lagrangian_coef_rate,
        initial_lambda_lagr=args.initial_lambda_lagr,
        cost_weight=args.cost_weight if args.alg in ("ppo_cost", "ppo-cost") else None,
    )


if __name__ == "__main__":
    main()




