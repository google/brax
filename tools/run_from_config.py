#!/usr/bin/env python3
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run experiment from JSON config.")
    parser.add_argument("config", help="Path to JSON config file")
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Ensure repo root on path so `tools.*` imports work when running from repo root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Throughput sweep if num_envs_list present, else training/eval
    if "num_envs_list" in cfg:
        from tools.throughput_sweep import sweep_throughput
        sweep_throughput(
            env_name=cfg["env"],
            alg=cfg["alg"],
            out_dir=cfg["out_dir"],
            num_envs_list=cfg["num_envs_list"],
            warmup_timesteps=cfg.get("warmup_timesteps", 200_000),
            measure_timesteps=cfg.get("measure_timesteps", 400_000),
            episode_length=cfg.get("episode_length", 1000),
            action_repeat=cfg.get("action_repeat", 1),
            unroll_length=cfg.get("unroll_length", 10),
            batch_size=cfg.get("batch_size", 512),
            num_minibatches=cfg.get("num_minibatches", 32),
            num_updates_per_batch=cfg.get("num_updates_per_batch", 4),
            learning_rate=cfg.get("learning_rate", 3e-4),
            entropy_cost=cfg.get("entropy_cost", 1e-3),
            discounting=cfg.get("discounting", 0.99),
            reward_scaling=cfg.get("reward_scaling", 1.0),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clipping_epsilon=cfg.get("clipping_epsilon", 0.3),
            normalize_observations=cfg.get("normalize_observations", True),
            seed=cfg.get("seed", 0),
            env_kwargs_json=json.dumps(cfg.get("env_kwargs", {})) if cfg.get("env_kwargs") else None,
            safety_bound=cfg.get("safety_bound"),
            lagrangian_coef_rate=cfg.get("lagrangian_coef_rate"),
            initial_lambda_lagr=cfg.get("initial_lambda_lagr"),
        )
    else:
        from tools.collect_metrics import run_experiment
        run_experiment(
            env_name=cfg["env"],
            alg=cfg["alg"],
            out_dir=cfg["out_dir"],
            num_timesteps=cfg.get("num_timesteps", 3_000_000),
            episode_length=cfg.get("episode_length", 1000),
            num_envs=cfg.get("num_envs", 1024),
            action_repeat=cfg.get("action_repeat", 1),
            unroll_length=cfg.get("unroll_length", 10),
            batch_size=cfg.get("batch_size", 512),
            num_minibatches=cfg.get("num_minibatches", 32),
            num_updates_per_batch=cfg.get("num_updates_per_batch", 4),
            learning_rate=cfg.get("learning_rate", 3e-4),
            entropy_cost=cfg.get("entropy_cost", 1e-3),
            discounting=cfg.get("discounting", 0.99),
            reward_scaling=cfg.get("reward_scaling", 1.0),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clipping_epsilon=cfg.get("clipping_epsilon", 0.3),
            normalize_observations=cfg.get("normalize_observations", True),
            num_evals=cfg.get("num_evals", 5),
            num_eval_envs=cfg.get("num_eval_envs", 128),
            deterministic_eval=cfg.get("deterministic_eval", False),
            training_metrics_steps=cfg.get("training_metrics_steps"),
            seed=cfg.get("seed", 0),
            env_kwargs_json=json.dumps(cfg.get("env_kwargs", {})) if cfg.get("env_kwargs") else None,
            safety_bound=cfg.get("safety_bound"),
            lagrangian_coef_rate=cfg.get("lagrangian_coef_rate"),
            initial_lambda_lagr=cfg.get("initial_lambda_lagr"),
            cost_weight=cfg.get("cost_weight"),
        )


if __name__ == "__main__":
    main()
