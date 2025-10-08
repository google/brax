"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import argparse
import csv
import functools
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from matplotlib import pyplot as plt

from brax import envs
from brax.envs import Env
from brax.envs.base import State as BraxState, Wrapper

# Import training modules conditionally to handle missing modules gracefully
try:
    from brax.training.agents.ppo.train import train as ppo_train
except ImportError:
    try:
        from brax.training.agents.ppo import train as ppo_train
    except ImportError:
        try:
            from brax.training.agents import ppo

            ppo_train = ppo.train
        except:
            ppo_train = None

try:
    from brax.training.agents.ppo.ppo_cost import train_ppo_cost, RewardMinusCostWrapper
except ImportError:
    try:
        from brax.training.agents.ppo import train_ppo_cost
        from brax.training.agents.ppo import RewardMinusCostWrapper  # type: ignore
    except ImportError:
        train_ppo_cost = None
        RewardMinusCostWrapper = None  # type: ignore

try:
    from brax.training.agents.ppo_lagrange_v3 import train as ppo_lagrange_v3_train
except ImportError:
    try:
        from brax.training.agents import ppo_lagrange_v3

        ppo_lagrange_v3_train = ppo_lagrange_v3.train
    except ImportError:
        ppo_lagrange_v3_train = None

try:
    from brax.training.agents.ppo_lagrange_v2 import train as ppo_lagrange_v2_train
except ImportError:
    try:
        from brax.training.agents import ppo_lagrange_v2

        ppo_lagrange_v2_train = ppo_lagrange_v2.train
    except ImportError:
        ppo_lagrange_v2_train = None

try:
    from brax.training.agents.ppo_lagrange import train as ppo_lagrange_train
except ImportError:
    try:
        from brax.training.agents import ppo_lagrange

        ppo_lagrange_train = ppo_lagrange.train
    except ImportError:
        ppo_lagrange_train = None
from brax.io import model as brax_model
from brax.io import json as brax_json
import wandb
from ml_collections import config_dict


# Configure environment for GPU usage
def setup_gpu_environment():
    """Setup GPU environment for MuJoCo and XLA."""
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    os.environ['MUJOCO_GL'] = 'egl'

    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    # Check installation
    try:
        print('Checking that the installation succeeded:')
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print('Installation successful.')
    except Exception as e:
        raise RuntimeError(
            'Something went wrong during installation. Check the error message above '
            'for more information.'
        ) from e


def get_default_env_config() -> config_dict.ConfigDict:
    """Returns the default config for PointHazardGoal environment."""
    config = config_dict.create(
        # New safety-gymnasium reward parameters
        reward_distance=3,
        reward_goal=10.0,
        goal_size=0.7,
        reward_orientation=False,
        reward_orientation_scale=0.002,
        reward_orientation_body='agent',
        ctrl_cost_weight=0.001,
        hazard_size=0.7,
        # Other parameters
        terminate_when_unhealthy=True,
        healthy_z_range=(0.05, 0.3),
        reset_noise_scale=0.005,
        exclude_current_positions_from_observation=True,
        max_velocity=5.0,
        debug=False,
    )
    return config


class CostExtraWrapper(Wrapper):
    """Wrapper that moves cost from info to extras for PPO Lagrange."""

    def step(self, state: BraxState, action: jax.Array) -> BraxState:
        next_state = self.env.step(state, action)

        # PPO Lagrange expects cost in state.info during collection
        if 'cost' not in next_state.info:
            if 'cost' in next_state.metrics:
                next_state.info['cost'] = next_state.metrics['cost']
            else:
                next_state.info['cost'] = jnp.zeros_like(next_state.reward)

        return next_state

    def reset(self, rng: jax.Array) -> BraxState:
        state = self.env.reset(rng)
        # Ensure cost is initialized in info
        if 'cost' not in state.info:
            state.info['cost'] = jnp.zeros_like(state.reward)
        return state


def wrap_env_with_cost(env: envs.Env) -> envs.Env:
    """Wrap environment with cost handling for PPO Lagrange."""
    return CostExtraWrapper(env)


class MetricsBuffer:
    """Buffer for collecting fine-grained metrics and logging to wandb periodically."""

    def __init__(self, wandb_log_interval: int = 50000):
        self.wandb_log_interval = wandb_log_interval
        self.last_wandb_log_step = 0
        self.metrics_history = []

    def should_log_to_wandb(self, current_step: int) -> bool:
        """Determine if we should log to wandb based on the interval."""
        return (current_step - self.last_wandb_log_step) >= self.wandb_log_interval

    def get_aggregated_metrics(self, metrics_since_last_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics collected since last wandb log."""
        if not metrics_since_last_log:
            return {}

        aggregated = {}

        # Group metrics by key
        metric_groups = {}
        for metric_dict in metrics_since_last_log:
            for key, value in metric_dict.items():
                if key == 'step':
                    continue
                if key not in metric_groups:
                    metric_groups[key] = []
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_groups[key].append(value)

        # Aggregate each metric group
        for key, values in metric_groups.items():
            if values:
                # For most metrics, use the latest value to avoid over-smoothing
                # For some specific metrics, we might want mean/max/min
                if any(x in key.lower() for x in ['loss', 'violation', 'lambda']):
                    # For loss-like metrics, use mean for better stability
                    aggregated[key] = np.mean(values)
                else:
                    # For other metrics (rewards, episode lengths, etc.), use latest
                    aggregated[key] = values[-1]

        return aggregated


# Global metrics buffer instance
_metrics_buffer = MetricsBuffer()


import re
import numpy as np
from typing import Callable, Dict, Optional, Union, Pattern, Tuple, Any

Reducer = Callable[[np.ndarray], float]
ReducerSpec = Union[str, Reducer]
ReducerOverrides = Dict[Union[str, Pattern[str]], ReducerSpec]

def _get_named_reducer(name: str) -> Reducer:
    if name == "last":
        return lambda a: float(a.reshape(-1)[-1])
    if name == "mean":
        return lambda a: float(a.mean())
    if name == "median":
        return lambda a: float(np.median(a))
    if name == "max":
        return lambda a: float(a.max())
    if name == "min":
        return lambda a: float(a.min())
    raise ValueError(f"Unknown reducer name: {name}")

def _resolve_reducer(
        key: str,
        overrides: Optional[ReducerOverrides],
        default: ReducerSpec
) -> Reducer:
    if overrides:
        for patt, spec in overrides.items():
            if isinstance(patt, str):
                matched = (patt == key)
            else:  # compiled regex
                matched = bool(patt.search(key))
            if matched:
                return spec if callable(spec) else _get_named_reducer(spec)
    return default if callable(default) else _get_named_reducer(default)

def reduce_for_logging(
        key: str,
        value: Any,
        *,
        default: ReducerSpec = "last",
        overrides: Optional[ReducerOverrides] = None
) -> Optional[float]:
    """
    Convert any value to a single finite float according to reducer policy.
    - default: reducer name or callable, used when no override matches.
    - overrides: { exact_key or regex: reducer_name|callable }
    Returns None if empty or all non-finite.
    """
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    # keep only finite
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    reducer = _resolve_reducer(key, overrides, default)
    return reducer(finite)


def custom_progress_fn(num_steps: int, metrics: Dict[str, Any],
                       metrics_list: Optional[List] = None,
                       use_wandb: bool = False,
                       verbose: bool = True,
                       wandb_log_interval: int = 50000,
                       default_reducer: ReducerSpec = "last",
                       reducer_overrides: Optional[ReducerOverrides] = None) -> None:
    """
    Progress function to print metrics and log to Weights & Biases.

    Args:
        num_steps: Current training step
        metrics: Metrics dictionary
        metrics_list: List to append metrics to (for local storage)
        use_wandb: Whether to use wandb logging
        verbose: Whether to print metrics
        wandb_log_interval: Steps between wandb logs (default 50000)
    """
    global _metrics_buffer

    # make sure num_steps is a plain int
    try:
        num_steps_int = int(np.asarray(num_steps).reshape(()))
    except Exception:
        num_steps_int = int(num_steps)

    # Update buffer interval if different
    if _metrics_buffer.wandb_log_interval != wandb_log_interval:
        _metrics_buffer.wandb_log_interval = wandb_log_interval

    if verbose:
        print(f"Step {num_steps}:")

    wandb_log_data = {}
    for key, value in metrics.items():
        scalar = reduce_for_logging(
            key, value, default=default_reducer, overrides=reducer_overrides
        )
        if scalar is None:
            continue

        if verbose and any(tok in key for tok in ("lambda", "cost", "constraint")):
            # purely for console visibility; not a policy decision
            print(f"  {key}: {scalar}")

        # namespacing, unchanged
        if not (key.startswith("episode/") or key.startswith("eval/") or key.startswith("training/")):
            wandb_log_data[f"training_batch/{key}"] = scalar
        else:
            wandb_log_data[key] = scalar

    if metrics_list is not None:
        row = {"step": num_steps_int}
        for k, v in metrics.items():
            s = reduce_for_logging(k, v, default=default_reducer, overrides=reducer_overrides)
            if s is not None:
                row[k] = s
        metrics_list.append(row)

    _metrics_buffer.metrics_history.append({"step": num_steps_int, **wandb_log_data})

    is_eval = any(k.startswith("eval/") for k in wandb_log_data.keys())
    if use_wandb and wandb.run is not None and wandb_log_data:
        if is_eval or _metrics_buffer.should_log_to_wandb(num_steps_int):
            payload = wandb_log_data if is_eval else {k: v for k, v in _metrics_buffer.metrics_history[-1].items() if k != "step"}
            wandb.log(payload, step=num_steps_int)
            _metrics_buffer.last_wandb_log_step = num_steps_int
            if len(_metrics_buffer.metrics_history) > 1000:
                _metrics_buffer.metrics_history = _metrics_buffer.metrics_history[-1000:]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config."""
    config = base_config.copy()
    config.update(override_config)
    return config


def get_algorithm_train_fn(alg_name: str):
    """Get the appropriate training function based on algorithm name."""
    # Try to find the best available PPO-Lagrange version
    if ppo_lagrange_v3_train is not None:
        default_ppol = ppo_lagrange_v3_train
    elif ppo_lagrange_v2_train is not None:
        default_ppol = ppo_lagrange_v2_train
    elif ppo_lagrange_train is not None:
        default_ppol = ppo_lagrange_train
    else:
        default_ppol = None

    alg_map = {
        'ppo': ppo_train,
        'ppo_cost': train_ppo_cost,
        'ppoc': train_ppo_cost,  # Alias
        'ppo_lagrange': default_ppol,
        'ppo_lagrange_v2': ppo_lagrange_v2_train or default_ppol,
        'ppo_lagrange_v3': ppo_lagrange_v3_train or default_ppol,
        'ppol': default_ppol,  # Alias
        'ppol_v3': ppo_lagrange_v3_train or default_ppol,  # Alias
    }

    train_fn = alg_map.get(alg_name)
    if train_fn is None:
        available = [k for k, v in alg_map.items() if v is not None]
        raise ValueError(f"Algorithm '{alg_name}' not available or not installed. Available: {available}")

    return train_fn


def train_from_config(config: Dict[str, Any], seed: int,
                      use_wandb: bool = True, verbose: bool = True) -> tuple[Any, Any, Any, list[Any], Env]:
    """
    Train an agent using the provided configuration.

    Returns:
        Tuple of (make_inference_fn, params, final_eval_metrics)
    """
    # Extract config values with notebook defaults
    # Prefer 'env_name' if provided, fallback to 'env' for backward compatibility
    env_name = config.get('env_name', config.get('env'))
    if env_name is None:
        raise ValueError("Config must include 'env' or 'env_name'.")
    alg_name = config.get('alg', 'ppo')

    num_timesteps = config.get('num_timesteps', 100_000_000)

    # Create environments (pass through env_kwargs if provided)
    env_kwargs = config.get('env_kwargs', {})
    train_environment = envs.get_environment(env_name, **env_kwargs)
    eval_env = envs.get_environment(env_name, **env_kwargs)

    print(f"Training environment '{env_name}' instantiated.")
    print(f"Evaluation environment '{env_name}' instantiated.")

    # Setup wandb if requested
    if use_wandb:
        # Prepare wandb config
        wandb_config = config.copy()
        wandb_config['seed'] = seed

        # Add environment config if available
        env_config = get_default_env_config().to_dict()
        # Merge any environment overrides provided via config
        if isinstance(env_kwargs, dict):
            cfg_over = env_kwargs.get('config_overrides', {})
            if isinstance(cfg_over, dict):
                env_config.update(cfg_over)
        wandb_config.update(env_config)

        # Initialize wandb
        run_name = f"{env_name}_{alg_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
        wandb_project = config.get('wandb_project', 'safe_brax')
        wandb_group = config.get('wandb_group', None)
        wandb_tags = config.get('wandb_tags', [])

        wandb.init(
            project=wandb_project,
            name=run_name,
            config=wandb_config,
            group=wandb_group,
            tags=wandb_tags,
        )

    # Setup metrics collection
    metrics_list = []
    wandb_log_interval = int(config.get('wandb_log_interval', 50000))
    bound_progress_fn = functools.partial(
        custom_progress_fn,
        metrics_list=metrics_list,
        use_wandb=use_wandb,
        verbose=verbose,
        wandb_log_interval=wandb_log_interval
    )

    # Get the appropriate training function
    train_fn_base = get_algorithm_train_fn(alg_name)

    # Prepare training function arguments
    train_kwargs = {
        # core
        'num_timesteps': num_timesteps,
        'max_devices_per_host': config.get('max_devices_per_host', None),
        'seed': seed,

        # high-level control
        'wrap_env': config.get('wrap_env', True),
        'madrona_backend': config.get('madrona_backend', False),
        'augment_pixels': config.get('augment_pixels', False),

        # env wrapper
        'num_envs': config.get('num_envs', 2048),
        'episode_length': config.get('episode_length', 2000),
        'action_repeat': config.get('action_repeat', 1),
        'wrap_env_fn': None,  # keep default
        'randomization_fn': None,  # keep default

        # algo params
        'learning_rate': config.get('learning_rate', 5e-4),
        'entropy_cost': config.get('entropy_cost', 5e-3),
        'discounting': config.get('discounting', 0.99),
        'unroll_length': config.get('unroll_length', 8),
        'batch_size': config.get('batch_size', 1024),
        'num_minibatches': config.get('num_minibatches', 32),
        'num_updates_per_batch': config.get('num_updates_per_batch', 6),
        'num_resets_per_eval': config.get('num_resets_per_eval', 0),
        'normalize_observations': config.get('normalize_observations', True),
        'reward_scaling': config.get('reward_scaling', 0.1),
        'max_grad_norm': config.get('max_grad_norm', None),
        'normalize_advantage': config.get('normalize_advantage', True),

        # eval
        'num_evals': config.get('num_evals', 5),
        'eval_env': eval_env,
        'num_eval_envs': config.get('num_eval_envs', 128),
        'deterministic_eval': config.get('deterministic_eval', False),

        # training metrics
        'log_training_metrics': True,
        'training_metrics_steps': int(config.get('training_metrics_steps', 10_000)),

        # callbacks
        'progress_fn': bound_progress_fn,
        'policy_params_fn': (lambda *args, **kwargs: None),

        # checkpointing
        'save_checkpoint_path': config.get('save_checkpoint_path', None),
        'restore_checkpoint_path': config.get('restore_checkpoint_path', None),
        'restore_params': None,  # programmatic only
        'restore_value_fn': config.get('restore_value_fn', True),
    }

    # Add algorithm-specific parameters
    if 'ppo' in alg_name:
        train_kwargs['gae_lambda'] = config.get('gae_lambda', 0.95)
        train_kwargs['clipping_epsilon'] = config.get('clipping_epsilon', 0.3)

    # PPO-Lagrange specific parameters
    if 'lagrange' in alg_name:
        train_kwargs['safety_bound'] = config.get('safety_bound', 0.2)
        train_kwargs['lagrangian_coef_rate'] = config.get('lagrangian_coef_rate', 0.01)
        train_kwargs['initial_lambda_lagr'] = config.get('initial_lambda_lagr', 0.0)

    # Create the training function
    train_fn = functools.partial(train_fn_base, **train_kwargs)

    # Train the agent
    print(f"Starting {alg_name} training for {env_name}...")
    make_inference_fn, params, final_eval_metrics = train_fn(
        environment=train_environment,
        eval_env=eval_env,
        progress_fn=bound_progress_fn
    )
    print("Training finished.")
    # print(f"Final evaluation metrics: {final_eval_metrics}")

    # Log final metrics to wandb
    if use_wandb and wandb.run is not None and final_eval_metrics:
        final_log_data = {}
        for key, value in final_eval_metrics.items():
            # average across parallel eval envs (or whatever you prefer)
            scalar = reduce_for_logging(key, value, default="mean")
            if scalar is not None:
                final_log_data[key] = scalar
        if final_log_data:
            wandb.log(final_log_data, step=int(float(np.asarray(num_timesteps).reshape(()))))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = config.get('model_dir', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = f'{model_dir}/{env_name.lower()}_{alg_name}_seed{seed}_{timestamp}'
    brax_model.save_params(model_path, params)
    print(f"Trained model parameters saved to: {model_path}")

    # Save metrics
    metrics_dir = config.get('out_dir', 'runs/metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_filename = f"{metrics_dir}/training_metrics_{env_name}_{alg_name}_seed{seed}_{timestamp}.csv"

    if metrics_list:
        # Unify all metric keys across steps to avoid CSV fieldname errors
        all_keys = set()
        for row in metrics_list:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)
        with open(metrics_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_list:
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        print(f"Metrics saved to {metrics_filename}")

    return make_inference_fn, params, final_eval_metrics, metrics_list, eval_env


def collect_rollout_metrics(env_name: str, make_inference_fn, params,
                            num_steps: int = 5000, seed: int = None,
                            save_trajectory: bool = True,
                            save_plots: bool = True,
                            env_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
    """
    Collect detailed metrics during a rollout.

    Returns:
        Dictionary containing all collected metrics
    """
    # Create evaluation environment (respect env_kwargs)
    eval_environment = envs.get_environment(env_name, **(env_kwargs or {}))

    # JIT compile reset and step
    jit_eval_reset = jax.jit(eval_environment.reset)
    jit_eval_step = jax.jit(eval_environment.step)

    # Create inference function
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    print(f"Inference function for rollout created for {env_name}.")

    # Initialize data collection
    rollout_frames = []
    rollout_metrics_data = {
        'distance_to_goal': [],
        'last_dist_goal': [],
        'reward': [],
        'dist_reward': [],
        'goal_reward': [],
        'orientation_reward': [],
        'ctrl_cost': [],
        'x_position': [],
        'y_position': [],
        'agent_pos_x': [],
        'agent_pos_y': [],
        'goal_pos_x': [],
        'goal_pos_y': [],
        'x_velocity': [],
        'y_velocity': [],
        'goals_reached_count': [],
        'cost': []
    }
    actions = []

    # Initialize rollout
    if seed is None:
        seed = int(time.time())
    rng_rollout = jax.random.PRNGKey(seed)
    eval_state = jit_eval_reset(rng_rollout)

    print(f"Starting rollout for {num_steps} steps...")
    for i in range(num_steps):
        act_rng, rng_rollout = jax.random.split(rng_rollout)
        action, _ = jit_inference_fn(eval_state.obs, act_rng)
        actions.append(action)

        eval_state = jit_eval_step(eval_state, action)
        rollout_frames.append(eval_state.pipeline_state)

        # Collect metrics from eval_state.metrics
        rollout_metrics_data['distance_to_goal'].append(eval_state.metrics.get('distance_to_goal', np.nan))
        rollout_metrics_data['reward'].append(eval_state.metrics.get('reward', np.nan))
        rollout_metrics_data['cost'].append(eval_state.metrics.get('cost', np.nan))
        rollout_metrics_data['dist_reward'].append(eval_state.metrics.get('dist_reward', np.nan))
        rollout_metrics_data['goal_reward'].append(eval_state.metrics.get('goal_reward', np.nan))
        rollout_metrics_data['orientation_reward'].append(eval_state.metrics.get('orientation_reward', np.nan))
        rollout_metrics_data['ctrl_cost'].append(eval_state.metrics.get('ctrl_cost', np.nan))
        rollout_metrics_data['x_position'].append(eval_state.metrics.get('x_position', np.nan))
        rollout_metrics_data['y_position'].append(eval_state.metrics.get('y_position', np.nan))
        rollout_metrics_data['x_velocity'].append(eval_state.metrics.get('x_velocity', np.nan))
        rollout_metrics_data['y_velocity'].append(eval_state.metrics.get('y_velocity', np.nan))
        rollout_metrics_data['goals_reached_count'].append(eval_state.metrics.get('goals_reached_count', np.nan))

        # Collect metrics from eval_state.info
        rollout_metrics_data['last_dist_goal'].append(eval_state.info.get('last_dist_goal', np.nan))
        current_agent_pos = eval_state.info.get('agent_pos', np.array([np.nan, np.nan, np.nan]))
        current_goal_pos = eval_state.info.get('goal_pos', np.array([np.nan, np.nan, np.nan]))
        rollout_metrics_data['agent_pos_x'].append(current_agent_pos[0])
        rollout_metrics_data['agent_pos_y'].append(current_agent_pos[1])
        rollout_metrics_data['goal_pos_x'].append(current_goal_pos[0])
        rollout_metrics_data['goal_pos_y'].append(current_goal_pos[1])

        if i % 100 == 0 or i == num_steps - 1:
            print(
                f"Rollout step {i + 1}/{num_steps} completed. Goals reached: {eval_state.metrics.get('goals_reached_count', 0)}")

        if eval_state.done:
            print(f"Rollout terminated early at step {i + 1} due to done signal.")
            remaining_steps = num_steps - (i + 1)
            for key_metric in rollout_metrics_data.keys():
                rollout_metrics_data[key_metric].extend([np.nan] * remaining_steps)
            break

    print("Rollout finished.")

    # Save trajectory if requested
    if save_trajectory:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('trajectories', exist_ok=True)
        rollout_trajectory_path = f'trajectories/{env_name}_rollout_{timestamp}.json'
        brax_json.save(rollout_trajectory_path, eval_environment.sys, rollout_frames)
        print(f"Rollout trajectory saved to {rollout_trajectory_path}")

    # Create plots if requested
    if save_plots:
        create_rollout_plots(rollout_metrics_data, env_name)

    return rollout_metrics_data


def verify_ppoc_shaping(env_name: str, make_inference_fn, params,
                        num_steps: int, seed: int,
                        cost_weight: float = 1.0,
                        env_kwargs: Optional[Dict[str, Any]] = None,
                        out_dir: str = 'runs/smoke') -> str:
    """Runs a short rollout on a RewardMinusCost-wrapped env and logs per-step
    raw_reward, shaped_reward, and cost for verifying shaped ≈ raw - cost.

    Returns the CSV path written.
    """
    # Create evaluation environment and wrap it if available
    eval_environment = envs.get_environment(env_name, **(env_kwargs or {}))
    if RewardMinusCostWrapper is not None:
        eval_environment = RewardMinusCostWrapper(eval_environment, cost_weight=cost_weight)

    # JIT compile reset and step
    jit_eval_reset = jax.jit(eval_environment.reset)
    jit_eval_step = jax.jit(eval_environment.step)

    # Create inference function
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    # Initialize rollout
    if seed is None:
        seed = int(time.time())
    rng_rollout = jax.random.PRNGKey(seed)
    eval_state = jit_eval_reset(rng_rollout)

    # Prepare CSV logging
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{out_dir}/ppoc_verify_{env_name}_seed{seed}_{timestamp}.csv"
    fieldnames = [
        'step', 'raw_reward', 'shaped_reward', 'cost',
        'reward_delta_vs_raw_minus_cost'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_steps):
            act_rng, rng_rollout = jax.random.split(rng_rollout)
            action, _ = jit_inference_fn(eval_state.obs, act_rng)
            eval_state = jit_eval_step(eval_state, action)

            # Extract values
            info = eval_state.info
            raw_reward = info.get('raw_reward', eval_state.reward)
            shaped_reward = info.get('shaped_reward', eval_state.reward)
            cost = info.get('cost', eval_state.metrics.get('cost', 0.0))

            # Convert to Python floats
            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return float(getattr(x, 'item', lambda: 0.0)())

            rr = to_float(raw_reward)
            sr = to_float(shaped_reward)
            cc = to_float(cost)
            delta = sr - (rr - cc * cost_weight)

            writer.writerow({
                'step': i,
                'raw_reward': rr,
                'shaped_reward': sr,
                'cost': cc,
                'reward_delta_vs_raw_minus_cost': delta,
            })

            if eval_state.done:
                # continue but metrics may be meaningless; break could be used
                pass

    print(f"PPO-C verify log written to: {csv_path}")
    return csv_path


def create_rollout_plots(rollout_metrics_data: Dict[str, List], env_name: str) -> None:
    """Create and save plots from rollout metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path_base = f'{plot_dir}/{env_name}_rollout_{timestamp}'

    num_steps = len(rollout_metrics_data['distance_to_goal'])
    time_steps = np.arange(num_steps)

    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot 1: Distance and Last Distance to Goal
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, rollout_metrics_data['distance_to_goal'], label='Current Distance to Goal', linestyle='-')
    plt.plot(time_steps, rollout_metrics_data['last_dist_goal'], label='Last Distance to Goal', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Distance")
    plt.title(f"{env_name} - Rollout: Goal Tracking")
    plt.legend()
    plt.tight_layout()
    goal_tracking_plot_path = f'{plot_path_base}_goal_distances.png'
    plt.savefig(goal_tracking_plot_path)
    plt.close()
    print(f"Goal tracking plot saved to: {goal_tracking_plot_path}")

    # Plot 2: Cost Plot
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, rollout_metrics_data['cost'], label='Cost', linestyle='-')
    plt.xlabel("Time Step")
    plt.ylabel("Cost")
    plt.title(f"{env_name} - Rollout: Cost")
    plt.legend()
    plt.tight_layout()
    cost_plot_path = f'{plot_path_base}_cost.png'
    plt.savefig(cost_plot_path)
    plt.close()
    print(f"Cost plot saved to: {cost_plot_path}")

    # Plot 3: Cumulative Cost
    cumulative_cost = np.cumsum(rollout_metrics_data['cost'])
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, cumulative_cost, label='Cumulative Cost', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Cost")
    plt.title(f"{env_name} - Rollout: Cumulative Cost Over Time")
    plt.legend()
    plt.tight_layout()
    cumulative_cost_plot_path = f'{plot_path_base}_cumulative_cost.png'
    plt.savefig(cumulative_cost_plot_path)
    plt.close()
    print(f"Cumulative cost plot saved to: {cumulative_cost_plot_path}")

    # Plot 4: Reward Component Breakdown
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, rollout_metrics_data['dist_reward'], label='Distance Reward', alpha=0.7)
    plt.plot(time_steps, rollout_metrics_data['goal_reward'], label='Goal Reward', alpha=0.7)
    plt.plot(time_steps, rollout_metrics_data['orientation_reward'], label='Orientation Reward', alpha=0.7)
    plt.plot(time_steps, -np.array(rollout_metrics_data['ctrl_cost']), label='Negative Control Cost', alpha=0.7)
    plt.plot(time_steps, rollout_metrics_data['reward'], label='Total Reward', linestyle='--', color='black',
             linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Reward Value")
    plt.title(f"{env_name} - Rollout: Reward Component Breakdown")
    plt.legend()
    plt.tight_layout()
    reward_breakdown_plot_path = f'{plot_path_base}_reward_breakdown.png'
    plt.savefig(reward_breakdown_plot_path)
    plt.close()
    print(f"Reward breakdown plot saved to: {reward_breakdown_plot_path}")

    # Plot 5: X-Y Trajectory
    plt.figure(figsize=(10, 8))
    valid_x = np.array(rollout_metrics_data['x_position'])
    valid_y = np.array(rollout_metrics_data['y_position'])
    goal_x_series = np.array(rollout_metrics_data['goal_pos_x'])
    goal_y_series = np.array(rollout_metrics_data['goal_pos_y'])

    # Filter out NaNs
    valid_indices_agent = ~(np.isnan(valid_x) | np.isnan(valid_y))
    valid_x_agent = valid_x[valid_indices_agent]
    valid_y_agent = valid_y[valid_indices_agent]

    valid_indices_goal = ~(np.isnan(goal_x_series) | np.isnan(goal_y_series))
    valid_x_goal = goal_x_series[valid_indices_goal]
    valid_y_goal = goal_y_series[valid_indices_goal]

    if len(valid_x_agent) > 0 and len(valid_y_agent) > 0:
        plt.plot(valid_x_agent, valid_y_agent, 'k-', alpha=0.7, label='Agent Path')
        plt.scatter(valid_x_agent[0], valid_y_agent[0], c='green', s=100, label='Agent Start', zorder=5, marker='o')
        plt.scatter(valid_x_agent[-1], valid_y_agent[-1], c='red', s=100, label='Agent End', zorder=5, marker='x')

        if len(valid_x_goal) > 0 and len(valid_y_goal) > 0:
            plt.scatter(valid_x_goal[0], valid_y_goal[0], c='blue', s=150, label='Initial Goal', zorder=4, marker='*')
            if any(g_x != valid_x_goal[0] for g_x in valid_x_goal) or any(
                    g_y != valid_y_goal[0] for g_y in valid_y_goal):
                plt.plot(valid_x_goal, valid_y_goal, 'b--', alpha=0.5, label='Goal Path')
                plt.scatter(valid_x_goal[-1], valid_y_goal[-1], c='purple', s=150, label='Final Goal', zorder=4,
                            marker='*')

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"{env_name} - Rollout: X-Y Trajectory")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No valid position data for trajectory plot", ha='center', va='center')

    plt.tight_layout()
    trajectory_plot_path = f'{plot_path_base}_xy_trajectory.png'
    plt.savefig(trajectory_plot_path)
    plt.close()
    print(f"X-Y trajectory plot saved to: {trajectory_plot_path}")


def record_episode_video(
        env,  # a Brax env (same as for eval)
        make_inference_fn,
        params,
        steps: int = 2500,
        camera: str | int = 0,  # camera name or id
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        out_name: str = "rollout.mp4",
        log_to_wandb: bool = True,
        seed: int = 0,
):
    """
    Renders a fresh eval rollout controlled by your trained policy.
    We step the *env* for observations/actions, and in parallel step a MuJoCo
    simulator for pretty pixels.
    """
    # 1) Ensure headless GPU rendering (you might need to do this before importing mujoco)
    os.environ.setdefault("MUJOCO_GL", "egl")

    # 2) JIT policy
    infer = make_inference_fn(params)
    infer = jax.jit(infer)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    # 3) Rollout
    key = jax.random.PRNGKey(seed)
    state = reset(key)
    frames = []

    for t in range(steps):
        key, sk = jax.random.split(key)
        action, _ = infer(state.obs, sk)
        state = step(state, action)
        frames.append(state.pipeline_state)

        if getattr(state, "done", False):
            break

    # 4) Render
    rendering = env.render(frames, width=width, height=height, camera=camera)

    # 5) Save mp4 (and log)
    os.makedirs("videos", exist_ok=True)
    mp4_path = os.path.join("videos", out_name)
    iio.imwrite(mp4_path, np.stack(rendering), fps=fps)

    if log_to_wandb and wandb.run is not None:
        wandb.log({"rollout/video": wandb.Video(mp4_path, fps=fps, format="mp4")})
    return mp4_path


def _json_type(s):
    if s is None or isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string for env_kwargs: {e}")


def main():
    """Main function to run training from command line."""
    parser = argparse.ArgumentParser(description='Train Safe-Brax agents from config files')

    # --- Core ---
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # --- Experiment Control ---
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--skip-rollout", action="store_true", help="Skip rollout evaluation after training")
    parser.add_argument("--skip-video", action="store_true", help="Skip video recording after training")
    parser.add_argument("--out_dir", type=str, default="runs/experimental_results",
                        help="Directory for metrics/outputs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model parameters")

    # --- Environment ---
    parser.add_argument("--env_name", type=str, default="safe_point_goal", help="Env name (preferred)")
    parser.add_argument("--env", type=str, default="point_resetting_goal_random_hazard_lidar_sensor_obs",
                        help="Alternate env key (legacy)")
    parser.add_argument("--env_kwargs", type=_json_type, default={
        "config_overrides": {
            "ctrl_cost_weight": 0.001,
            "goal_size": 0.7,
            "reward_goal": 10.0,
            "reward_distance": 3,
            "reward_orientation": False,
            "reward_orientation_scale": 0.002,
        },
        # "num_hazards": 8,
    }, help="JSON string or path for env_kwargs")

    # --- Algorithm ---
    parser.add_argument("--alg", type=str, default="ppo_lagrange", help="Algorithm name (e.g., ppo, ppo_lagrange)")
    parser.add_argument("--max_devices_per_host", type=int, default=None, help="Limit devices per host")

    # --- Training Scale / Rollout ---
    parser.add_argument("--num_timesteps", type=float, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--episode_length", type=int, default=2000, help="Episode length")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel envs")
    parser.add_argument("--unroll_length", type=int, default=8, help="Unroll length")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_minibatches", type=int, default=32, help="Number of minibatches")
    parser.add_argument("--num_updates_per_batch", type=int, default=6, help="SGD updates per batch")
    parser.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=5000,
                        help="Steps for post-training rollout evaluation")

    # --- Optimization / PPO Core ---
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--entropy_cost", type=float, default=5e-3, help="Entropy coefficient")
    parser.add_argument("--discounting", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--reward_scaling", type=float, default=0.1, help="Reward scaling")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clipping_epsilon", type=float, default=0.3, help="PPO clipping epsilon")
    parser.add_argument("--normalize_observations", type=bool, default=True,
                        help="Normalize observations (true/false)")

    # --- Evaluation / Logging cadence ---
    parser.add_argument("--num_evals", type=int, default=5, help="Number of eval passes during training")
    parser.add_argument("--num_eval_envs", type=int, default=128, help="Parallel envs during eval")
    parser.add_argument("--deterministic_eval", type=bool, default=False, help="Deterministic eval policy")
    parser.add_argument("--training_metrics_steps", type=float, default=10_000,
                        help="Env steps between training metrics logs")

    # --- PPO-Lagrange ---
    parser.add_argument("--safety_bound", type=float, default=0.2, help="Safety constraint bound")
    parser.add_argument("--lagrangian_coef_rate", type=float, default=0.001, help="Lagrange multiplier LR")
    parser.add_argument("--initial_lambda_lagr", type=float, default=0.0, help="Initial lambda value")

    # --- PPO-Cost verification ---
    parser.add_argument("--ppoc-verify-log-steps", type=int, default=0,
                        help="If >0 and alg is ppo_cost, run verify rollout and log shaping")
    parser.add_argument("--ppoc-cost-weight", type=float, default=1.0,
                        help="Cost weight used for PPO-C verify logging")

    # --- WandB ---
    parser.add_argument("--use_wandb", type=bool, default=True, help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="safe-brax-experimental-results", help="W&B project")
    parser.add_argument("--wandb_group", type=str, default="pointgoal-baselines", help="W&B group")
    parser.add_argument("--wandb_tags", type=str, nargs='+', help="JSON list or path of tags")
    parser.add_argument("--wandb_log_interval", type=int, default=50000,
                        help="Steps between wandb logs (keeps fine-grained local metrics but reduces wandb frequency)")

    # --- Video Recording ---
    parser.add_argument("--camera", type=str, default="fixedfar",
                        help="Camera name or id (string name or numeric string index)")
    parser.add_argument("--video_width", type=int, default=320, help="Output video width")
    parser.add_argument("--video_height", type=int, default=240, help="Output video height")
    parser.add_argument("--video_fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--video_length", type=int, default=2500, help="Number of frames in the video")

    args = parser.parse_args()

    # Setup GPU environment
    setup_gpu_environment()

    # Load config
    config = load_config(args.config)

    # Override config with command line args if provided
    for key, value in vars(args).items():
        config[key] = value

    # Run training for each seed
    for seed in args.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        # Train the agent
        make_inference_fn, params, final_metrics, metrics_list, eval_env = train_from_config(
            config=config,
            seed=seed,
            use_wandb=not args.no_wandb,
            verbose=not args.quiet
        )

        # Perform rollout evaluation if not skipped
        if not args.skip_rollout:
            print(f"\nPerforming rollout evaluation...")
            rollout_env_name = config.get('env_name', config.get('env'))
            rollout_metrics = collect_rollout_metrics(
                env_name=rollout_env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=args.rollout_steps,
                seed=seed,
                save_trajectory=True,
                save_plots=True,
                env_kwargs=config.get('env_kwargs', {})
            )

        if not args.skip_video:
            vid = record_episode_video(
                env=eval_env,
                make_inference_fn=make_inference_fn,
                params=params,
                steps=args.video_length,
                camera=config.get("camera", 0),
                width=config.get("video_width", 320),
                height=config.get("video_height", 240),
                fps=int(config.get("video_fps", 30)),
                out_name=f"{config.get('env_name', 'env')}_{config.get('alg', 'algo')}_seed{seed}.mp4",
                log_to_wandb=not args.no_wandb,
                seed=seed,
            )
            print("Saved video:", vid)

        # PPO-C verify shaping log if requested
        if config.get('alg') in ('ppo_cost', 'ppoc') and args.ppoc_verify_log_steps > 0:
            print("\nRunning PPO-C shaping verification rollout...")
            rollout_env_name = config.get('env_name', config.get('env'))
            _ = verify_ppoc_shaping(
                env_name=rollout_env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=args.ppoc_verify_log_steps,
                seed=seed,
                cost_weight=float(config.get('cost_weight', args.ppoc_cost_weight)),
                env_kwargs=config.get('env_kwargs', {}),
                out_dir=config.get('out_dir', 'runs/smoke')
            )

        # Finish wandb run if active
        if not args.no_wandb and wandb.run is not None:
            wandb.finish()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
