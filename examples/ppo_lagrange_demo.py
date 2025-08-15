#!/usr/bin/env python3
"""Simple demo of PPO-Lagrange for constrained reinforcement learning in Brax.

This script demonstrates training a policy with safety constraints.
"""

from ast import Tuple
from collections.abc import Callable
import sys
from typing import Optional
sys.path.append('..')  # Add parent directory to path

from brax import envs
from brax.training.agents.ppo_lagrange import train
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper, PipelineEnv
from typing import Callable, Optional, Tuple
from brax.base import System


def create_constrained_cartpole():
  """Create a CartPole environment with position constraints."""
  
  base_env = envs.get_environment('inverted_pendulum')
  
  class ConstrainedCartPole(PipelineEnv):
    """CartPole with position constraints."""
    
    def reset(self, rng: jnp.ndarray) -> envs.State:
      state = self.env.reset(rng)
      return state
    
    def step(self, state: envs.State, action: jnp.ndarray) -> envs.State:
      next_state = self.env.step(state, action)
      
      # Add cost signal - penalize if cart moves too far from center
      position = state.obs[0] if state.obs.ndim == 1 else state.obs[..., 0]
      cost = jnp.where(jnp.abs(position) > 1.0, 1.0, 0.0)
      
      # Put cost in extras for training
      extras = next_state.info.copy()
      extras['cost'] = cost
      
      return next_state.replace(info=extras)
  
  return ConstrainedCartPole(base_env)


def main():
  """Train PPO-Lagrange on constrained CartPole."""
  
  print("PPO-Lagrange Demo: Constrained CartPole")
  print("=" * 50)
  
  # Create constrained environment
  env = create_constrained_cartpole()
  
  # Custom wrapper to put costs in extras
  def wrap_env_fn(
    env: Env, 
    episode_length: int, 
    action_repeat: int, 
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None
  ) -> Wrapper:
    class CostExtraWrapper(envs.Wrapper):
      def step(self, state: envs.State, action: jnp.ndarray) -> envs.State:
        next_state = self.env.step(state, action)
        extras = state.info.copy()
        if 'cost' in next_state.info:
          extras['cost'] = next_state.info['cost']
        else:
          extras['cost'] = jnp.zeros_like(next_state.reward)
        return next_state.replace(info=extras)
    return CostExtraWrapper(env)
  
  # Training configuration
  config = {
      'num_timesteps': 1_000_000,
      'episode_length': 1000,
      'cost_limit': 10.0,  # Maximum allowed cost per episode
      'num_envs': 128,
      'learning_rate': 3e-4,
      'unroll_length': 20,
      'batch_size': 256,
      'num_minibatches': 8,
      'num_updates_per_batch': 4,
      'num_evals': 10,
      'lambda_lr': 0.05,
      'lambda_init': 0.0,
  }
  
  print("\nTraining Configuration:")
  for key, value in config.items():
    print(f"  {key}: {value}")
  
  # Track progress
  metrics_history = []
  
  def progress_fn(steps: int, metrics: dict):
    """Progress callback."""
    metrics_history.append((steps, metrics))
    
    if steps % 100_000 == 0:
      print(f"\nStep {steps:,}:")
      print(f"  Reward: {metrics.get('reward_return', 0):.2f}")
      print(f"  Cost: {metrics.get('cost_return', 0):.2f}")
      print(f"  Lambda: {metrics.get('lambda', 0):.3f}")
      print(f"  Constraint violation: {metrics.get('constraint_violation', 0):.2f}")
  
  # Train PPO-Lagrange
  print("\nStarting training...")
  make_policy, params, final_metrics = train.train(
      environment=env,
      wrap_env_fn=wrap_env_fn,
      progress_fn=progress_fn,
      **config
  )
  
  print("\n" + "=" * 50)
  print("Training Complete!")
  print(f"Final reward: {final_metrics.get('reward_return', 0):.2f}")
  print(f"Final cost: {final_metrics.get('cost_return', 0):.2f}")
  print(f"Final lambda: {final_metrics.get('lambda', 0):.3f}")
  
  # Simple evaluation
  print("\nEvaluating trained policy...")
  policy = make_policy(params, deterministic=True)
  
  total_reward = 0
  total_cost = 0
  num_episodes = 10
  
  for i in range(num_episodes):
    rng = jax.random.PRNGKey(i)
    state = env.reset(rng)
    episode_reward = 0
    episode_cost = 0
    
    for _ in range(1000):
      action, _ = policy(state.obs, rng)
      state = env.step(state, action)
      episode_reward += float(state.reward)
      episode_cost += float(state.info.get('cost', 0))
      
      if state.done:
        break
    
    total_reward += episode_reward
    total_cost += episode_cost
    print(f"  Episode {i+1}: Reward={episode_reward:.1f}, Cost={episode_cost:.1f}")
  
  print(f"\nAverage over {num_episodes} episodes:")
  print(f"  Mean reward: {total_reward/num_episodes:.2f}")
  print(f"  Mean cost: {total_cost/num_episodes:.2f}")
  
  # Save training curve data
  if metrics_history:
    import matplotlib.pyplot as plt
    
    steps = [m[0] for m in metrics_history]
    rewards = [m[1].get('reward_return', 0) for m in metrics_history]
    costs = [m[1].get('cost_return', 0) for m in metrics_history]
    lambdas = [m[1].get('lambda', 0) for m in metrics_history]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    
    ax1.plot(steps, rewards, 'b-')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Reward')
    ax1.grid(True)
    
    ax2.plot(steps, costs, 'r-')
    ax2.axhline(y=config['cost_limit'], color='k', linestyle='--', label='Cost Limit')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Episode Cost')
    ax2.set_title('Training Cost')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(steps, lambdas, 'g-')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Lambda')
    ax3.set_title('Lagrange Multiplier')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_lagrange_demo_curves.png')
    print("\nTraining curves saved to 'ppo_lagrange_demo_curves.png'")


if __name__ == '__main__':
  main() 