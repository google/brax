"""Example of using PPO-Lagrange for constrained reinforcement learning."""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax import base
from brax.training.agents.ppo_lagrange import train as ppo_lagrange


class ConstrainedCartpole(envs.Env):
  """Cartpole environment with position constraint.
  
  This is a simple example of a constrained environment where the cart
  must stay within certain position bounds. Violating these bounds
  incurs a cost.
  """
  
  def __init__(self, backend='generalized', **kwargs):
    # Use standard cartpole as base
    self._env = envs.get_environment('cartpole', backend=backend, **kwargs)
    self._position_limit = 2.0  # Cart position constraint
    
  @property
  def observation_size(self) -> int:
    return self._env.observation_size
  
  @property
  def action_size(self) -> int:
    return self._env.action_size
  
  @property
  def backend(self) -> str:
    return self._env.backend
  
  def reset(self, rng: jnp.ndarray) -> base.State:
    """Reset the environment."""
    state = self._env.reset(rng)
    # Add cost field to state info
    state.info['cost'] = jnp.zeros_like(state.reward)
    return state
  
  def step(self, state: base.State, action: jnp.ndarray) -> base.State:
    """Step the environment with constraint checking."""
    # Take a step in the base environment
    next_state = self._env.step(state, action)
    
    # Calculate constraint cost
    # Cost is 1.0 if cart position exceeds limits, 0.0 otherwise
    cart_position = next_state.pipeline_state.q[0]  # x position of cart
    cost = jnp.where(
        jnp.abs(cart_position) > self._position_limit,
        1.0,
        0.0
    )
    
    # Add cost to state info
    next_state.info['cost'] = cost
    
    return next_state


def main():
  """Train PPO-Lagrange on constrained cartpole."""
  
  # Create constrained environment
  env = ConstrainedCartpole()
  
  # Wrap environment to include cost in extras
  def wrap_env_with_cost(env):
    """Wrapper that adds cost to transition extras."""
    wrapped_env = envs.training.wrap(env, episode_length=200)
    
    # Override step to include cost in extras
    original_step = wrapped_env.step
    
    def step_with_cost(state, action):
      next_state = original_step(state, action)
      # Add cost to extras
      if 'cost' in next_state.info:
        next_state.extras['cost'] = next_state.info['cost']
      return next_state
    
    wrapped_env.step = step_with_cost
    return wrapped_env
  
  # Train PPO-Lagrange
  train_fn = functools.partial(
      ppo_lagrange.train,
      num_timesteps=1_000_000,
      num_evals=10,
      reward_scaling=1.0,
      cost_scaling=1.0,
      cost_limit=25.0,  # Maximum allowed cost per episode
      episode_length=200,
      num_envs=64,
      learning_rate=3e-4,
      entropy_cost=1e-3,
      unroll_length=10,
      batch_size=64,
      num_minibatches=8,
      num_updates_per_batch=4,
      normalize_observations=True,
      # Lagrange multiplier parameters
      lambda_init=0.0,
      lambda_lr=0.05,
      lambda_max=10.0,
      # Logging
      progress_fn=lambda step, metrics: print(f"Step {step}: {metrics}"),
      wrap_env_fn=wrap_env_with_cost,
  )
  
  # Run training
  make_policy, params, metrics = train_fn(environment=env)
  
  print("Training completed!")
  print(f"Final metrics: {metrics}")
  
  # Create policy for evaluation
  policy = make_policy(params, deterministic=True)
  
  # Evaluate trained policy
  eval_env = env
  rng = jax.random.PRNGKey(0)
  state = eval_env.reset(rng)
  
  total_reward = 0.0
  total_cost = 0.0
  for _ in range(200):
    rng, action_rng = jax.random.split(rng)
    action, _ = policy(state.obs, action_rng)
    state = eval_env.step(state, action)
    total_reward += state.reward
    total_cost += state.info.get('cost', 0.0)
  
  print(f"\nEvaluation results:")
  print(f"Total reward: {total_reward}")
  print(f"Total cost: {total_cost}")
  print(f"Constraint satisfied: {total_cost <= 25.0}")


if __name__ == "__main__":
  main() 