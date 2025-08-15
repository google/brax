# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Velocity-constrained ant locomotion environment."""

import jax
from jax import numpy as jp
from brax.envs.ant import Ant
from brax.envs.base import State


class AntVelocityConstrained(Ant):
  """Ant locomotion environment with velocity constraints.
  
  This environment extends the standard Ant environment by adding velocity
  constraints that serve as safety constraints for constrained RL training.
  The constraint violation (cost) is computed based on the ant's velocity
  exceeding a specified threshold.
  """

  def __init__(
      self,
      max_velocity: float = 0.5,  # Set below ant's natural max velocity (~0.81) to ensure violations
      velocity_cost_weight: float = 1.0,
      **kwargs,
  ):
    """Initialize the velocity-constrained ant environment.
    
    Args:
      max_velocity: Maximum allowed velocity magnitude (m/s).
      velocity_cost_weight: Weight for velocity constraint violation cost.
      **kwargs: Additional arguments passed to parent Ant class.
    """
    super().__init__(**kwargs)
    self._max_velocity = max_velocity
    self._velocity_cost_weight = velocity_cost_weight

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics with velocity constraints."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    # Calculate velocity components
    velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
    # Handle both single env and vectorized env cases
    if velocity.ndim == 1:
        # Single environment case
        velocity_magnitude = jp.linalg.norm(velocity[:2])  # Only consider x,y velocity
        forward_reward = velocity[0]
    else:
        # Vectorized environment case  
        velocity_magnitude = jp.linalg.norm(velocity[:, :2], axis=1)  # Only consider x,y velocity
        forward_reward = velocity[:, 0]
    
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
    
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
      
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = 0.0
    
    # Velocity constraint cost (this becomes the "safety constraint")
    velocity_violation = jp.maximum(0.0, velocity_magnitude - self._max_velocity)
    velocity_cost = self._velocity_cost_weight * velocity_violation

    obs = self._get_obs(pipeline_state)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    # Ensure all metrics have consistent shapes by expanding to match reward shape
    reward_shape = jp.shape(reward)
    
    # Update metrics with velocity-related information
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=jp.linalg.norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[0] if velocity.ndim == 1 else velocity[:, 0],
        y_velocity=velocity[1] if velocity.ndim == 1 else velocity[:, 1],
        velocity_magnitude=jp.broadcast_to(velocity_magnitude, reward_shape),
        velocity_violation=jp.broadcast_to(velocity_violation, reward_shape),
        velocity_cost=jp.broadcast_to(velocity_cost, reward_shape),
        forward_reward=forward_reward,
        # Cost for PPO Lagrange (constraint violation)
        cost=jp.broadcast_to(velocity_cost, reward_shape),
    )
    
    # Update info dictionary with cost (required for PPO Lagrange v2)
    # Keep JAX arrays - don't convert to Python scalars during compilation
    
    # Get previous info or create new one
    current_info = getattr(state, 'info', {})
    step_count = current_info.get('step_count', 0) + 1
    
    # Update info dictionary (copy existing and update)
    new_info = current_info.copy() if isinstance(current_info, dict) else {}
    new_info.update({
        "cost": velocity_cost,  # Cost for PPO Lagrange v2 (keep as JAX array)
        "velocity_magnitude": velocity_magnitude,
        "velocity_violation": velocity_violation,
        "step_count": step_count,
    })
    
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
    )

  def reset(self, rng: jax.Array) -> State:
    """Reset the environment with velocity constraint metrics."""
    state = super().reset(rng)
    
    # Initialize velocity-related metrics with consistent shapes
    reward_shape = jp.shape(state.reward)
    zero = jp.zeros(reward_shape)
    state.metrics.update(
        velocity_magnitude=zero,
        velocity_violation=zero, 
        velocity_cost=zero,
        cost=zero,  # Initialize cost for PPO Lagrange
    )
    
    # Initialize info dictionary with cost (required for PPO Lagrange v2)
    # Use JAX arrays, not Python scalars
    info = {
        "cost": zero,  # Initialize cost in info (JAX array)
        "velocity_magnitude": zero,
        "velocity_violation": zero,
        "step_count": 0,  # This can remain a Python int
    }
    
    return state.replace(info=info) 