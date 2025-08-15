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

"""Height-constrained humanoid locomotion environment."""

import jax
from jax import numpy as jp
from brax.envs.humanoidstandup import HumanoidStandup
from brax.envs.base import State


class HumanoidHeightConstrained(HumanoidStandup):
  """Humanoid locomotion environment with height constraints.
  
  This environment extends the standard Humanoid environment by adding height
  constraints that serve as safety constraints for constrained RL training.
  The constraint violation (cost) is computed based on the humanoid's height
  exceeding a specified threshold, forcing it to learn to crouch or crawl.
  """

  def __init__(
      self,
      max_height: float = 1.3,  # Set below humanoid's natural max height (~2.0) to force crouching
      height_cost_weight: float = 1.0,
      **kwargs,
  ):
    """Initialize the height-constrained humanoid environment.
    
    Args:
      max_height: Maximum allowed height (z-coordinate of torso center of mass) in meters.
      height_cost_weight: Weight for height constraint violation cost.
      **kwargs: Additional arguments passed to parent Humanoid class.
    """
    super().__init__(**kwargs)
    self._max_height = max_height
    self._height_cost_weight = height_cost_weight

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics with height constraints."""
    # Scale action from [-1,1] to actuator limits
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min

    pipeline_state = self.pipeline_step(state.pipeline_state, action)

    # Get current height (z coordinate of torso)
    current_height = pipeline_state.x.pos[0, 2]
    
    # Calculate forward movement reward (similar to humanoid, not just upward like humanoidstandup)
    pipeline_state0 = state.pipeline_state
    if pipeline_state0 is not None:
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = velocity[0]  # Reward forward movement
    else:
        forward_reward = 0.0
    
    # Control cost
    ctrl_cost = 0.01 * jp.sum(jp.square(action))

    # Height constraint cost (this becomes the "safety constraint")
    height_violation = jp.maximum(0.0, current_height - self._max_height)
    height_cost = self._height_cost_weight * height_violation

    obs = self._get_obs(pipeline_state, action)
    
    # Reward structure: encourage forward movement while staying low
    reward = forward_reward + 1.0 - ctrl_cost  # Base reward of 1.0 like humanoidstandup
    done = 0.0
    
    # Ensure all metrics have consistent shapes by expanding to match reward shape
    reward_shape = jp.shape(reward)
    
    # Update metrics with height-related information
    state.metrics.update(
        reward_linup=forward_reward,  # Keep humanoidstandup naming
        reward_quadctrl=-ctrl_cost,   # Keep humanoidstandup naming
        forward_reward=forward_reward,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=jp.linalg.norm(pipeline_state.x.pos[0, :2]),
        x_velocity=velocity[0] if pipeline_state0 is not None else 0.0,
        y_velocity=velocity[1] if pipeline_state0 is not None else 0.0,
        height=jp.broadcast_to(current_height, reward_shape),
        height_violation=jp.broadcast_to(height_violation, reward_shape),
        height_cost=jp.broadcast_to(height_cost, reward_shape),
        # Cost for PPO Lagrange (constraint violation)
        cost=jp.broadcast_to(height_cost, reward_shape),
    )
    
    # Update info dictionary with cost (required for PPO Lagrange v2)
    # Get previous info or create new one
    current_info = getattr(state, 'info', {})
    step_count = current_info.get('step_count', 0) + 1
    
    # Update info dictionary (copy existing and update)
    new_info = current_info.copy() if isinstance(current_info, dict) else {}
    new_info.update({
        "cost": height_cost,  # Cost for PPO Lagrange v2 (keep as JAX array)
        "height": current_height,
        "height_violation": height_violation,
        "step_count": step_count,
    })
    
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
    )

  def reset(self, rng: jax.Array) -> State:
    """Reset the environment with height constraint metrics."""
    state = super().reset(rng)
    
    # Initialize height-related metrics with consistent shapes
    reward_shape = jp.shape(state.reward)
    zero = jp.zeros(reward_shape)
    
    # Add height-specific metrics to the existing humanoidstandup metrics
    state.metrics.update(
        forward_reward=zero,         # Add forward movement tracking
        x_position=zero,             # Add position tracking
        y_position=zero,
        distance_from_origin=zero,
        x_velocity=zero,             # Add velocity tracking  
        y_velocity=zero,
        height=zero,                 # Height constraint metrics
        height_violation=zero, 
        height_cost=zero,
        cost=zero,                   # Initialize cost for PPO Lagrange
    )
    
    # Initialize info dictionary with cost (required for PPO Lagrange v2)
    # Use JAX arrays, not Python scalars
    info = {
        "cost": zero,  # Initialize cost in info (JAX array)
        "height": zero,
        "height_violation": zero,
        "step_count": 0,  # This can remain a Python int
    }
    
    return state.replace(info=info) 