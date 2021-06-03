# Copyright 2021 The Brax Authors.
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

"""A brax environment for training and inference."""

import abc
from typing import Dict

from flax import struct
import jax
import jax.numpy as jnp
import brax


@struct.dataclass
class State:
  """Environment state for training and inference."""
  rng: jnp.ndarray
  qp: brax.QP
  info: brax.Info
  obs: jnp.ndarray
  reward: jnp.ndarray
  done: jnp.ndarray
  steps: jnp.ndarray
  metrics: Dict[str, jnp.ndarray]


class Env(abc.ABC):
  """API for driving a brax system for training and inference."""

  def __init__(self,
               config: brax.Config,
               batch_size: int = 0,
               action_repeat: int = 1,
               episode_length: int = 1000):
    config.dt *= action_repeat
    config.substeps *= action_repeat
    self.sys = brax.System(config)
    self.batch_size = batch_size
    self.action_repeat = action_repeat
    self.episode_length = episode_length

    if batch_size:
      self.reset = jax.vmap(self.reset)
      self.step = jax.vmap(self.step)

  @abc.abstractmethod
  def reset(self, rng: jnp.ndarray) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jnp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

  @property
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    rng = jax.random.PRNGKey(0)
    if self.batch_size:
      rng = jax.random.split(rng, self.batch_size)
    reset_state = self.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def action_size(self) -> int:
    """The size of the action vector expected by step."""
    return self.sys.num_joint_dof
