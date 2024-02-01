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

"""A brax environment for training and inference."""

import abc
from typing import Any, Dict, Optional

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1 import pytree
from flax import struct

from google.protobuf import text_format


@struct.dataclass
class State:
  """Environment state for training and inference."""
  qp: brax.QP
  obs: jp.ndarray
  reward: jp.ndarray
  done: jp.ndarray
  metrics: Dict[str, jp.ndarray] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)


@pytree.register
class Env(abc.ABC):
  """API for driving a brax system for training and inference."""


  def __init__(self, config: Optional[str], *args, **kwargs):
    if config:
      config = text_format.Parse(config, brax.Config())
      self.sys = brax.System(config, *args, **kwargs)

  @abc.abstractmethod
  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

  @property
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    rng = jp.random_prngkey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def action_size(self) -> int:
    """The size of the action vector expected by step."""
    return self.sys.num_joint_dof + self.sys.num_forces_dof

  @property
  def unwrapped(self) -> 'Env':
    return self


class Wrapper(Env):
  """Wraps the environment to allow modular transformations."""

  def __init__(self, env: Env):
    super().__init__(config=None)
    self.env = env

  def reset(self, rng: jp.ndarray) -> State:
    return self.env.reset(rng)

  def step(self, state: State, action: jp.ndarray) -> State:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> int:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Env:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)
