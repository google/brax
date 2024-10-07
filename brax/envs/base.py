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

# pylint:disable=g-multiple-import
"""A brax environment for training and inference."""

import abc
from typing import Any, Dict, List, Optional, Sequence, Union

from brax import base
from brax.generalized import pipeline as g_pipeline
from brax.io import image
from brax.mjx import pipeline as m_pipeline
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline
from flax import struct
import jax
import numpy as np


@struct.dataclass
class State(base.Base):
  """Environment state for training and inference."""

  pipeline_state: Optional[base.State]
  obs: jax.Array
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)


class Env(abc.ABC):
  """Interface for driving training and inference."""

  @abc.abstractmethod
  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""

  @property
  @abc.abstractmethod
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""

  @property
  @abc.abstractmethod
  def action_size(self) -> int:
    """The size of the action vector expected by step."""

  @property
  @abc.abstractmethod
  def backend(self) -> str:
    """The physics backend that this env was instantiated with."""

  @property
  def unwrapped(self) -> 'Env':
    return self


class PipelineEnv(Env):
  """API for driving a brax system for training and inference."""

  __pytree_ignore__ = (
      '_backend',
      '_pipeline',
  )

  def __init__(
      self,
      sys: base.System,
      backend: str = 'mjx',
      n_frames: int = 1,
      debug: bool = False,
  ):
    """Initializes PipelineEnv.

    Args:
      sys: system defining the kinematic tree and other properties
      backend: string specifying the physics pipeline
      n_frames: the number of times to step the physics pipeline for each
        environment step
      debug: whether to get debug info from the pipeline init/step
    """
    self.sys = sys

    pipeline = {
        'generalized': g_pipeline,
        'spring': s_pipeline,
        'positional': p_pipeline,
        'mjx': m_pipeline,
    }
    if backend not in pipeline:
      raise ValueError(f'backend should be in {pipeline.keys()}.')

    self._backend = backend
    self._pipeline = pipeline[backend]
    self._n_frames = n_frames
    self._debug = debug

  def pipeline_init(
      self,
      q: jax.Array,
      qd: jax.Array,
      act: Optional[jax.Array] = None,
      ctrl: Optional[jax.Array] = None,
  ) -> base.State:
    """Initializes the pipeline state."""
    return self._pipeline.init(self.sys, q, qd, act, ctrl, self._debug)

  def pipeline_step(self, pipeline_state: Any, action: jax.Array) -> base.State:
    """Takes a physics step using the physics pipeline."""

    def f(state, _):
      return (
          self._pipeline.step(self.sys, state, action, self._debug),
          None,
      )

    return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

  @property
  def dt(self) -> jax.Array:
    """The timestep used for each env step."""
    return self.sys.opt.timestep * self._n_frames  # pytype: disable=attribute-error

  @property
  def observation_size(self) -> int:
    rng = jax.random.PRNGKey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def action_size(self) -> int:
    return self.sys.act_size()

  @property
  def backend(self) -> str:
    return self._backend

  def render(
      self,
      trajectory: List[base.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
  ) -> Sequence[np.ndarray]:
    """Renders a trajectory using the MuJoCo renderer."""
    return image.render_array(self.sys, trajectory, height, width, camera)


class Wrapper(Env):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Env):
    self.env = env

  def reset(self, rng: jax.Array) -> State:
    return self.env.reset(rng)

  def step(self, state: State, action: jax.Array) -> State:
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

  @property
  def backend(self) -> str:
    return self.unwrapped.backend

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)
