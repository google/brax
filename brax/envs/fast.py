# Copyright 2023 The Brax Authors.
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

"""Gotta go fast!  This trivial Env is meant for unit testing."""

from brax import base
from brax.envs import env
import jax.numpy as jnp


class Fast(env.PipelineEnv):
  """Trains an agent to go fast."""

  def __init__(self, **kwargs):
    self._dt = 0.02
    self._reset_count = 0
    self._step_count = 0

  def reset(self, rng: jnp.ndarray) -> env.State:  # pytype: disable=signature-mismatch  # jax-ndarray
    self._reset_count += 1
    pipeline_state = base.State(
        q=jnp.zeros(1),
        qd=jnp.zeros(1),
        x=base.Transform.create(pos=jnp.zeros(3)),
        xd=base.Motion.create(vel=jnp.zeros(3)),
        contact=None
    )
    obs = jnp.zeros(2)
    reward, done = jnp.array(0.0), jnp.array(0.0)
    return env.State(pipeline_state, obs, reward, done)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:  # pytype: disable=signature-mismatch  # jax-ndarray
    self._step_count += 1
    vel = state.pipeline_state.xd.vel + (action > 0) * self._dt
    pos = state.pipeline_state.x.pos + vel * self._dt

    qp = state.pipeline_state.replace(
        x=state.pipeline_state.x.replace(pos=pos),
        xd=state.pipeline_state.xd.replace(vel=vel),
    )
    obs = jnp.array([pos[0], vel[0]])
    reward = pos[0]

    return state.replace(pipeline_state=qp, obs=obs, reward=reward)

  @property
  def reset_count(self):
    return self._reset_count

  @property
  def step_count(self):
    return self._step_count

  @property
  def observation_size(self):
    return 2

  @property
  def action_size(self):
    return 1
