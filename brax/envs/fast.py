# Copyright 2022 The Brax Authors.
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

import brax
from brax.envs import env
import jax.numpy as jnp


class Fast(env.Env):
  """Trains an agent to go fast."""

  def __init__(self, **kwargs):
    super().__init__(config='dt: .02', **kwargs)

  def reset(self, rng: jnp.ndarray) -> env.State:
    zero = jnp.zeros(1)
    qp = brax.QP(pos=zero, vel=zero, rot=zero, ang=zero)
    obs = jnp.zeros(2)
    reward, done = jnp.zeros(2)
    return env.State(qp, obs, reward, done)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    vel = state.qp.vel + (action > 0) * self.sys.config.dt
    pos = state.qp.pos + vel * self.sys.config.dt

    qp = state.qp.replace(pos=pos, vel=vel)
    obs = jnp.array([pos[0], vel[0]])
    reward = pos[0]

    return state.replace(qp=qp, obs=obs, reward=reward)

  @property
  def observation_size(self):
    return 2

  @property
  def action_size(self):
    return 1
