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

"""HalfCheetah that jumps."""

from brax.v1 import jumpy as jp
from brax.v1.envs import env
from brax.v1.envs import halfcheetah


class JumpCheetah(halfcheetah.Halfcheetah):
  """Run + Jump HalfCheetah."""

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_ctrl_cost': zero,
        'reward_forward': zero,
        'reward_hop': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    x_before = state.qp.pos[0, 0]
    x_after = qp.pos[0, 0]
    forward_reward = (x_after - x_before) / self.sys.config.dt
    ctrl_cost = -.1 * jp.sum(jp.square(action))
    z_before = state.qp.pos[0, 2]
    z_after = qp.pos[0, 2]
    hop_reward = jp.abs(z_after - z_before) / self.sys.config.dt
    reward = forward_reward + ctrl_cost + hop_reward
    state.metrics.update(
        reward_ctrl_cost=ctrl_cost,
        reward_forward=forward_reward,
        reward_hop=hop_reward)

    return state.replace(qp=qp, obs=obs, reward=reward)
