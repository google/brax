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

"""Trains n agents to keep a ball away from a heuristic agent that chases it."""

import brax
from brax import jumpy as jp
from brax.envs import env
from brax.ben_utils.utils import make_config


class PITM(env.Env):
  """Trains a single agent n-player piggy in the middle game"""

  def __init__(self, **kwargs):
    self.default_qp = kwargs.pop('default_qp')
    config = kwargs.pop('config')
    n_players = kwargs.pop('n_players')
    # if not config: 
    #     config = _SYSTEM_CONFIG 
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.default_qp
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'piggy_to_ball_cost': zero,
        'piggy_reach_ball_cost': zero,
        'player_to_ball_cost': zero,
        'reward_ctrl_cost': zero,
        'reward_contact_cost': zero,
        'reward_survive': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)
    
    # penalty for piggy approaching the ball
    x_dist_before = abs(state.qp.pos[idx['piggy'], 0] - state.qp.pos[idx['ball'], 0])
    x_dist_after = abs(qp.pos[idx['piggy'], 0] - qp.pos[idx['ball'], 0])
    y_dist_before = abs(state.qp.pos[idx['piggy'], 1] - state.qp.pos[idx['ball'], 1])
    y_dist_after = abs(qp.pos[idx['piggy'], 1] - qp.pos[idx['ball'], 1])
    dist_before = abs((x_dist_before**2 + y_dist_before**2)**0.5)
    dist_after = abs((x_dist_after**2 + y_dist_after**2)**0.5)
    piggy_ball_cost = (dist_before - dist_after) / self.sys.config.dt  # +ve means ball is closer
    
    # big penalty for piggy reaching ball
    piggy_reach_ball_cost =  (dist_after < 1.5) * 100

    # small reward for players approaching the ball
    player_ball_reward = 0
    for n in range(n_players):
      x_dist_before = abs(state.qp.pos[idx['p%d'%n], 0] - state.qp.pos[idx['ball'], 0])
      x_dist_after = abs(qp.pos[idx['p%d'%n], 0] - qp.pos[idx['ball'], 0])
      y_dist_before = abs(state.qp.pos[idx['p%d'%n], 1] - state.qp.pos[idx['ball'], 1])
      y_dist_after = abs(qp.pos[idx['p%d'%n], 1] - qp.pos[idx['ball'], 1])
      dist_before = abs((x_dist_before**2 + y_dist_before**2)**0.5)
      dist_after = abs((x_dist_after**2 + y_dist_after**2)**0.5)
      player_ball_reward += (dist_before - dist_after) / self.sys.config.dt  # +ve means ball is closer
    
    ctrl_cost = .5 * jp.sum(jp.square(action)) # dependent on torque
    
    # contact cost - leave in for now
    contact_cost = 0.5 * 1e-3 * jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)))
    survive_reward = jp.float32(1)
    
    reward = -piggy_ball_cost - piggy_reach_ball_cost + player_ball_reward - ctrl_cost - contact_cost + survive_reward

    # termination - these shouldn't matter for our ball
    done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
    # done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done) # don't want it to stop when it bounces...
    state.metrics.update(
        piggy_ball_cost=piggy_ball_cost,
        piggy_reach_ball_cost=piggy_reach_ball_cost,
        player_ball_reward=player_ball_reward,
        reward_ctrl_cost=ctrl_cost,
        reward_contact_cost=contact_cost,
        reward_survive=survive_reward)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observes body position and velocities."""
    # # some pre-processing to pull joint angles and velocities
    # (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    # # qpos:
    # # Z of the torso (1,)
    # # orientation of the torso as quaternion (4,)
    # # joint angles (8,)
    # qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

    # # qvel:
    # # velocity of the torso (3,)
    # # angular velocity of the torso (3,)
    # # joint angle velocities (8,)
    # qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # # external contact forces:
    # # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
    # # ignores
    # cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
    # # flatten bottom dimension
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
    
    # obs = jp.concatenate(qpos + qvel + cfrc)
    
    ###############################################################################################
    
    # Trying something - observe everything?
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    pos, rot, vel, ang = [qp.pos.flatten(), joint_angle], [qp.rot.flatten()], [qp.vel.flatten()], [qp.ang.flatten(), joint_vel]
    cfrc = []
    # cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc] # flatten bottom dimension
    obs = jp.concatenate(pos + rot + vel + ang + cfrc)

    return obs