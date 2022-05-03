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


class PITM(env.Env):
  """Trains a single agent n-player piggy in the middle game"""

  def __init__(self, **kwargs):
    self.default_qp = kwargs.pop('default_qp')
    config = kwargs.pop('config')
    self.n_players = kwargs.pop('n_players')
    self.body_idx = kwargs.pop('body_idx')
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.default_qp
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = dict(
        piggy_ball_cost=zero,
        piggy_reach_ball_cost=zero,
        player_ball_reward=zero,
        ctrl_cost=zero,
        contact_cost=zero,
        survive_reward=zero,
    )
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    idx = self.body_idx

    # adding impulse to piggy - moves towards ball
    x_dist_before = state.qp.pos[idx['ball'], 0] - state.qp.pos[idx['piggy'], 0]
    y_dist_before = state.qp.pos[idx['ball'], 1] - state.qp.pos[idx['piggy'], 1]
    acc = 10 # force (acceleration * mass of 1.0)
    vec = jp.array([x_dist_before, y_dist_before, 0.])
    vec = vec / jp.sum(vec**2)**0.5 # normalize vector
    acc = acc * vec
    # action is an array of length 2*n_players, x and y for each
    act = [] # list of actions (x,y,z for piggy + each player)
    for i in range(self.n_players):
      act.append(action[2*i])   # x
      act.append(action[2*i+1]) # y
      act.append(0.)            # z
    act = jp.concatenate([acc, jp.array(act)])

    print('Size of action vec: ', action.shape, action)
    print('Act vec: ', act.shape, act)

    qp, info = self.sys.step(state.qp, act)
    obs = self._get_obs(qp, info)
    
    # penalty for piggy approaching the ball
    x_dist_after = qp.pos[idx['ball'], 0] - qp.pos[idx['piggy'], 0]
    y_dist_after = qp.pos[idx['ball'], 1] - qp.pos[idx['piggy'], 1]
    dist_before = abs((x_dist_before**2 + y_dist_before**2)**0.5)
    dist_after = abs((x_dist_after**2 + y_dist_after**2)**0.5)
    piggy_ball_cost = (dist_before - dist_after) / self.sys.config.dt  # +ve means ball is closer
    piggy_ball_cost *= 10.
    
    # big penalty for piggy reaching ball
    piggy_reach_ball_cost =  (dist_after < 1.5) * 100.

    # small reward for players approaching the ball
    player_ball_reward = 0
    for n in range(1, self.n_players+1):
      x_dist_before = abs(state.qp.pos[idx['p%d'%n], 0] - state.qp.pos[idx['ball'], 0])
      x_dist_after = abs(qp.pos[idx['p%d'%n], 0] - qp.pos[idx['ball'], 0])
      y_dist_before = abs(state.qp.pos[idx['p%d'%n], 1] - state.qp.pos[idx['ball'], 1])
      y_dist_after = abs(qp.pos[idx['p%d'%n], 1] - qp.pos[idx['ball'], 1])
      dist_before = abs((x_dist_before**2 + y_dist_before**2)**0.5)
      dist_after = abs((x_dist_after**2 + y_dist_after**2)**0.5)
      player_ball_reward += (dist_before - dist_after) / self.sys.config.dt  # +ve means ball is closer
    player_ball_reward *= 0.1
    
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
        ctrl_cost=ctrl_cost,
        contact_cost=contact_cost,
        survive_reward=survive_reward,
    )
    print(state.metrics)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return self.n_players * 2 # x and y for each player -- z force always zero, piggy always moves towards ball

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observes body position and velocities."""
    # Trying something - observe everything?
    pos, rot, vel, ang = [qp.pos.flatten()], [qp.rot.flatten()], [qp.vel.flatten()], [qp.ang.flatten()]
    cfrc = []
    # cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc] # flatten bottom dimension
    obs = jp.concatenate(pos + rot + vel + ang + cfrc)

    return obs