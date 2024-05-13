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
"""Trains a halfcheetah to run in the +x direction."""

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Halfcheetah(PipelineEnv):



  # pyformat: disable
  """
  ### Description

  This environment is based on the work by P. Wawrzyński in
  ["A Cat-Like Robot Real-Time Learning to Run"](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).

  The HalfCheetah is a 2-dimensional robot consisting of 9 links and 8 joints
  connecting them (including two paws).

  The goal is to apply a torque on the joints to make the cheetah run forward
  (right) as fast as possible, with a positive reward allocated based on the
  distance moved forward and a negative reward allocated for moving backward.

  The torso and head of the cheetah are fixed, and the torque can only be
  applied on the other 6 joints over the front and back thighs (connecting to
  the torso), shins (connecting to the thighs) and feet (connecting to the
  shins).

  ### Action Space

  The agents take a 6-element vector for actions. The action space is a
  continuous `(action, action, action, action, action, action)` all in
  `[-1.0, 1.0]`, where `action` represents the numerical torques applied
  between *links*

  | Num | Action                                  | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|-----------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                         | hinge | torque (N m) |
  | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                          | hinge | torque (N m) |
  | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                          | hinge | torque (N m) |
  | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                         | hinge | torque (N m) |
  | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                          | hinge | torque (N m) |
  | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                          | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  cheetah, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(18,)` where the elements
  correspond to the following:

  | Num | Observation                          | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|--------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the center of mass   | -Inf | Inf | rootx                          | slide | position (m)             |
  | 1   | w-orientation of the front tip       | -Inf | Inf | rooty                          | hinge | angle (rad)              |
  | 2   | y-orientation of the front tip       | -Inf | Inf | rooty                          | hinge | angle (rad)              |
  | 3   | angle of the back thigh rotor        | -Inf | Inf | bthigh                         | hinge | angle (rad)              |
  | 4   | angle of the back shin rotor         | -Inf | Inf | bshin                          | hinge | angle (rad)              |
  | 5   | angle of the back foot rotor         | -Inf | Inf | bfoot                          | hinge | angle (rad)              |
  | 6   | velocity of the tip along the y-axis | -Inf | Inf | fthigh                         | hinge | angle (rad)              |
  | 7   | angular velocity of front tip        | -Inf | Inf | fshin                          | hinge | angle (rad)              |
  | 8   | angular velocity of second rotor     | -Inf | Inf | ffoot                          | hinge | angle (rad)              |
  | 9   | x-coordinate of the front tip        | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
  | 10  | y-coordinate of the front tip        | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
  | 11  | angle of the front tip               | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
  | 12  | angle of the second rotor            | -Inf | Inf | bthigh                         | hinge | angular velocity (rad/s) |
  | 13  | angle of the second rotor            | -Inf | Inf | bshin                          | hinge | angular velocity (rad/s) |
  | 14  | velocity of the tip along the x-axis | -Inf | Inf | bfoot                          | hinge | angular velocity (rad/s) |
  | 15  | velocity of the tip along the y-axis | -Inf | Inf | fthigh                         | hinge | angular velocity (rad/s) |
  | 16  | angular velocity of front tip        | -Inf | Inf | fshin                          | hinge | angular velocity (rad/s) |
  | 17  | angular velocity of second rotor     | -Inf | Inf | ffoot                          | hinge | angular velocity (rad/s) |

  ### Rewards

  The reward consists of two parts:

  - *reward_run*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.05*. This reward would be
    positive if the cheetah runs forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the cheetah if it takes
    actions that are too large. It is measured as *-coefficient x
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.1

  ### Starting State

  All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,) with a noise added to the initial
  state for stochasticity. As seen before, the first 8 values in the state are
  positional and the last 9 values are velocity. A uniform noise in the range of
  [-0.1, 0.1] is added to the positional values while a standard normal noise
  with a mean of 0 and standard deviation of 0.1 is added to the initial
  velocity values of all zeros.

  ### Episode Termination

  The episode terminates when the episode length is greater than 1000.
  """
  # pyformat: enable


  def __init__(
      self,
      forward_reward_weight=1.0,
      ctrl_cost_weight=0.1,
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs
  ):
    path = epath.resource_path('brax') / 'envs/assets/half_cheetah.xml'
    sys = mjcf.load(path)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.003125})
      n_frames = 16
      gear = jp.array([120, 90, 60, 120, 100, 100])
      sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'x_position': zero,
        'x_velocity': zero,
        'reward_ctrl': zero,
        'reward_run': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Runs one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0  is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    x_velocity = (
        pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
    ) / self.dt
    forward_reward = self._forward_reward_weight * x_velocity
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(pipeline_state)
    reward = forward_reward - ctrl_cost
    state.metrics.update(
        x_position=pipeline_state.x.pos[0, 0],
        x_velocity=x_velocity,
        reward_run=forward_reward,
        reward_ctrl=-ctrl_cost,
    )

    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Returns the environment observations."""
    position = pipeline_state.q
    velocity = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      position = position[1:]

    return jp.concatenate((position, velocity))
