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

# pylint:disable=g-multiple-import
"""An inverted pendulum environment."""

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class InvertedPendulum(PipelineEnv):



  # pyformat: disable
  """### Description

  This environment is the cartpole environment based on the work done by Barto,
  Sutton, and Anderson in
  ["Neuronlike adaptive elements that can solve difficult learning control
  problems"](https://ieeexplore.ieee.org/document/6313077).

  This environment involves a cart that can moved linearly, with a pole fixed on
  it at one end and having another end free. The cart can be pushed left or
  right, and the goal is to balance the pole on the top of the cart by applying
  forces on the cart.

  ### Action Space

  The agent take a 1-element vector for actions. The action space is a
  continuous `(action)` in `[-3, 3]`, where `action` represents the numerical
  force applied to the cart (with magnitude representing the amount of force and
  sign representing the direction)

  | Num | Action                    | Control Min | Control Max | Name (in
  corresponding config) | Joint | Unit      |
  |-----|---------------------------|-------------|-------------|--------------------------------|-------|-----------|
  | 0   | Force applied on the cart | -1          | 1           | thruster
  | slide | Force (N) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  pendulum system, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(4,)` where the elements correspond
  to the following:

  | Num | Observation                                   | Min  | Max | Name (in
  corresponding config) | Joint | Unit                     |
  |-----|-----------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | position of the cart along the linear surface | -Inf | Inf | thruster
  | slide | position (m)             |
  | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge
  | hinge | angle (rad)              |
  | 2   | linear velocity of the cart                   | -Inf | Inf | thruster
  | slide | velocity (m/s)           |
  | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge
  | hinge | angular velocity (rad/s) |


  ### Rewards

  The goal is to make the inverted pendulum stand upright (within a certain
  angle limit) as long as possible - as such a reward of +1 is awarded for each
  timestep that the pole is upright.

  ### Starting State
  All observations start in state (0.0, 0.0, 0.0, 0.0) with a uniform noise in
  the range of [-0.01, 0.01] added to the values for stochasticity.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches 1000 timesteps.
  2. The absolute value of the vertical angle between the pole and the cart is
  greater than 0.2 radians.
  """
  # pyformat: enable


  def __init__(self, backend='generalized', **kwargs):
    path = epath.resource_path('brax') / 'envs/assets/inverted_pendulum.xml'
    sys = mjcf.load(path)

    n_frames = 2

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 4

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
    )
    qd = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
    )
    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)
    reward, done = jp.zeros(2)
    metrics = {}

    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state = self.pipeline_step(state.pipeline_state, action)
    obs = self._get_obs(pipeline_state)
    reward = 1.0
    done = jp.where(jp.abs(obs[1]) > 0.2, 1.0, 0.0)
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])
