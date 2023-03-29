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

"""An inverted pendulum environment."""

from brax import base
from brax.envs import env
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class InvertedPendulum(env.PipelineEnv):



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

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder).

  ```
  env = gym.make('InvertedPendulum-v2')
  ```

  There is no v3 for InvertedPendulum, unlike the robot environments where a v3
  and beyond take gym.make kwargs such as ctrl_cost_weight, reset_noise_scale
  etc.

  There is a v4 version that uses the mujoco-bindings

  ```
  env = gym.make('InvertedPendulum-v4')
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('InvertedPendulum-v5')
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v3: support for gym.make kwargs such as ctrl_cost_weight, reset_noise_scale
    etc. rgb rendering comes from tracking camera (so agent does not run away
    from screen)
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks (including inverted
    pendulum)
  * v0: Initial versions release (1.0.0)
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

  def reset(self, rng: jp.ndarray) -> env.State:
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

    return env.State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
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

  def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])
