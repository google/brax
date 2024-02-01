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

"""An inverted pendulum environment."""

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class InvertedDoublePendulum(env.Env):



  """
  ### Description

  This environment originates from control theory and builds on the cartpole
  environment based on the work done by Barto, Sutton, and Anderson in
  ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077).

  This environment involves a cart that can moved linearly, with a pole fixed on
  it and a second pole fixed on the other end of the first one (leaving the
  second pole as the only one with one free end). The cart can be pushed left or
  right, and the goal is to balance the second pole on top of the first pole,
  which is in turn on top of the cart, by applying continuous forces on the
  cart.

  ### Action Space

  The agent take a 1-element vector for actions.

  The action space is a continuous `(action)` in `[-3, 3]`, where `action`
  represents the numerical force applied to the cart (with magnitude
  representing the amount of force and sign representing the direction)

  | Num | Action                    | Control Min | Control Max | Name (in corresponding config) | Joint | Unit      |
  |-----|---------------------------|-------------|-------------|--------------------------------|-------|-----------|
  | 0   | Force applied on the cart | -3          | 3           | slider                         | slide | Force (N) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  pendulum system, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(11,)` where the elements
  correspond to the following:

  | Num | Observation                                                       | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|-------------------------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | position of the cart along the linear surface                     | -Inf | Inf | thruster                       | slide | position (m)             |
  | 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                     | hinge | unitless                 |
  | 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                    | hinge | unitless                 |
  | 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                     | hinge | unitless                 |
  | 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                    | hinge | unitless                 |
  | 5   | velocity of the cart                                              | -Inf | Inf | thruster                       | slide | velocity (m/s)           |
  | 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                          | hinge | angular velocity (rad/s) |
  | 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                         | hinge | angular velocity (rad/s) |

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
  env = gym.make('InvertedDoublePendulum-v2')
  ```

  There is no v3 for InvertedDoublePendulum, unlike the robot environments where
  a v3 and beyond take gym.make kwargs such as ctrl_cost_weight,
  reset_noise_scale etc.

  There is a v4 version that uses the mujoco-bindings

  ```
  env = gym.make('InvertedDoublePendulum-v4')
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('InvertedDoublePendulum-v5')
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


  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

    obs = self._get_obs(qp)
    reward, done = jp.zeros(2)
    metrics = {}

    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, _ = self.sys.step(state.qp, action)
    _, joint_vel = self.sys.joints[0].angle_vel(qp)

    tip_pos = jp.take(qp, 2).to_world(jp.array([0, 0, .3]))  # pytype: disable=wrong-arg-types  # jax-ndarray
    (x, _, y), _ = tip_pos
    dist_penalty = 0.01 * x**2 + (y - 2)**2
    v1, v2 = joint_vel
    vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
    alive_bonus = 10.0

    obs = self._get_obs(qp)
    reward = alive_bonus - dist_penalty - vel_penalty
    done = jp.where(y <= 1, jp.float32(1), jp.float32(0))

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return 1

  def _get_obs(self, qp: brax.QP) -> jp.ndarray:
    """Observe cartpole body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    # qpos: position and orientation of the torso and the joint angles
    qpos = [qp.pos[0, :1], jp.sin(joint_angle), jp.cos(joint_angle)]

    # qvel: velocity of the torso and the joint angle velocities.
    qvel = [qp.vel[0, :1], joint_vel]

    return jp.concatenate(qpos + qvel)

  def _noise(self, rng):
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), -0.01, 0.01)


_SYSTEM_CONFIG = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      capsule {
        radius: 0.1
        length: 0.4
      }
    }
    frozen { position { x:0 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge"
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  bodies {
    name: "pole2"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge2"
    parent: "pole"
    child: "pole2"
    parent_offset { z: .3 }
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  forces {
    name: "thruster"
    body: "cart"
    strength: 500.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.05
  substeps: 12
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      capsule {
        radius: 0.1
        length: 0.4
      }
    }
    frozen { position { x:0 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge"
    stiffness: 30000.0
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    spring_damping: 500.0
    angle_limit { min: 0.0 max: 0.0 }
  }
  bodies {
    name: "pole2"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge2"
    stiffness: 30000.0
    parent: "pole"
    child: "pole2"
    parent_offset { z: .3 }
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    spring_damping: 500.0
    angle_limit { min: 0.0 max: 0.0 }
  }
  forces {
    name: "thruster"
    body: "cart"
    strength: 500.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.05
  substeps: 12
  dynamics_mode: "legacy_spring"
  """
