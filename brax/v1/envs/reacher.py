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

"""Trains a reacher to reach a target.

Based on the OpenAI Gym MuJoCo Reacher environment.
"""

from typing import Tuple

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class Reacher(env.Env):



  """
  ### Description

  "Reacher" is a two-jointed robot arm. The goal is to move the robot's end
  effector (called *fingertip*) close to a target that is spawned at a random
  position.

  ### Action Space

  The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)`
  represents the torques applied at the hinge joints.

  | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1          | 1           | joint0                         | hinge | torque (N m) |
  | 1   | Torque applied at the second hinge (connecting the two links)                   | -1          | 1           | joint1                         | hinge | torque (N m) |

  ### Observation Space

  Observations consist of:

  - The cosine of the angles of the two arms
  - The sine of the angles of the two arms
  - The coordinates of the target
  - The angular velocities of the arms
  - The vector between the target and the reacher's fingertip (3 dimensional
    with the last element being 0)

  The observation is a `ndarray` with shape `(11,)` where the elements
  correspond to the following:

  | Num | Observation                                                                                    | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|------------------------------------------------------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                    | hinge | unitless                 |
  | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                    | hinge | unitless                 |
  | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | cos(joint0)                    | hinge | unitless                 |
  | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | cos(joint1)                    | hinge | unitless                 |
  | 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                       | slide | position (m)             |
  | 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                       | slide | position (m)             |
  | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                         | hinge | angular velocity (rad/s) |
  | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                         | hinge | angular velocity (rad/s) |
  | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                             | slide | position (m)             |
  | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                             | slide | position (m)             |
  | 10  | z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both) | -Inf | Inf | NA                             | slide | position (m)             |

  ### Rewards

  The reward consists of two parts:

  - *reward_dist*: This reward is a measure of how far the *fingertip*
    of the reacher (the unattached end) is from the target, with a more negative
    value assigned for when the reacher's *fingertip* is further away from the
    target. It is calculated as the negative vector norm of (position of
    the fingertip - position of target), or *-norm("fingertip" - "target")*.
  - *reward_ctrl*: A negative reward for penalising the walker if it takes
    actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

  Unlike other environments, Reacher does not allow you to specify weights for
  the individual reward terms. However, `info` does contain the keys
  *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms, you
  should create a wrapper that computes the weighted reward from `info`.

  ### Starting State

  All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0) with a noise added for stochasticity. A uniform noise in the range
  [-0.1, 0.1] is added to the positional attributes, while the target position
  is selected uniformly at random in a disk of radius 0.2 around the origin.

  Independent, uniform noise in the range of [-0.005, 0.005] is added to the
  velocities, and the last element ("fingertip" - "target") is calculated at the
  end once everything is set.

  The default setting has a *dt = 0.02*

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder)..

  ```
  env = gym.make('Reacher-v2')
  ```

  There is no v3 for Reacher, unlike the robot environments where a v3 and
  beyond take gym.make kwargs such as ctrl_cost_weight, reset_noise_scale etc.

  There is a v4 version that uses the mujoco-bindings

  ```
  env = gym.make('Reacher-v4')
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('Reacher-v5')
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks (not including
    reacher, which has a max_time_steps of 50). Added reward_threshold to
    environments.
  * v0: Initial versions release (1.0.0)
  """


  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)
    self._target_idx = self.sys.body.index['target']
    self._arm_idx = self.sys.body.index['body1']

  def reset(self, rng: jp.ndarray) -> env.State:
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)

    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    _, target = self._random_target(rng)
    pos = jp.index_update(qp.pos, self._target_idx, target)

    qp = qp.replace(pos=pos)
    obs = self._get_obs(qp, self.sys.info(qp))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_dist': zero,
        'reward_ctrl': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # vector from tip to target is last 3 entries of obs vector
    reward_dist = -jp.norm(obs[-3:])
    reward_ctrl = -jp.square(action).sum()
    reward = reward_dist + reward_ctrl

    state.metrics.update(
        reward_dist=reward_dist,
        reward_ctrl=reward_ctrl,
    )

    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Egocentric observation of target and arm body."""
    joint_angle, _ = self.sys.joints[0].angle_vel(qp)

    # qpos:
    # x,y coord of target
    qpos = [qp.pos[self._target_idx, :2]]

    # dist to target and speed of tip
    arm_qps = jp.take(qp, jp.array(self._arm_idx))
    tip_pos, tip_vel = arm_qps.to_world(jp.array([0.11, 0., 0.]))
    tip_to_target = [tip_pos - qp.pos[self._target_idx]]
    cos_sin_angle = [jp.cos(joint_angle), jp.sin(joint_angle)]

    # qvel:
    # velocity of tip
    qvel = [tip_vel[:2]]

    return jp.concatenate(cos_sin_angle + qpos + qvel + tip_to_target)

  def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns a target location in a random circle slightly above xy plane."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    dist = .2 * jp.random_uniform(rng1)
    ang = jp.pi * 2. * jp.random_uniform(rng2)
    target_x = dist * jp.cos(ang)
    target_y = dist * jp.sin(ang)
    target_z = .01
    target = jp.array([target_x, target_y, target_z]).transpose()
    return rng, target


_SYSTEM_CONFIG = """
  bodies {
    name: "ground"
    colliders {
      plane {
      }
    }
    mass: 1.0
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    frozen {
      all: true
    }
  }
  bodies {
    name: "body0"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "body1"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    colliders {
      position { x: .11 }
      sphere {
        radius: 0.01
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "target"
    colliders {
      position {
      }
      sphere {
        radius: 0.009
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
  }
  joints {
    name: "joint0"
    parent: "ground"
    child: "body0"
    parent_offset {
      z: 0.01
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
        min: -360
        max: 360
      }
  }
  joints {
    name: "joint1"
    parent: "body0"
    child: "body1"
    parent_offset {
      x: 0.1
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -360
      max: 360
    }
  }
  actuators {
    name: "joint0"
    joint: "joint0"
    strength: 25.0
    torque {
    }
  }
  actuators {
    name: "joint1"
    joint: "joint1"
    strength: 25.0
    torque {
    }
  }
  collide_include {
  }
  gravity {
    z: -9.81
  }
  dt: 0.02
  substeps: 4
  frozen {
    position {
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
    }
  }
  dynamics_mode: "pbd"
  """


_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "ground"
    colliders {
      plane {
      }
    }
    mass: 1.0
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    frozen {
      all: true
    }
  }
  bodies {
    name: "body0"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "body1"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    colliders {
      position { x: .11 }
      sphere {
        radius: 0.01
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "target"
    colliders {
      position {
      }
      sphere {
        radius: 0.009
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
  }
  joints {
    name: "joint0"
    stiffness: 100.0
    parent: "ground"
    child: "body0"
    parent_offset {
      z: 0.01
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
        min: -360
        max: 360
      }
    limit_strength: 0.0
    spring_damping: 3.0
  }
  joints {
    name: "joint1"
    stiffness: 100.0
    parent: "body0"
    child: "body1"
    parent_offset {
      x: 0.1
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -360
      max: 360
    }
    limit_strength: 0.0
    spring_damping: 3.0
  }
  actuators {
    name: "joint0"
    joint: "joint0"
    strength: 25.0
    torque {
    }
  }
  actuators {
    name: "joint1"
    joint: "joint1"
    strength: 25.0
    torque {
    }
  }
  collide_include {
  }
  gravity {
    z: -9.81
  }
  baumgarte_erp: 0.1
  dt: 0.02
  substeps: 4
  frozen {
    position {
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
    }
  }
  dynamics_mode: "legacy_spring"
  """
