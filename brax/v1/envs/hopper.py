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

"""Trains a hopper to run in the +x direction."""

from typing import Tuple

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1.envs import env


class Hopper(env.Env):



  """
  ### Description

  This environment is based on the work done by Erez, Tassa, and Todorov in
  ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
  increase the number of independent state and control variables as compared to
  the classic control environments.

  The hopper is a two-dimensional one-legged figure that consist of four main
  body parts - the torso at the top, the thigh in the middle, the leg in the
  bottom, and a single foot on which the entire body rests.

  The goal is to make hops that move in the forward (right) direction by
  applying torques on the three hinges connecting the four body parts.

  ### Action Space

  The agent take a 3-element vector for actions. The action space is a
  continuous `(action, action, action)` all in `[-1, 1]`, where `action`
  represents the numerical torques applied between *links*

  | Num | Action                             | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                    | hinge | torque (N m) |
  | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                      | hinge | torque (N m) |
  | 3   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                     | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  hopper, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(11,)` where the elements
  correspond to the following:

  | Num | Observation                                      | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|--------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz                          | slide | position (m)             |
  | 1   | angle of the top                                 | -Inf | Inf | rooty                          | hinge | angle (rad)              |
  | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                    | hinge | angle (rad)              |
  | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                      | hinge | angle (rad)              |
  | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                     | hinge | angle (rad)              |
  | 5   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
  | 6   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
  | 7   | angular velocity of the angle of the top         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
  | 8   | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                    | hinge | angular velocity (rad/s) |
  | 9   | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                      | hinge | angular velocity (rad/s) |
  | 10  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                     | hinge | angular velocity (rad/s) |

  ### Rewards

  The reward consists of three parts:

  - *reward_healthy*: Every timestep that the hopper is alive, it gets a reward
    of 1,
  - *reward_forward*: A reward of hopping forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.008*. This reward would be
    positive if the hopper hops forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the hopper if it takes
    actions that are too large. It is measured as *-coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.001

  ### Starting State

  All observations start in state (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0) with a uniform noise in the range of [-0.005, 0.005] added to the
  values for stochasticity.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The height of the hopper becomes less than 0.7 metres (hopper has hopped
     too low).
  3. The absolute value of the angle (index 2) is less than 0.2 radians (hopper
     has fallen down).

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder
  (or by changing the path to a modified XML file in another folder).

  ```
  env = gym.make('Hopper-v2')
  ```

  v3, v4, and v5 take gym.make kwargs such as ctrl_cost_weight,
  reset_noise_scale etc.

  ```
  env = gym.make('Hopper-v5', ctrl_cost_weight=0.1, ....)
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight,
        reset_noise_scale etc. rgb rendering comes from tracking camera (so
        agent does not run away from screen)
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks. Added
        reward_threshold to environments.
  * v0: Initial versions release (1.0.0)
  """


  def __init__(self,
               forward_reward_weight: float = 1.0,
               ctrl_cost_weight: float = 1e-3,
               healthy_reward: float = 1.0,
               terminate_when_unhealthy: bool = True,
               healthy_z_range: Tuple[float, float] = (0.7, float('inf')),
               healthy_angle_range=(-0.2, 0.2),
               reset_noise_scale=5e-3,
               exclude_current_positions_from_observation=True,
               legacy_spring=False,
               **kwargs):
    """Creates a Hopper environment.

    Args:
      forward_reward_weight: Weight for the forward reward, i.e. velocity in
        x-direction.
      ctrl_cost_weight: Weight for the control cost.
      healthy_reward: Reward for staying healthy, i.e. respecting the posture
        constraints.
      terminate_when_unhealthy: Done bit will be set when unhealthy if true.
      healthy_z_range: Range of the z-position for being healthy.
      healthy_angle_range: Range of joint angles for being healthy.
      reset_noise_scale: Scale of noise to add to reset states.
      exclude_current_positions_from_observation: x-position will be hidden
        from the observations if true.
      legacy_spring: if True, reverts to legacy spring dynamics instead of pbd.
      **kwargs: Arguments that are passed to the base class.
    """
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._healthy_angle_range = healthy_angle_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)

    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_ctrl': zero,
        'reward_healthy': zero,
        'x_position': zero,
        'x_velocity': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, _ = self.sys.step(state.qp, action)

    x_velocity = (qp.pos[0, 0] - state.qp.pos[0, 0]) / self.sys.config.dt
    forward_reward = self._forward_reward_weight * x_velocity

    min_z, max_z = self._healthy_z_range
    min_angle, max_angle = self._healthy_angle_range
    ang_y = math.quat_to_euler(qp.rot[0])[1]
    is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)  # pytype: disable=wrong-arg-types  # jax-ndarray
    is_healthy = jp.where(ang_y > max_angle, x=0.0, y=is_healthy)  # pytype: disable=wrong-arg-types  # jax-ndarray
    is_healthy = jp.where(ang_y < min_angle, x=0.0, y=is_healthy)  # pytype: disable=wrong-arg-types  # jax-ndarray
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(qp)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_ctrl=-ctrl_cost,
        reward_healthy=healthy_reward,
        x_position=qp.pos[0, 0],
        x_velocity=x_velocity)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP) -> jp.ndarray:
    """Returns the environment observations."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)
    ang_y = math.quat_to_euler(qp.rot[0])[1:2]

    # qpos: position and orientation of the torso and the joint angles
    if self._exclude_current_positions_from_observation:
      qpos = [qp.pos[0, 2:], ang_y, joint_angle]
    else:
      qpos = [qp.pos[0, (0, 2)], ang_y, joint_angle]

    # qvel: velocity of the torso and the joint angle velocities
    qvel = [qp.vel[0, (0, 2)], qp.ang[0, 1:2], joint_vel]

    return jp.concatenate(qpos + qvel)

  def _noise(self, rng):
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)


_SYSTEM_CONFIG = """
  bodies {
    name: "torso"
    colliders {
      capsule {
        radius: 0.05
        length: 0.5
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 3.6651914
  }
  bodies {
    name: "thigh"
    colliders {
      position { z: -0.225 }
      rotation {}
      capsule {
        radius: 0.05
        length: 0.55
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 4.0578904
  }
  bodies {
    name: "leg"
    colliders {
      position {}
      rotation {}
      capsule {
        radius: 0.04
        length: 0.58
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 2.7813568
  }
  bodies {
    name: "foot"
    colliders {
      position {
        x: -0.065
        y: -0.13
        z: -0.03
      }
      rotation { y: 90.0 }
      capsule {
        radius: 0.06
        length: 0.51
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 5.3155746
  }
  bodies {
    name: "floor"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    frozen { all: true }
  }
  joints {
    name: "thigh_joint"
    parent: "torso"
    child: "thigh"
    parent_offset { z: -0.2 }
    rotation { z: -90.0 }
    angle_limit { min: -150.0 }
  }
  joints {
    name: "leg_joint"
    parent: "thigh"
    child: "leg"
    parent_offset { z: -0.45 }
    child_offset { z: 0.25 }
    rotation { z: -90.0 }
    angle_limit { min: -150.0 }
  }
  joints {
    name: "foot_joint"
    parent: "leg"
    child: "foot"
    parent_offset { z: -0.25 }
    child_offset {
      x: -0.13
      y: -0.13
      z: -0.03
    }
    rotation { z: -90.0 }
    angle_limit { min: -45.0 max: 45.0 }
  }
  actuators {
    name: "thigh_joint"
    joint: "thigh_joint"
    strength: 200.0
    torque {}
  }
  actuators {
    name: "leg_joint"
    joint: "leg_joint"
    strength: 200.0
    torque {}
  }
  actuators {
    name: "foot_joint"
    joint: "foot_joint"
    strength: 200.0
    torque {}
  }
  friction: 0.94868329805
  gravity { z: -9.81 }
  velocity_damping: 1.0
  angular_damping: -0.05
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "thigh"
  }
  collide_include {
    first: "floor"
    second: "leg"
  }
  collide_include {
    first: "floor"
    second: "foot"
  }
  dt: 0.008
  substeps: 4
  frozen {
    position { y: 1.0 }
    rotation { x: 1.0 z: 1.0 }
  }
  defaults {
    qps { name: "torso" pos { z: 1.25 } }
    angles { name: "thigh_joint" angle {} }
    angles { name: "leg_joint" angle {} }
  }
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "torso"
    colliders {
      capsule {
        radius: 0.05
        length: 0.5
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 3.6651914
  }
  bodies {
    name: "thigh"
    colliders {
      position { z: -0.225 }
      rotation {}
      capsule {
        radius: 0.05
        length: 0.55
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 4.0578904
  }
  bodies {
    name: "leg"
    colliders {
      position {}
      rotation {}
      capsule {
        radius: 0.04
        length: 0.58
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 2.7813568
  }
  bodies {
    name: "foot"
    colliders {
      position {
        x: -0.065
        y: -0.13
        z: -0.03
      }
      rotation { y: 90.0 }
      capsule {
        radius: 0.06
        length: 0.51
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 5.3155746
  }
  bodies {
    name: "floor"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    frozen { all: true }
  }
  joints {
    name: "thigh_joint"
    stiffness: 10000.0
    parent: "torso"
    child: "thigh"
    parent_offset { z: -0.2 }
    rotation { z: -90.0 }
    angle_limit { min: -150.0 }
    angular_damping: 20.0
  }
  joints {
    name: "leg_joint"
    stiffness: 10000.0
    parent: "thigh"
    child: "leg"
    parent_offset { z: -0.45 }
    child_offset { z: 0.25 }
    rotation { z: -90.0 }
    angle_limit { min: -150.0 }
    angular_damping: 20.0
  }
  joints {
    name: "foot_joint"
    stiffness: 10000.0
    parent: "leg"
    child: "foot"
    parent_offset { z: -0.25 }
    child_offset {
      x: -0.13
      y: -0.13
      z: -0.03
    }
    rotation { z: -90.0 }
    angle_limit { min: -45.0 max: 45.0 }
    angular_damping: 20.0
  }
  actuators {
    name: "thigh_joint"
    joint: "thigh_joint"
    strength: 200.0
    torque {}
  }
  actuators {
    name: "leg_joint"
    joint: "leg_joint"
    strength: 200.0
    torque {}
  }
  actuators {
    name: "foot_joint"
    joint: "foot_joint"
    strength: 200.0
    torque {}
  }
  friction: 0.94868329805
  gravity { z: -9.81 }
  velocity_damping: 1.0
  angular_damping: -0.05
  baumgarte_erp: 0.1
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "thigh"
  }
  collide_include {
    first: "floor"
    second: "leg"
  }
  collide_include {
    first: "floor"
    second: "foot"
  }
  dt: 0.008
  substeps: 2
  frozen {
    position { y: 1.0 }
    rotation { x: 1.0 z: 1.0 }
  }
  defaults {
    qps { name: "torso" pos { z: 1.25 } }
    angles { name: "thigh_joint" angle {} }
    angles { name: "leg_joint" angle {} }
  }
  dynamics_mode: "legacy_spring"
  """
