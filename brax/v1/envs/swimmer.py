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

"""Trains a swimmer to swim in the +x direction."""

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1.envs import env


class Swimmer(env.Env):



  """
  ### Description

  This environment corresponds to the Swimmer environment described in Rémi
  Coulom's PhD thesis
  ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).

  The environment aims to increase the number of independent state and control
  variables as compared to the classic control environments. The swimmers
  consist of three or more segments ('***links***') and one less articulation
  joints ('***rotors***') - one rotor joint connecting exactly two links to
  form a linear chain.

  The swimmer is suspended in a two dimensional pool and always starts in the
  same position (subject to some deviation drawn from an uniform distribution),
  and the goal is to move as fast as possible towards the right by applying
  torque on the rotors and using the fluids friction.

  ### Notes

  The problem parameters are:

  * *n*: number of body parts
  * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
  * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
  * *k*: viscous-friction coefficient

  While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1,
  and *k* = 0.1. It is possible to tweak the MuJoCo XML files to increase the
  number of links, or to tweak any of the parameters.

  ### Action Space

  The agent take a 2-element vector for actions. The action space is a
  continuous `(action, action)` in `[-1, 1]`, where `action` represents the
  numerical torques applied between *links*

  | Num | Action                             | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the first rotor  | -1          | 1           | rot2                           | hinge | torque (N m) |
  | 1   | Torque applied on the second rotor | -1          | 1           | rot3                           | hinge | torque (N m) |

  ### Observation Space

  The state space consists of:

  * A<sub>0</sub>: position of first point
  * θ<sub>i</sub>: angle of part *i* with respect to the *x* axis
  * A<sub>0</sub>, θ<sub>i</sub>: their derivatives with respect to time (velocity and angular velocity)

  The observation is a `ndarray` with shape `(12,)` where the elements correspond to the following:

  | Num | Observation                          | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|---------------------------------------|------|-----|-------------------------------|-------|--------------------------|
  | 0   | angle of the front tip                | -Inf | Inf | rot                           | hinge | angle (rad)              |
  | 1   | angle of the second rotor             | -Inf | Inf | rot2                          | hinge | angle (rad)              |
  | 2   | angle of the second rotor             | -Inf | Inf | rot3                          | hinge | angle (rad)              |
  | 3   | velocity of the tip along the x-axis  | -Inf | Inf | slider1                       | slide | velocity (m/s)           |
  | 4   | velocity of the tip along the y-axis  | -Inf | Inf | slider2                       | slide | velocity (m/s)           |
  | 5   | velocity of the mid along the x-axis  | -Inf | Inf | N/A                           | slide | velocity (m/s)           |
  | 6   | velocity of the back along the y-axis | -Inf | Inf | N/A                           | slide | velocity (m/s)           |
  | 7   | velocity of the back along the x-axis | -Inf | Inf | N/A                           | slide | velocity (m/s)           |
  | 8   | velocity of the tip along the y-axis  | -Inf | Inf | N/A                           | slide | velocity (m/s)           |
  | 9   | angular velocity of front tip         | -Inf | Inf | rot                           | hinge | angular velocity (rad/s) |
  | 10  | angular velocity of second rotor      | -Inf | Inf | rot2                          | hinge | angular velocity (rad/s) |
  | 11  | angular velocity of third rotor       | -Inf | Inf | rot3                          | hinge | angular velocity (rad/s) |

  ### Rewards

  The reward consists of two parts:

  - *reward_fwd*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action) / dt*. *dt* is the
    time between actions - the default *dt = 0.04*. This reward would be positive
    if the swimmer swims right as desired.
  - *reward_control*: A negative reward for penalising the swimmer if it takes
    actions that are too large. It is measured as *-coefficient x
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.0001

  ### Starting State

  All observations start in state (0,0,0,0,0,0,0,0) with a Uniform noise in the
  range of [-0.1, 0.1] is added to the initial state for stochasticity.

  ### Episode Termination

  The episode terminates when the episode length is greater than 1000.

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder
  (or by changing the path to a modified XML file in another folder).

  ```
  gym.make('Swimmer-v2')
  ```

  v3 and v4 take gym.make kwargs such as ctrl_cost_weight, reset_noise_scale
  etc.

  ```
  env = gym.make('Swimmer-v4', ctrl_cost_weight=0.1, ....)
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('Swimmer-v5')
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v3: support for gym.make kwargs such as ctrl_cost_weight, reset_noise_scale
    etc. rgb rendering comes from tracking camera (so agent does not run away
    from screen)
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks. Added
    reward_threshold to environments.
  * v0: Initial versions release (1.0.0)
  """


  def __init__(self,
               forward_reward_weight=1.0,
               ctrl_cost_weight=1e-4,
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True,
               legacy_reward=False,
               legacy_spring=False,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation)

    # these parameters were derived from the mujoco swimmer:
    viscosity = 0.1
    density = 10.0
    inertia = (
        0.17278759594743870,
        3.5709436495803999,
        3.5709436495803999,
    )
    body_mass = 34.557519189487735

    # convert inertia to box
    inertia = jp.array([
        inertia[1] + inertia[2] - inertia[0],
        inertia[0] + inertia[1] - inertia[2],
        inertia[0] + inertia[2] - inertia[1],
    ])
    inertia = jp.sqrt(inertia / (body_mass * 6))

    # spherical drag
    self._spherical_drag = -3 * jp.pi * jp.mean(inertia) * viscosity

    # corrections to spherical drag force due to shape of capsules
    self._fix_drag = 0.5 * density * jp.array([
        inertia[1] * inertia[2],
        inertia[0] * inertia[2],
        inertia[0] * inertia[1]
    ])

  def reset(self, rng: jp.ndarray) -> env.State:
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        "reward_fwd": zero,
        "reward_ctrl": zero,
        "x_position": zero,
        "y_position": zero,
        "distance_from_origin": zero,
        "x_velocity": zero,
        "y_velocity": zero,
        "forward_reward": zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    force = self._get_viscous_force(state.qp)
    act = jp.concatenate([action, force.reshape(-1)], axis=0)
    qp, info = self.sys.step(state.qp, act)

    com_before = self._center_of_mass(state.qp)
    com_after = self._center_of_mass(qp)
    velocity = (com_after - com_before) / self.sys.config.dt
    forward_reward = self._forward_reward_weight * velocity[0]
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(qp, info)
    reward = forward_reward - ctrl_cost
    state.metrics.update(
        reward_fwd=forward_reward,
        reward_ctrl=-ctrl_cost,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.norm(qp.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )

    return state.replace(qp=qp, obs=obs, reward=reward)

  @property
  def action_size(self):
    return 2

  def _get_viscous_force(self, qp):
    """Calculate viscous force to apply to each body."""
    # ignore the floor
    qp = jp.take(qp, jp.arange(0, qp.vel.shape[0] - 1))

    # spherical drag force
    force = qp.vel * self._spherical_drag

    # corrections to spherical drag force due to shape of capsules
    vel = jp.vmap(math.rotate)(qp.vel, math.quat_inv(qp.rot))
    force -= jp.diag(self._fix_drag * jp.abs(vel) * vel)
    force = jp.vmap(math.rotate)(force, qp.rot)
    force = jp.clip(force, -5., 5.)  # pytype: disable=wrong-arg-types  # jax-ndarray

    return force

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe swimmer body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    # convert quaternion to rotation angle about z axis
    ang_z = math.quat_to_euler(qp.rot[0])[2:3]

    # qpos: position and orientation of the torso and the joint angles
    if self._exclude_current_positions_from_observation:
      qpos = [ang_z, joint_angle]
    else:
      qpos = [qp.pos[0, :2], ang_z, joint_angle]

    # # qvel: velocity of the bodies and the joint angle velocities
    qvel = [qp.vel[0, :2].ravel(), qp.ang[0, 2:], joint_vel]

    return jp.concatenate(qpos + qvel)

  def _center_of_mass(self, qp):
    """Returns the center of mass position of a swimmer with state qp."""
    mass, pos = self.sys.body.mass[:-1], qp.pos[:-1]
    return jp.sum(jp.vmap(jp.multiply)(mass, pos), axis=0) / jp.sum(mass)

  def _noise(self, rng):
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)


_SYSTEM_CONFIG = """
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "mid"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "back"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "floor"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  joints {
    name: "rot2"
    parent: "torso"
    child: "mid"
    parent_offset {
      x: 0.5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  joints {
    name: "rot3"
    parent: "mid"
    child: "back"
    parent_offset {
      x: .5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  actuators {
    name: "rot2"
    joint: "rot2"
    strength: 30.0
    torque {
    }
  }
  actuators {
    name: "rot3"
    joint: "rot3"
    strength: 30.0
    torque {
    }
  }
  forces {
    name: "torso_viscosity_thruster"
    body: "torso"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "mid_viscosity_thruster"
    body: "mid"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "back_viscosity_thruster"
    body: "back"
    strength: 1.0
    thruster {}
  }
  frozen {
    rotation { x: 1.0 y: 1.0 }
  }
  friction: 0.6
  angular_damping: -0.05
  collide_include { }
  dt: 0.04
  substeps: 12
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "floor"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "mid"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "back"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  joints {
    name: "rot2"
    stiffness: 10000.0
    parent: "torso"
    child: "mid"
    parent_offset {
      x: 0.5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  joints {
    name: "rot3"
    stiffness: 10000.0
    parent: "mid"
    child: "back"
    parent_offset {
      x: .5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  actuators {
    name: "rot2"
    joint: "rot2"
    strength: 30.0
    torque {
    }
  }
  actuators {
    name: "rot3"
    joint: "rot3"
    strength: 30.0
    torque {
    }
  }
  forces {
    name: "torso_viscosity_thruster"
    body: "torso"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "mid_viscosity_thruster"
    body: "mid"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "back_viscosity_thruster"
    body: "back"
    strength: 1.0
    thruster {}
  }
  frozen {
    rotation { x: 1.0 y: 1.0 }
  }
  friction: 0.6
  angular_damping: -0.05
  baumgarte_erp: 0.1
  collide_include { }
  dt: 0.04
  substeps: 12
  dynamics_mode: "legacy_spring"
  """
