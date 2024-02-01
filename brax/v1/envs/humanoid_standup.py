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

"""Trains a humanoid to stand up."""

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class HumanoidStandup(env.Env):



  """
  ### Description

  This environment is based on the environment introduced by Tassa, Erez and
  Todorov in
  ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).

  The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen)
  with a pair of legs and arms. The legs each consist of two links, and so the
  arms (representing the knees and elbows respectively).

  The environment starts with the humanoid laying on the ground, and then the
  goal of the environment is to make the humanoid standup and then keep it
  standing by applying torques on the various hinges.

  ### Action Space

  The agent take a 17-element vector for actions. The action space is a
  continuous `(action, ...)` all in `[-1, 1]`, where `action` represents the
  numerical torques applied at the hinge joints.

  | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
  |-----|------------------------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
  | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -1.0        | 1.0         | abdomen_yz                       | hinge | torque (N m) |
  | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -1.0        | 1.0         | abdomen_yz                       | hinge | torque (N m) |
  | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -1.0        | 1.0         | abdomen_x                        | hinge | torque (N m) |
  | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -1.0        | 1.0         | right_hip_xyz (right_thigh)      | hinge | torque (N m) |
  | 4   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -1.0        | 1.0         | right_hip_xyz (right_thigh)      | hinge | torque (N m) |
  | 5   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -1.0        | 1.0         | right_hip_xyz (right_thigh)      | hinge | torque (N m) |
  | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -1.0        | 1.0         | right_knee                       | hinge | torque (N m) |
  | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -1.0        | 1.0         | left_hip_xyz (left_thigh)        | hinge | torque (N m) |
  | 8   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -1.0        | 1.0         | left_hip_xyz (left_thigh)        | hinge | torque (N m) |
  | 9   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -1.0        | 1.0         | left_hip_xyz (left_thigh)        | hinge | torque (N m) |
  | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -1.0        | 1.0         | left_knee                        | hinge | torque (N m) |
  | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -1.0        | 1.0         | right_shoulder12                 | hinge | torque (N m) |
  | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -1.0        | 1.0         | right_shoulder12                 | hinge | torque (N m) |
  | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -1.0        | 1.0         | right_elbow                      | hinge | torque (N m) |
  | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -1.0        | 1.0         | left_shoulder12                  | hinge | torque (N m) |
  | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -1.0        | 1.0         | left_shoulder12                  | hinge | torque (N m) |
  | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -1.0        | 1.0         | left_elbow                       | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  Humanoid, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(376,)` where the elements
  correspond to the following:

  | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
  |-----|-----------------------------------------------------------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)             |
  | 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
  | 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
  | 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
  | 4   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
  | 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_yz                       | hinge | angle (rad)              |
  | 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_yy                       | hinge | angle (rad)              |
  | 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)              |
  | 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_xyz                    | hinge | angle (rad)              |
  | 9   | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_xyz                    | hinge | angle (rad)              |
  | 10  | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_xyz                    | hinge | angle (rad)              |
  | 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)              |
  | 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_xyz                     | hinge | angle (rad)              |
  | 13  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_xyz                     | hinge | angle (rad)              |
  | 14  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_xyz                     | hinge | angle (rad)              |
  | 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)              |
  | 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder12                 | hinge | angle (rad)              |
  | 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder12                 | hinge | angle (rad)              |
  | 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)              |
  | 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder12                  | hinge | angle (rad)              |
  | 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder12                  | hinge | angle (rad)              |
  | 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)              |
  | 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)           |
  | 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)           |
  | 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)           |
  | 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s) |
  | 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s) |
  | 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s) |
  | 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | angular velocity (rad/s) |
  | 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | angular velocity (rad/s) |
  | 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | angular velocity (rad/s) |
  | 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_xyz                    | hinge | angular velocity (rad/s) |
  | 32  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | angular velocity (rad/s) |
  | 33  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | angular velocity (rad/s) |
  | 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | angular velocity (rad/s) |
  | 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_xyz                     | hinge | angular velocity (rad/s) |
  | 36  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | angular velocity (rad/s) |
  | 37  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | angular velocity (rad/s) |
  | 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | angular velocity (rad/s) |
  | 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder12                 | hinge | angular velocity (rad/s) |
  | 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder12                 | hinge | angular velocity (rad/s) |
  | 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | angular velocity (rad/s) |
  | 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder12                  | hinge | angular velocity (rad/s) |
  | 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder12                  | hinge | angular velocity (rad/s) |
  | 44  | angular velocity of the angle between left upper arm and left_lower_arm                                         | -Inf | Inf | left_elbow                       | hinge | angular velocity (rad/s) |

  Additionally, after all the positional and velocity based values in the table,
  the state_space consists of (in order):

  - *cinert:* Mass and inertia of a single rigid body relative to the center of
    mass (this is an intermediate result of transition). It has shape 14*10
    (*nbody * 10*) and hence adds to another 140 elements in the state space.
  - *cvel:* Center of mass based velocity. It has shape 14 * 6 (*nbody * 6*) and
    hence adds another 84 elements in the state space
  - *qfrc_actuator:* Constraint force generated as the actuator force. This has
    shape `(23,)`  *(nv * 1)* and hence adds another 23 elements to the state
    space.

  The (x,y,z) coordinates are translational DOFs while the orientations are
  rotational DOFs expressed as quaternions.

  ### Rewards

  The reward consists of three parts:

  - *reward_linup*: A reward for moving upward (in an attempt to stand up). This
    is not a relative reward which measures how much upward it has moved from
    the last timestep, but it is an absolute reward which measures how much
    upward the Humanoid has moved overall. It is measured as *(z coordinate
    after action - 0) / (atomic timestep)*, where *z coordinate after action* is
    index 0 in the state, and *atomic timestep* is the time for one frame of
    movement.
  - *reward_quadctrl*: A negative reward for penalising the humanoid if it has
    too large of a control force. If there are *nu* actuators/controls, then the
    control has shape  `nu x 1`. It is measured as *0.1 **x**
    sum(control<sup>2</sup>)*.

  ### Starting State

  All observations start in state (0.0, 0.0,  0.105, 1.0, 0.0  ... 0.0) with a
  uniform noise in the range of [-0.01, 0.01] added to the positional and
  velocity values (values in the table) for stochasticity.

  Note that the initial z coordinate is intentionally selected to be low,
  thereby indicating a laying down humanoid. The initial orientation is designed
  to make it face forward as well.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder).

  ```
  env = gym.make('HumanoidStandup-v2')
  ```

  There is no v3 for HumanoidStandup, unlike the robot environments where a v3
  take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

  There is a v4 version that uses the mujoco-bindings

  ```
  env = gym.make('HumanoidStandup-v4')
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('HumanoidStandup-v5')
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
               legacy_spring=False,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)

    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp, self.sys.info(qp), jp.zeros(self.action_size))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_linup': zero,
        'reward_quadctrl': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)

    # ignore the floor at last index.
    pos_after = qp.pos[0, 2]  # z coordinate of torso
    uph_cost = (pos_after - 0) / self.sys.config.dt
    quad_ctrl_cost = .01 * jp.sum(jp.square(action))

    obs = self._get_obs(qp, info, action)
    reward = uph_cost + 1 - quad_ctrl_cost
    state.metrics.update(
        reward_linup=uph_cost,
        reward_quadctrl=-quad_ctrl_cost)

    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info,
               action: jp.ndarray) -> jp.ndarray:
    """Observe humanoid body position, velocities, and angles."""
    angle_vels = [j.angle_vel(qp) for j in self.sys.joints]

    # qpos: position and orientation of the torso and the joint angles.
    joint_angles = [angle for angle, _ in angle_vels]
    qpos = [qp.pos[0, 2:], qp.rot[0]] + joint_angles

    # qvel: velocity of the torso and the joint angle velocities.
    joint_velocities = [vel for _, vel in angle_vels]
    qvel = [qp.vel[0], qp.ang[0]] + joint_velocities

    # center of mass obs:
    com = self._center_of_mass(qp)
    mass_sum = jp.sum(self.sys.body.mass[:-1])

    def com_vals(body, qp):
      d = qp.pos - com
      com_inr = body.mass * jp.eye(3) * jp.norm(d) ** 2
      com_inr += jp.diag(body.inertia) - jp.outer(d, d)
      com_vel = body.mass * qp.vel / mass_sum
      com_ang = jp.cross(d, qp.vel) / (1e-7 + jp.norm(d) ** 2)

      return com_inr, com_vel, com_ang

    com_inr, com_vel, com_ang = jp.vmap(com_vals)(self.sys.body, qp)
    cinert = [com_inr[:-1].ravel()]
    cvel = [com_vel[:-1].ravel(), com_ang[:-1].ravel()]

    # actuator forces
    qfrc_actuator = []
    for act in self.sys.actuators:
      torque = jp.take(action, act.act_index)
      torque = torque.reshape(torque.shape[:-2] + (-1,))
      torque *= jp.repeat(act.strength, act.act_index.shape[-1])
      qfrc_actuator.append(torque)

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # can be calculated in brax like so:
    # cfrc = [
    #     jp.clip(info.contact.vel, -1, 1),
    #     jp.clip(info.contact.ang, -1, 1)
    # ]
    # flatten bottom dimension
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
    # then add it to the jp.concatenate below

    return jp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator)

  def _center_of_mass(self, qp):
    mass, pos = self.sys.body.mass[:-1], qp.pos[:-1]
    return jp.sum(jp.vmap(jp.multiply)(mass, pos), axis=0) / jp.sum(mass)

  def _noise(self, rng):
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), -0.01, 0.01)


_SYSTEM_CONFIG = """
  bodies {
    name: "torso"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.07
        length: 0.28
      }
    }
    colliders {
      position {
        z: 0.19
      }
      capsule {
        radius: 0.09
        length: 0.18
      }
    }
    colliders {
      position {
        x: -0.01
        z: -0.12
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 8.907463
  }
  bodies {
    name: "lwaist"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 2.2619467
  }
  bodies {
    name: "pelvis"
    colliders {
      position {
        x: -0.02
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.09
        length: 0.32
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 6.6161942
  }
  bodies {
    name: "right_thigh"
    colliders {
      position {
        y: 0.005
        z: -0.17
      }
      rotation {
        x: -178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "right_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "left_thigh"
    colliders {
      position {
        y: -0.005
        z: -0.17
      }
      rotation {
        x: 178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "left_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "right_upper_arm"
    colliders {
      position {
        x: 0.08
        y: -0.08
        z: -0.08
      }
      rotation {
        x: 135.0
        y: 35.26439
        z: -75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "right_lower_arm"
    colliders {
      position {
        x: 0.09
        y: 0.09
        z: 0.09
      }
      rotation {
        x: -45.0
        y: 35.26439
        z: 15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: 0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "left_upper_arm"
    colliders {
      position {
        x: 0.08
        y: 0.08
        z: -0.08
      }
      rotation {
        x: -135.0
        y: 35.26439
        z: 75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "left_lower_arm"
    colliders {
      position {
        x: 0.09
        y: -0.09
        z: 0.09
      }
      rotation {
        x: 45.0
        y: 35.26439
        z: -15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: -0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "floor"
    colliders {
      plane {
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
    name: "abdomen_yz"
    parent: "torso"
    child: "lwaist"
    parent_offset {
      x: -0.01
      z: -0.195
    }
    child_offset {
      z: 0.065
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -45.0
      max: 45.0
    }
    angle_limit {
      min: -65.0
      max: 30.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "abdomen_x"
    parent: "lwaist"
    child: "pelvis"
    parent_offset {
      z: -0.065
    }
    child_offset {
      z: 0.1
    }
    rotation {
      x: 90.0
    }
    angle_limit {
      min: -35.0
      max: 35.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_hip_xyz"
    parent: "pelvis"
    child: "right_thigh"
    parent_offset {
      y: -0.1
      z: -0.04
    }
    child_offset {
    }
    rotation {
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_knee"
    parent: "right_thigh"
    child: "right_shin"
    parent_offset {
      y: 0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angle_limit {
      min: -160.0
      max: -2.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_hip_xyz"
    parent: "pelvis"
    child: "left_thigh"
    parent_offset {
      y: 0.1
      z: -0.04
    }
    child_offset {
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_knee"
    parent: "left_thigh"
    child: "left_shin"
    parent_offset {
      y: -0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angle_limit {
      min: -160.0
      max: -2.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_shoulder12"
    parent: "torso"
    child: "right_upper_arm"
    parent_offset {
      y: -0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 135.0
      y: 35.26439
    }
    angle_limit {
      min: -85.0
      max: 60.0
    }
    angle_limit {
      min: -70.0
      max: 50.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_elbow"
    parent: "right_upper_arm"
    child: "right_lower_arm"
    parent_offset {
      x: 0.18
      y: -0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 135.0
      z: 90.0
    }
    angle_limit {
      min: -90.0
      max: 50.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_shoulder12"
    parent: "torso"
    child: "left_upper_arm"
    parent_offset {
      y: 0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 45.0
      y: -35.26439
    }
    angle_limit {
      min: -60.0
      max: 85.0
    }
    angle_limit {
      min: -50.0
      max: 70.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_elbow"
    parent: "left_upper_arm"
    child: "left_lower_arm"
    parent_offset {
      x: 0.18
      y: 0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 45.0
      z: -90.0
    }
    angle_limit {
      min: -90.0
      max: 50.0
    }
    angular_damping: 30.0
  }
  actuators {
    name: "abdomen_yz"
    joint: "abdomen_yz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "abdomen_x"
    joint: "abdomen_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_hip_xyz"
    joint: "right_hip_xyz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_knee"
    joint: "right_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_hip_xyz"
    joint: "left_hip_xyz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_knee"
    joint: "left_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_shoulder12"
    joint: "right_shoulder12"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "right_elbow"
    joint: "right_elbow"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_shoulder12"
    joint: "left_shoulder12"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_elbow"
    joint: "left_elbow"
    strength: 100.0
    torque {
    }
  }
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "left_shin"
  }
  collide_include {
    first: "floor"
    second: "right_shin"
  }
  collide_include {
    first: "floor"
    second: "left_lower_arm"
  }
  collide_include {
    first: "floor"
    second: "right_lower_arm"
  }
  defaults {
    angles {
      name: "left_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_hip_xyz"
      angle { x: 0 y: 5 z: 0 }
    }
    angles {
      name: "left_hip_xyz"
      angle { x: 0 y: 5 z: 0 }
    }
    angles {
      name: "right_shoulder12"
      angle { x: 0 y: -45 z: 0 }
    }
    angles {
      name: "left_shoulder12"
      angle { x: 0 y: 45 z: 0 }
    }
    qps {
      name: "torso"
      pos {x: -0.58 y: 0.00 z: 0.09 }
      rot {x: 180 y: -90 z: 180 }
    }
    qps {
      name: "lwaist"
      pos {x: -0.32 y: 0.00 z: 0.10 }
      rot {x: 180 y: -72 z: 180 }
    }
    qps {
      name: "pelvis"
      pos {x: -0.17 y: 0.00 z: 0.15 }
      rot {x: 180 y: -72 z: 180 }
    }
    qps {
      name: "right_thigh"
      pos {x: -0.13 y: -0.10 z: 0.16 }
      rot {x: 180 y: -77 z: 180 }
    }
    qps {
      name: "right_shin"
      pos {x: 0.27 y: -0.09 z: 0.24 }
      rot {x: -0 y: -78 z: -0 }
    }
    qps {
      name: "left_thigh"
      pos {x: -0.13 y: 0.10 z: 0.16 }
      rot {x: 180 y: -77 z: 180 }
    }
    qps {
      name: "left_shin"
      pos {x: 0.27 y: 0.09 z: 0.24 }
      rot {x: -0 y: -78 z: -0 }
    }
    qps {
      name: "right_upper_arm"
      pos {x: -0.64 y: -0.17 z: 0.09 }
      rot {x: 16 y: -59 z: -16 }
    }
    qps {
      name: "right_lower_arm"
      pos {x: -0.42 y: -0.39 z: 0.04 }
      rot {x: 24 y: -45 z: -24 }
    }
    qps {
      name: "left_upper_arm"
      pos {x: -0.64 y: 0.17 z: 0.09 }
      rot {x: -16 y: -59 z: 16 }
    }
    qps {
      name: "left_lower_arm"
      pos {x: -0.42 y: 0.39 z: 0.04 }
      rot {x: -24 y: -45 z: 24 }
    }
    qps {
      name: "floor"
      pos {x: 0.00 y: 0.00 z: 0.00 }
      rot {x: 0 y: 0 z: 0 }
    }
  }
  friction: 1.0
  gravity {
    z: -9.81
  }
  angular_damping: -0.05
  dt: 0.015
  substeps: 8
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "torso"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.07
        length: 0.28
      }
    }
    colliders {
      position {
        z: 0.19
      }
      capsule {
        radius: 0.09
        length: 0.18
      }
    }
    colliders {
      position {
        x: -0.01
        z: -0.12
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 8.907463
  }
  bodies {
    name: "lwaist"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 2.2619467
  }
  bodies {
    name: "pelvis"
    colliders {
      position {
        x: -0.02
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.09
        length: 0.32
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 6.6161942
  }
  bodies {
    name: "right_thigh"
    colliders {
      position {
        y: 0.005
        z: -0.17
      }
      rotation {
        x: -178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "right_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "left_thigh"
    colliders {
      position {
        y: -0.005
        z: -0.17
      }
      rotation {
        x: 178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "left_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "right_upper_arm"
    colliders {
      position {
        x: 0.08
        y: -0.08
        z: -0.08
      }
      rotation {
        x: 135.0
        y: 35.26439
        z: -75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "right_lower_arm"
    colliders {
      position {
        x: 0.09
        y: 0.09
        z: 0.09
      }
      rotation {
        x: -45.0
        y: 35.26439
        z: 15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: 0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "left_upper_arm"
    colliders {
      position {
        x: 0.08
        y: 0.08
        z: -0.08
      }
      rotation {
        x: -135.0
        y: 35.26439
        z: 75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "left_lower_arm"
    colliders {
      position {
        x: 0.09
        y: -0.09
        z: 0.09
      }
      rotation {
        x: 45.0
        y: 35.26439
        z: -15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: -0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "floor"
    colliders {
      plane {
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
    name: "abdomen_yz"
    stiffness: 27000
    parent: "torso"
    child: "lwaist"
    parent_offset {
      x: -0.01
      z: -0.195
    }
    child_offset {
      z: 0.065
    }
    rotation {
      y: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -45.0
      max: 45.0
    }
    angle_limit {
      min: -75.0
      max: 30.0
    }
  }
  joints {
    name: "abdomen_x"
    stiffness: 27000
    parent: "lwaist"
    child: "pelvis"
    parent_offset {
      z: -0.065
    }
    child_offset {
      z: 0.1
    }
    rotation {
      x: 90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -35.0
      max: 35.0
    }
  }
  joints {
    name: "right_hip_xyz"
    stiffness: 27000
    parent: "pelvis"
    child: "right_thigh"
    parent_offset {
      y: -0.1
      z: -0.04
    }
    child_offset {
    }
    rotation {
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
  }
  joints {
    name: "right_knee"
    stiffness: 27000
    parent: "right_thigh"
    child: "right_shin"
    parent_offset {
      y: 0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -160.0
      max: -2.0
    }
  }
  joints {
    name: "left_hip_xyz"
    stiffness: 27000
    parent: "pelvis"
    child: "left_thigh"
    parent_offset {
      y: 0.1
      z: -0.04
    }
    child_offset {
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
  }
  joints {
    name: "left_knee"
    stiffness: 27000
    parent: "left_thigh"
    child: "left_shin"
    parent_offset {
      y: -0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -160.0
      max: -2.0
    }
  }
  joints {
    name: "right_shoulder12"
    stiffness: 27000
    parent: "torso"
    child: "right_upper_arm"
    parent_offset {
      y: -0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 135.0
      y: 35.26439
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -85.0
      max: 60.0
    }
    angle_limit {
      min: -85.0
      max: 60.0
    }
  }
  joints {
    name: "right_elbow"
    stiffness: 27000
    parent: "right_upper_arm"
    child: "right_lower_arm"
    parent_offset {
      x: 0.18
      y: -0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 135.0
      z: 90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -90.0
      max: 50.0
    }
  }
  joints {
    name: "left_shoulder12"
    stiffness: 27000
    parent: "torso"
    child: "left_upper_arm"
    parent_offset {
      y: 0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 45.0
      y: -35.26439
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -60.0
      max: 85.0
    }
    angle_limit {
      min: -60.0
      max: 85.0
    }
  }
  joints {
    name: "left_elbow"
    stiffness: 27000
    parent: "left_upper_arm"
    child: "left_lower_arm"
    parent_offset {
      x: 0.18
      y: 0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 45.0
      z: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -90.0
      max: 50.0
    }
  }
  actuators {
    name: "abdomen_yz"
    joint: "abdomen_yz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "abdomen_x"
    joint: "abdomen_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_hip_xyz"
    joint: "right_hip_xyz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_knee"
    joint: "right_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_hip_xyz"
    joint: "left_hip_xyz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_knee"
    joint: "left_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_shoulder12"
    joint: "right_shoulder12"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "right_elbow"
    joint: "right_elbow"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_shoulder12"
    joint: "left_shoulder12"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_elbow"
    joint: "left_elbow"
    strength: 100.0
    torque {
    }
  }
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "left_shin"
  }
  collide_include {
    first: "floor"
    second: "right_shin"
  }
  collide_include {
    first: "floor"
    second: "left_lower_arm"
  }
  collide_include {
    first: "floor"
    second: "right_lower_arm"
  }
  defaults {
    angles {
      name: "left_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_hip_xyz"
      angle { x: 0 y: 5 z: 0 }
    }
    angles {
      name: "left_hip_xyz"
      angle { x: 0 y: 5 z: 0 }
    }
    angles {
      name: "right_shoulder12"
      angle { x: 0 y: -45 z: 0 }
    }
    angles {
      name: "left_shoulder12"
      angle { x: 0 y: 45 z: 0 }
    }
    qps {
      name: "torso"
      pos {x: -0.58 y: 0.00 z: 0.09 }
      rot {x: 180 y: -90 z: 180 }
    }
    qps {
      name: "lwaist"
      pos {x: -0.32 y: 0.00 z: 0.10 }
      rot {x: 180 y: -72 z: 180 }
    }
    qps {
      name: "pelvis"
      pos {x: -0.17 y: 0.00 z: 0.15 }
      rot {x: 180 y: -72 z: 180 }
    }
    qps {
      name: "right_thigh"
      pos {x: -0.13 y: -0.10 z: 0.16 }
      rot {x: 180 y: -77 z: 180 }
    }
    qps {
      name: "right_shin"
      pos {x: 0.27 y: -0.09 z: 0.24 }
      rot {x: -0 y: -78 z: -0 }
    }
    qps {
      name: "left_thigh"
      pos {x: -0.13 y: 0.10 z: 0.16 }
      rot {x: 180 y: -77 z: 180 }
    }
    qps {
      name: "left_shin"
      pos {x: 0.27 y: 0.09 z: 0.24 }
      rot {x: -0 y: -78 z: -0 }
    }
    qps {
      name: "right_upper_arm"
      pos {x: -0.64 y: -0.17 z: 0.09 }
      rot {x: 16 y: -59 z: -16 }
    }
    qps {
      name: "right_lower_arm"
      pos {x: -0.42 y: -0.39 z: 0.04 }
      rot {x: 24 y: -45 z: -24 }
    }
    qps {
      name: "left_upper_arm"
      pos {x: -0.64 y: 0.17 z: 0.09 }
      rot {x: -16 y: -59 z: 16 }
    }
    qps {
      name: "left_lower_arm"
      pos {x: -0.42 y: 0.39 z: 0.04 }
      rot {x: -24 y: -45 z: 24 }
    }
    qps {
      name: "floor"
      pos {x: 0.00 y: 0.00 z: 0.00 }
      rot {x: 0 y: 0 z: 0 }
    }
  }
  friction: 1.0
  gravity {
    z: -9.81
  }
  angular_damping: -0.05
  baumgarte_erp: 0.1
  dt: 0.015
  substeps: 8
  dynamics_mode: "legacy_spring"
  """
