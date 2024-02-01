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

"""Trains an ant to run in the +x direction."""

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class Ant(env.Env):



  """
  ### Description

  This environment is based on the environment introduced by Schulman, Moritz,
  Levine, Jordan and Abbeel in
  ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).

  The ant is a 3D robot consisting of one torso (free rotational body) with four
  legs attached to it with each leg having two links.

  The goal is to coordinate the four legs to move in the forward (right)
  direction by applying torques on the eight hinges connecting the two links of
  each leg and the torso (nine parts and eight hinges).

  ### Action Space

  The agent take a 8-element vector for actions.

  The action space is a continuous `(action, action, action, action, action,
  action, action, action)` all in `[-1, 1]`, where `action` represents the
  numerical torques applied at the hinge joints.

  | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
  |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
  | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
  | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | ankle_1 (front_left_leg)         | hinge | torque (N m) |
  | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
  | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | ankle_2 (front_right_leg)        | hinge | torque (N m) |
  | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
  | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | ankle_3 (back_leg)               | hinge | torque (N m) |
  | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
  | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | ankle_4 (right_back_leg)         | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  ant, followed by the velocities of those individual parts (their derivatives)
  with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(27,)` where the elements correspond to the following:

  | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
  |-----|--------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
  | 1   | w-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 2   | x-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 3   | y-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 4   | z-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
  | 5   | angle between torso and first link on front left             | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 6   | angle between the two links on the front left                | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 7   | angle between torso and first link on front right            | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 8   | angle between the two links on the front right               | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 9   | angle between torso and first link on back left              | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 10  | angle between the two links on the back left                 | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 11  | angle between torso and first link on back right             | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 12  | angle between the two links on the back right                | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
  | 13  | x-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 14  | y-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 15  | z-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
  | 16  | x-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 17  | y-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 18  | z-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
  | 19  | angular velocity of angle between torso and front left link  | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
  | 20  | angular velocity of the angle between front left links       | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
  | 21  | angular velocity of angle between torso and front right link | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
  | 22  | angular velocity of the angle between front right links      | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
  | 23  | angular velocity of angle between torso and back left link   | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
  | 24  | angular velocity of the angle between back left links        | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
  | 25  | angular velocity of angle between torso and back right link  | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
  | 26  | angular velocity of the angle between back right links       | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |

  The (x,y,z) coordinates are translational DOFs while the orientations are
  rotational DOFs expressed as quaternions.

  If use_contact_forces=True, contact forces are added to the observation:
  external forces and torques applied to the center of mass of each of the
  links. 60 extra dimensions: the torso link, 8 leg links, and the ground link
  (10 total), with 6 external forces each (force x, y, z, torque x, y, z).

  ### Rewards

  The reward consists of three parts:

  - *reward_survive*: Every timestep that the ant is alive, it gets a reward of
    1.
  - *reward_forward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.05*. This reward would be
    positive if the ant moves forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the ant if it takes actions
    that are too large. It is measured as *coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.5.
  - *contact_cost*: A negative reward for penalising the ant if the external
    contact force is too large. It is calculated *0.5 * 0.001 *
    sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

  ### Starting State

  All observations start in state (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a
  uniform noise in the range of [-0.1, 0.1] added to the positional values and
  standard normal noise with 0 mean and 0.1 standard deviation added to the
  velocity values for stochasticity.

  Note that the initial z coordinate is intentionally selected to be slightly
  high, thereby indicating a standing up ant. The initial orientation is
  designed to make it face forward as well.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The y-orientation (index 2) in the state is **not** in the range
     `[0.2, 1.0]`

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder).

  ```
  env = gym.make('Ant-v2')
  ```

  v3, v4, and v5 take gym.make kwargs such as ctrl_cost_weight,
  reset_noise_scale etc.

  ```
  env = gym.make('Ant-v5', ctrl_cost_weight=0.1, ....)
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
               ctrl_cost_weight=0.5,
               use_contact_forces=False,
               contact_cost_weight=5e-4,
               healthy_reward=1.0,
               terminate_when_unhealthy=True,
               healthy_z_range=(0.2, 1.0),
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True,
               legacy_spring=False,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
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
    obs = self._get_obs(qp, self.sys.info(qp))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_survive': zero,
        'reward_ctrl': zero,
        'reward_contact': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)

    velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
    forward_reward = velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)  # pytype: disable=wrong-arg-types  # jax-ndarray
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = (self._contact_cost_weight *
                    jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))  # pytype: disable=wrong-arg-types  # jax-ndarray
    obs = self._get_obs(qp, info)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=qp.pos[0, 0],
        y_position=qp.pos[0, 1],
        distance_from_origin=jp.norm(qp.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe ant body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    # qpos: position and orientation of the torso and the joint angles.
    if self._exclude_current_positions_from_observation:
      qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]
    else:
      qpos = [qp.pos[0], qp.rot[0], joint_angle]

    # qvel: velocity of the torso and the joint angle velocities.
    qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    if self._use_contact_forces:
      cfrc = [
          jp.clip(info.contact.vel, -1, 1),  # pytype: disable=wrong-arg-types  # jax-ndarray
          jp.clip(info.contact.ang, -1, 1)  # pytype: disable=wrong-arg-types  # jax-ndarray
      ]
      # flatten bottom dimension
      cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
    else:
      cfrc = []

    return jp.concatenate(qpos + qvel + cfrc)

  def _noise(self, rng):
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)

# TODO: name bodies in config according to mujoco xml

_SYSTEM_CONFIG = """
  bodies {
    name: "$ Torso"
    colliders {
      capsule {
        radius: 0.25
        length: 0.5
        end: 1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 10
  }
  bodies {
    name: "Aux 1"
    colliders {
      rotation { x: 90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 4"
    colliders {
      rotation { x: 90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 2"
    colliders {
      rotation { x: 90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 7"
    colliders {
      rotation { x: 90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 3"
    colliders {
      rotation { x: -90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 10"
    colliders {
      rotation { x: -90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 4"
    colliders {
      rotation { x: -90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 13"
    colliders {
      rotation { x: -90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Ground"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  joints {
    name: "hip_1"
    parent_offset { x: 0.2 y: 0.2 }
    child_offset { x: -0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 1"
    angle_limit { min: -30.0 max: 30.0 }
    rotation { y: -90 }
    angular_damping: 20
  }
  joints {
    name: "ankle_1"
    parent_offset { x: 0.1 y: 0.1 }
    child_offset { x: -0.2 y: -0.2 }
    parent: "Aux 1"
    child: "$ Body 4"
    rotation: { z: 135 }
    angle_limit {
      min: 30.0
      max: 70.0
    }
    angular_damping: 20
  }
  joints {
    name: "hip_2"
    parent_offset { x: -0.2 y: 0.2 }
    child_offset { x: 0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 2"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_2"
    parent_offset { x: -0.1 y: 0.1 }
    child_offset { x: 0.2 y: -0.2 }
    parent: "Aux 2"
    child: "$ Body 7"
    rotation { z: 45 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "hip_3"
    parent_offset { x: -0.2 y: -0.2 }
    child_offset { x: 0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 3"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_3"
    parent_offset { x: -0.1 y: -0.1 }
    child_offset {
      x: 0.2
      y: 0.2
    }
    parent: "Aux 3"
    child: "$ Body 10"
    rotation { z: 135 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "hip_4"
    parent_offset { x: 0.2 y: -0.2 }
    child_offset { x: -0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 4"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_4"
    parent_offset { x: 0.1 y: -0.1 }
    child_offset { x: -0.2 y: 0.2 }
    parent: "Aux 4"
    child: "$ Body 13"
    rotation { z: 45 }
    angle_limit { min: 30.0 max: 70.0 }
    angular_damping: 20
  }
  actuators {
    name: "hip_1"
    joint: "hip_1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_1"
    joint: "ankle_1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_2"
    joint: "hip_2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_2"
    joint: "ankle_2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_3"
    joint: "hip_3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_3"
    joint: "ankle_3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_4"
    joint: "hip_4"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_4"
    joint: "ankle_4"
    strength: 350.0
    torque {}
  }
  friction: 1.0
  gravity { z: -9.8 }
  angular_damping: -0.05
  collide_include {
    first: "$ Torso"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 4"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 7"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 10"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 13"
    second: "Ground"
  }
  dt: 0.05
  substeps: 10
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
bodies {
  name: "$ Torso"
  colliders {
    capsule {
      radius: 0.25
      length: 0.5
      end: 1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 10
}
bodies {
  name: "Aux 1"
  colliders {
    rotation { x: 90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 4"
  colliders {
    rotation { x: 90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 2"
  colliders {
    rotation { x: 90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 7"
  colliders {
    rotation { x: 90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 3"
  colliders {
    rotation { x: -90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 10"
  colliders {
    rotation { x: -90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 4"
  colliders {
    rotation { x: -90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 13"
  colliders {
    rotation { x: -90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Ground"
  colliders {
    plane {}
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
  frozen { all: true }
}
joints {
  name: "$ Torso_Aux 1"
  parent_offset { x: 0.2 y: 0.2 }
  child_offset { x: -0.1 y: -0.1 }
  parent: "$ Torso"
  child: "Aux 1"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  angle_limit { min: -30.0 max: 30.0 }
  rotation { y: -90 }
}
joints {
  name: "Aux 1_$ Body 4"
  parent_offset { x: 0.1 y: 0.1 }
  child_offset { x: -0.2 y: -0.2 }
  parent: "Aux 1"
  child: "$ Body 4"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation: { z: 135 }
  angle_limit {
    min: 30.0
    max: 70.0
  }
}
joints {
  name: "$ Torso_Aux 2"
  parent_offset { x: -0.2 y: 0.2 }
  child_offset { x: 0.1 y: -0.1 }
  parent: "$ Torso"
  child: "Aux 2"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 2_$ Body 7"
  parent_offset { x: -0.1 y: 0.1 }
  child_offset { x: 0.2 y: -0.2 }
  parent: "Aux 2"
  child: "$ Body 7"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { z: 45 }
  angle_limit { min: -70.0 max: -30.0 }
}
joints {
  name: "$ Torso_Aux 3"
  parent_offset { x: -0.2 y: -0.2 }
  child_offset { x: 0.1 y: 0.1 }
  parent: "$ Torso"
  child: "Aux 3"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 3_$ Body 10"
  parent_offset { x: -0.1 y: -0.1 }
  child_offset {
    x: 0.2
    y: 0.2
  }
  parent: "Aux 3"
  child: "$ Body 10"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { z: 135 }
  angle_limit { min: -70.0 max: -30.0 }
}
joints {
  name: "$ Torso_Aux 4"
  parent_offset { x: 0.2 y: -0.2 }
  child_offset { x: -0.1 y: 0.1 }
  parent: "$ Torso"
  child: "Aux 4"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 4_$ Body 13"
  parent_offset { x: 0.1 y: -0.1 }
  child_offset { x: -0.2 y: 0.2 }
  parent: "Aux 4"
  child: "$ Body 13"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { z: 45 }
  angle_limit { min: 30.0 max: 70.0 }
}
actuators {
  name: "$ Torso_Aux 1"
  joint: "$ Torso_Aux 1"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 1_$ Body 4"
  joint: "Aux 1_$ Body 4"
  strength: 350.0
  torque {}
}
actuators {
  name: "$ Torso_Aux 2"
  joint: "$ Torso_Aux 2"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 2_$ Body 7"
  joint: "Aux 2_$ Body 7"
  strength: 350.0
  torque {}
}
actuators {
  name: "$ Torso_Aux 3"
  joint: "$ Torso_Aux 3"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 3_$ Body 10"
  joint: "Aux 3_$ Body 10"
  strength: 350.0
  torque {}
}
actuators {
  name: "$ Torso_Aux 4"
  joint: "$ Torso_Aux 4"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 4_$ Body 13"
  joint: "Aux 4_$ Body 13"
  strength: 350.0
  torque {}
}
friction: 1.0
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "$ Torso"
  second: "Ground"
}
collide_include {
  first: "$ Body 4"
  second: "Ground"
}
collide_include {
  first: "$ Body 7"
  second: "Ground"
}
collide_include {
  first: "$ Body 10"
  second: "Ground"
}
collide_include {
  first: "$ Body 13"
  second: "Ground"
}
dt: 0.05
substeps: 10
dynamics_mode: "legacy_spring"
"""
