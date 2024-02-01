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

"""Trains a halfcheetah to run in the +x direction."""

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class Halfcheetah(env.Env):



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

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder
  (or by changing the path to a modified XML file in another folder).

  ```
  env = gym.make('HalfCheetah-v2')
  ```

  v3, v4, and v5 take gym.make kwargs such as ctrl_cost_weight,
  reset_noise_scale etc.

  ```
  env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, ....)

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight,
    reset_noise_scale etc. rgb rendering comes from tracking camera (so agent
    does not run away from screen)
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks. Added
    reward_threshold to environments.
  * v0: Initial versions release (1.0.0)
  """


  def __init__(self,
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.1,
               legacy_spring=False,
               exclude_current_positions_from_observation=True,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
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
        'x_position': zero,
        'x_velocity': zero,
        'reward_ctrl': zero,
        'reward_run': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)

    velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
    forward_reward = self._forward_reward_weight * velocity[0]
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(qp, info)
    reward = forward_reward - ctrl_cost
    state.metrics.update(
        x_position=qp.pos[0, 0],
        x_velocity=velocity[0],
        reward_run=forward_reward,
        reward_ctrl=-ctrl_cost)

    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe halfcheetah body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    # qpos: position and orientation of the torso and the joint angles
    # TODO: convert rot to just y-ang component
    if self._exclude_current_positions_from_observation:
      qpos = [qp.pos[0, 2:], qp.rot[0, (0, 2)], joint_angle]
    else:
      qpos = [qp.pos[0, (0, 2)], qp.rot[0, (0, 2)], joint_angle]

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
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 1.0920000076293945
      }
    }
    colliders {
      position {
        x: 0.6000000238418579
        z: 0.10000000149011612
      }
      rotation {
        y: 49.847328186035156
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.9447969794273376
      y: 0.9447969794273376
      z: 0.9447969794273376
    }
    mass: 9.457332611083984
  }
  bodies {
    name: "bthigh"
    colliders {
      position {
        x: 0.10000000149011612
        z: -0.12999999523162842
      }
      rotation {
        x: -180.0
        y: 37.723960876464844
        z: -180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.38199999928474426
      }
    }
    inertia {
      x: 0.029636280611157417
      y: 0.029636280611157417
      z: 0.029636280611157417
    }
    mass: 2.335526943206787
  }
  bodies {
    name: "bshin"
    colliders {
      position {
        x: -0.14000000059604645
        z: -0.07000000029802322
      }
      rotation {
        x: 180.0
        y: -63.68956756591797
        z: 180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.032029107213020325
      y: 0.032029107213020325
      z: 0.032029107213020325
    }
    mass: 2.402003049850464
  }
  bodies {
    name: "bfoot"
    colliders {
      position {
        x: 0.029999999329447746
        z: -0.09700000286102295
      }
      rotation {
        y: -15.469860076904297
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.2800000011920929
      }
    }
    inertia {
      x: 0.0117056118324399
      y: 0.0117056118324399
      z: 0.0117056118324399
    }
    mass: 1.6574708223342896
  }
  bodies {
    name: "fthigh"
    colliders {
      position {
        x: -0.07000000029802322
        z: -0.11999999731779099
      }
      rotation {
        y: 29.793806076049805
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3580000102519989
      }
    }
    inertia {
      x: 0.024391336366534233
      y: 0.024391336366534233
      z: 0.024391336366534233
    }
    mass: 2.1759843826293945
  }
  bodies {
    name: "fshin"
    colliders {
      position {
        x: 0.06499999761581421
        z: -0.09000000357627869
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.30399999022483826
      }
    }
    inertia {
      x: 0.014954624697566032
      y: 0.014954624697566032
      z: 0.014954624697566032
    }
    mass: 1.8170133829116821
  }
  bodies {
    name: "ffoot"
    colliders {
      position {
        x: 0.04500000178813934
        z: -0.07000000029802322
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.23199999332427979
      }
    }
    inertia {
      x: 0.006711110472679138
      y: 0.006711110472679138
      z: 0.006711110472679138
    }
    mass: 1.3383854627609253
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
    frozen {
      position { x: 1.0 y: 1.0 z: 1.0 }
      rotation { x: 1.0 y: 1.0 z: 1.0 }
    }
  }
  joints {
    name: "bthigh"
    parent: "torso"
    child: "bthigh"
    parent_offset {
      x: -0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -29.793806076049805
      max: 60.16056823730469
    }
    }
  joints {
    name: "bshin"
    parent: "bthigh"
    child: "bshin"
    parent_offset {
      x: 0.1599999964237213
      z: -0.25
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -44.97718811035156
      max: 44.97718811035156
    }
    }
  joints {
    name: "bfoot"
    parent: "bshin"
    child: "bfoot"
    parent_offset {
      x: -0.2800000011920929
      z: -0.14000000059604645
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -22.918312072753906
      max: 44.97718811035156
    }
    }
  joints {
    name: "fthigh"
    parent: "torso"
    child: "fthigh"
    parent_offset {
      x: 0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -57.295780181884766
      max: 40.1070442199707
    }
    }
  joints {
    name: "fshin"
    parent: "fthigh"
    child: "fshin"
    parent_offset {
      x: -0.14000000059604645
      z: -0.23999999463558197
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -68.75493621826172
      max: 49.847328186035156
    }
    }
  joints {
    name: "ffoot"
    parent: "fshin"
    child: "ffoot"
    parent_offset {
      x: 0.12999999523162842
      z: -0.18000000715255737
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -28.647890090942383
      max: 28.647890090942383
    }
    }
  actuators {
    name: "bthigh"
    joint: "bthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "bshin"
    joint: "bshin"
    strength: 90.0
    torque {
    }
  }
  actuators {
    name: "bfoot"
    joint: "bfoot"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "fthigh"
    joint: "fthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "fshin"
    joint: "fshin"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "ffoot"
    joint: "ffoot"
    strength: 30.0
    torque {
    }
  }
  friction: 0.77459666924
  gravity {
    z: -9.8100004196167
  }
  angular_damping: -0.009999999776482582
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "bfoot"
  }
  collide_include {
    first: "floor"
    second: "ffoot"
  }
  collide_include {
    first: "floor"
    second: "bthigh"
  }
  collide_include {
    first: "floor"
    second: "fthigh"
  }
  collide_include {
    first: "floor"
    second: "bshin"
  }
  collide_include {
    first: "floor"
    second: "fshin"
  }
  collide_include {
    first: "bfoot"
    second: "ffoot"
  }
  dt: 0.05
  substeps: 16
  frozen {
    position {
      y: 1.0
    }
    rotation {
      x: 1.0
      z: 1.0
    }
  }
  dynamics_mode: "pbd"
"""

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 1.0920000076293945
      }
    }
    colliders {
      position {
        x: 0.6000000238418579
        z: 0.10000000149011612
      }
      rotation {
        y: 49.847328186035156
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.9447969794273376
      y: 0.9447969794273376
      z: 0.9447969794273376
    }
    mass: 9.457332611083984
  }
  bodies {
    name: "bthigh"
    colliders {
      position {
        x: 0.10000000149011612
        z: -0.12999999523162842
      }
      rotation {
        x: -180.0
        y: 37.723960876464844
        z: -180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.38199999928474426
      }
    }
    inertia {
      x: 0.029636280611157417
      y: 0.029636280611157417
      z: 0.029636280611157417
    }
    mass: 2.335526943206787
  }
  bodies {
    name: "bshin"
    colliders {
      position {
        x: -0.14000000059604645
        z: -0.07000000029802322
      }
      rotation {
        x: 180.0
        y: -63.68956756591797
        z: 180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.032029107213020325
      y: 0.032029107213020325
      z: 0.032029107213020325
    }
    mass: 2.402003049850464
  }
  bodies {
    name: "bfoot"
    colliders {
      position {
        x: 0.029999999329447746
        z: -0.09700000286102295
      }
      rotation {
        y: -15.469860076904297
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.2800000011920929
      }
    }
    inertia {
      x: 0.0117056118324399
      y: 0.0117056118324399
      z: 0.0117056118324399
    }
    mass: 1.6574708223342896
  }
  bodies {
    name: "fthigh"
    colliders {
      position {
        x: -0.07000000029802322
        z: -0.11999999731779099
      }
      rotation {
        y: 29.793806076049805
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3580000102519989
      }
    }
    inertia {
      x: 0.024391336366534233
      y: 0.024391336366534233
      z: 0.024391336366534233
    }
    mass: 2.1759843826293945
  }
  bodies {
    name: "fshin"
    colliders {
      position {
        x: 0.06499999761581421
        z: -0.09000000357627869
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.30399999022483826
      }
    }
    inertia {
      x: 0.014954624697566032
      y: 0.014954624697566032
      z: 0.014954624697566032
    }
    mass: 1.8170133829116821
  }
  bodies {
    name: "ffoot"
    colliders {
      position {
        x: 0.04500000178813934
        z: -0.07000000029802322
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.23199999332427979
      }
    }
    inertia {
      x: 0.006711110472679138
      y: 0.006711110472679138
      z: 0.006711110472679138
    }
    mass: 1.3383854627609253
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
    frozen {
      position { x: 1.0 y: 1.0 z: 1.0 }
      rotation { x: 1.0 y: 1.0 z: 1.0 }
    }
  }
  joints {
    name: "bthigh"
    stiffness: 25000
    parent: "torso"
    child: "bthigh"
    parent_offset {
      x: -0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -29.793806076049805
      max: 60.16056823730469
    }
    limit_strength: 1000
    }
  joints {
    name: "bshin"
    stiffness: 25000
    parent: "bthigh"
    child: "bshin"
    parent_offset {
      x: 0.1599999964237213
      z: -0.25
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -44.97718811035156
      max: 44.97718811035156
    }
    limit_strength: 1000
    }
  joints {
    name: "bfoot"
    stiffness: 25000
    parent: "bshin"
    child: "bfoot"
    parent_offset {
      x: -0.2800000011920929
      z: -0.14000000059604645
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -22.918312072753906
      max: 44.97718811035156
    }
    limit_strength: 1000
    }
  joints {
    name: "fthigh"
    stiffness: 25000
    parent: "torso"
    child: "fthigh"
    parent_offset {
      x: 0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -57.295780181884766
      max: 40.1070442199707
    }
    limit_strength: 1000
    }
  joints {
    name: "fshin"
    stiffness: 25000
    parent: "fthigh"
    child: "fshin"
    parent_offset {
      x: -0.14000000059604645
      z: -0.23999999463558197
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 80.0
    angle_limit {
      min: -68.75493621826172
      max: 49.847328186035156
    }
    limit_strength: 1000
    }
  joints {
    name: "ffoot"
    stiffness: 25000
    parent: "fshin"
    child: "ffoot"
    parent_offset {
      x: 0.12999999523162842
      z: -0.18000000715255737
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 50.0
    angle_limit {
      min: -28.647890090942383
      max: 28.647890090942383
    }
    limit_strength: 1000
    }
  actuators {
    name: "bthigh"
    joint: "bthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "bshin"
    joint: "bshin"
    strength: 90.0
    torque {
    }
  }
  actuators {
    name: "bfoot"
    joint: "bfoot"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "fthigh"
    joint: "fthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "fshin"
    joint: "fshin"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "ffoot"
    joint: "ffoot"
    strength: 30.0
    torque {
    }
  }
  friction: 0.77459666924
  gravity {
    z: -9.8100004196167
  }
  angular_damping: -0.009999999776482582
  baumgarte_erp: 0.20000000149011612
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "bfoot"
  }
  collide_include {
    first: "floor"
    second: "ffoot"
  }
  collide_include {
    first: "floor"
    second: "bthigh"
  }
  collide_include {
    first: "floor"
    second: "fthigh"
  }
  collide_include {
    first: "floor"
    second: "bshin"
  }
  collide_include {
    first: "floor"
    second: "fshin"
  }
  collide_include {
    first: "bfoot"
    second: "ffoot"
  }
  dt: 0.05
  substeps: 16
  frozen {
    position {
      y: 1.0
    }
    rotation {
      x: 1.0
      z: 1.0
    }
  }
  dynamics_mode: "legacy_spring"
  """
