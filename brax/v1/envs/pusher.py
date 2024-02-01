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

"""Trains a robot arm to push a ball to a target."""

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class Pusher(env.Env):



  """
  ### Description

  "Pusher" is a multi-jointed robot arm which is very similar to that of a
  human.

  The goal is to move a target cylinder (called *object*) to a goal position
  using the robot's end effector (called *fingertip*). The robot consists of
  shoulder, elbow, forearm, and wrist joints.

  ### Action Space

  The action space is a `Box(-2, 2, (7,), float32)`. An action `(a, b)`
  represents the torques applied at the hinge joints.

  | Num | Action                                        | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|-----------------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Rotation of the panning the shoulder          | -1          | 1           | r_shoulder_pan_joint           | hinge | torque (N m) |
  | 1   | Rotation of the shoulder lifting joint        | -1          | 1           | r_shoulder_lift_joint          | hinge | torque (N m) |
  | 2   | Rotation of the shoulder rolling joint        | -1          | 1           | r_upper_arm_roll_joint         | hinge | torque (N m) |
  | 3   | Rotation of hinge joint that flexed the elbow | -1          | 1           | r_elbow_flex_joint             | hinge | torque (N m) |
  | 4   | Rotation of hinge that rolls the forearm      | -1          | 1           | r_forearm_roll_joint           | hinge | torque (N m) |
  | 5   | Rotation of flexing the wrist                 | -1          | 1           | r_wrist_flex_joint             | hinge | torque (N m) |
  | 6   | Rotation of rolling the wrist                 | -1          | 1           | r_wrist_roll_joint             | hinge | torque (N m) |

  ### Observation Space

  Observations consist of

  - Angle of rotational joints on the pusher
  - Angular velocities of rotational joints on the pusher
  - The coordinates of the fingertip of the pusher
  - The coordinates of the object to be moved
  - The coordinates of the goal position

  The observation is a `ndarray` with shape `(23,)` where the elements
  correspond to the table below. An analogy can be drawn to a human arm in order
  to help understand the state space, with the words flex and roll meaning the
  same as human joints.

  | Num | Observation                                              | Min  | Max | Name (in corresponding config) | Joint    | Unit                     |
  |-----|----------------------------------------------------------|------|-----|--------------------------------|----------|--------------------------|
  | 0   | Rotation of the panning the shoulder                     | -Inf | Inf | r_shoulder_pan_joint           | hinge    | angle (rad)              |
  | 1   | Rotation of the shoulder lifting joint                   | -Inf | Inf | r_shoulder_lift_joint          | hinge    | angle (rad)              |
  | 2   | Rotation of the shoulder rolling joint                   | -Inf | Inf | r_upper_arm_roll_joint         | hinge    | angle (rad)              |
  | 3   | Rotation of hinge joint that flexed the elbow            | -Inf | Inf | r_elbow_flex_joint             | hinge    | angle (rad)              |
  | 4   | Rotation of hinge that rolls the forearm                 | -Inf | Inf | r_forearm_roll_joint           | hinge    | angle (rad)              |
  | 5   | Rotation of flexing the wrist                            | -Inf | Inf | r_wrist_flex_joint             | hinge    | angle (rad)              |
  | 6   | Rotation of rolling the wrist                            | -Inf | Inf | r_wrist_roll_joint             | hinge    | angle (rad)              |
  | 7   | Rotational velocity of the panning the shoulder          | -Inf | Inf | r_shoulder_pan_joint           | hinge    | angular velocity (rad/s) |
  | 8   | Rotational velocity of the shoulder lifting joint        | -Inf | Inf | r_shoulder_lift_joint          | hinge    | angular velocity (rad/s) |
  | 9   | Rotational velocity of the shoulder rolling joint        | -Inf | Inf | r_upper_arm_roll_joint         | hinge    | angular velocity (rad/s) |
  | 10  | Rotational velocity of hinge joint that flexed the elbow | -Inf | Inf | r_elbow_flex_joint             | hinge    | angular velocity (rad/s) |
  | 11  | Rotational velocity of hinge that rolls the forearm      | -Inf | Inf | r_forearm_roll_joint           | hinge    | angular velocity (rad/s) |
  | 12  | Rotational velocity of flexing the wrist                 | -Inf | Inf | r_wrist_flex_joint             | hinge    | angular velocity (rad/s) |
  | 13  | Rotational velocity of rolling the wrist                 | -Inf | Inf | r_wrist_roll_joint             | hinge    | angular velocity (rad/s) |
  | 14  | x-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                       | slide    | position (m)             |
  | 15  | y-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                       | slide    | position (m)             |
  | 16  | z-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                       | slide    | position (m)             |
  | 17  | x-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidex)            | slide    | position (m)             |
  | 18  | y-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidey)            | slide    | position (m)             |
  | 19  | z-coordinate of the object to be moved                   | -Inf | Inf | object                         | cylinder | position (m)             |
  | 20  | x-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidex)             | slide    | position (m)             |
  | 21  | y-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidey)             | slide    | position (m)             |
  | 22  | z-coordinate of the goal position of the object          | -Inf | Inf | goal                           | sphere   | position (m)             |


  ### Rewards

  The reward consists of two parts:

  - *reward_near*: This reward is a measure of how far the *fingertip* of the
  pusher (the unattached end) is from the object, with a more negative value
  assigned for when the pusher's *fingertip* is further away from the target.
  It is calculated as the negative vector norm of (position of the fingertip -
  position of target), or *-norm("fingertip" - "target")*.
  - *reward_dist*: This reward is a measure of how far the object is from the
  target goal position, with a more negative value assigned for object is
  further away from the target. It is calculated as the negative vector norm of
  (position of the object - position of goal), or *-norm("object" - "target")*.
  - *reward_control*: A negative reward for penalising the pusher if it takes
  actions that are too large. It is measured as the negative squared Euclidean
  norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

  Unlike other environments, Pusher does not allow you to specify weights for
  the individual reward terms. However, `info` does contain the keys
  *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms, you
  should create a wrapper that computes the weighted reward from `info`.

  ### Starting State

  All pusher (not including object and goal) states start in (0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0). A uniform noise in the
  range [-0.005, 0.005] is added to the velocity attributes only. The velocities
  of the object and goal are permanently set to 0. The object's x-position is
  selected uniformly between [-0.3, 0] while the y-position is selected
  uniformly between [-0.2, 0.2], and this process is repeated until the vector
  norm between the object's (x,y) position and origin is not greater than 0.17.
  The goal always have the same position of (0.45, -0.05, -0.323).

  The default *dt = 0.05*.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps.

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder (or by changing
  the path to a modified XML file in another folder)..

  ```
  env = gym.make('Pusher-v2')
  ```

  There is no v3 for Pusher, unlike the robot environments where a v3 and
  beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

  There is a v4 version that uses the mujoco-bindings
  ```
  env = gym.make('Pusher-v4')
  ```

  And a v5 version that uses Brax:

  ```
  env = gym.make('Pusher-v5')
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
    del legacy_spring
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self._object_idx = self.sys.body.index['object']
    self._tips_arm_idx = self.sys.body.index['r_wrist_roll_link']
    self._goal_idx = self.sys.body.index['goal']
    self._table_idx = self.sys.body.index['table']
    self._goal_pos = jp.array([0.45, 0.05, 0.05])

  def reset(self, rng: jp.ndarray) -> env.State:
    rng, rng1, rng2 = jp.random_split(rng, 3)

    # randomly orient object
    cylinder_pos = jp.concatenate([
        jp.random_uniform(rng, (1,), -0.3, 0),
        jp.random_uniform(rng1, (1,), -0.2, 0.2),
        jp.ones(1) * 0.0
    ])

    # constraint maximum distance of object
    norm = jp.norm(cylinder_pos)
    scale = jp.where(norm > .17, .17 / norm, 1.)  # pytype: disable=wrong-arg-types  # jax-ndarray
    cylinder_pos = scale * cylinder_pos + jp.array([0., 0., .05])
    qpos = self.sys.default_angle()

    qvel = jp.concatenate([
        jp.random_uniform(rng2, (self.sys.num_joint_dof - 4,), -0.005, 0.005),
        jp.zeros(4)
    ])
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

    # position cylinder and goal
    pos = jp.index_update(qp.pos, self._goal_idx, self._goal_pos)
    pos = jp.index_update(pos, self._object_idx, cylinder_pos)
    pos = jp.index_update(pos, self._table_idx, jp.zeros(3))

    qp = qp.replace(pos=pos)

    obs = self._get_obs(qp)
    reward, done, zero = jp.zeros(3)
    metrics = {'reward_dist': zero, 'reward_ctrl': zero, 'reward_near': zero}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    vec_1 = state.qp.pos[self._object_idx] - state.qp.pos[self._tips_arm_idx]
    vec_2 = state.qp.pos[self._object_idx] - state.qp.pos[self._goal_idx]

    reward_near = -jp.norm(vec_1)
    reward_dist = -jp.norm(vec_2)
    reward_ctrl = -jp.square(action).sum()

    qp, _ = self.sys.step(state.qp, action)
    obs = self._get_obs(qp)
    reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
    state.metrics.update(
        reward_near=reward_near,
        reward_dist=reward_dist,
        reward_ctrl=reward_ctrl,
    )
    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP) -> jp.ndarray:
    """Observe pusher body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    return jp.concatenate([
        joint_angle,
        joint_vel,
        qp.pos[self._tips_arm_idx],
        qp.pos[self._object_idx],
        qp.pos[self._goal_idx],
    ])


_SYSTEM_CONFIG = """
bodies {
  name: "table"
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
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
  }
}
bodies {
  name: "r_shoulder_pan_link"
  colliders {
    position {
      z: -0.10000000149011612
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.10000000149011612
      length: 0.800000011920929
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "r_shoulder_lift_link"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.10000000149011612
      length: 0.4000000059604645
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "r_upper_arm_roll_link"
  colliders {
    position {
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.019999999552965164
      length: 0.23999999463558197
    }
  }
  colliders {
    position {
      x: 0.20000000298023224
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.05999999865889549
      length: 0.5199999809265137
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "r_elbow_flex_link"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.05999999865889549
      length: 0.1599999964237213
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "r_forearm_roll_link"
  colliders {
    position {
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.019999999552965164
      length: 0.23999999463558197
    }
  }
  colliders {
    position {
      x: 0.14550000429153442
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.05000000074505806
      length: 0.39100000262260437
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "r_wrist_flex_link"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.009999999776482582
      length: 0.05999999865889549
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "r_wrist_roll_link"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.019999999552965164
      length: 0.23999999463558197
    }
  }
  colliders {
    position {
      x: 0.05000000074505806
      y: -0.10000000149011612
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.019999999552965164
      length: 0.14000000059604645
    }
  }
  colliders {
    position {
      x: 0.05000000074505806
      y: 0.10000000149011612
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.019999999552965164
      length: 0.14000000059604645
    }
  }
  colliders {
    position {
      x: 0.10000000149011612
      y: -0.10000000149011612
    }
    sphere {
      radius: 0.009999999776482582
    }
  }
  colliders {
    position {
      x: 0.10000000149011612
      y: 0.10000000149011612
    }
    sphere {
      radius: 0.009999999776482582
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "object"
  colliders {
    position {
    }
    sphere {
      radius: 0.05000000074505806
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position { z: 1 }
    rotation { x: 1 y: 1 z: 1 }
  }
}
bodies {
  name: "goal"
  colliders {
    position {
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.07999999821186066
      length: 0.16
    }
    color: "red"
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
bodies {
  name: "mount"
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "r_shoulder_pan_joint"
  parent: "mount"
  child: "r_shoulder_pan_link"
  parent_offset {
    y: -0.6000000238418579
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -130.9437713623047
    max: 98.23945617675781
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "r_shoulder_lift_joint"
  parent: "r_shoulder_pan_link"
  child: "r_shoulder_lift_link"
  parent_offset {
    x: 0.10000000149011612
  }
  child_offset {
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -30.00006866455078
    max: 80.0020980834961
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "r_upper_arm_roll_joint"
  parent: "r_shoulder_lift_link"
  child: "r_upper_arm_roll_link"
  parent_offset {
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -85.94367218017578
    max: 97.40282440185547
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "r_elbow_flex_joint"
  parent: "r_upper_arm_roll_link"
  child: "r_elbow_flex_link"
  parent_offset {
    x: 0.4000000059604645
  }
  child_offset {
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -133.00070190429688
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "r_forearm_roll_joint"
  parent: "r_elbow_flex_link"
  child: "r_forearm_roll_link"
  parent_offset {
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -85.94367218017578
    max: 85.94367218017578
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "r_wrist_flex_joint"
  parent: "r_forearm_roll_link"
  child: "r_wrist_flex_link"
  parent_offset {
    x: 0.32100000977516174
  }
  child_offset {
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -62.681583404541016
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "r_wrist_roll_joint"
  parent: "r_wrist_flex_link"
  child: "r_wrist_roll_link"
  parent_offset {
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -85.94367218017578
    max: 85.94367218017578
  }
  reference_rotation {
    y: -0.0
  }
}
actuators {
  name: "r_shoulder_pan_joint"
  joint: "r_shoulder_pan_joint"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "r_shoulder_lift_joint"
  joint: "r_shoulder_lift_joint"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "r_upper_arm_roll_joint"
  joint: "r_upper_arm_roll_joint"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "r_elbow_flex_joint"
  joint: "r_elbow_flex_joint"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "r_forearm_roll_joint"
  joint: "r_forearm_roll_joint"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "r_wrist_flex_joint"
  joint: "r_wrist_flex_joint"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "r_wrist_roll_joint"
  joint: "r_wrist_roll_joint"
  strength: 100.0
  torque {
  }
}
gravity {  # zero-gravity works in this env because the `object` body is constrained to the x-y plane via `frozen` fields
}
collide_include {
  first: "table"
  second: "object"
}
collide_include {
  first: "r_wrist_roll_link"
  second: "object"
}
collide_include {
  first: "r_forearm_roll_link"
  second: "object"
}
collide_include {
  first: "r_forearm_roll_link"
  second: "table"
}
collide_include {
  first: "r_wrist_roll_link"
  second: "table"
}
dt: 0.05000000074505806
substeps: 50
solver_scale_pos: 0.20000000298023224
solver_scale_ang: 0.20000000298023224
solver_scale_collide: 0.20000000298023224
dynamics_mode: "pbd",
"""
