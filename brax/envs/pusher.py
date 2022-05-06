# Copyright 2022 The Brax Authors.
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

"""Trains a robot arm to push a ball to a target.

Based on the OpenAI Gym MuJoCo Pusher environment.
"""

import brax
from brax import jumpy as jp
from brax.envs import env


class Pusher(env.Env):
  """Trains a robot arm to push a ball to a target."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self.object_idx = self.sys.body.index['object']
    self.tips_arm_idx = self.sys.body.index['r_wrist_roll_link']
    self.goal_idx = self.sys.body.index['goal']
    self.table_idx = self.sys.body.index['table']

    self.goal_pos = jp.array([0.45, 0.05, 0.05])

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
    scale = jp.where(norm > .17, .17 / norm, 1.)
    cylinder_pos = scale * cylinder_pos + jp.array([0., 0., .05])
    qpos = self.sys.default_angle()

    qvel = jp.concatenate([
        jp.random_uniform(rng2, (self.sys.num_joint_dof - 4,), -0.005, 0.005),
        jp.zeros(4)
    ])
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

    # position cylinder and goal
    pos = jp.index_update(qp.pos, self.goal_idx, self.goal_pos)
    pos = jp.index_update(pos, self.object_idx, cylinder_pos)
    pos = jp.index_update(pos, self.table_idx, jp.zeros(3))

    qp = qp.replace(pos=pos)

    obs = self._get_obs(qp)
    reward, done, zero = jp.zeros(3)
    metrics = {'rewardDist': zero, 'rewardCtrl': zero, 'rewardNear': zero}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:

    vec_1 = state.qp.pos[self.object_idx] - state.qp.pos[self.tips_arm_idx]
    vec_2 = state.qp.pos[self.object_idx] - state.qp.pos[self.goal_idx]

    reward_near = -jp.norm(vec_1)
    reward_dist = -jp.norm(vec_2)
    reward_ctrl = -jp.square(action).sum()
    reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

    qp, _ = self.sys.step(state.qp, action)
    obs = self._get_obs(qp)

    state.metrics.update(
        rewardDist=reward_dist,
        rewardCtrl=reward_ctrl,
        rewardNear=reward_near,
    )
    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP) -> jp.ndarray:
    """Observe pusher body position and velocities."""

    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    return jp.concatenate([
        joint_angle,
        joint_vel,
        qp.pos[self.tips_arm_idx],
        qp.pos[self.object_idx],
        qp.pos[self.goal_idx],
    ])


_SYSTEM_CONFIG = """bodies {
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
"""
