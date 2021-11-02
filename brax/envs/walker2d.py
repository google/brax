# Copyright 2021 The Brax Authors.
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

"""Trains a 2D walker to run in the +x direction."""

from typing import Optional, Tuple

import brax
from brax import jumpy as jp
from brax.envs import env as brax_env
from brax.physics import bodies


class Walker2d(brax_env.Env):
  """Trains a 2D walker to run in the +x direction.

  This is similar to the Walker2d-V3 Mujoco environment in OpenAI Gym, which is
  a variant of Hopper with two legs. The two legs do not collide with each
  other.
  """

  # TODO: Add healthy_angle_range.
  def __init__(self,
               forward_reward_weight: float = 1.0,
               ctrl_cost_weight: float = 1e-3,
               healthy_reward: float = 1.0,
               terminate_when_unhealthy: bool = True,
               healthy_z_range: Tuple[float, float] = (0.7, 2.0),
               exclude_current_positions_from_observation: bool = True,
               system_config: Optional[str] = None,
               **kwargs):
    """Creates a Walker environment.

    Args:
      forward_reward_weight: Weight for the forward reward, i.e. velocity in
        x-direction.
      ctrl_cost_weight: Weight for the control cost.
      healthy_reward: Reward for staying healthy, i.e. respecting the posture
        constraints.
      terminate_when_unhealthy: Done bit will be set when unhealthy if true.
      healthy_z_range: Range of the z-position for being healthy.
      exclude_current_positions_from_observation: x-position will not be exposed
        in the observations if true.
      system_config: System config to use. If None, then _SYSTEM_CONFIG defined
        in this file will be used.
      **kwargs: Arguments that are passed to the base class.
    """
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation)

    super().__init__(system_config or _SYSTEM_CONFIG, **kwargs)

    config = self.sys.config
    body = bodies.Body(config)
    assert config.bodies[-1].name == 'floor'
    body = jp.take(body, body.idx[:-1])  # Skip the floor body.
    self.mass = body.mass.reshape(-1, 1)
    self.inertia = body.inertia

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.005, .005)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.005, .005)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_ctrl': zero,
        'reward_healthy': zero,
    }
    return brax_env.State(qp, obs, reward, done, metrics)

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    """Run one timestep of the environment's dynamics."""
    # Reverse torque improves performance over a range of hparams.
    qp, _ = self.sys.step(state.qp, -action)
    obs = self._get_obs(qp)

    # Ignore the floor at last index.
    pos_before = state.qp.pos[:-1]
    pos_after = qp.pos[:-1]
    com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
    com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
    x_velocity = (com_after[0] - com_before[0]) / self.sys.config.dt
    forward_reward = self._forward_reward_weight * x_velocity

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    rewards = forward_reward + healthy_reward

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    costs = ctrl_cost

    reward = rewards - costs

    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    state.metrics.update(
        reward_forward=forward_reward,
        reward_ctrl=-ctrl_cost,
        reward_healthy=healthy_reward)
    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP) -> jp.ndarray:
    """Returns the environment observations."""
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    # qpos: position and orientation of the torso and the joint angles.
    qpos = ([] if self._exclude_current_positions_from_observation else
            [qp.pos[0, 0:1]])
    qpos += [qp.pos[0, 2:], qp.rot[0], joint_angle]
    # qvel: velocity of the torso and the joint angle velocities.
    qvel = [qp.vel[0], joint_vel]
    return jp.concatenate(qpos + qvel)


_SYSTEM_CONFIG = """
bodies {
  name: "torso"
  colliders {
    position {}
    rotation {}
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
      x: -0.1
      y: -0.2
      z: -0.1
    }
    rotation { y: 90.0 }
    capsule {
      radius: 0.06
      length: 0.32
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 3.1667254
}
bodies {
  name: "thigh_left"
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
  name: "leg_left"
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
  name: "foot_left"
  colliders {
    position {
      x: -0.1
      y: -0.2
      z: -0.1
    }
    rotation { y: 90.0 }
    capsule {
      radius: 0.06
      length: 0.32
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 3.1667254
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
    x: -0.2
    y: -0.2
    z: -0.1
  }
  rotation { z: -90.0 }
  angle_limit { min: -45.0 max: 45.0 }
  angular_damping: 20.0
}
joints {
  name: "thigh_left_joint"
  stiffness: 10000.0
  parent: "torso"
  child: "thigh_left"
  parent_offset { z: -0.2 }
  rotation { z: -90.0 }
  angle_limit { min: -150.0 }
  angular_damping: 20.0
}
joints {
  name: "leg_left_joint"
  stiffness: 10000.0
  parent: "thigh_left"
  child: "leg_left"
  parent_offset { z: -0.45 }
  child_offset { z: 0.25 }
  rotation { z: -90.0 }
  angle_limit { min: -150.0 }
  angular_damping: 20.0
}
joints {
  name: "foot_left_joint"
  stiffness: 10000.0
  parent: "leg_left"
  child: "foot_left"
  parent_offset { z: -0.25 }
  child_offset {
    x: -0.2
    y: -0.2
    z: -0.1
  }
  rotation { z: -90.0 }
  angle_limit { min: -45.0 max: 45.0 }
  angular_damping: 20.0
}
actuators {
  name: "thigh_joint"
  joint: "thigh_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "leg_joint"
  joint: "leg_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "foot_joint"
  joint: "foot_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "thigh_left_joint"
  joint: "thigh_left_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "leg_left_joint"
  joint: "leg_left_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "foot_left_joint"
  joint: "foot_left_joint"
  strength: 100.0
  torque {}
}
friction: 0.9
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
collide_include {
  first: "floor"
  second: "thigh_left"
}
collide_include {
  first: "floor"
  second: "leg_left"
}
collide_include {
  first: "floor"
  second: "foot_left"
}
dt: 0.02
substeps: 4
frozen {
  position { y: 1.0 }
  rotation { x: 1.0 z: 1.0 }
}
defaults {
  qps { name: "torso" pos { z: 1.19 } }
  angles { name: "thigh_joint" angle {} }
  angles { name: "leg_joint" angle {} }
  angles { name: "thigh_left_joint" angle {} }
  angles { name: "leg_left_joint" angle {} }
}
"""
