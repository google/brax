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

"""Trains an ant to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env


class Ant(env.Env):
  """Trains an ant to run in the +x direction."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1, .1)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_ctrl_cost': zero,
        'reward_contact_cost': zero,
        'reward_forward': zero,
        'reward_survive': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    x_before = state.qp.pos[0, 0]
    x_after = qp.pos[0, 0]
    forward_reward = (x_after - x_before) / self.sys.config.dt
    ctrl_cost = .5 * jp.sum(jp.square(action))
    contact_cost = (0.5 * 1e-3 *
                    jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
    survive_reward = jp.float32(1)
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward

    done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
    done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done)
    state.metrics.update(
        reward_ctrl_cost=ctrl_cost,
        reward_contact_cost=contact_cost,
        reward_forward=forward_reward,
        reward_survive=survive_reward)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe ant body position and velocities."""
    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    # qpos:
    # Z of the torso (1,)
    # orientation of the torso as quaternion (4,)
    # joint angles (8,)
    qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

    # qvel:
    # velcotiy of the torso (3,)
    # angular velocity of the torso (3,)
    # joint angle velocities (8,)
    qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
    # ignores
    cfrc = [
        jp.clip(info.contact.vel, -1, 1),
        jp.clip(info.contact.ang, -1, 1)
    ]
    # flatten bottom dimension
    cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]

    return jp.concatenate(qpos + qvel + cfrc)

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
"""
