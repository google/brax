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

"""Trains a halfcheetah to run in the +x direction."""

import jax.numpy as jnp

import brax
from brax.envs import env

from google.protobuf import text_format


class Halfcheetah(env.Env):
  """Trains a halfcheetah to run in the +x direction."""

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)

  def reset(self, rng: jnp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, steps = jnp.zeros(3)
    metrics = {}
    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    rng = state.rng
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    x_before = state.qp.pos[0, 0]
    x_after = qp.pos[0, 0]
    forward_reward = (x_after - x_before) / self.sys.config.dt
    ctrl_cost = -.1 * jnp.sum(jnp.square(action))
    reward = forward_reward + ctrl_cost

    steps = state.steps + self.action_repeat
    done = jnp.where(steps >= self.episode_length, x=1.0, y=0.0)
    metrics = {}

    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Observe halfcheetah body position and velocities."""
    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vel,) = self.sys.joint_revolute.angle_vel(qp)

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

    return jnp.concatenate(qpos + qvel)

_SYSTEM_CONFIG = """
bodies {
  name: "torso"
  colliders {
    rotation {
      y: 90.0
    }
    capsule {
      radius: 0.046
      length: 1.092
    }
  }
  colliders {
    position {
      x: 0.6
      z: 0.1
    }
    rotation {
      y: 49.84733
    }
    capsule {
      radius: 0.046
      length: 0.392
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 9.457333
}
bodies {
  name: "bthigh"
  colliders {
    position {
      x: 0.1
      z: -0.13
    }
    rotation {
      x: -180.0
      y: 37.72396
      z: -180.0
    }
    capsule {
      radius: 0.046
      length: 0.382
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.335527
}
bodies {
  name: "bshin"
  colliders {
    position {
      x: -0.14
      z: -0.07
    }
    rotation {
      x: 180.0
      y: -63.689568
      z: 180.0
    }
    capsule {
      radius: 0.046
      length: 0.392
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.402003
}
bodies {
  name: "bfoot"
  colliders {
    position {
      x: 0.03
      z: -0.097
    }
    rotation {
      y: -15.46986
    }
    capsule {
      radius: 0.046
      length: 0.28
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6574708
}
bodies {
  name: "fthigh"
  colliders {
    position {
      x: -0.07
      z: -0.12
    }
    rotation {
      y: 29.793806
    }
    capsule {
      radius: 0.046
      length: 0.358
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.1759844
}
bodies {
  name: "fshin"
  colliders {
    position {
      x: 0.065
      z: -0.09
    }
    rotation {
      y: -34.37747
    }
    capsule {
      radius: 0.046
      length: 0.304
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.8170134
}
bodies {
  name: "ffoot"
  colliders {
    position {
      x: 0.045
      z: -0.07
    }
    rotation {
      y: -34.37747
    }
    capsule {
      radius: 0.046
      length: 0.232
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.3383855
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
  frozen { all: true }
}
joints {
  name: "bthigh"
  stiffness: 15000.0
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
  angular_damping: 20.0
  angle_limit {
    min: -29.793806
    max: 60.16057
  }
}
joints {
  name: "bshin"
  stiffness: 15000.0
  parent: "bthigh"
  child: "bshin"
  parent_offset {
    x: 0.16
    z: -0.25
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -44.97719
    max: 44.97719
  }
}
joints {
  name: "bfoot"
  stiffness: 15000.0
  parent: "bshin"
  child: "bfoot"
  parent_offset {
    x: -0.28
    z: -0.14
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.918312
    max: 44.97719
  }
}
joints {
  name: "fthigh"
  stiffness: 15000.0
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
  angular_damping: 20.0
  angle_limit {
    min: -57.29578
    max: 40.107044
  }
}
joints {
  name: "fshin"
  stiffness: 15000.0
  parent: "fthigh"
  child: "fshin"
  parent_offset {
    x: -0.14
    z: -0.24
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -68.75494
    max: 49.84733
  }
}
joints {
  name: "ffoot"
  stiffness: 15000.0
  parent: "fshin"
  child: "ffoot"
  parent_offset {
    x: 0.13
    z: -0.18
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -28.64789
    max: 28.64789
  }
}
actuators {
  name: "bthigh"
  joint: "bthigh"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "bshin"
  joint: "bshin"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "bfoot"
  joint: "bfoot"
  strength: 240.0
  torque {
  }
}
actuators {
  name: "fthigh"
  joint: "fthigh"
  strength: 350.0
  torque {
  }
}
actuators {
  name: "fshin"
  joint: "fshin"
  strength: 240.0
  torque {
  }
}
actuators {
  name: "ffoot"
  joint: "ffoot"
  strength: 240.0
  torque {
  }
}
friction: 0.6
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
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
dt: 0.05
substeps: 12
frozen {
  position { x: 0.0 y: 1.0 z: 0.0 }
  rotation { x: 1.0 y: 0.0 z: 1.0 }
}
"""
