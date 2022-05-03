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

"""Trains an ant to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env


class ThrusterCheck(env.Env):
  """"""

  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_ctrl_cost': zero,
        'reward_contact_cost': zero,
        'reward_forward': zero,
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
    # # some pre-processing to pull joint angles and velocities
    # (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    # # qpos:
    # # Z of the torso (1,)
    # # orientation of the torso as quaternion (4,)
    # # joint angles (8,)
    # qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

    # # qvel:
    # # velocity of the torso (3,)
    # # angular velocity of the torso (3,)
    # # joint angle velocities (8,)
    # qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # # external contact forces:
    # # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
    # # ignores
    # cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
    # # flatten bottom dimension
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]


    return jp.concatenate([qp.pos[0]] + [qp.vel[0]] + [qp.rot[0]] + [qp.ang[0]])


_SYSTEM_CONFIG = """
bodies {
  name: "p1"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
    material {
      elasticity: 1.0
      friction: 0.10000000149011612
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
    }
    rotation {
    }
  }
}
 bodies {
  name: "ground"
  colliders {
    plane {
    }
    material {
      elasticity: 1.0
      friction: 0.10000000149011612
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
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
    all: true
  }
}
elasticity: 1.0
friction: 0.10000000149011612
gravity {
  z: -9.800000190734863
}
angular_damping: -0.05
dt: 0.05000000074505806
substeps: 20
frozen {
}
forces {
  name: "p1_thrust"
  body: "p1"
  strength: 1.0
  thruster {
  }
}
dynamics_mode: "pbd"

""" 
