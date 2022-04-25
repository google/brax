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

"""Trains an agent to locomote to a target location."""

from typing import Tuple

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env


class SphereFetch(env.Env):
  """SphereFetch trains an actuated sphere to run to a target location."""

  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)
    self.target_idx = self.sys.body.index['Target']
    self.torso_idx = self.sys.body.index['p1']
    self.target_radius = 2
    self.target_distance = 15

  def reset(self, rng: jp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    pos = jp.index_update(qp.pos, self.target_idx, target)
    qp = qp.replace(pos=pos)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'hits': zero,
        'weightedHits': zero,
        'movingToTarget': zero,
        # 'torsoIsUp': zero,
        # 'torsoHeight': zero
    }
    info = {'rng': rng}
    return env.State(qp, obs, reward, done, metrics, info)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # small reward for torso moving towards target
    torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
    target_rel = qp.pos[self.target_idx] - qp.pos[self.torso_idx]
    target_dist = jp.norm(target_rel)
    target_dir = target_rel / (1e-6 + target_dist)
    moving_to_target = .1 * jp.dot(torso_delta, target_dir)

    # # small reward for torso being up
    # up = jp.array([0., 0., 1.])
    # torso_up = math.rotate(up, qp.rot[self.torso_idx])
    # torso_is_up = .1 * self.sys.config.dt * jp.dot(torso_up, up)

    # # small reward for torso height
    # torso_height = .1 * self.sys.config.dt * qp.pos[0, 2]

    # big reward for reaching target (don't care about facing it)
    fwd = jp.array([1., 0., 0.])
    # torso_fwd = math.rotate(fwd, qp.rot[self.torso_idx])
    # torso_facing = jp.dot(target_dir, torso_fwd)
    target_hit = target_dist < self.target_radius
    target_hit = jp.where(target_hit, jp.float32(1), jp.float32(0))
    weighted_hit = target_hit # * torso_facing

    reward = moving_to_target + weighted_hit # + torso_is_up + torso_height

    state.metrics.update(
        hits=target_hit,
        weightedHits=weighted_hit,
        movingToTarget=moving_to_target,
        # torsoIsUp=torso_is_up,
        # torsoHeight=torso_height,
        )

    # teleport any hit targets
    rng, target = self._random_target(state.info['rng'])
    target = jp.where(target_hit, target, qp.pos[self.target_idx])
    pos = jp.index_update(qp.pos, self.target_idx, target)
    qp = qp.replace(pos=pos)
    state.info.update(rng=rng)
    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Egocentric observation of target and the dog's body."""
    torso_fwd = math.rotate(jp.array([1., 0., 0.]), qp.rot[self.torso_idx])
    # torso_up = math.rotate(jp.array([0., 0., 1.]), qp.rot[self.torso_idx])

    v_inv_rotate = jp.vmap(math.inv_rotate, include=(True, False))

    pos_local = qp.pos - qp.pos[self.torso_idx]
    pos_local = v_inv_rotate(pos_local, qp.rot[self.torso_idx])
    vel_local = v_inv_rotate(qp.vel, qp.rot[self.torso_idx])

    target_local = pos_local[self.target_idx]
    target_local_mag = jp.reshape(jp.norm(target_local), -1)
    target_local_dir = target_local / (1e-6 + target_local_mag)

    pos_local = jp.reshape(pos_local, -1)
    vel_local = jp.reshape(vel_local, -1)

    contact_mag = jp.sum(jp.square(info.contact.vel), axis=-1)
    contacts = jp.where(contact_mag > 0.00001, 1, 0)

    return jp.concatenate([
        torso_fwd, # torso_up, 
        target_local_mag, target_local_dir, pos_local,
        vel_local, contacts
    ])

  def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns a target location in a random circle on xz plane."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    dist = self.target_radius + self.target_distance * jp.random_uniform(rng1)
    ang = jp.pi * 2. * jp.random_uniform(rng2)
    target_x = dist * jp.cos(ang)
    target_y = dist * jp.sin(ang)
    target_z = 1.0
    target = jp.array([target_x, target_y, target_z]).transpose()
    return rng, target

_SYSTEM_CONFIG = """

bodies {
  name: "p1"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  mass: 1.0
}
bodies {
  name: "p1_roll"
  mass: 0.01
}
bodies {
  name: "p1_pitch"
  mass: 0.01
}
bodies {
name: "Target"
colliders { sphere { radius: 2 }}
frozen { all: true }
}
bodies {
  name: "ground"
  colliders {
    plane {
    }
  }
  frozen {
    all: true
  }
}
joints {
  name: "joint1"
  parent: "p1_roll"
  child: "p1"
  angle_limit {
    min: -180.0
    max: 180.0
  }
}
joints {
  name: "joint2"
  parent: "p1_pitch"
  child: "p1"
  rotation {
    z: -90.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
}
actuators {
  name: "torque1"
  joint: "joint1"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "torque2"
  joint: "joint2"
  strength: 100.0
  torque {
  }
}
elasticity: 1.0
friction: 3.0
gravity {
  z: -9.8
}
dt: 0.05
substeps: 20
dynamics_mode: "pbd"

"""