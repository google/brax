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

# pylint:disable=g-multiple-import
"""A brax system."""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from brax.physics import actuators
from brax.physics import colliders
from brax.physics import config_pb2
from brax.physics import integrators
from brax.physics import joints
from brax.physics import tree
from brax.physics.base import Info, P, QP, validate_config, vec_to_np


class System:
  """A brax system."""

  def __init__(self, config: config_pb2.Config):
    self.config = validate_config(config)

    self.num_bodies = len(config.bodies)
    self.body_idx = {b.name: i for i, b in enumerate(config.bodies)}

    self.active_pos = 1. * jnp.logical_not(
        jnp.array([vec_to_np(b.frozen.position) for b in config.bodies]))
    self.active_rot = 1. * jnp.logical_not(
        jnp.array([vec_to_np(b.frozen.rotation) for b in config.bodies]))

    self.box_plane = colliders.BoxPlane(config)
    self.capsule_plane = colliders.CapsulePlane(config)
    self.capsule_capsule = colliders.CapsuleCapsule(config)

    self.num_joints = len(config.joints)
    self.joint_revolute = joints.Revolute.from_config(config)
    self.joint_universal = joints.Universal.from_config(config)
    self.joint_spherical = joints.Spherical.from_config(config)

    self.num_actuators = len(config.actuators)
    self.num_joint_dof = sum(len(j.angle_limit) for j in config.joints)

    self.angle_1d = actuators.Angle.from_config(config, self.joint_revolute)
    self.angle_2d = actuators.Angle.from_config(config, self.joint_universal)
    self.angle_3d = actuators.Angle.from_config(config, self.joint_spherical)
    self.torque_1d = actuators.Torque.from_config(config, self.joint_revolute)
    self.torque_2d = actuators.Torque.from_config(config, self.joint_universal)
    self.torque_3d = actuators.Torque.from_config(config, self.joint_spherical)

  @functools.partial(jax.jit, static_argnums=(0,))
  def default_qp(self) -> QP:
    """Returns a default state for the system."""
    root = tree.Link.from_config(self.config).to_world()

    # raise any sub-trees above the ground plane
    child_min_z = [(child, child.min_z()) for child in root.children]

    # construct qps array that matches bodies order
    qps = []
    for body in self.config.bodies:
      pos = jnp.array([0., 0., 0.])
      rot = jnp.array([1., 0., 0., 0.])
      for child, min_z in child_min_z:
        link = child.rfind(body.name)
        if link:
          pos = link.pos - min_z * jnp.array([0., 0., 1.])
          rot = link.rot
      qp = QP(pos=pos, rot=rot, vel=jnp.zeros(3), ang=jnp.zeros(3))
      qps.append(qp)

    return jax.tree_multimap((lambda *args: jnp.stack(args)), *qps)

  @functools.partial(jax.jit, static_argnums=(0,))
  def info(self, qp: QP) -> Info:
    """Return info about a system state."""
    dp_c = self.box_plane.apply(qp, 1.)
    dp_c += self.capsule_plane.apply(qp, 1.)
    dp_c += self.capsule_capsule.apply(qp, 1.)

    dp_j = self.joint_revolute.apply(qp)
    dp_j += self.joint_universal.apply(qp)
    dp_j += self.joint_spherical.apply(qp)

    dp_a = P(jnp.zeros((self.num_bodies, 3)), jnp.zeros((self.num_bodies, 3)))

    info = Info(contact=dp_c, joint=dp_j, actuator=dp_a)
    return info

  @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, qp: QP, act: jnp.ndarray) -> Tuple[QP, Info]:
    """Calculates a physics step for a system, returns next state and info."""

    def substep(carry, _):
      qp, info = carry
      dt = self.config.dt / self.config.substeps

      # apply kinetic step
      qp = integrators.kinetic(self.config, qp, dt, self.active_pos,
                               self.active_rot)

      # apply impulses arising from joints and actuators
      dp_j = self.joint_revolute.apply(qp)
      dp_j += self.joint_universal.apply(qp)
      dp_j += self.joint_spherical.apply(qp)
      dp_a = self.angle_1d.apply(qp, act)
      dp_a += self.angle_2d.apply(qp, act)
      dp_a += self.angle_3d.apply(qp, act)
      dp_a += self.torque_1d.apply(qp, act)
      dp_a += self.torque_2d.apply(qp, act)
      dp_a += self.torque_3d.apply(qp, act)
      qp = integrators.potential(self.config, qp, dp_j + dp_a, dt,
                                 self.active_pos, self.active_rot)

      # apply collision velocity updates
      dp_c = self.box_plane.apply(qp, dt)
      dp_c += self.capsule_plane.apply(qp, dt)
      dp_c += self.capsule_capsule.apply(qp, dt)
      qp = integrators.potential_collision(self.config, qp, dp_c,
                                           self.active_pos, self.active_rot)

      info = Info(
          contact=info.contact + dp_c,
          joint=info.joint + dp_j,
          actuator=info.actuator + dp_a)

      return (qp, info), ()

    zero = P(jnp.zeros((self.num_bodies, 3)), jnp.zeros((self.num_bodies, 3)))
    info = Info(contact=zero, joint=zero, actuator=zero)
    (qp, info), _ = jax.lax.scan(substep, (qp, info), (), self.config.substeps)
    return qp, info
