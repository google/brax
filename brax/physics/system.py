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
from brax.physics import bodies
from brax.physics import colliders
from brax.physics import config_pb2
from brax.physics import integrators
from brax.physics import joints
from brax.physics import math
from brax.physics import tree
from brax.physics.base import Info, P, Q, QP, euler_to_quat, validate_config, vec_to_np


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
    # TODO: add capsule <> heightmap
    self.box_heightMap = colliders.BoxHeightMap(config)
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

  @functools.partial(jax.jit, static_argnums=(0, 1))
  def default_qp(self, default_index: int = 0) -> QP:
    """Returns a default state for the system."""
    qps = {}

    # do we have default overrides in the config?
    default_angles, default_qps = {}, {}
    if default_index < len(self.config.defaults):
      defaults = self.config.defaults[default_index]
      default_qps = {qp.name: qp for qp in defaults.qps}
      default_angles = {angle.name: angle for angle in defaults.angles}

    # make the kinematic tree of bodies
    root = tree.Node.from_config(self.config)

    parent_joint = {j.child: j for j in self.config.joints}
    rot_axes = jax.jit(jax.vmap(math.rotate, in_axes=[0, None]))

    for body in root.depth_first():
      if body.name in default_qps:
        # set qp if found in defaults.qps override
        qp = default_qps[body.name]
        qps[body.name] = QP(
            pos=vec_to_np(qp.pos),
            rot=euler_to_quat(qp.rot),
            vel=vec_to_np(qp.vel),
            ang=vec_to_np(qp.ang))
      elif body.name in parent_joint:
        # pos/rot can be determined if the body is the child of a joint
        joint = parent_joint[body.name]
        axes = rot_axes(jnp.eye(3), euler_to_quat(joint.rotation))
        if joint.name in default_angles:
          angles = vec_to_np(default_angles[joint.name].angle) * jnp.pi / 180
        else:
          angles = [(l.min + l.max) * jnp.pi / 360 for l in joint.angle_limit]

        # for each joint angle, rotate by that angle.
        # these are euler intrinsic rotations, so the axes are rotated too
        local_rot = jnp.array([1., 0., 0., 0.])
        for axis, angle in zip(axes, angles):
          axis_rotated = math.rotate(axis, local_rot)
          next_rot = math.quat_rot_axis(axis_rotated, angle)
          local_rot = math.qmult(next_rot, local_rot)

        local_offset = math.rotate(vec_to_np(joint.child_offset), local_rot)
        local_pos = vec_to_np(joint.parent_offset) - local_offset

        parent_qp = qps[joint.parent]
        rot = math.qmult(parent_qp.rot, local_rot)
        pos = parent_qp.pos + math.rotate(local_pos, parent_qp.rot)
        qps[body.name] = QP(
            pos=pos, rot=rot, vel=jnp.zeros(3), ang=jnp.zeros(3))
      else:
        qps[body.name] = QP.zero()

    # any trees that have no body qp overrides in the config are set just above
    # the xy plane.  this convenience operation may be removed in the future.
    body_map = {body.name: body for body in self.config.bodies}
    for node in root.children:
      children = [node.name] + [n.name for n in node.depth_first()]
      if any(c in default_qps for c in children):
        continue  # ignore a tree if some part of it is overriden

      zs = jnp.array([bodies.min_z(qps[c], body_map[c]) for c in children])
      min_z = jnp.min(zs)
      for body in children:
        qp = qps[body]
        pos = qp.pos - min_z * jnp.array([0., 0., 1.])
        qps[body] = qp.replace(pos=pos)

    qps = [qps[body.name] for body in self.config.bodies]
    return jax.tree_multimap((lambda *args: jnp.stack(args)), *qps)

  @functools.partial(jax.jit, static_argnums=(0,))
  def info(self, qp: QP) -> Info:
    """Return info about a system state."""
    dp_c = self.box_plane.apply(qp, 1.)
    dp_c += self.box_heightMap.apply(qp, 1.)
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
      dp_c += self.box_heightMap.apply(qp, dt)
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
