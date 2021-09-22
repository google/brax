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
"""Joints connect bodies and constrain their movement."""

import abc
from typing import Any, List, Tuple

from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics import math
from brax.physics import pytree
from brax.physics.base import P, QP, euler_to_quat, take, vec_to_np
import jax
import jax.numpy as jnp


class Joint(abc.ABC):
  """A joint connects two bodies and constrains their movement.

  This constraint is determined by axes that define how to bodies may move in
  relation to one-another.
  """

  __pytree_ignore__ = ('index', 'dof')

  def __init__(self, joints: List[config_pb2.Joint], body: bodies.Body,
               spring_damping_coeff: float = 2.0):
    """Creates a Joint that connects two bodies and constrains their movement.

    Args:
      joints: list of joints (all of the same type) to batch together
      body: batched body that contain the parents and children of each joint
      spring_damping_coeff: coefficient for setting default spring damping
    """
    self.stiffness = jnp.array([j.stiffness for j in joints])
    self.angular_damping = jnp.array([j.angular_damping for j in joints])
    self.spring_damping = jnp.array([
        j.spring_damping if j.HasField('spring_damping') else
        spring_damping_coeff * jnp.sqrt(j.stiffness) for j in joints
    ])
    self.limit_strength = jnp.array([
        j.limit_strength if j.HasField('limit_strength') else j.stiffness
        for j in joints
    ])
    self.limit = jnp.array([[[i.min, i.max]
                             for i in j.angle_limit]
                            for j in joints]) / 180.0 * jnp.pi
    self.body_p = take(body, jnp.array([body.index[j.parent] for j in joints]))
    self.body_c = take(body, jnp.array([body.index[j.child] for j in joints]))
    self.off_p = jnp.array([vec_to_np(j.parent_offset) for j in joints])
    self.off_c = jnp.array([vec_to_np(j.child_offset) for j in joints])
    self.index = {j.name: i for i, j in enumerate(joints)}
    self.dof = len(joints[0].angle_limit)

  def apply(self, qp: QP) -> P:
    """Returns impulses to constrain and align bodies connected by a joint.

    Args:
      qp: State data for system

    Returns:
      dP: Impulses on all bodies to maintain joint constraints
    """
    qp_p = take(qp, self.body_p.idx)
    qp_c = take(qp, self.body_c.idx)
    dp_p, dp_c = self.apply_reduced(qp_p, qp_c)

    # sum together all impulse contributions across parents and children
    body_idx = jnp.concatenate((self.body_p.idx, self.body_c.idx))
    dp_vel = jnp.concatenate((dp_p.vel, dp_c.vel))
    dp_ang = jnp.concatenate((dp_p.ang, dp_c.ang))
    dp_vel = jax.ops.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = jax.ops.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    return P(vel=dp_vel, ang=dp_ang)

  def angle_vel(self, qp: QP) -> Tuple[Any, Any]:
    """Returns joint angle and velocity.

    Args:
      qp: State data for system

    Returns:
      angle: n-tuple of joint angles where n = # DoF of the joint
      vel: n-tuple of joint velocities where n = # DoF of the joint
    """
    @jax.vmap
    def op(joint, qp_p, qp_c):
      axes, angles = joint.axis_angle(qp_p, qp_c)
      vels = tuple([jnp.dot(qp_p.ang - qp_c.ang, axis) for axis in axes])
      return angles, vels

    qp_p = take(qp, self.body_p.idx)
    qp_c = take(qp, self.body_c.idx)
    angles, vels = op(self, qp_p, qp_c)

    return angles, vels

  @abc.abstractmethod
  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns impulses to constrain and align bodies connected by a joint.

    Operates in reduced joint space.

    Args:
      qp_p: Joint parent state data
      qp_c: Joint child state data

    Returns:
      dp_p: Joint parent impulse
      dp_c: Joint child impulse
    """

  @abc.abstractmethod
  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint.

    vmap across axis_angle to get all joints.

    Args:
      qp_p: State for parent body
      qp_c: State for child body

    Returns:
      axis: n-tuple of joint axes where n = # DoF of the joint
      angle: n-tuple of joint angles where n = # DoF of the joint
    """


@pytree.register
class Revolute(Joint):
  """A revolute joint constrains two bodies around a single axis."""

  def __init__(self, joints: List[config_pb2.Joint], body: bodies.Body):
    if not joints:
      return
    super().__init__(joints, body, .5)
    rot = jnp.array([euler_to_quat(j.rotation) for j in joints])
    vrotate = jax.vmap(math.rotate, in_axes=(None, 0))
    self.axis = vrotate(jnp.array([1., 0., 0.]), rot)

    ref_axes = [jnp.array([-a[1], a[0], 0.]) for a in self.axis]
    # one of these two vectors is guaranteed to be orthogonal to axis
    for i, (axis, ref_axis) in enumerate(zip(self.axis, ref_axes)):
      if jnp.abs(jnp.dot(axis, ref_axis)) > 1e-4 or jnp.allclose(
          ref_axis, 0, atol=1e-3):
        ref_axes[i] = jnp.array([-axis[2], 0., axis[0]])
    self.ref = jnp.array([r / (1e-6 + jnp.linalg.norm(r)) for r in ref_axes])

  @jax.vmap
  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    pos_p, vel_p = math.to_world(qp_p, self.off_p)
    pos_c, vel_c = math.to_world(qp_c, self.off_c)

    # push the bodies towards their offsets
    # TODO: cap this damping so that it can't overcorrect
    impulse = (pos_p - pos_c) * self.stiffness + self.spring_damping * (
        vel_p - vel_c)
    dp_p = self.body_p.impulse(qp_p, -impulse, pos_p)
    dp_c = self.body_c.impulse(qp_c, impulse, pos_c)

    # torque the bodies to align their axes
    (axis,), (angle,) = self.axis_angle(qp_p, qp_c)
    axis_c = math.rotate(self.axis, qp_c.rot)
    torque = self.stiffness * jnp.cross(axis, axis_c)

    # torque the bodies to stay within angle limits
    dang = jnp.where(angle < self.limit[0][0], self.limit[0][0] - angle, 0)
    dang = jnp.where(angle > self.limit[0][1], self.limit[0][1] - angle, dang)
    torque -= self.limit_strength * axis * dang

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dang_p = jnp.matmul(self.body_p.inertia, torque)
    dang_c = jnp.matmul(self.body_c.inertia, -torque)
    dp_p = dp_p.replace(ang=dp_p.ang + dang_p)
    dp_c = dp_c.replace(ang=dp_c.ang + dang_c)

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    axis_p = math.rotate(self.axis, qp_p.rot)
    angle = math.signed_angle(qp_p, qp_c, axis_p, self.ref)
    return (axis_p,), (angle,)


@pytree.register
class Universal(Joint):
  """A revolute joint constrains two bodies around two axes."""

  def __init__(self, joints: List[config_pb2.Joint], body: bodies.Body):
    if not joints:
      return
    super().__init__(joints, body)
    rot = jnp.array([euler_to_quat(j.rotation) for j in joints])
    vrotate = jax.vmap(math.rotate, in_axes=(None, 0))
    self.axis_1 = vrotate(jnp.array([1., 0., 0.]), rot)
    self.axis_2 = vrotate(jnp.array([0., 1., 0.]), rot)
    self.ref = jax.vmap(jnp.cross)(self.axis_2, self.axis_1)

  @jax.vmap
  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    pos_p, vel_p = math.to_world(qp_p, self.off_p)
    pos_c, vel_c = math.to_world(qp_c, self.off_c)

    # push the bodies towards their offsets
    # TODO: cap this damping so that it can't overcorrect
    impulse = (pos_p - pos_c) * self.stiffness + self.spring_damping * (
        vel_p - vel_c)
    dp_p = self.body_p.impulse(qp_p, -impulse, pos_p)
    dp_c = self.body_c.impulse(qp_c, impulse, pos_c)

    # torque the bodies to align to a joint plane
    (axis_1, axis_2), (angle_1, angle_2) = self.axis_angle(qp_p, qp_c)
    axis_c_proj = axis_2 - jnp.dot(axis_2, axis_1) * axis_1
    axis_c_proj = axis_c_proj / math.safe_norm(axis_c_proj)
    torque = (self.limit_strength / 5.) * jnp.cross(axis_c_proj, axis_2)

    # torque the bodies to stay within angle limits
    limit_1, limit_2 = self.limit
    dang_1 = jnp.where(angle_1 < limit_1[0], limit_1[0] - angle_1, 0)
    dang_1 = jnp.where(angle_1 > limit_1[1], limit_1[1] - angle_1, dang_1)
    dang_2 = jnp.where(angle_2 < limit_2[0], limit_2[0] - angle_2, 0)
    dang_2 = jnp.where(angle_2 > limit_2[1], limit_2[1] - angle_2, dang_2)
    torque -= self.limit_strength * (axis_1 * dang_1 + axis_2 * dang_2)

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dang_p = jnp.matmul(self.body_p.inertia, torque)
    dang_c = jnp.matmul(self.body_c.inertia, -torque)
    dp_p = dp_p.replace(ang=dp_p.ang + dang_p)
    dp_c = dp_c.replace(ang=dp_c.ang + dang_c)

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    axis_1_c = math.rotate(self.axis_1, qp_c.rot)
    axis_2_p = math.rotate(self.axis_2, qp_p.rot)
    child_in_plane = jnp.cross(axis_2_p, axis_1_c)
    ref_c = math.rotate(self.ref, qp_c.rot)
    angle_1 = math.signed_angle(qp_p, qp_c, axis_2_p, self.ref)
    angle_2 = jnp.arctan2(
        jnp.dot(jnp.cross(child_in_plane, ref_c), axis_1_c),
        jnp.dot(child_in_plane, ref_c))
    return (axis_2_p, axis_1_c), (angle_1, angle_2)


@pytree.register
class Spherical(Joint):
  """A spherical joint constrains two bodies around three axes."""

  def __init__(self, joints: List[config_pb2.Joint], body: bodies.Body):
    if not joints:
      return
    super().__init__(joints, body)
    rot = jnp.array([euler_to_quat(j.rotation) for j in joints])
    vrotate = jax.vmap(math.rotate, in_axes=(None, 0))
    self.axis_1 = vrotate(jnp.array([1., 0., 0.]), rot)
    self.axis_2 = vrotate(jnp.array([0., 1., 0.]), rot)
    self.axis_3 = vrotate(jnp.array([0., 0., 1.]), rot)

  @jax.vmap
  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    pos_p, vel_p = math.to_world(qp_p, self.off_p)
    pos_c, vel_c = math.to_world(qp_c, self.off_c)

    # push the bodies towards their offsets
    # TODO: cap this damping so that it can't overcorrect
    impulse = (pos_p - pos_c) * self.stiffness + self.spring_damping * (
        vel_p - vel_c)
    dp_p = self.body_p.impulse(qp_p, -impulse, pos_p)
    dp_c = self.body_c.impulse(qp_c, impulse, pos_c)

    # torque the bodies to stay within angle limits
    axis, angle = self.axis_angle(qp_p, qp_c)
    angle_1, angle_2, angle_3 = angle
    limit_1, limit_2, limit_3 = self.limit
    dang_1 = jnp.where(angle_1 < limit_1[0], limit_1[0] - angle_1, 0)
    dang_1 = jnp.where(angle_1 > limit_1[1], limit_1[1] - angle_1, dang_1)
    dang_2 = jnp.where(angle_2 < limit_2[0], limit_2[0] - angle_2, 0)
    dang_2 = jnp.where(angle_2 > limit_2[1], limit_2[1] - angle_2, dang_2)
    dang_3 = jnp.where(angle_3 < limit_3[0], limit_3[0] - angle_3, 0)
    dang_3 = jnp.where(angle_3 > limit_3[1], limit_3[1] - angle_3, dang_3)

    # TODO: fully decouple different torque axes
    torque = axis[0] * dang_1 + axis[1] * dang_2 + axis[2] * dang_3
    torque *= self.limit_strength

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dp_p = dp_p.replace(ang=dp_p.ang +
                        jnp.matmul(self.body_p.inertia, torque))
    dp_c = dp_c.replace(ang=dp_c.ang +
                        jnp.matmul(self.body_c.inertia, -torque))

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    axis_1_p = math.rotate(self.axis_1, qp_p.rot)
    axis_2_p = math.rotate(self.axis_2, qp_p.rot)
    axis_3_p = math.rotate(self.axis_3, qp_p.rot)
    axis_1_c = math.rotate(self.axis_1, qp_c.rot)
    axis_2_c = math.rotate(self.axis_2, qp_c.rot)
    axis_3_c = math.rotate(self.axis_3, qp_c.rot)

    axis_2_in_plane = axis_2_c - jnp.dot(axis_2_c, axis_3_p) * axis_3_p
    axis_2_in_projected_length = jnp.linalg.norm(axis_2_in_plane)
    axis_2_in_plane = axis_2_in_plane / (1e-7 +
                                         jnp.linalg.norm(axis_2_in_plane))

    angle_z = jnp.arctan2(
        jnp.dot(axis_2_in_plane, axis_1_p),
        jnp.dot(axis_2_in_plane, axis_2_p))  # parent z axis torque
    angle_x = -1. * jnp.arctan2(
        jnp.dot(axis_2_c, axis_3_p),
        axis_2_in_projected_length)  # child x axis torque
    axis_3_in_child_xz = axis_3_p - jnp.dot(axis_3_p, axis_2_c) * axis_2_c
    axis_3_in_child_xz = axis_3_in_child_xz / (
        1e-7 + jnp.linalg.norm(axis_3_in_child_xz))
    angle_y = jnp.arctan2(
        jnp.dot(
            axis_3_in_child_xz -
            jnp.dot(axis_3_in_child_xz, axis_3_c) * axis_3_c, axis_1_c),
        jnp.dot(
            axis_3_in_child_xz -
            jnp.dot(axis_3_in_child_xz, axis_1_c) * axis_1_c,
            axis_3_c))  # child y torque

    axis = (axis_1_c, axis_2_c, axis_3_p,)
    angle = (angle_x, angle_y, angle_z)

    return axis, angle


def get(config: config_pb2.Config, body: bodies.Body) -> List[Joint]:
  """Creates all joints given a config."""
  joints = {}
  for joint in config.joints:
    dof = len(joint.angle_limit)
    if dof not in joints:
      joints[dof] = []
    joints[dof].append(joint)

  # ensure stable order for joint application: dof
  joints = sorted(joints.items(), key=lambda kv: kv[0])
  ret = []
  for k, v in joints:
    if k == 1:
      ret.append(Revolute(v, body))
    elif k == 2:
      ret.append(Universal(v, body))
    elif k == 3:
      ret.append(Spherical(v, body))
    else:
      raise RuntimeError(f'invalid number of joint limits: {k}')

  return ret
