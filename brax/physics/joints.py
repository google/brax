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

  def __init__(self,
               joints: List[config_pb2.Joint],
               body: bodies.Body,
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
    v_rot = jax.vmap(math.rotate, in_axes=[0, None])
    relative_quats = jnp.array(
        [euler_to_quat(j.reference_rotation) for j in joints])
    self.axis_c = jnp.array(
        [v_rot(jnp.eye(3), euler_to_quat(j.rotation)) for j in joints])
    self.axis_p = jnp.array(
        [v_rot(j, r) for j, r in zip(self.axis_c, relative_quats)])

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
  """A revolute joint constrains two bodies around a single axis.

  Constructs a revolute joint where the parent's local x-axis is constrained
  to point in the same direction as the child's local x-axis.  This construction
  follows the line of nodes convention shared by the universal and spherical
  joints for x-y'-z'' intrinsic euler angles.
  """

  def __init__(self, joints: List[config_pb2.Joint], body: bodies.Body):
    super().__init__(joints, body, .5)

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
    axis_c = math.rotate(self.axis_c[0], qp_c.rot)
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
    axis_p = math.rotate(self.axis_p[0], qp_p.rot)
    ref_p = math.rotate(self.axis_p[2], qp_p.rot)
    ref_c = math.rotate(self.axis_c[2], qp_c.rot)
    # algebraically the same as the calculation in `Spherical`, but simpler
    # because child local-x and parent local-x are constrained to be the same
    psi = math.signed_angle(axis_p, ref_p, ref_c)
    return (axis_p,), (psi,)


@pytree.register
class Universal(Joint):
  """A revolute joint constrains two bodies around two axes.

  Constructs a universal joint defined as the first two degrees of freedom
  of a spherical joint.  See `Spherical` for details.
  """

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
    v_rot = jax.vmap(math.rotate, in_axes=[0, None])
    axis_p_rotated = v_rot(self.axis_p, qp_p.rot)
    axis_c_rotated = v_rot(self.axis_c, qp_c.rot)
    axis_1_p = axis_p_rotated[0]
    axis_2_p = axis_p_rotated[1]
    axis_1_c = axis_c_rotated[0]
    axis_2_c = axis_c_rotated[1]
    axis_3_c = axis_c_rotated[2]

    line_of_nodes = jnp.cross(axis_3_c, axis_1_p)
    line_of_nodes = line_of_nodes / (1e-10 + math.safe_norm(line_of_nodes))

    y_N_normal = axis_1_p

    psi = math.signed_angle(y_N_normal, axis_2_p, line_of_nodes)

    axis_1_p_in_xz_c = jnp.dot(axis_1_p, axis_1_c) * axis_1_c + jnp.dot(
        axis_1_p, axis_2_c) * axis_2_c
    axis_1_p_in_xz_c = axis_1_p_in_xz_c / (1e-10 +
                                           math.safe_norm(axis_1_p_in_xz_c))
    theta = jnp.arccos(jnp.clip(jnp.dot(axis_1_p_in_xz_c, axis_1_p), -1,
                                1)) * jnp.sign(jnp.dot(axis_1_p, axis_3_c))

    axis = (axis_1_p, axis_2_c)
    angle = (psi, theta)

    return axis, angle


@pytree.register
class Spherical(Joint):
  """A spherical joint constrains two bodies around three axes.

  Constructs a spherical joint which returns intrinsic euler angles in the
    x-y'-z'' convention between the parent and child.  Uses the line of nodes
    construction described in section 3.2.3.2 here:
    https://www.sedris.org/wg8home/Documents/WG80485.pdf
    """

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
    torque *= self.limit_strength * -1.

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dp_p = dp_p.replace(ang=dp_p.ang + jnp.matmul(self.body_p.inertia, torque))
    dp_c = dp_c.replace(ang=dp_c.ang + jnp.matmul(self.body_c.inertia, -torque))

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    v_rot = jax.vmap(math.rotate, in_axes=[0, None])
    axis_p_rotated = v_rot(self.axis_p, qp_p.rot)
    axis_c_rotated = v_rot(self.axis_c, qp_c.rot)
    axis_1_p = axis_p_rotated[0]
    axis_2_p = axis_p_rotated[1]
    axis_1_c = axis_c_rotated[0]
    axis_2_c = axis_c_rotated[1]
    axis_3_c = axis_c_rotated[2]

    line_of_nodes = jnp.cross(axis_3_c, axis_1_p)
    line_of_nodes = line_of_nodes / (1e-10 + math.safe_norm(line_of_nodes))

    y_N_normal = axis_1_p

    psi = math.signed_angle(y_N_normal, axis_2_p, line_of_nodes)

    axis_1_p_in_xz_c = jnp.dot(axis_1_p, axis_1_c) * axis_1_c + jnp.dot(
        axis_1_p, axis_2_c) * axis_2_c
    axis_1_p_in_xz_c = axis_1_p_in_xz_c / (1e-10 +
                                           math.safe_norm(axis_1_p_in_xz_c))

    ang_between_1_p_xz_c = jnp.dot(axis_1_p_in_xz_c, axis_1_p)
    theta = jnp.arccos(jnp.clip(ang_between_1_p_xz_c, -1, 1)) * jnp.sign(
        jnp.dot(axis_1_p, axis_3_c))
    yc_N_normal = -1. * axis_3_c

    phi = math.signed_angle(yc_N_normal, axis_2_c, line_of_nodes)

    axis = (axis_1_p, axis_2_c, axis_3_c)
    angle = (psi, theta, phi)

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
