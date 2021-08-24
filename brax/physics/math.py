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
"""Some general purpose math functions."""

from typing import Tuple

import jax.numpy as jnp

from brax.physics.base import QP


def safe_norm(d, axis=None):
  """Calculates a jnp.linalg.norm(d) that's safe for gradients at d=0.

  These gymnastics are to avoid a poorly defined gradient for jnp.linal.norm(0)
  see https://github.com/google/jax/issues/3058 for details

  Args:
    d: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    Equivalent to jnp.linalg.norm(d)
  """
  is_zero = jnp.allclose(d, 0.)
  d = jnp.where(is_zero, jnp.ones_like(d), d)  # replace d with ones if is_zero
  l = jnp.linalg.norm(d, axis=axis)
  l = jnp.where(is_zero, 0., l)  # replace norm with zero if is_zero

  return l


def rotate(vec: jnp.ndarray, quat: jnp.ndarray):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: jnp.ndarray (3)
    quat: jnp.ndarray (4) (w,x,y,z)

  Returns:
    A jnp.ndarry(3) containing vec rotated by quat.
  """

  u = quat[1:]
  s = quat[0]
  return 2 * (jnp.dot(u, vec) *
              u) + (s * s - jnp.dot(u, u)) * vec + 2 * s * jnp.cross(u, vec)


def inv_rotate(vec: jnp.ndarray, quat: jnp.ndarray):
  """Rotates a vector by the inverse of a unit quaternion.

  Args:
    vec: jnp.ndarray
    quat: jnp.ndarray

  Returns:
    A vector rotated by quat^{-1}
  """
  u = -1. * quat[1:]
  s = quat[0]
  return 2 * (jnp.dot(u, vec) *
              u) + (s * s - jnp.dot(u, u)) * vec + 2 * s * jnp.cross(u, vec)


def ang_to_quat(ang: jnp.ndarray):
  """Converts angular velocity to a quaternion.

  Args:
    ang: angular velocity

  Returns:
    A rotation quaternion.
  """
  return jnp.array([[0., -ang[0], -ang[1], -ang[2]],
                    [ang[0], 0, -ang[2], ang[1]], [ang[1], ang[2], 0., -ang[0]],
                    [ang[2], -ang[1], ang[0], 0.]])


def quat_to_axis_angle(q: jnp.ndarray):
  """Returns the axis-angle representation of a quaternion.

  Args:
    q: (4,) a quaternion

  Returns:
    The angle of axis-angle of this quaternion, in the range [-pi, pi].
  """
  # TODO: replace with more accurate safe function
  # avoid the singularity at 0:
  epsilon = 0.00001
  # safety 1e-6 jitter added because both sqrt and arctan2 have bad gradients
  denom = jnp.sqrt(epsilon + 1 - q[0] * q[0])
  angle = 2. * jnp.arctan2(
      jnp.sqrt(epsilon + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), q[0])
  angle += jnp.where(angle > jnp.pi, x=-2. * jnp.pi, y=0.)
  angle += jnp.where(angle < -jnp.pi, x=2. * jnp.pi, y=0.)
  return jnp.array([q[1] / denom, q[2] / denom, q[3] / denom]), angle


def to_world(qp: QP, rpos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Returns world information about a point relative to a part.

  Args:
    qp: Part from which to offset point.
    rpos: Point relative to center of mass of part.

  Returns:
    A 2-tuple containing:
      * World-space coordinates of rpos
      * World-space velocity of rpos
  """
  rpos_off = rotate(rpos, qp.rot)
  rvel = jnp.cross(qp.ang, rpos_off)
  return (qp.pos + rpos_off, qp.vel + rvel)


def world_velocity(qp: QP, pos: jnp.ndarray) -> jnp.ndarray:
  """Returns the velocity of the point on a rigidbody in world space.

  Args:
    qp: Part from which to extract a velocity.
    pos: World space position which to use for velocity calculation.
  """
  return qp.vel + jnp.cross(qp.ang, pos - qp.pos)


def signed_angle(qp_p: QP, qp_c: QP, normal_axis: jnp.ndarray,
                 ref_vector) -> float:
  """Calculates the signed angle between two parts.

  This calculation works by checking how much a parent and child part's quats
  rotate a reference vector.

  Args:
    qp_p: State data for the parent part
    qp_c: State data for the child part
    normal_axis: Common axis around which to calculate change in angle
    ref_vector: A reference vector pointing at 0-degrees offset

  Returns:
    The signed angle between two parts.
  """
  ref_vector_p = rotate(ref_vector, qp_p.rot)
  ref_vector_c = rotate(ref_vector, qp_c.rot)
  angle = jnp.arctan2(
      jnp.dot(jnp.cross(ref_vector_p, ref_vector_c), normal_axis),
      jnp.dot(ref_vector_p, ref_vector_c))
  return angle


def qmult(u, v):
  """Multiplies two quaternions.

  Args:
    u: jnp.ndarray (4) (w,x,y,z)
    v: jnp.ndarray (4) (w,x,y,z)

  Returns:
    A quaternion u*v.
  """
  return jnp.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


def quat_rot_axis(v, angle):
  """Provides a quaternion that describes rotating around axis v by angle.

  Args:
    v: jnp.ndarray (3) (x,y,z)
    angle: float angle to rotate by

  Returns:
    A quaternion that rotates around v by angle
  """
  qx = v[0] * jnp.sin(angle / 2.)
  qy = v[1] * jnp.sin(angle / 2.)
  qz = v[2] * jnp.sin(angle / 2.)
  qw = jnp.cos(angle / 2.)
  return jnp.array([qw, qx, qy, qz])


def quat_rot_between_vec(u, v):
  """Provides quaternion that describes rotation from u to v.

  Args:
    u: jnp.ndarray (3) (x,y,z)
    v: jnp.ndarray (4) (x,y,z)

  Returns:
    A quaternion describing rotation from u to v.
  """
  angle = jnp.arccos(
      jnp.vdot(u, v) / jnp.sqrt(jnp.dot(u, u)) / jnp.sqrt(jnp.dot(v, v)))
  direction = jnp.cross(u, v)
  direction /= jnp.sqrt(jnp.vdot(direction, direction))
  direction *= jnp.sin(angle / 2.)
  return jnp.array([
      jnp.cos(angle / 2.),
      direction[0],
      direction[1],
      direction[2],
  ])


def quat_rot_between_xy_vec(u, v):
  """Provides quaternion that describes rotation from u to v around the z axis.

  Args:
    u: jnp.ndarray (2) (x,y)
    v: jnp.ndarray (2) (x,y)

  Returns:
    A quaternion describing rotation from u to v.
  """
  angle = jnp.arctan2(v[1], v[0]) - jnp.arctan2(u[1], u[0])
  direction = jnp.array([0., 0., 1.])
  direction *= jnp.sin(angle / 2.)
  return jnp.array([
      jnp.cos(jnp.abs(angle) / 2.),
      direction[0],
      direction[1],
      direction[2],
  ])


# TODO: discern from jax team why this is faster than linalg.det
def det(r1, r2, r3):
  """Calculates the determinant of a 3x3 matrix with rows as args.

  Args:
    r1: First row
    r2: Second row
    r3: Third row

  Returns:
    Determinant of matrix [r1,r2,r3].  Functionally equivalent to
    jnp.linalg.det(jnp.array([r1,r2,r3])), but jits 10x faster for large batch.

  """
  return r1[0] * r2[1] * r3[2] + r1[1] * r2[2] * r3[0] + r1[2] * r2[0] * r3[
      1] - r1[2] * r2[1] * r3[0] - r1[0] * r2[2] * r3[1] - r1[1] * r2[0] * r3[2]


def relative_quat(quat_1, quat_2):
  """Calculates the relative quaternion rotation from quat_1 to quat_2.

  Args:
    quat_1: First quaterion
    quat_2: Second quaternion

  Returns:
    A quaternion that is equivalent to the rotation achieved by inverting quat_1
    and then applying quat_2.
  """
  inv_quat_1 = inv_quat(quat_1)
  new_quat = qmult(quat_2, inv_quat_1)
  return new_quat


def inv_quat(q):
  """Calculates the inverse of quaternion q.

  Args:
    q: Some quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * jnp.array([1., -1., -1., -1.])
