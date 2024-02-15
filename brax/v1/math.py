# Copyright 2024 The Brax Authors.
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

# pylint:disable=redefined-builtin
"""Common math functions used by multiple brax modules."""
from typing import Tuple

from brax.v1 import jumpy as jp

Vector3 = jp.ndarray
Quaternion = jp.ndarray


def rotate(vec: Vector3, quat: Quaternion) -> Vector3:
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise AssertionError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (jp.dot(u, vec) * u) + (s * s - jp.dot(u, u)) * vec
  r = r + 2 * s * jp.cross(u, vec)
  return r


def inv_rotate(vec: Vector3, quat: Quaternion) -> Vector3:
  """Rotates a vector by the inverse of a unit quaternion.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    A vector rotated by quat^{-1}
  """
  return rotate(vec, quat_inv(quat))


def ang_to_quat(ang: Vector3) -> Quaternion:
  """Converts angular velocity to a quaternion.

  Args:
    ang: (3,) angular velocity

  Returns:
    A rotation quaternion.
  """
  return jp.array([0, ang[0], ang[1], ang[2]])


def euler_to_quat(v: Vector3) -> Quaternion:
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = jp.cos(v * jp.pi / 360)
  s1, s2, s3 = jp.sin(v * jp.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return jp.array([w, x, y, z])


def quat_to_euler(q: Quaternion) -> Vector3:
  """Converts quaternions to euler rotations in radians."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''

  z = jp.arctan2(-2 * q[1] * q[2] + 2 * q[0] * q[3],
                 q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2])
  # TODO: Investigate why quaternions go so big we need to clip.
  y = jp.safe_arcsin(jp.clip(2 * q[1] * q[3] + 2 * q[0] * q[2], -1., 1.))  # pytype: disable=wrong-arg-types  # jax-ndarray
  x = jp.arctan2(-2 * q[2] * q[3] + 2 * q[0] * q[1],
                 q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0])

  return jp.array([x, y, z])


def quat_to_axis_angle(q: Quaternion) -> Tuple[Vector3, jp.ndarray]:
  """Returns the axis-angle representation of a quaternion.

  Args:
    q: (4,) a quaternion

  Returns:
    The angle of axis-angle of this quaternion, in the range [-pi, pi].
  """
  # TODO: replace with more accurate safe function
  # avoid the singularity at 0:
  epsilon = 1e-10
  # safety 1e-10 jitter added because both sqrt and arctan2 have bad gradients
  denom = jp.safe_norm(q[1:])
  angle = 2. * jp.arctan2(
      jp.sqrt(epsilon + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), q[0])
  angle += jp.where(angle > jp.pi, x=-2 * jp.pi, y=0)  # pytype: disable=wrong-arg-types  # jax-ndarray
  angle += jp.where(angle < -jp.pi, x=2 * jp.pi, y=0)  # pytype: disable=wrong-arg-types  # jax-ndarray
  scale = jp.where(denom == 0., 0., 1. / denom)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return q[1:] * scale, angle


def signed_angle(axis: Vector3, ref_p: Vector3, ref_c: Vector3) -> jp.ndarray:
  """Calculates the signed angle between two vectors along an axis.

  Args:
    axis: (3,) common axis around which to calculate change in angle
    ref_p: (3,) vector pointing at 0-degrees offset in the parent's frame
    ref_c: (3,) vector pointing at 0-degrees offset in the child's frame

  Returns:
    The signed angle between two parts.
  """
  return jp.arctan2(jp.dot(jp.cross(ref_p, ref_c), axis), jp.dot(ref_p, ref_c))


def quat_mul(u: Quaternion, v: Quaternion) -> Quaternion:
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return jp.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


def vec_quat_mul(u: Vector3, v: Quaternion) -> Quaternion:
  """Multiplies a vector and a quaternion.

  This is a convenience method for multiplying two quaternions when
  one of the quaternions has a 0-value w-part, i.e.:
  quat_mul([0.,a,b,c], [d,e,f,g])

  It is slightly more efficient than constructing a 0-w-part quaternion
  from the vector.

  Args:
    u: (3,) vector representation of the quaternion (0.,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return jp.array([
      -u[0] * v[1] - u[1] * v[2] - u[2] * v[3],
      u[0] * v[0] + u[1] * v[3] - u[2] * v[2],
      -u[0] * v[3] + u[1] * v[0] + u[2] * v[1],
      u[0] * v[2] - u[1] * v[1] + u[2] * v[0],
  ])


def quat_rot_axis(axis: Vector3, angle: jp.ndarray) -> Quaternion:
  """Provides a quaternion that describes rotating around axis v by angle.

  Args:
    axis: (3,) axis (x,y,z)
    angle: () float angle to rotate by

  Returns:
    A quaternion that rotates around v by angle
  """
  qx = axis[0] * jp.sin(angle / 2)
  qy = axis[1] * jp.sin(angle / 2)
  qz = axis[2] * jp.sin(angle / 2)
  qw = jp.cos(angle / 2)
  return jp.array([qw, qx, qy, qz])


def quat_inv(q: Quaternion) -> Quaternion:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * jp.array([1, -1, -1, -1])


def relative_quat(q1: Quaternion, q2: Quaternion) -> Quaternion:
  """Returns the relative quaternion from q1 to q2."""
  return quat_mul(q2, quat_inv(q1))


def normalize(v: Vector3, epsilon=1e-6) -> Vector3:
  """Normalizes a vector."""
  return v / (epsilon + jp.safe_norm(v))
