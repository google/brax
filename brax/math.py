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

"""Some useful math functions."""

from typing import Tuple, Optional, Union

import jax
from jax import custom_jvp
from jax import numpy as jp
import numpy as np


def rotate(vec: jax.Array, quat: jax.Array):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (jp.dot(u, vec) * u) + (s * s - jp.dot(u, u)) * vec
  r = r + 2 * s * jp.cross(u, vec)
  return r


def inv_rotate(vec: jax.Array, quat: jax.Array):
  """Rotates a vector vec by an inverted unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by the inverse of quat.
  """
  return rotate(vec, quat_inv(quat))


def rotate_np(vec: np.ndarray, quat: np.ndarray):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
  r = r + 2 * s * np.cross(u, vec)
  return r


def ang_to_quat(ang: jax.Array):
  """Converts angular velocity to a quaternion.

  Args:
    ang: (3,) angular velocity

  Returns:
    A rotation quaternion.
  """
  return jp.array([0, ang[0], ang[1], ang[2]])


def quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
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


def quat_mul_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return np.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


def quat_inv(q: jax.Array) -> jax.Array:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * jp.array([1, -1, -1, -1])


def quat_rot_axis(axis: jax.Array, angle: jax.Array) -> jax.Array:
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


def quat_to_3x3(q: jax.Array) -> jax.Array:
  """Converts quaternion to 3x3 rotation matrix."""
  d = jp.dot(q, q)
  w, x, y, z = q
  s = 2 / d
  xs, ys, zs = x * s, y * s, z * s
  wx, wy, wz = w * xs, w * ys, w * zs
  xx, xy, xz = x * xs, x * ys, x * zs
  yy, yz, zz = y * ys, y * zs, z * zs

  return jp.array([
      jp.array([1 - (yy + zz), xy - wz, xz + wy]),
      jp.array([xy + wz, 1 - (xx + zz), yz - wx]),
      jp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
  ])


def quat_mul_ang(q: jax.Array, ang: jax.Array) -> jax.Array:
  """Multiplies a quat by an angular velocity."""
  mat = jp.array([
      [-q[2], q[1], -q[0], q[3]],
      [-q[3], q[0], q[1], -q[2]],
      [-q[0], -q[3], q[2], q[1]],
  ])
  return jp.dot(ang, mat)


def signed_angle(
    axis: jax.Array, ref_p: jax.Array, ref_c: jax.Array
) -> jax.Array:
  """Calculates the signed angle between two vectors along an axis.

  Args:
    axis: (3,) common axis around which to calculate change in angle
    ref_p: (3,) vector pointing at 0-degrees offset in the parent's frame
    ref_c: (3,) vector pointing at 0-degrees offset in the child's frame

  Returns:
    The signed angle between two parts.
  """
  return jp.arctan2(jp.dot(jp.cross(ref_p, ref_c), axis), jp.dot(ref_p, ref_c))


@custom_jvp
def safe_arccos(x: jax.Array) -> jax.Array:
  """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
  return jp.arccos(x)


@safe_arccos.defjvp
def _safe_arccos_jvp(primal, tangent):
  (x,) = primal
  (x_dot,) = tangent
  primal_out = safe_arccos(x)
  tangent_out = -x_dot / jp.sqrt(1.0 - jp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
  return primal_out, tangent_out


@custom_jvp
def safe_arcsin(x: jax.Array) -> jax.Array:
  """Trigonometric inverse sine, element-wise with safety clipping in grad."""
  return jp.arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
  (x,) = primal
  (x_dot,) = tangent
  primal_out = safe_arcsin(x)
  tangent_out = x_dot / jp.sqrt(1.0 - jp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
  return primal_out, tangent_out


def inv_3x3(m) -> jax.Array:
  """Inverse specialized for 3x3 matrices."""
  det = jp.linalg.det(m)
  adjugate = jp.array([
      [
          m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1],
          m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2],
          m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1],
      ],
      [
          m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2],
          m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
          m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2],
      ],
      [
          m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0],
          m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1],
          m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0],
      ],
  ])
  return adjugate / (det + 1e-10)


def orthogonals(a: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Returns orthogonal vectors `b` and `c`, given a normal vector `a`."""
  y, z = jp.array([0, 1, 0]), jp.array([0, 0, 1])
  b = jp.where((-0.5 < a[1]) & (a[1] < 0.5), y, z)
  b = b - a * a.dot(b)
  # make b a normal vector. however if a is a zero vector, zero b as well.
  b = normalize(b)[0] * jp.any(a)
  return b, jp.cross(a, b)


def solve_pgs(a: jax.Array, b: jax.Array, num_iters: int) -> jax.Array:
  """Projected Gauss-Seidel solver for a MLCP defined by matrix A and vector b.
  """
  num_rows = b.shape[0]
  x = jp.zeros((num_rows,))

  def get_x(x, xs):
    i, a_i, b_i = xs
    residual = b_i + jp.dot(a_i, x)
    x_i = x[i] - residual / a_i[i]
    x_i = jp.maximum(x_i, 0.0)
    x = x.at[i].set(x_i)

    return x, None

  # TODO: turn this into a scan
  for _ in range(num_iters):
    x, _ = jax.lax.scan(get_x, x, (jp.arange(num_rows), a, b))

  return x


def inv_approximate(
    a: jax.Array, a_inv: jax.Array, num_iter: int = 10
) -> jax.Array:
  """Use Newton-Schulz iteration to solve ``A^-1``.

  Args:
    a: 2D array to invert
    a_inv: approximate solution to A^-1
    num_iter: number of iterations

  Returns:
    A^-1 inverted matrix
  """

  def body_fn(carry, _):
    a_inv, r, err = carry
    a_inv_next = a_inv @ (np.eye(a.shape[0]) + r)
    r_next = np.eye(a.shape[0]) - a @ a_inv_next
    err_next = safe_norm(r_next)
    a_inv_next = jp.where(err_next < err, a_inv_next, a_inv)
    return (a_inv_next, r_next, err_next), None

  # ensure ||I - X0 @ A|| < 1, in order to guarantee convergence
  r0 = jp.eye(a.shape[0]) - a @ a_inv
  a_inv = jp.where(safe_norm(r0) > 1, 0.5 * a.T / jp.trace(a @ a.T), a_inv)
  (a_inv, _, _), _ = jax.lax.scan(body_fn, (a_inv, r0, 1.0), None, num_iter)

  return a_inv


def safe_norm(
    x: jax.Array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> jax.Array:
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.

  Avoids a poorly defined gradient for jnp.linal.norm(0) see
  https://github.com/google/jax/issues/3058 for details
  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    Norm of the array x.
  """

  is_zero = jp.allclose(x, 0.0)
  # temporarily swap x with ones if is_zero, then swap back
  x = x + is_zero * 1.0
  n = jp.linalg.norm(x) * (1.0 - is_zero)

  return n


def normalize(
    x: jax.Array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> Tuple[jax.Array, jax.Array]:
  """Normalizes an array.

  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    A tuple of (normalized array x, the norm).
  """
  norm = safe_norm(x, axis=axis)
  n = x / (norm + 1e-6 * (norm == 0.0))
  return n, norm


def from_to(v1: jax.Array, v2: jax.Array) -> jax.Array:
  """Calculates the quaternion that rotates unit vector v1 to unit vector v2."""
  xyz = jp.cross(v1, v2)
  w = 1.0 + jp.dot(v1, v2)
  rnd = jax.random.uniform(jax.random.PRNGKey(0), (3,))
  v1_o = rnd - jp.dot(rnd, v1) * v1
  xyz = jp.where(w < 1e-6, v1_o, xyz)
  rot = jp.append(w, xyz)
  return rot / jp.linalg.norm(rot)


def euler_to_quat(v: jax.Array) -> jax.Array:
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = jp.cos(v * jp.pi / 360)
  s1, s2, s3 = jp.sin(v * jp.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return jp.array([w, x, y, z])


def quat_to_euler(q: jax.Array) -> jax.Array:
  """Converts quaternions to euler rotations in radians."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''

  z = jp.arctan2(
      -2 * q[1] * q[2] + 2 * q[0] * q[3],
      q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2],
  )
  # TODO: Investigate why quaternions go so big we need to clip.
  y = safe_arcsin(jp.clip(2 * q[1] * q[3] + 2 * q[0] * q[2], -1.0, 1.0))
  x = jp.arctan2(
      -2 * q[2] * q[3] + 2 * q[0] * q[1],
      q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0],
  )

  return jp.array([x, y, z])


def vec_quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
  """Multiplies a vector u and a quaternion v.

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


def relative_quat(q1: jax.Array, q2: jax.Array) -> jax.Array:
  """Returns the relative quaternion from q1 to q2."""
  return quat_mul(q2, quat_inv(q1))
