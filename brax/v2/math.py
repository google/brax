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

"""Some useful math functions."""

from typing import Tuple, Optional, Union

import jax
from jax import custom_jvp
from jax import numpy as jp
import numpy as np


def rotate(vec: jp.ndarray, quat: jp.ndarray):
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


def ang_to_quat(ang: jp.ndarray):
  """Converts angular velocity to a quaternion.

  Args:
    ang: (3,) angular velocity

  Returns:
    A rotation quaternion.
  """
  return jp.array([0, ang[0], ang[1], ang[2]])


def quat_mul(u: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
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


def quat_inv(q: jp.ndarray) -> jp.ndarray:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * jp.array([1, -1, -1, -1])


def quat_rot_axis(axis: jp.ndarray, angle: jp.ndarray) -> jp.ndarray:
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


def quat_to_3x3(q: jp.ndarray) -> jp.ndarray:
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


def quat_from_3x3(m: jp.ndarray) -> jp.ndarray:
  """Converts 3x3 rotation matrix to quaternion."""
  w = jp.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
  x = (m[2][1] - m[1][2]) / (w * 4)
  y = (m[0][2] - m[2][0]) / (w * 4)
  z = (m[1][0] - m[0][1]) / (w * 4)
  return jp.array([w, x, y, z])


def quat_mul_ang(q: jp.ndarray, ang: jp.ndarray) -> jp.ndarray:
  """Multiplies a quat by an angular velocity."""
  mat = jp.array([
      [-q[2], q[1], -q[0], q[3]],
      [-q[3], q[0], q[1], -q[2]],
      [-q[0], -q[3], q[2], q[1]],
  ])
  return jp.dot(ang, mat)


def signed_angle(
    axis: jp.ndarray, ref_p: jp.ndarray, ref_c: jp.ndarray
) -> jp.ndarray:
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
def safe_arccos(x: jp.ndarray) -> jp.ndarray:
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
def safe_arcsin(x: jp.ndarray) -> jp.ndarray:
  """Trigonometric inverse sine, element-wise with safety clipping in grad."""
  return jp.arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
  (x,) = primal
  (x_dot,) = tangent
  primal_out = safe_arccos(x)
  tangent_out = x_dot / jp.sqrt(1.0 - jp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
  return primal_out, tangent_out


def inv_3x3(m) -> jp.ndarray:
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


def orthogonals(n: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
  """Given a normal n of a plane, returns orthogonal p, q on the plane."""
  n_sqr = n[2] * n[2]
  a = n[1] * n[1] + jp.where(n_sqr > 0.5, n_sqr, n[0] * n[0])
  k = jp.sqrt(a)

  p_gt = jp.array([0, -n[2], n[1]])
  p_lt = jp.array([-n[1], n[0], n[1]])
  p = jp.where(a > 0.5, p_gt, p_lt) * k

  # set q = n x p
  q_gt = jp.array([a * k, -n[0] * p[2], n[0] * p[1]])
  q_lt = jp.array([-n[2] * p[1], n[2] * p[0], a * k])
  q = jp.where(a > 0.5, q_gt, q_lt)

  return p, q


def solve_pgs(a: jp.ndarray, b: jp.ndarray, num_iters: int) -> jp.ndarray:
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
    a: jp.ndarray, a_inv: jp.ndarray, tol: float = 1e-12, maxiter: int = 10
) -> jp.ndarray:
  """Use Newton-Schulz iteration to solve ``A^-1``.

  Args:
    a: 2D array to invert
    a_inv: approximate solution to A^-1
    tol: tolerance for convergance, ``norm(residual) <= tol``.
    maxiter: maximum number of iterations.  Iteration will stop after maxiter
      steps even if the specified tolerance has not been achieved.

  Returns:
    A^-1 inverted matrix
  """

  def cond_fn(value):
    # TODO: test whether it's better for convergence to check
    # ||I - Xn @ A || > tol - this is certainly faster and results seem OK
    _, k, err = value
    return (err > tol) & (k < maxiter)

  def body_fn(value):
    a_inv, k, _ = value
    a_inv_new = 2 * a_inv - a_inv @ a.T @ a_inv
    return a_inv_new, k + 1, jp.linalg.norm(a_inv_new - a_inv)

  # ensure ||I - X0 @ A|| < 1, in order to guarantee convergence
  r0 = jp.eye(a.shape[0]) - a @ a_inv
  a_inv = jp.where(jp.linalg.norm(r0) > 1, 0.5 * a.T / jp.trace(a @ a.T), a_inv)

  a_inv, *_ = jax.lax.while_loop(cond_fn, body_fn, (a_inv, 0, 1.0))

  return a_inv


def safe_norm(
    x: jp.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> jp.ndarray:
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
  x = jp.where(is_zero, jp.ones_like(x), x)
  n = jp.linalg.norm(x, axis=axis)
  n = jp.where(is_zero, 0.0, n)
  return n


def normalize(
    x: jp.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> Tuple[jp.ndarray, jp.ndarray]:
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


def from_to(v1: jp.ndarray, v2: jp.ndarray) -> jp.ndarray:
  """Calculates the quaternion that rotates unit vector v1 to unit vector v2."""
  rot = jp.append(1.0 + v1.dot(v2), jp.cross(v1, v2))

  # handle v1.dot(v2) == -1
  x, y = jp.array([1.0, 0.0, 0.0]), jp.array([0.0, 1.0, 0.0])
  rot_axis = jp.where(
      jp.abs(v1.dot(x)) > 0.99, jp.cross(v1, y), jp.cross(v1, x)
  )
  rot = jp.where(rot[0] < 1e-6, quat_rot_axis(rot_axis, jp.pi), rot)

  return rot / jp.linalg.norm(rot)


def euler_to_quat(v: jp.ndarray) -> jp.ndarray:
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = jp.cos(v * jp.pi / 360)
  s1, s2, s3 = jp.sin(v * jp.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return jp.array([w, x, y, z])


def quat_to_euler(q: jp.ndarray) -> jp.ndarray:
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
