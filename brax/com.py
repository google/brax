# Copyright 2026 The Brax Authors.
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

"""Helper functions for physics calculations in maximal coordinates."""

# pylint:disable=g-multiple-import
from typing import Tuple

from brax import _fork_lane as lane
from brax import math
from brax.base import Motion, System, Transform
import jax
from jax import numpy as jp


def _shift(pos, rot_q, off):
  """Rigid shift of a frame by a link-frame offset, in lane form.

  ``Transform.create(pos=off)`` carries an IDENTITY rotation, so
  ``x.do(Transform.create(pos=off))`` is a full ``quat_mul`` against
  ``[1, 0, 0, 0]`` and the result's rotation is ``x.rot`` unchanged (every
  term of the product is either ``u_i * 1`` or ``u_j * 0``, so this is exact,
  not an approximation).  Likewise the companion
  ``Transform.create(pos=d).do(motion)`` rotates by ``quat_inv(identity)``,
  which is the identity map on a finite vector.  Dropping both leaves the
  arithmetic that is actually doing something.
  """
  r = lane.rotate(off, rot_q)
  return (pos[0] + r[0], pos[1] + r[1], pos[2] + r[2])


def _carry_velocity(d, ang, vel):
  """``Transform.create(pos=d).vmap().do(Motion(ang, vel))`` in lane form."""
  c = lane.cross3(d, ang)
  return ang, (vel[0] - c[0], vel[1] - c[1], vel[2] - c[2])


def from_world_vecform(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion]:
  """brax's original, kept as the reference implementation and for A/B."""
  x_i = x.vmap().do(Transform.create(pos=sys.link.inertia.transform.pos))
  xd_i = Transform.create(pos=x_i.pos - x.pos).vmap().do(xd)
  return x_i, xd_i


def to_world_vecform(
    sys: System, x_i: Transform, xd_i: Motion
) -> Tuple[Transform, Motion]:
  """brax's original, kept as the reference implementation and for A/B."""
  x = x_i.vmap().do(Transform.create(pos=-sys.link.inertia.transform.pos))
  xd = Transform.create(pos=x.pos - x_i.pos).vmap().do(xd_i)
  return x, xd


def inv_inertia_vecform(sys, x) -> jax.Array:
  """brax's original, kept as the reference implementation and for A/B."""

  @jax.vmap
  def inv_i(link_inertia, x_rot):
    ri = math.quat_mul(x_rot, link_inertia.transform.rot)
    i = link_inertia.i
    d = jp.diagonal(i)
    i = i - jp.diag(d) + jp.diag(d ** (1 - sys.spring_inertia_scale))
    i_inv_mx = _sym3_inv(i)
    i_rot_row = jax.vmap(math.rotate, in_axes=[0, None])(i_inv_mx, ri)
    i_rot_col = jax.vmap(math.rotate, in_axes=[0, None])(i_rot_row.T, ri)
    return i_rot_col

  return inv_i(sys.link.inertia, x.rot)


def from_world(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion]:
  """Converts link transform and motion from world frame to com frame."""
  if not lane.ENABLED:
    return from_world_vecform(sys, x, xd)
  off = lane.const3(sys.link.inertia.transform.pos)
  pos, rot_q = lane.vec3(x.pos), lane.quat(x.rot)
  pos_i = _shift(pos, rot_q, off)
  ang, vel = _carry_velocity(
      lane.sub3(pos_i, pos), lane.vec3(xd.ang), lane.vec3(xd.vel))
  return (Transform(pos=lane.stack3(pos_i), rot=x.rot),
          Motion(ang=xd.ang, vel=lane.stack3(vel)))


def to_world(
    sys: System, x_i: Transform, xd_i: Motion
) -> Tuple[Transform, Motion]:
  """Converts link transform and motion from com frame to world frame."""
  if not lane.ENABLED:
    return to_world_vecform(sys, x_i, xd_i)
  off = lane.neg3(lane.const3(sys.link.inertia.transform.pos))
  pos_i, rot_q = lane.vec3(x_i.pos), lane.quat(x_i.rot)
  pos = _shift(pos_i, rot_q, off)
  ang, vel = _carry_velocity(
      lane.sub3(pos, pos_i), lane.vec3(xd_i.ang), lane.vec3(xd_i.vel))
  return (Transform(pos=lane.stack3(pos), rot=x_i.rot),
          Motion(ang=xd_i.ang, vel=lane.stack3(vel)))


def _sym3_inv(m: jax.Array) -> jax.Array:
  """Inverse of a 3x3 symmetric positive-definite matrix, by the adjugate.

  explore_bench fork: brax inverted only `diagonal(i)`, which was exact while
  every link inertia was MuJoCo's principal-axis diagonal. Folding
  `dof_armature` in adds `armature * a a^T`, which is not diagonal unless the
  joint axis happens to be a principal axis, so the off-diagonal terms have to
  be carried. Closed-form rather than `jp.linalg.inv`: it is a handful of
  fused multiplies under vmap instead of an LU decomposition.
  """
  c00 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
  c01 = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
  c02 = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
  det = m[0, 0] * c00 + m[1, 0] * c01 + m[2, 0] * c02
  c11 = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
  c12 = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]
  c22 = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
  adj = jp.array([[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]])
  # a zero-inertia link keeps brax's previous behaviour of an infinite inverse
  return adj / jp.where(det == 0.0, 1e-30, det)

def _sym3_inv_batch(m: jax.Array) -> jax.Array:
  """``_sym3_inv`` for a stack of matrices, without a vmap.

  Same expressions in the same order; only the batching differs, so the result
  is bit-identical to ``jax.vmap(_sym3_inv)``.
  """
  g = lambda r, c: m[..., r, c]
  c00 = g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)
  c01 = g(0, 2) * g(2, 1) - g(0, 1) * g(2, 2)
  c02 = g(0, 1) * g(1, 2) - g(0, 2) * g(1, 1)
  det = g(0, 0) * c00 + g(1, 0) * c01 + g(2, 0) * c02
  c11 = g(0, 0) * g(2, 2) - g(0, 2) * g(2, 0)
  c12 = g(0, 2) * g(1, 0) - g(0, 0) * g(1, 2)
  c22 = g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0)
  adj = jp.stack([
      jp.stack([c00, c01, c02], axis=-1),
      jp.stack([c01, c11, c12], axis=-1),
      jp.stack([c02, c12, c22], axis=-1),
  ], axis=-2)
  return adj / jp.where(det == 0.0, 1e-30, det)[..., None, None]


def body_inv_inertia(sys) -> jax.Array:
  """The link-frame inverse inertia, which does not depend on the state.

  ``sys.link.inertia.i`` is a model constant, so its (symmetric) inverse is
  one too.  brax recomputed it inside a per-link vmap on every call, and
  ``inv_inertia`` is called three times per solver sweep.
  """
  i = sys.link.inertia.i
  d = jp.diagonal(i, axis1=-2, axis2=-1)
  eye = jp.eye(3)
  i = i - d[..., None] * eye + (d ** (1 - sys.spring_inertia_scale))[..., None] * eye
  return _sym3_inv_batch(i)


def inv_inertia_lane(sys, x_rot_q):
  """Lane-form world-frame inverse inertia: 9 ``(..., n_link)`` arrays.

  ``x_rot_q`` is the link rotation as a 4-tuple of components.  Returns the
  rows of ``R M R^T`` as three 3-tuples, exactly the quantity
  ``inv_inertia`` packs into an ``(n_link, 3, 3)`` array -- and the form every
  consumer in ``positional/joints.py`` actually wants, since all of them
  immediately contract it with a 3-vector.
  """
  m = body_inv_inertia(sys)
  ri = lane.qmul(x_rot_q, lane.quat(sys.link.inertia.transform.rot))
  # rows of M rotated by R, i.e. i_rot_row = M R^T
  rows = [lane.rotate((m[..., k, 0], m[..., k, 1], m[..., k, 2]), ri)
          for k in range(3)]
  # then rows of (M R^T)^T rotated by R, i.e. R M R^T
  return tuple(
      lane.rotate((rows[0][j], rows[1][j], rows[2][j]), ri) for j in range(3)
  )


def inv_inertia(sys, x) -> jax.Array:
  """Gets the inverse inertia at the center of mass in world frame."""
  if not lane.ENABLED:
    return inv_inertia_vecform(sys, x)
  cols = inv_inertia_lane(sys, lane.quat(x.rot))
  return jp.stack([lane.stack3(c) for c in cols], axis=-2)
