# explore_bench fork of brax: a batch-first collision narrowphase.
"""Hand-written, branch-free narrowphase for the geom-type pairs these scenes use.

Why this exists
---------------
Measured on ``dynmanip/juggle-gripper``, one v4 chip, N=256 worlds, per world
per substep::

    full substep                    762.41 us   100%
    contact.get -> mjx.collision    690.70 us    90.6%
    contact resolve (brax's own)      ~0          0%
    all other brax dynamics          10.46 us     1.4%

The scene has ~755 narrowphase pairs after culling, so mjx costs ~917 ns of
wall clock per pair-test.  That is not a pair-count problem -- halving the
pairs only halved the cost -- it is the SHAPE of mjx's collision code: thirteen
small per-group kernels, a generic convex/ConvexInfo machinery that materialises
8 verts x 6 faces x 12 edges per box, Sutherland-Hodgman polygon clipping with
per-plane gathers, and an op count in the thousands on tensors of a few hundred
rows.  On a TPU that is all fixed per-op overhead and HBM traffic on tiny
tensors.

This module implements the same collisions directly, with three rules:

1. **Closed form instead of enumeration.**  The support radius of a box along
   an axis is ``|n . R| . h`` -- no 8-vertex reduction.  The closest point on a
   box to a point is ``clip(p, -h, h)`` -- no 6-face scan.  A box-box
   separating-axis test is 15 axes of pure arithmetic.
2. **One fused computation per geom-type pair, shaped ``(n_pairs, ...)``.**  No
   Python loop over pairs, no ``lax.cond``, no data-dependent shapes.  The vmap
   over worlds then gives ``(n_worlds, n_pairs, ...)``.
3. **Fixed contact count per pair type**, identical to the ``ncon`` of the mjx
   routine it replaces, so the emitted ``Contact`` has exactly the same shape
   and the two paths are directly A/B-able.

Contract of a routine here: given batched geom poses/sizes it returns
``(dist, pos, frame)`` with shapes ``(n, ncon)``, ``(n, ncon, 3)``,
``(n, ncon, 3, 3)``; the caller flattens pair-major, which is what mjx's
``collider`` wrapper does with ``jp.concatenate``.  ``frame[0]`` is the contact
normal and points from ``geom1`` to ``geom2``, as mjx defines it.

The mjx path is NOT removed -- it is the correctness reference, and any pair
type not in ``_NARROWPHASE`` below still goes through it.  See
``.work/narrowphase_report.md`` for the per-pair-type validation against
``mujoco.mjx``'s own collision functions and for what is approximated.
"""

import os as _os
from typing import Tuple

import jax
from jax import numpy as jp
import numpy as np

# ---------------------------------------------------------------------------
# batched vector helpers.  Everything takes a leading pair axis (and whatever
# vmap adds in front of it) and never reduces over it.
# ---------------------------------------------------------------------------

_HUGE = 1e6


def _dot(a, b):
  return jp.sum(a * b, axis=-1)


def _norm(x):
  """|x| along the last axis, with a defined gradient at zero (as mjx's does)."""
  s = jp.sum(x * x, axis=-1)
  return jp.sqrt(jp.where(s == 0.0, 1.0, s)) * (s != 0.0)


def _normalize(x) -> Tuple[jax.Array, jax.Array]:
  n = _norm(x)
  return x / jp.expand_dims(n + 1e-6 * (n == 0.0), -1), n


def _cross(a, b):
  return jp.stack([
      a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
      a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
  ], axis=-1)


def _rot(m, v):
  """m @ v for m (..., 3, 3) and v (..., 3)."""
  return jp.sum(m * v[..., None, :], axis=-1)


def _rotT(m, v):
  """m.T @ v for m (..., 3, 3) and v (..., 3)."""
  return jp.sum(m * v[..., :, None], axis=-2)


def _rot_pts(m, v):
  """m @ v for m (n, 3, 3) and a stack of points v (n, k, 3)."""
  return jp.sum(m[..., None, :, :] * v[..., None, :], axis=-1)


def _rotT_pts(m, v):
  """m.T @ v for m (n, 3, 3) and a stack of points v (n, k, 3)."""
  return jp.sum(m[..., None, :, :] * v[..., :, None], axis=-2)


def _matmulT(a, b):
  """a.T @ b for (n, 3, 3) matrices."""
  return jp.einsum('nki,nkj->nij', a, b)


#: MEASURED (`.work/layout_probe.py`, one v4 chip, a (256, 578) grid): the same
#: argmax-then-select costs 1.494 ms as a `take_along_axis` and 0.077 ms as a
#: one-hot masked reduction -- **19x**.  A dynamic per-row index over a short
#: axis is a `gather` in HLO, which on TPU cannot stay in the vector unit: it
#: serialises into address arithmetic over a minor axis that is 3 or 2 elements
#: wide inside a 128-lane register.  The one-hot form is a multiply and a sum
#: over the same short axis, stays fused with the arithmetic that produced it,
#: and the 128-lane waste is then shared with everything around it.  So NOTHING
#: in this module indexes with a traced index; every selection goes through
#: these three.


def _select_many(x, idx, k):
  """x (..., k, d) selected by idx (..., m) along the k axis -> (..., m, d)."""
  oh = jax.nn.one_hot(idx, k, dtype=x.dtype)
  return jp.einsum('...mk,...kd->...md', oh, x)


def _take_many(x, idx):
  """x (..., k, d) selected by idx (..., m) along the k axis -> (..., m, d)."""
  return _select_many(x, idx, x.shape[-2])


def _take(x, idx):
  """x (..., k, d) selected by one index per row idx (...,) -> (..., d)."""
  oh = jax.nn.one_hot(idx, x.shape[-2], dtype=x.dtype)
  return jp.einsum('...k,...kd->...d', oh, x)


def _take1(x, idx):
  """Same, for x with no trailing vector axis: x (..., k) -> (...,)."""
  oh = jax.nn.one_hot(idx, x.shape[-1], dtype=x.dtype)
  return jp.sum(oh * x, axis=-1)


def _take1_many(x, idx, k=None):
  """x (..., k) selected by idx (..., m) -> (..., m)."""
  oh = jax.nn.one_hot(idx, k or x.shape[-1], dtype=x.dtype)
  return jp.einsum('...mk,...k->...m', oh, x)


def _take1_many_bool(x, idx, k=None):
  """Boolean version of `_take1_many`."""
  return _take1_many(x.astype(jp.float32), idx, k) > 0.5


def _make_frame(n):
  """mjx's `math.make_frame`, batched over the leading axes."""
  a, _ = _normalize(n)
  y = jp.array([0.0, 1.0, 0.0])
  z = jp.array([0.0, 0.0, 1.0])
  cond = ((a[..., 1] > -0.5) & (a[..., 1] < 0.5))[..., None]
  b = jp.where(cond, y, z)
  b = b - a * _dot(a, b)[..., None]
  b, _ = _normalize(b)
  b = b * jp.any(a != 0.0, axis=-1)[..., None]
  return jp.stack([a, b, _cross(a, b)], axis=-2)


def _closest_segment_point(a, b, pt):
  """mjx's `math.closest_segment_point`, batched."""
  ab = b - a
  t = _dot(pt - a, ab) / (_dot(ab, ab) + 1e-6)
  return a + jp.clip(t, 0.0, 1.0)[..., None] * ab


def _closest_segment_to_segment(a0, a1, b0, b1):
  """mjx's `math.closest_segment_to_segment_points`, batched."""
  dir_a, len_a = _normalize(a1 - a0)
  dir_b, len_b = _normalize(b1 - b0)
  half_a, half_b = len_a * 0.5, len_b * 0.5
  a_mid = a0 + dir_a * half_a[..., None]
  b_mid = b0 + dir_b * half_b[..., None]
  trans = a_mid - b_mid

  ab = _dot(dir_a, dir_b)
  at = _dot(dir_a, trans)
  bt = _dot(dir_b, trans)
  denom = 1.0 - ab * ab
  ta = (-at + ab * bt) / (denom + 1e-6)
  tb = bt + ta * ab
  ta = jp.clip(ta, -half_a, half_a)
  tb = jp.clip(tb, -half_b, half_b)
  best_a = a_mid + dir_a * ta[..., None]
  best_b = b_mid + dir_b * tb[..., None]

  # mjx re-projects to resolve the both-clipped case; keep it bit-for-bit.
  new_a = _closest_segment_point(a0, a1, best_b)
  d1 = _dot(best_b - new_a, best_b - new_a)
  new_b = _closest_segment_point(b0, b1, best_a)
  d2 = _dot(best_a - new_b, best_a - new_b)
  swap = (d1 < d2)[..., None]
  return jp.where(swap, new_a, best_a), jp.where(swap, best_b, new_b)


def _sphere_sphere_core(p1, r1, p2, r2):
  n, d = _normalize(p2 - p1)
  n = jp.where((d == 0.0)[..., None], jp.array([1.0, 0.0, 0.0]), n)
  dist = d - (r1 + r2)
  pos = p1 + n * (r1 + dist * 0.5)[..., None]
  return dist, pos, n


# corners of the unit box, in mjx's `mesh.box` order (itertools.product).
_BOX_SIGNS = np.array(list(np.ndindex(2, 2, 2)), dtype=np.float64) * 2.0 - 1.0


def _manifold_points3(poly, mask, poly_norm):
  """mjx's `_manifold_points` (max-area 4-subset), batched. poly (n, k, 3)."""
  dm = jp.where(mask, 0.0, -_HUGE)
  a_i = jp.argmax(dm, axis=-1)
  a = _take(poly, a_i)
  ap = a[..., None, :] - poly
  b_i = jp.argmax(jp.sum(ap * ap, axis=-1) + dm, axis=-1)
  b = _take(poly, b_i)
  ab = _cross(poly_norm, a - b)
  c_i = jp.argmax(jp.abs(_dot(ap, ab[..., None, :])) + dm, axis=-1)
  c = _take(poly, c_i)
  ac = _cross(poly_norm, a - c)
  bc = _cross(poly_norm, b - c)
  bp = b[..., None, :] - poly
  d_i = jp.argmax(
      jp.abs(_dot(bp, bc[..., None, :])) + dm
      + jp.abs(_dot(ap, ac[..., None, :])) + dm,
      axis=-1,
  )
  return jp.stack([a_i, b_i, c_i, d_i], axis=-1)


def _manifold_points2(pts, mask):
  """Same selection rule in 2D (poly_norm = +z, so cross() is a perp)."""
  perp = lambda v: jp.stack([-v[..., 1], v[..., 0]], axis=-1)
  dm = jp.where(mask, 0.0, -_HUGE)
  a_i = jp.argmax(dm, axis=-1)
  a = _take(pts, a_i)
  ap = a[..., None, :] - pts
  b_i = jp.argmax(jp.sum(ap * ap, axis=-1) + dm, axis=-1)
  b = _take(pts, b_i)
  ab = perp(a - b)
  c_i = jp.argmax(jp.abs(_dot(ap, ab[..., None, :])) + dm, axis=-1)
  c = _take(pts, c_i)
  ac, bc = perp(a - c), perp(b - c)
  bp = b[..., None, :] - pts
  d_i = jp.argmax(
      jp.abs(_dot(bp, bc[..., None, :])) + dm
      + jp.abs(_dot(ap, ac[..., None, :])) + dm,
      axis=-1,
  )
  return jp.stack([a_i, b_i, c_i, d_i], axis=-1)


def _first_occurrence(idx):
  """mjx's uniqueness mask: True where idx[i] has not appeared before."""
  eq = idx[..., :, None] == idx[..., None, :]
  return jp.sum(jp.tril(eq), axis=-1) == 1


# ---------------------------------------------------------------------------
# PLANE x *
# ---------------------------------------------------------------------------


def plane_sphere(p1, m1, s1, p2, m2, s2):
  """Signed distance along the plane normal.  Exact; matches mjx bitwise."""
  n = m1[..., :, 2]
  r = s2[..., 0]
  dist = _dot(p2 - p1, n) - r
  pos = p2 - n * (r + 0.5 * dist)[..., None]
  frame = _make_frame(n)
  return dist[..., None], pos[..., None, :], frame[..., None, :, :]


def plane_capsule(p1, m1, s1, p2, m2, s2):
  """Two plane-sphere tests at the capsule's two end spheres (mjx's rule)."""
  n = m1[..., :, 2]
  axis = m2[..., :, 2]
  b, b_norm = _normalize(axis - n * _dot(n, axis)[..., None])
  y = jp.array([0.0, 1.0, 0.0])
  z = jp.array([0.0, 0.0, 1.0])
  alt = jp.where((((n[..., 1] > -0.5) & (n[..., 1] < 0.5)))[..., None], y, z)
  b = jp.where((b_norm < 0.5)[..., None], alt, b)
  frame = jp.stack([n, b, _cross(n, b)], axis=-2)

  seg = axis * s2[..., 1:2]
  r = s2[..., 0]
  d0 = _dot(p2 + seg - p1, n) - r
  d1 = _dot(p2 - seg - p1, n) - r
  q0 = p2 + seg - n * (r + 0.5 * d0)[..., None]
  q1 = p2 - seg - n * (r + 0.5 * d1)[..., None]
  dist = jp.stack([d0, d1], axis=-1)
  pos = jp.stack([q0, q1], axis=-2)
  return dist, pos, jp.stack([frame, frame], axis=-3)


def plane_box(p1, m1, s1, p2, m2, s2):
  """Plane against a box: mjx's `plane_convex`, specialised to the 8 corners."""
  signs = jp.asarray(_BOX_SIGNS, dtype=p1.dtype)          # (8, 3)
  vert = signs * s2[..., None, :]                          # (n, 8, 3)
  n_w = m1[..., :, 2]
  plane_local = _rotT(m2, p1 - p2)
  n_local = _rotT(m2, n_w)
  support = _dot(plane_local[..., None, :] - vert, n_local[..., None, :])
  keep = support > jp.maximum(0.0, jp.max(support, axis=-1, keepdims=True) - 1e-3)
  idx = _manifold_points3(vert, keep, n_local)             # (n, 4)

  pos_local = _take_many(vert, idx)                        # (n, 4, 3)
  pos = p2[..., None, :] + _rot_pts(m2, pos_local)
  sup = _take1_many(support, idx)
  dist = jp.where(_first_occurrence(idx), -sup, 1.0)
  pos = pos - 0.5 * dist[..., None] * n_w[..., None, :]
  frame = _make_frame(n_w)[..., None, :, :]
  return dist, pos, jp.broadcast_to(frame, dist.shape + (3, 3))


# ---------------------------------------------------------------------------
# SPHERE x *
# ---------------------------------------------------------------------------


def sphere_sphere(p1, m1, s1, p2, m2, s2):
  dist, pos, n = _sphere_sphere_core(p1, s1[..., 0], p2, s2[..., 0])
  return dist[..., None], pos[..., None, :], _make_frame(n)[..., None, :, :]


def sphere_capsule(p1, m1, s1, p2, m2, s2):
  seg = m2[..., :, 2] * s2[..., 1:2]
  pt = _closest_segment_point(p2 - seg, p2 + seg, p1)
  dist, pos, n = _sphere_sphere_core(p1, s1[..., 0], pt, s2[..., 0])
  return dist[..., None], pos[..., None, :], _make_frame(n)[..., None, :, :]


def sphere_box(p1, m1, s1, p2, m2, s2):
  """Sphere against a box, closed form.

  The closest point on a box to a point is ``clip(p, -h, h)`` in the box frame;
  mjx reaches the same point through a six-face support scan and a polygon
  edge projection.  When the sphere centre is inside the box the closest exit
  is the nearest face, which is the smallest of ``h - |c|``.
  """
  h = s2
  r = s1[..., 0]
  c = _rotT(m2, p1 - p2)                                   # sphere in box frame
  clamped = jp.clip(c, -h, h)
  delta = c - clamped
  d = _norm(delta)
  outside = d > 1e-9

  q = h - jp.abs(c)                                        # per-axis depth
  k = jp.argmin(q, axis=-1)
  pen = jp.min(q, axis=-1)
  ek = jax.nn.one_hot(k, 3, dtype=c.dtype)
  f_in = jp.where(_dot(c, ek)[..., None] < 0, -ek, ek)     # nearest face normal

  n_local = jp.where(outside[..., None], -delta / (d + ~outside)[..., None], -f_in)
  d_signed = jp.where(outside, d, -pen)
  pt = jp.where(outside[..., None], clamped, c + f_in * pen[..., None])
  spt = c + n_local * r[..., None]
  pos_local = (pt + spt) * 0.5

  # mjx reports dist=1 (rather than the true clearance) as soon as any face
  # slab separates the two, so match that: it is the value the solver sees.
  sep = jp.max(jp.abs(c) - h, axis=-1) - r
  dist = jp.where(sep >= 0, 1.0, d_signed - r)
  pos = p2 + _rot(m2, pos_local)
  n = _rot(m2, n_local)
  return dist[..., None], pos[..., None, :], _make_frame(n)[..., None, :, :]


# ---------------------------------------------------------------------------
# CAPSULE x *
# ---------------------------------------------------------------------------


def capsule_capsule(p1, m1, s1, p2, m2, s2):
  seg1 = m1[..., :, 2] * s1[..., 1:2]
  seg2 = m2[..., :, 2] * s2[..., 1:2]
  pt1, pt2 = _closest_segment_to_segment(p1 - seg1, p1 + seg1,
                                         p2 - seg2, p2 + seg2)
  dist, pos, n = _sphere_sphere_core(pt1, s1[..., 0], pt2, s2[..., 0])
  return dist[..., None], pos[..., None, :], _make_frame(n)[..., None, :, :]


_CB_ITERS = 8


def _segment_box_closest(a0, a1, h):
  """Closest points between a segment and an origin-centred box, batched.

  The distance from a point to a convex set is convex, and the segment is
  affine, so ``t -> |clip(s(t)) - s(t)|`` is convex on [0, 1]: alternating
  projection (clip to the box, then project back onto the segment) is a
  contraction to the true closest pair whenever the two are disjoint.  A fixed
  unrolled iteration count keeps it branch-free; there is no ``while``.
  """
  dvec = a1 - a0
  dd = _dot(dvec, dvec)
  t = jp.full(a0.shape[:-1], 0.5, dtype=a0.dtype)
  for _ in range(_CB_ITERS):
    s = a0 + t[..., None] * dvec
    q = jp.clip(s, -h, h)
    t = jp.clip(_dot(q - a0, dvec) / (dd + 1e-12), 0.0, 1.0)
  s = a0 + t[..., None] * dvec
  q = jp.clip(s, -h, h)
  return s, q


def capsule_box(p1, m1, s1, p2, m2, s2):
  """Capsule against a box: a face manifold plus an exact edge/corner witness.

  Face branch: pick the box face with the largest support against the capsule's
  inner segment (mjx's rule), then clip the segment to that face's rectangle.
  Because the face is axis aligned in the box frame the clip is a 1-D interval
  intersection, not a four-plane polygon clip with gathers.

  Edge branch: mjx scans all twelve box edges for the shallow case.  Here the
  closest point between the segment and the box is computed directly, which is
  the exact witness for every non-penetrating configuration, edge and corner
  alike.
  """
  h = s2
  r = s1[..., 0]
  axis = _rotT(m2, m1[..., :, 2])
  centre = _rotT(m2, p1 - p2)
  seg = axis * s1[..., 1:2]
  a0, a1 = centre - seg, centre + seg

  lo = jp.minimum(a0, a1)
  hi = jp.maximum(a0, a1)
  # support of the six signed faces: max over faces is the separating value.
  sup = jp.concatenate([lo - h - r[..., None], -hi - h - r[..., None]], axis=-1)
  best = jp.argmax(sup, axis=-1)
  has_support = jp.max(sup, axis=-1) < 0
  k = best % 3
  sigma = jp.where(best < 3, 1.0, -1.0)
  ek = jax.nn.one_hot(k, 3, dtype=h.dtype)
  f = sigma[..., None] * ek                                # outward face normal
  hk = _dot(h, ek)

  # clip the segment parameter to the two tangential slabs of that face
  dvec = a1 - a0
  tang = 1.0 - ek
  inv = 1.0 / jp.where(dvec == 0.0, 1.0, dvec)
  t_a = (-h - a0) * inv
  t_b = (h - a0) * inv
  t_lo = jp.minimum(t_a, t_b)
  t_hi = jp.maximum(t_a, t_b)
  inside0 = jp.abs(a0) <= h
  t_lo = jp.where(dvec == 0.0, jp.where(inside0, -_HUGE, _HUGE), t_lo)
  t_hi = jp.where(dvec == 0.0, jp.where(inside0, _HUGE, -_HUGE), t_hi)
  t_lo = jp.where(tang > 0, t_lo, -_HUGE)
  t_hi = jp.where(tang > 0, t_hi, _HUGE)
  tmin = jp.clip(jp.max(t_lo, axis=-1), 0.0, 1.0)
  tmax = jp.clip(jp.min(t_hi, axis=-1), 0.0, 1.0)
  valid = (jp.min(t_hi, axis=-1) >= jp.max(t_lo, axis=-1)) & has_support

  cp = a0[..., None, :] + jp.stack([tmin, tmax], axis=-1)[..., None] * dvec[..., None, :]
  cap_pts = cp - f[..., None, :] * r[..., None, None]
  face_pen = hk[..., None] + r[..., None] - sigma[..., None] * _dot(cp, ek[..., None, :])
  face_pts = cap_pts + f[..., None, :] * face_pen[..., None]
  face_pos = (cap_pts + face_pts) * 0.5
  face_pen = jp.where(valid[..., None], face_pen, -1.0)
  face_n = jp.broadcast_to(-f[..., None, :], face_pos.shape)

  # shallow edge / corner contact from the true segment-to-box witness
  sp, qp = _segment_box_closest(a0, a1, h)
  e_vec = qp - sp
  e_axis, e_d = _normalize(e_vec)
  e_pen = r - e_d
  e_pos = (qp + sp + e_axis * r[..., None]) * 0.5
  parallel = jp.abs(_dot(e_axis, f)) > 0.99
  min_face_pen = jp.min(face_pen, axis=-1)
  use_edge = (
      (e_pen > 0)
      & (e_d > 1e-9)
      & jp.where(min_face_pen > 0, e_pen < min_face_pen, True)
      & ~parallel
  )

  pen = jp.where(use_edge[..., None],
                 jp.stack([e_pen, -jp.ones_like(e_pen)], axis=-1), face_pen)
  pos = jp.where(use_edge[..., None, None],
                 jp.stack([e_pos, face_pos[..., 1, :]], axis=-2), face_pos)
  n = jp.where(use_edge[..., None, None],
               jp.stack([e_axis, face_n[..., 1, :]], axis=-2), face_n)

  pos = p2[..., None, :] + _rot_pts(m2, pos)
  n = _rot_pts(m2, n)
  return -pen, pos, _make_frame(n)


# ---------------------------------------------------------------------------
# BOX x BOX -- the dominant group
# ---------------------------------------------------------------------------


def _box_box_axes(c, absc, t, h1, h2):
  """The 15 separating axes and their signed separations, in box-1's frame.

  ``sep`` is the true gap along each axis (negative = overlap), which needs the
  axis norm; Gottschalk's classic test skips the normalisation because it only
  needs the sign, but the contact depth and the axis ranking both need metric
  values.  Support radii are closed form -- ``|n . R| . h`` -- so no vertex
  enumeration happens anywhere here.
  """
  n = t.shape[0]
  eye = jp.broadcast_to(jp.eye(3, dtype=t.dtype), (n, 3, 3))
  b_ax = jp.swapaxes(c, -1, -2)                            # rows are B's axes
  # cross(e_i, b_j) for the nine (i, j)
  ei = jp.repeat(eye, 3, axis=1)                           # (n, 9, 3)
  bj = jp.tile(b_ax, (1, 3, 1))                            # (n, 9, 3)
  cross = _cross(ei, bj)
  axes = jp.concatenate([eye, b_ax, cross], axis=1)        # (n, 15, 3)

  anorm = _norm(axes)
  unit = axes / (anorm + (anorm == 0.0))[..., None]
  ra = jp.sum(jp.abs(unit) * h1[..., None, :], axis=-1)
  # |unit @ C| . h2 : the support radius of box 2 along each axis
  ub = jp.einsum('nkj,nji->nki', unit, c)
  rb = jp.sum(jp.abs(ub) * h2[..., None, :], axis=-1)
  proj = _dot(unit, t[..., None, :])
  sep = jp.abs(proj) - (ra + rb)
  degenerate = anorm < 1e-6
  sep = jp.where(degenerate, -_HUGE, sep)
  return unit, sep, proj


def box_box_vecform(p1, m1, s1, p2, m2, s2):
  """Separating-axis test over 15 axes, then a clipped face manifold.

  Face case: the reference face is axis aligned in the reference box's frame,
  so the incident quad is clipped against a rectangle.  Twelve candidate
  contact points -- the two Liang-Barsky endpoints of each of the four incident
  edges (which recovers both the incident corners inside the rectangle and
  every edge crossing) plus the four rectangle corners inside the quad -- are
  reduced to four by mjx's own maximal-area rule.

  Edge case: instead of mjx's "reuse the deepest clipped face point", the two
  witness edges are read straight off the signs of the separating axis and
  their closest points are computed exactly.  One contact, three inert.
  """
  n = p1.shape[0]
  c = _matmulT(m1, m2)                                     # box2 axes in box1
  t = _rotT(m1, p2 - p1)
  absc = jp.abs(c)
  unit, sep, proj = _box_box_axes(c, absc, t, s1, s2)

  best = jp.argmax(sep, axis=-1)
  sep_best = jp.max(sep, axis=-1)
  ax = _take(unit, best)
  sgn = jp.where(_take1(proj, best) < 0, -1.0, 1.0)
  n_ab = ax * sgn[..., None]                               # box1 -> box2, box1 frame
  n_world = _rot(m1, n_ab)

  best_face = jp.argmax(sep[:, :6], axis=-1)
  face_ax = _take(unit[:, :6], best_face)
  is_edge = (best >= 6) & (jp.abs(_dot(face_ax, ax)) < 0.99)

  # ---- reference / incident assignment (the owner of the best face axis) ----
  swap = (best_face >= 3)[..., None]
  pr = jp.where(swap, p2, p1)
  pi = jp.where(swap, p1, p2)
  hr = jp.where(swap, s2, s1)
  hi = jp.where(swap, s1, s2)
  rr = jp.where(swap[..., None], m2, m1)
  ri = jp.where(swap[..., None], m1, m2)
  # outward normal of the reference box, pointing at the incident box
  nout_w = jp.where(swap, -n_world, n_world)

  nout_r = _rotT(rr, nout_w)
  k = jp.argmax(jp.abs(nout_r), axis=-1)
  ek = jax.nn.one_hot(k, 3, dtype=t.dtype)
  eu = jax.nn.one_hot((k + 1) % 3, 3, dtype=t.dtype)
  ev = jax.nn.one_hot((k + 2) % 3, 3, dtype=t.dtype)
  sigma = jp.where(_dot(nout_r, ek) < 0, -1.0, 1.0)
  hk, hu, hv = _dot(hr, ek), _dot(hr, eu), _dot(hr, ev)

  # incident face: the one most anti-aligned with the reference normal
  g = _rotT(ri, nout_w)
  mm = jp.argmax(jp.abs(g), axis=-1)
  em = jax.nn.one_hot(mm, 3, dtype=t.dtype)
  ea = jax.nn.one_hot((mm + 1) % 3, 3, dtype=t.dtype)
  eb = jax.nn.one_hot((mm + 2) % 3, 3, dtype=t.dtype)
  tau = jp.where(_dot(g, em) < 0, 1.0, -1.0)
  ic = em * (tau * _dot(hi, em))[..., None]
  da = ea * _dot(hi, ea)[..., None]
  db = eb * _dot(hi, eb)[..., None]
  quad_i = jp.stack([ic + da + db, ic - da + db, ic - da - db, ic + da - db],
                    axis=-2)                               # (n, 4, 3) CCW cycle

  # into the reference frame
  mrel = _matmulT(rr, ri)
  orel = _rotT(rr, pi - pr)
  quad = _rot_pts(mrel, quad_i) + orel[..., None, :]
  qx = _dot(quad, eu[..., None, :])
  qy = _dot(quad, ev[..., None, :])

  # incident plane in (x, y) -> z, for the depth of an arbitrary candidate
  ni = _rot(mrel, em * tau[..., None])
  nx, ny, nz = _dot(ni, eu), _dot(ni, ev), _dot(ni, ek)
  q0 = quad[:, 0]
  c0 = nx * qx[:, 0] + ny * qy[:, 0] + nz * _dot(q0, ek)
  nz_safe = nz + 1e-9 * (jp.abs(nz) < 1e-9) * jp.where(nz < 0, -1.0, 1.0)

  # ---- candidate set -----------------------------------------------------
  # (a) each incident edge, clipped to the rectangle (Liang-Barsky).
  ex0 = jp.stack([qx, qy], axis=-1)                        # (n, 4, 2)
  ex1 = jp.roll(ex0, -1, axis=1)
  ed = ex1 - ex0
  lim = jp.stack([hu, hv], axis=-1)[..., None, :]
  inv = 1.0 / jp.where(ed == 0.0, 1.0, ed)
  ta_ = (-lim - ex0) * inv
  tb_ = (lim - ex0) * inv
  tlo = jp.minimum(ta_, tb_)
  thi = jp.maximum(ta_, tb_)
  in0 = jp.abs(ex0) <= lim
  tlo = jp.where(ed == 0.0, jp.where(in0, -_HUGE, _HUGE), tlo)
  thi = jp.where(ed == 0.0, jp.where(in0, _HUGE, -_HUGE), thi)
  tmin = jp.max(tlo, axis=-1)
  tmax = jp.min(thi, axis=-1)
  ok = (tmax >= tmin) & (jp.minimum(tmax, 1.0) >= jp.maximum(tmin, 0.0))
  tmin = jp.clip(tmin, 0.0, 1.0)
  tmax = jp.clip(tmax, 0.0, 1.0)
  cand_a = jp.concatenate([
      ex0 + tmin[..., None] * ed,
      ex0 + tmax[..., None] * ed,
  ], axis=1)                                               # (n, 8, 2)
  mask_a = jp.concatenate([ok, ok], axis=1)

  # (b) rectangle corners that fall inside the incident quad.
  cx = jp.stack([hu, -hu, -hu, hu], axis=-1)
  cy = jp.stack([hv, hv, -hv, -hv], axis=-1)
  cand_b = jp.stack([cx, cy], axis=-1)                     # (n, 4, 2)
  side = (ed[..., None, 0] * (cy[..., None, :] - ex0[..., None, 1])
          - ed[..., None, 1] * (cx[..., None, :] - ex0[..., None, 0]))
  mask_b = jp.all(side >= 0, axis=1) | jp.all(side <= 0, axis=1)

  cand = jp.concatenate([cand_a, cand_b], axis=1)          # (n, 12, 2)
  mask = jp.concatenate([mask_a, mask_b], axis=1)
  z = (c0[..., None] - nx[..., None] * cand[..., 0]
       - ny[..., None] * cand[..., 1]) / nz_safe[..., None]
  pen = hk[..., None] - sigma[..., None] * z
  mask = mask & (pen > 1e-6)

  idx = _manifold_points2(cand, mask)
  sel = _take_many(cand, idx)                              # (n, 4, 2)
  sel_pen = _take1_many(pen, idx)
  keep = _first_occurrence(idx) & _take1_many_bool(mask, idx)
  face_dist = jp.where(keep, -sel_pen, 1.0)
  face_local = (sel[..., 0:1] * eu[..., None, :]
                + sel[..., 1:2] * ev[..., None, :]
                + (sigma * hk)[..., None, None] * ek[..., None, :])
  face_pos = pr[..., None, :] + _rot_pts(rr, face_local)

  # ---- exact edge-edge witness ------------------------------------------
  # the witness edge of box 1 runs along the axis whose cross product produced
  # the separating axis; the other two coordinates sit at the extreme corner in
  # the direction of the normal.
  i_ax = (best - 6) // 3
  j_ax = (best - 6) % 3
  i_ax = jp.clip(i_ax, 0, 2)
  j_ax = jp.clip(j_ax, 0, 2)
  eia = jax.nn.one_hot(i_ax, 3, dtype=t.dtype)
  eja = jax.nn.one_hot(j_ax, 3, dtype=t.dtype)
  s_a = jp.where(n_ab < 0, -1.0, 1.0) * (1.0 - eia)
  corner_a = s_a * s1
  dir_a = eia * _dot(s1, eia)[..., None]
  cb = _rot(jp.swapaxes(c, -1, -2), n_ab)                  # normal in box2 axes
  s_b = jp.where(cb < 0, 1.0, -1.0) * (1.0 - eja)
  corner_b = t + _rot(c, s_b * s2)
  dir_b = _rot(c, eja * _dot(s2, eja)[..., None])
  wa, wb = _closest_segment_to_segment(corner_a - dir_a, corner_a + dir_a,
                                       corner_b - dir_b, corner_b + dir_b)
  edge_pos = p1 + _rot(m1, (wa + wb) * 0.5)

  one = jp.ones_like(sep_best)
  edge_dist = jp.stack([sep_best, one, one, one], axis=-1)
  edge_posn = jp.stack([edge_pos, edge_pos, edge_pos, edge_pos], axis=-2)

  dist = jp.where(is_edge[..., None], edge_dist, face_dist)
  pos = jp.where(is_edge[..., None, None], edge_posn, face_pos)
  frame = _make_frame(n_world)[..., None, :, :]
  return dist, pos, jp.broadcast_to(frame, dist.shape + (3, 3))


# ---------------------------------------------------------------------------
# BOX x BOX, lane form.  This is the routine that is actually used; the
# vector-form one above is kept because it is the readable statement of the
# same algorithm and the two are checked against each other.
# ---------------------------------------------------------------------------
#
# Why a second version exists.  A TPU vector register is 8 sublanes x 128
# lanes, and an array's last two axes are what get tiled into it.  A routine
# written per-pair and vmapped over worlds produces intermediates shaped
# ``(n_worlds, n_pairs, k, 3)``: the minor axis holds THREE useful values in a
# 128-lane register and the pair axis -- the only one that is actually wide --
# never reaches the lanes at all.  Every such tensor is ~42x larger in HBM than
# its contents and every op on it wastes ~97% of the vector unit.
#
# So this version never forms a trailing component axis.  A vector is three
# separate ``(n_pairs,)`` arrays, a matrix is nine; under the vmap each is
# exactly ``(n_worlds, n_pairs)``, which is the one shape a TPU likes.  Short
# axes (15 separating axes, 12 manifold candidates) appear only where a
# reduction needs them, and then as ``(..., k, n_pairs)`` so that the reduction
# runs over the SUBLANE axis while the pair axis keeps the lanes.
#
# MEASURED, one v4 chip, a (256 worlds, 578 pairs) grid -- see
# `.work/boxbox_bench.py` and `.work/narrowphase_report.md`.


def _lane_frame(nx, ny, nz):
  """mjx's `math.make_frame`, in lane form.  Returns 9 arrays, row-major."""
  inv = jax.lax.rsqrt(jp.maximum(nx * nx + ny * ny + nz * nz, 1e-24))
  ax, ay, az = nx * inv, ny * inv, nz * inv
  cond = (ay > -0.5) & (ay < 0.5)
  bx = jp.where(cond, 0.0, 0.0)
  by = jp.where(cond, 1.0, 0.0)
  bz = jp.where(cond, 0.0, 1.0)
  d = ax * bx + ay * by + az * bz
  bx, by, bz = bx - ax * d, by - ay * d, bz - az * d
  binv = jax.lax.rsqrt(jp.maximum(bx * bx + by * by + bz * bz, 1e-24))
  bx, by, bz = bx * binv, by * binv, bz * binv
  live = ((nx != 0.0) | (ny != 0.0) | (nz != 0.0)).astype(nx.dtype)
  bx, by, bz = bx * live, by * live, bz * live
  cx = ay * bz - az * by
  cy = az * bx - ax * bz
  cz = ax * by - ay * bx
  return (ax, ay, az, bx, by, bz, cx, cy, cz)


def _lane_pick(vals, idx):
  """Select one of `len(vals)` lane arrays per element, by a traced index.

  `vals` is a list of ``(..., n)`` arrays.  Stacking them on axis -2 puts the
  short axis in the sublanes and leaves `n` in the lanes, so the one-hot
  multiply-and-reduce below is a plain fused elementwise op -- not a gather,
  and not a batched matmul over a 100k-wide batch, both of which measured an
  order of magnitude worse.
  """
  k = len(vals)
  stack = jp.stack(vals, axis=-2)
  oh = jax.nn.one_hot(idx, k, axis=-2, dtype=stack.dtype)
  return jp.sum(oh * stack, axis=-2)


def _lane_argmax(vals):
  """argmax over a list of lane arrays -> (index, value)."""
  stack = jp.stack(vals, axis=-2)
  return jp.argmax(stack, axis=-2), jp.max(stack, axis=-2)


def _lane_seg_seg(a0, a1, b0, b1):
  """`math.closest_segment_to_segment_points`, lane form.  Points are triples."""
  def sub(u, v):
    return tuple(x - y for x, y in zip(u, v))

  def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

  def scale(u, s):
    return tuple(x * s for x in u)

  def add(u, v):
    return tuple(x + y for x, y in zip(u, v))

  def unit(u):
    n2 = dot(u, u)
    ln = jp.sqrt(n2)
    inv = jp.where(ln > 0.0, 1.0 / jp.where(ln > 0.0, ln, 1.0), 0.0)
    return scale(u, inv), ln

  def seg_pt(p0, p1, pt):
    ab = sub(p1, p0)
    t = dot(sub(pt, p0), ab) / (dot(ab, ab) + 1e-6)
    return add(p0, scale(ab, jp.clip(t, 0.0, 1.0)))

  dir_a, len_a = unit(sub(a1, a0))
  dir_b, len_b = unit(sub(b1, b0))
  half_a, half_b = len_a * 0.5, len_b * 0.5
  a_mid = add(a0, scale(dir_a, half_a))
  b_mid = add(b0, scale(dir_b, half_b))
  trans = sub(a_mid, b_mid)
  ab = dot(dir_a, dir_b)
  at = dot(dir_a, trans)
  bt = dot(dir_b, trans)
  ta = (-at + ab * bt) / (1.0 - ab * ab + 1e-6)
  tb = bt + ta * ab
  ta = jp.clip(ta, -half_a, half_a)
  tb = jp.clip(tb, -half_b, half_b)
  best_a = add(a_mid, scale(dir_a, ta))
  best_b = add(b_mid, scale(dir_b, tb))
  new_a = seg_pt(a0, a1, best_b)
  new_b = seg_pt(b0, b1, best_a)
  d1 = dot(sub(best_b, new_a), sub(best_b, new_a))
  d2 = dot(sub(best_a, new_b), sub(best_a, new_b))
  sw = d1 < d2
  wa = tuple(jp.where(sw, x, y) for x, y in zip(new_a, best_a))
  wb = tuple(jp.where(sw, x, y) for x, y in zip(best_b, new_b))
  return wa, wb


def box_box(p1, m1, s1, p2, m2, s2):
  """BOX x BOX: 15-axis separating-axis test plus a clipped face manifold.

  Semantically identical to `box_box_vecform` above -- same axes, same
  reference/incident assignment, same 12-candidate clip, same maximal-area
  4-subset, same exact edge-edge witness -- but written so that no intermediate
  ever carries a trailing size-3 axis.  See the note above this function.
  """
  f = p1.dtype
  # ---- unpack to lanes ---------------------------------------------------
  A = [[m1[..., i, j] for j in range(3)] for i in range(3)]
  B = [[m2[..., i, j] for j in range(3)] for i in range(3)]
  h1 = [s1[..., i] for i in range(3)]
  h2 = [s2[..., i] for i in range(3)]
  dp = [p2[..., i] - p1[..., i] for i in range(3)]
  P1 = [p1[..., i] for i in range(3)]
  P2 = [p2[..., i] for i in range(3)]

  # c[i][j] = (box1 axis i) . (box2 axis j);  t = (p2 - p1) in box1's frame
  c = [[sum(A[k][i] * B[k][j] for k in range(3)) for j in range(3)]
       for i in range(3)]
  ac = [[jp.abs(c[i][j]) for j in range(3)] for i in range(3)]
  t = [sum(A[k][i] * dp[k] for k in range(3)) for i in range(3)]

  one = jp.ones_like(t[0])
  zero = jp.zeros_like(t[0])

  # ---- the 15 separating axes -------------------------------------------
  sep, proj, axv = [], [], []          # axv[k] is a 3-tuple of lane arrays

  for i in range(3):                   # box 1's face normals
    rb = sum(ac[i][j] * h2[j] for j in range(3))
    proj.append(t[i])
    sep.append(jp.abs(t[i]) - (h1[i] + rb))
    axv.append(tuple(one if k == i else zero for k in range(3)))

  for j in range(3):                   # box 2's face normals
    ra = sum(ac[i][j] * h1[i] for i in range(3))
    pj = sum(t[i] * c[i][j] for i in range(3))
    proj.append(pj)
    sep.append(jp.abs(pj) - (ra + h2[j]))
    axv.append((c[0][j], c[1][j], c[2][j]))

  for i in range(3):                   # the nine edge-edge crossings
    i1, i2 = (i + 1) % 3, (i + 2) % 3
    for j in range(3):
      j1, j2 = (j + 1) % 3, (j + 2) % 3
      ra = h1[i1] * ac[i2][j] + h1[i2] * ac[i1][j]
      rb = h2[j1] * ac[i][j2] + h2[j2] * ac[i][j1]
      pj = t[i2] * c[i1][j] - t[i1] * c[i2][j]
      b = (c[0][j], c[1][j], c[2][j])
      # cross(e_i, b); |cross|^2 = 1 - (e_i . b)^2 because |b| = 1
      if i == 0:
        cr = (zero, -b[2], b[1])
      elif i == 1:
        cr = (b[2], zero, -b[0])
      else:
        cr = (-b[1], b[0], zero)
      n2 = jp.maximum(1.0 - c[i][j] * c[i][j], 0.0)
      degen = n2 < 1e-12
      inv = jax.lax.rsqrt(jp.where(degen, one, n2))
      proj.append(pj * inv)
      sep.append(jp.where(degen, -_HUGE, (jp.abs(pj) - ra - rb) * inv))
      axv.append(tuple(x * inv for x in cr))

  best, sep_best = _lane_argmax(sep)
  proj_best = _lane_pick(proj, best)
  ax = tuple(_lane_pick([axv[k][d] for k in range(15)], best) for d in range(3))
  sgn = jp.where(proj_best < 0, -one, one)
  n_ab = tuple(x * sgn for x in ax)                       # box1 -> box2, box1
  n_w = tuple(sum(A[k][i] * n_ab[i] for i in range(3)) for k in range(3))

  best_face, _ = _lane_argmax(sep[:6])
  fax = tuple(_lane_pick([axv[k][d] for k in range(6)], best_face)
              for d in range(3))
  is_edge = (best >= 6) & (jp.abs(sum(fax[d] * ax[d] for d in range(3))) < 0.99)

  # ---- reference / incident box -----------------------------------------
  swap = best_face >= 3
  pr = [jp.where(swap, P2[i], P1[i]) for i in range(3)]
  pi_ = [jp.where(swap, P1[i], P2[i]) for i in range(3)]
  hr = [jp.where(swap, h2[i], h1[i]) for i in range(3)]
  hi = [jp.where(swap, h1[i], h2[i]) for i in range(3)]
  rr = [[jp.where(swap, B[i][j], A[i][j]) for j in range(3)] for i in range(3)]
  ri = [[jp.where(swap, A[i][j], B[i][j]) for j in range(3)] for i in range(3)]
  nout = [jp.where(swap, -n_w[k], n_w[k]) for k in range(3)]

  nout_r = [sum(rr[k][i] * nout[k] for k in range(3)) for i in range(3)]
  kk, _ = _lane_argmax([jp.abs(x) for x in nout_r])
  ek = [(kk == i).astype(f) for i in range(3)]
  eu = [(((kk + 1) % 3) == i).astype(f) for i in range(3)]
  ev = [(((kk + 2) % 3) == i).astype(f) for i in range(3)]
  dotv = lambda u, v: u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
  sigma = jp.where(dotv(nout_r, ek) < 0, -one, one)
  hk, hu, hv = dotv(hr, ek), dotv(hr, eu), dotv(hr, ev)

  g = [sum(ri[k][i] * nout[k] for k in range(3)) for i in range(3)]
  mm, _ = _lane_argmax([jp.abs(x) for x in g])
  em = [(mm == i).astype(f) for i in range(3)]
  ea = [(((mm + 1) % 3) == i).astype(f) for i in range(3)]
  eb = [(((mm + 2) % 3) == i).astype(f) for i in range(3)]
  tau = jp.where(dotv(g, em) < 0, one, -one)
  ic = [em[i] * (tau * dotv(hi, em)) for i in range(3)]
  da = [ea[i] * dotv(hi, ea) for i in range(3)]
  db = [eb[i] * dotv(hi, eb) for i in range(3)]
  quad_i = [
      [ic[i] + da[i] + db[i] for i in range(3)],
      [ic[i] - da[i] + db[i] for i in range(3)],
      [ic[i] - da[i] - db[i] for i in range(3)],
      [ic[i] + da[i] - db[i] for i in range(3)],
  ]

  mrel = [[sum(rr[k][i] * ri[k][j] for k in range(3)) for j in range(3)]
          for i in range(3)]
  orel = [sum(rr[k][i] * (pi_[k] - pr[k]) for k in range(3)) for i in range(3)]
  quad = [[sum(mrel[i][j] * q[j] for j in range(3)) + orel[i]
           for i in range(3)] for q in quad_i]
  qx = [dotv(q, eu) for q in quad]
  qy = [dotv(q, ev) for q in quad]

  ni = [sum(mrel[i][j] * (em[j] * tau) for j in range(3)) for i in range(3)]
  nx, ny, nz = dotv(ni, eu), dotv(ni, ev), dotv(ni, ek)
  c0 = nx * qx[0] + ny * qy[0] + nz * dotv(quad[0], ek)
  nz_safe = nz + 1e-9 * (jp.abs(nz) < 1e-9) * jp.where(nz < 0, -one, one)

  # ---- 12 candidates: 8 Liang-Barsky endpoints + 4 rectangle corners -----
  cand_x, cand_y, mask = [], [], []
  lo, hi_ = [], []
  for q in range(4):
    q1 = (q + 1) % 4
    e = (qx[q1] - qx[q], qy[q1] - qy[q])
    o = (qx[q], qy[q])
    lim = (hu, hv)
    tlo, thi = [], []
    for d in range(2):
      degen = e[d] == 0.0
      inv = 1.0 / jp.where(degen, one, e[d])
      ta_ = (-lim[d] - o[d]) * inv
      tb_ = (lim[d] - o[d]) * inv
      inside = jp.abs(o[d]) <= lim[d]
      tlo.append(jp.where(degen, jp.where(inside, -_HUGE, _HUGE),
                          jp.minimum(ta_, tb_)))
      thi.append(jp.where(degen, jp.where(inside, _HUGE, -_HUGE),
                          jp.maximum(ta_, tb_)))
    tmin = jp.maximum(tlo[0], tlo[1])
    tmax = jp.minimum(thi[0], thi[1])
    ok = (tmax >= tmin) & (jp.minimum(tmax, 1.0) >= jp.maximum(tmin, 0.0))
    lo.append((jp.clip(tmin, 0.0, 1.0), e, o, ok))
    hi_.append(jp.clip(tmax, 0.0, 1.0))
  for q in range(4):
    tmin, e, o, ok = lo[q]
    cand_x.append(o[0] + tmin * e[0])
    cand_y.append(o[1] + tmin * e[1])
    mask.append(ok)
  for q in range(4):
    tmin, e, o, ok = lo[q]
    tmax = hi_[q]
    cand_x.append(o[0] + tmax * e[0])
    cand_y.append(o[1] + tmax * e[1])
    mask.append(ok)
  corners = ((hu, hv), (-hu, hv), (-hu, -hv), (hu, -hv))
  for (rx, ry) in corners:
    inside_pos, inside_neg = one > 0, one > 0
    for q in range(4):
      _, e, o, _ = lo[q]
      side = e[0] * (ry - o[1]) - e[1] * (rx - o[0])
      inside_pos = inside_pos & (side >= 0)
      inside_neg = inside_neg & (side <= 0)
    cand_x.append(rx)
    cand_y.append(ry)
    mask.append(inside_pos | inside_neg)

  z = [(c0 - nx * cand_x[k] - ny * cand_y[k]) / nz_safe for k in range(12)]
  pen = [hk - sigma * z[k] for k in range(12)]
  mask = [mask[k] & (pen[k] > 1e-6) for k in range(12)]

  # ---- maximal-area 4-subset (mjx's rule) --------------------------------
  dm = [jp.where(mask[k], 0.0, -_HUGE) for k in range(12)]
  ia, _ = _lane_argmax(dm)
  ax0 = _lane_pick(cand_x, ia)
  ay0 = _lane_pick(cand_y, ia)
  apx = [ax0 - cand_x[k] for k in range(12)]
  apy = [ay0 - cand_y[k] for k in range(12)]
  ib, _ = _lane_argmax([apx[k] * apx[k] + apy[k] * apy[k] + dm[k]
                        for k in range(12)])
  bx0 = _lane_pick(cand_x, ib)
  by0 = _lane_pick(cand_y, ib)
  abx, aby = -(ay0 - by0), (ax0 - bx0)            # perp(a - b)
  ic_, _ = _lane_argmax([jp.abs(apx[k] * abx + apy[k] * aby) + dm[k]
                         for k in range(12)])
  cx0 = _lane_pick(cand_x, ic_)
  cy0 = _lane_pick(cand_y, ic_)
  acx, acy = -(ay0 - cy0), (ax0 - cx0)
  bcx, bcy = -(by0 - cy0), (bx0 - cx0)
  bpx = [bx0 - cand_x[k] for k in range(12)]
  bpy = [by0 - cand_y[k] for k in range(12)]
  id_, _ = _lane_argmax([jp.abs(bpx[k] * bcx + bpy[k] * bcy) + dm[k]
                         + jp.abs(apx[k] * acx + apy[k] * acy) + dm[k]
                         for k in range(12)])
  idx = [ia, ib, ic_, id_]

  maskf = [mask[k].astype(f) for k in range(12)]
  sel_x = [_lane_pick(cand_x, i) for i in idx]
  sel_y = [_lane_pick(cand_y, i) for i in idx]
  sel_pen = [_lane_pick(pen, i) for i in idx]
  sel_msk = [_lane_pick(maskf, i) > 0.5 for i in idx]
  keep = []
  for j in range(4):
    k_ = sel_msk[j]
    for j2 in range(j):
      k_ = k_ & (idx[j] != idx[j2])
    keep.append(k_)

  face_dist = [jp.where(keep[j], -sel_pen[j], one) for j in range(4)]
  face_pos = []
  for j in range(4):
    loc = [sel_x[j] * eu[i] + sel_y[j] * ev[i] + sigma * hk * ek[i]
           for i in range(3)]
    face_pos.append([pr[k] + sum(rr[k][i] * loc[i] for i in range(3))
                     for k in range(3)])

  # ---- exact edge-edge witness ------------------------------------------
  i_ax = jp.clip((best - 6) // 3, 0, 2)
  j_ax = jp.clip((best - 6) % 3, 0, 2)
  eia = [(i_ax == i).astype(f) for i in range(3)]
  eja = [(j_ax == j).astype(f) for j in range(3)]
  corner_a = [jp.where(n_ab[i] < 0, -one, one) * (1.0 - eia[i]) * h1[i]
              for i in range(3)]
  dir_a = [eia[i] * dotv(h1, eia) for i in range(3)]
  cb = [sum(c[i][j] * n_ab[i] for i in range(3)) for j in range(3)]
  s_b = [jp.where(cb[j] < 0, one, -one) * (1.0 - eja[j]) * h2[j]
         for j in range(3)]
  corner_b = [t[i] + sum(c[i][j] * s_b[j] for j in range(3)) for i in range(3)]
  dbv = [eja[j] * dotv(h2, eja) for j in range(3)]
  dir_b = [sum(c[i][j] * dbv[j] for j in range(3)) for i in range(3)]
  wa, wb = _lane_seg_seg(
      tuple(corner_a[i] - dir_a[i] for i in range(3)),
      tuple(corner_a[i] + dir_a[i] for i in range(3)),
      tuple(corner_b[i] - dir_b[i] for i in range(3)),
      tuple(corner_b[i] + dir_b[i] for i in range(3)))
  mid = [(wa[i] + wb[i]) * 0.5 for i in range(3)]
  edge_pos = [P1[k] + sum(A[k][i] * mid[i] for i in range(3)) for k in range(3)]

  # ---- assemble ----------------------------------------------------------
  dist = jp.stack([
      jp.where(is_edge, sep_best if j == 0 else one, face_dist[j])
      for j in range(4)], axis=-1)
  pos = jp.stack([
      jp.stack([jp.where(is_edge, edge_pos[k], face_pos[j][k])
                for k in range(3)], axis=-1)
      for j in range(4)], axis=-2)
  fr = _lane_frame(*n_w)
  frame = jp.stack([jp.stack(fr[r * 3:r * 3 + 3], axis=-1) for r in range(3)],
                   axis=-2)
  return dist, pos, jp.broadcast_to(frame[..., None, :, :],
                                    dist.shape + (3, 3))


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

#: (geom type 1, geom type 2) -> (routine, ncon).  Keys follow mjx's ascending
#: GeomType convention (PLANE 0, SPHERE 2, CAPSULE 3, BOX 6).  ncon MUST equal
#: the ncon of the mjx routine being replaced, so both paths emit contact
#: buffers of identical shape and can be swapped without touching anything
#: downstream.
_NARROWPHASE = {
    (0, 2): (plane_sphere, 1),
    (0, 3): (plane_capsule, 2),
    (0, 6): (plane_box, 4),
    (2, 2): (sphere_sphere, 1),
    (2, 3): (sphere_capsule, 1),
    (2, 6): (sphere_box, 1),
    (3, 3): (capsule_capsule, 1),
    (3, 6): (capsule_box, 2),
    (6, 6): (box_box, 4),
}


#: Which pair types this module actually takes over.  Everything else -- and
#: everything in `_NARROWPHASE` but not in here -- still goes to mjx.
#:
#: Default is BOX x BOX ALONE, because that is what the measurement asks for:
#: BOX x BOX is 87% of the narrowphase on `dynmanip/juggle-gripper` and
#: CAPSULE x BOX is ~0%, so taking over anything else buys no measurable time
#: and only adds fidelity risk.  Two specific reasons to leave the rest off:
#:  * CAPSULE x BOX here is an 8-step bisection on the segment-box clearance,
#:    which disagrees with mjx on 0.27% of random configurations (see
#:    `.work/narrowphase_report.md`); mjx's exact routine costs nothing.
#:  * the primitive pairs (plane/sphere/capsule) reproduce mjx to 1e-16 but
#:    are already cheap, so switching them is churn without a payoff.
#: Set `ENABLED = frozenset(_NARROWPHASE)` to A/B the whole module.
ENABLED = frozenset({(6, 6)})

if _os.environ.get('BRAX_FORK_NARROWPHASE_ALL', '').lower() in ('1', 'true', 'on'):
  ENABLED = frozenset(_NARROWPHASE)   # A/B the whole module, CAPSULExBOX included


def supported(types) -> bool:
  key = (int(types[0]), int(types[1]))
  return key in _NARROWPHASE and key in ENABLED


def ncon(types) -> int:
  return _NARROWPHASE[(int(types[0]), int(types[1]))][1]


def run(sys, d, types, geom):
  """Runs one geom-type group.  Signature mirrors an mjx collision function.

  Args:
    sys: brax System / mjx Model
    d: mjx Data with geom_xpos / geom_xmat set
    types: the (t1, t2) geom type pair
    geom: (n_pairs, 2) geom indices

  Returns:
    dist (n_pairs * ncon,), pos (..., 3), frame (..., 3, 3), pair-major, which
    is the order mjx's `collider` produces and `jp.repeat(..., ncon)` expects.
  """
  fn, k = _NARROWPHASE[(int(types[0]), int(types[1]))]
  g1, g2 = geom.T
  dist, pos, frame = fn(
      d.geom_xpos[g1], d.geom_xmat[g1], sys.geom_size[g1],
      d.geom_xpos[g2], d.geom_xmat[g2], sys.geom_size[g2],
  )
  npair = dist.shape[0]
  return (dist.reshape(npair * k),
          pos.reshape(npair * k, 3),
          frame.reshape(npair * k, 3, 3))


def _aabb_clearance(sys, d, g1, g2):
  """Oriented-AABB clearance, the same broadphase metric the mjx path uses."""
  c1, c2 = sys.geom_aabb[g1, :3], sys.geom_aabb[g2, :3]
  h1, h2 = sys.geom_aabb[g1, 3:], sys.geom_aabb[g2, 3:]
  m1, m2 = d.geom_xmat[g1], d.geom_xmat[g2]
  p1 = d.geom_xpos[g1] + _rot(m1, c1)
  p2 = d.geom_xpos[g2] + _rot(m2, c2)
  e1 = _rot(jp.abs(m1), h1)
  e2 = _rot(jp.abs(m2), h2)
  return jp.max(jp.abs(p2 - p1) - (e1 + e2), axis=-1)


def collide(sys, d, max_geom_pairs: int = -1, max_contact_points: int = -1,
            by_type=()):
  """mjx's collision pass with this module's narrowphase in place of mjx's.

  Identical grouping, identical broadphase cull and identical contact
  parameters -- only the per-group narrowphase kernel differs, and any group
  whose geom-type pair is not implemented here still calls mjx's routine.  The
  emitted contact buffer has exactly the same layout and length as the mjx
  path, so this is a drop-in A/B.
  """
  from mujoco.mjx._src import collision_driver as cd

  if d._impl.ncon == 0:
    return d

  overrides = {(int(a), int(b)): int(v) for a, b, v in by_type}
  groups = cd._contact_groups(sys, d)  # pylint: disable=protected-access
  for key, contact in groups.items():
    t1, t2 = int(key.types[0]), int(key.types[1])
    budget = overrides.get((min(t1, t2), max(t1, t2)), max_geom_pairs)
    if (budget > -1
        and contact.geom.shape[0] > budget
        and not set(key.types) & cd._GEOM_NO_BROADPHASE):
      g1, g2 = contact.geom.T
      dist = _aabb_clearance(sys, d, g1, g2)
      _, idx = jax.lax.top_k(-dist, k=budget)
      contact = jax.tree_util.tree_map(lambda x, i=idx: x[i], contact)

    if supported(key.types):
      k = ncon(key.types)
      dist, pos, frame = run(sys, d, key.types, contact.geom)
    else:
      func = cd._COLLISION_FUNC[key.types]  # pylint: disable=protected-access
      k = func.ncon
      dist, pos, frame = func(sys, d, key, contact.geom)
    if k > 1:
      contact = jax.tree_util.tree_map(
          lambda x, r=k: jp.repeat(x, r, axis=0), contact)
    groups[key] = contact.replace(dist=dist, pos=pos, frame=frame)

  condim_groups = {}
  for key, contact in groups.items():
    condim_groups.setdefault(key.condim, []).append(contact)
  if max_contact_points > -1:
    for key, contacts in condim_groups.items():
      contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)
      if contact.geom.shape[0] > max_contact_points:
        _, idx = jax.lax.top_k(-contact.dist, k=max_contact_points)
        contact = jax.tree_util.tree_map(lambda x, i=idx: x[i], contact)
      condim_groups[key] = [contact]

  contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
  contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)
  return d.tree_replace({'_impl.contact': contact})
