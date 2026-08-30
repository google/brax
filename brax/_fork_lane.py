# explore_bench fork of brax: LANE-FORM primitives for the positional solver.
"""Component-split (\"lane form\") spatial algebra.

Why this file exists
--------------------
A TPU vector register is 8 sublanes x 128 lanes, tiled from an array's last
two axes.  brax's maximal-coordinate code is written per-link and vmapped over
worlds by the caller, so every intermediate has shape ``(n_worlds, n_link, 3)``
or ``(n_worlds, n_link, 4)``.  For the dynmanip scenes that is
``(256, 25, 3)`` -> padded to ``(256, 32, 128)``: **1,048,576 register slots
holding 19,200 useful values, 1.8% occupancy.**  Carrying each component as
its own ``(n_worlds, n_link)`` array instead gives ``(256, 25)`` -> padded
``(256, 128)``, three of them, i.e. 98,304 slots -- **10.7x fewer**.

The second, larger effect is fusion.  ``math.rotate`` contains two
``jp.dot``s and a ``jp.cross`` over the trailing size-3 axis; under vmap each
becomes a ``dot_general``/reduce, which is a fusion barrier, so a chain of
rotations materialises a ``(256, 25, 3)`` buffer per step.  Written
componentwise the whole chain is elementwise multiply-add on ``(256, 25)`` and
XLA emits a single fused loop.

This is the same finding `_fork_narrowphase.py` recorded (218 ns/pair ->
4.7 ns/pair with identical outputs, and compile 50.4 s -> 7.0 s).

Conventions
-----------
* a 3-vector is a python tuple ``(x, y, z)`` of arrays with the link axis last
* a quaternion is ``(w, x, y, z)``
* a 3x3 matrix is a tuple of 9 arrays in row-major order
* ``vec3``/``quat``/``stack3``/``stack4`` cross the boundary to brax's packed
  ``(..., 3)`` / ``(..., 4)`` arrays.

Every routine here reproduces the arithmetic of its ``brax.math``
counterpart operation by operation, so the results agree to float32 round-off
(the only freedom taken is the summation order inside a 3-element ``jp.dot``).
"""

import os

import jax
from jax import numpy as jp
import numpy as np

#: A/B switch.  ``BRAX_FORK_LANE=0`` restores brax's original packed-vector
#: implementations (kept in-tree as ``*_vecform``) so any measurement or
#: physics comparison can be run both ways in the same tree.
ENABLED = os.environ.get('BRAX_FORK_LANE', '1') not in ('0', 'false', 'False')

#: `positional/joints.position_update` separately, because it is the one
#: rewrite whose benefit depends on the model: it still has to hand a PACKED
#: Transform to `_three_dof_joint_update` (the one piece not converted), so it
#: pays a pack/unpack at the boundary that the other rewrites do not.  On a
#: 25-link scene that is amortised; on a 10-link one it is not.
ENABLE_POS = os.environ.get('BRAX_FORK_LANE_POS', '1') not in (
    '0', 'false', 'False')


# ---------------------------------------------------------------- boundary --


def vec3(a):
  """(..., 3) packed -> 3-tuple of (...,)."""
  return (a[..., 0], a[..., 1], a[..., 2])


def quat(a):
  """(..., 4) packed -> 4-tuple of (...,)."""
  return (a[..., 0], a[..., 1], a[..., 2], a[..., 3])


def stack3(v):
  return jp.stack(v, axis=-1)


def stack4(q):
  return jp.stack(q, axis=-1)


def const3(a):
  """A numpy/jax (..., 3) constant -> 3-tuple, kept as-is."""
  return (a[..., 0], a[..., 1], a[..., 2])


# ------------------------------------------------------------ vector algebra --


def add3(a, b):
  return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub3(a, b):
  return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def neg3(a):
  return (-a[0], -a[1], -a[2])


def scale3(a, s):
  return (a[0] * s, a[1] * s, a[2] * s)


def dot3(a, b):
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
  return (
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  )


def safe_norm3(a):
  """``math.safe_norm`` on a 3-vector, guard and all.

  The guard is not cosmetic: ``safe_norm`` calls ``jp.allclose(x, 0.0)``,
  whose default ``atol`` is 1e-8, and reports a norm of EXACTLY zero for any
  vector shorter than that.  A plain ``sqrt(dot(a, a))`` would instead return
  the tiny norm and ``normalize`` would divide by it, turning a 1e-12 vector
  into a unit one.  These joints do produce such vectors (a line of nodes at a
  joint's singular configuration), so the branch has to be reproduced.
  """
  z = ((jp.abs(a[0]) <= 1e-8) & (jp.abs(a[1]) <= 1e-8)
       & (jp.abs(a[2]) <= 1e-8)).astype(a[0].dtype)
  b = (a[0] + z, a[1] + z, a[2] + z)
  return jp.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) * (1.0 - z)


def norm3(a):
  return jp.sqrt(dot3(a, a))


def normalize3(a):
  """Mirrors ``math.normalize`` on a 3-vector (including its zero guard)."""
  n = safe_norm3(a)
  d = 1.0 / (n + 1e-6 * (n == 0.0))
  return (a[0] * d, a[1] * d, a[2] * d), n


def matvec(m, v):
  """3x3 (as three row 3-tuples) times a 3-vector, in lane form.

  ``i_inv @ v`` is a ``dot_general`` on TPU, i.e. a DEFAULT-PRECISION matmul
  that rounds both operands to bfloat16.  The positional solver's inverse
  inertias reach 1e6 for the gripper's finger links, so this is not a
  rounding detail: measured on ``dynmanip/juggle-gripper``, brax's own
  ``com.inv_inertia`` on TPU is 7.4e-03 relative away from the same
  computation on CPU float32, while the lane form is 5.7e-07.
  """
  return (dot3(m[0], v), dot3(m[1], v), dot3(m[2], v))


def scale4(q, s):
  return (q[0] * s, q[1] * s, q[2] * s, q[3] * s)


def add4(a, b):
  return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


# -------------------------------------------------------- quaternion algebra --


def rotate(v, q):
  """``math.rotate(v, q)``, componentwise."""
  s, ux, uy, uz = q
  d = dot3((ux, uy, uz), v)
  n = s * s - (ux * ux + uy * uy + uz * uz)
  c = cross3((ux, uy, uz), v)
  s2 = 2 * s
  return (
      2 * (d * ux) + n * v[0] + s2 * c[0],
      2 * (d * uy) + n * v[1] + s2 * c[1],
      2 * (d * uz) + n * v[2] + s2 * c[2],
  )


def qinv(q):
  return (q[0], -q[1], -q[2], -q[3])


def inv_rotate(v, q):
  """``math.inv_rotate(v, q)``."""
  return rotate(v, qinv(q))


def qmul(u, v):
  """``math.quat_mul(u, v)``."""
  return (
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  )


def vec_quat_mul(u, v):
  """``math.vec_quat_mul(u, v)`` -- quaternion product of ``(0, u)`` with v."""
  return (
      -u[0] * v[1] - u[1] * v[2] - u[2] * v[3],
      u[0] * v[0] + u[1] * v[3] - u[2] * v[2],
      -u[0] * v[3] + u[1] * v[0] + u[2] * v[1],
      u[0] * v[2] - u[1] * v[1] + u[2] * v[0],
  )


def qnormalize(q):
  """``math.normalize`` on a quaternion."""
  n = jp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
  inv = 1.0 / (n + 1e-6 * (n == 0.0))
  return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv), n


# ------------------------------------------------------------------ gathers --


def parent_gather_index(link_parents):
  """Static index vector selecting each link's parent row, -1 -> appended row."""
  n = len(link_parents)
  return np.array([p if p >= 0 else n for p in link_parents], dtype=np.int32)


def parent_gather_matrix(link_parents):
  """One-hot ``(n_link, n_link + 1)`` selecting each link's parent row.

  brax writes this as ``x.concatenate(Transform.zero((1,))).take(parent_idx)``.
  ``link_parents`` is STATIC, so the same selection is a constant matrix.

  **This must be contracted at HIGHEST precision.**  A default-precision
  ``dot`` on TPU rounds BOTH operands to bfloat16, and while a one-hot matrix
  survives that exactly, the data does not: measured on
  ``dynmanip/juggle-gripper``, a default-precision one-hot gather of the link
  positions differs from an exact gather by 8.1e-03 -- three decimal digits
  thrown away for a pure data movement.  See ``GATHER``.
  """
  n = len(link_parents)
  m = np.zeros((n, n + 1), dtype=np.float32)
  for i, p in enumerate(link_parents):
    m[i, p if p >= 0 else n] = 1.0
  # kept as numpy: a device array built at trace time and cached across traces
  # leaks a tracer, and XLA folds a numpy constant into the graph anyway.
  return m


#: How the (static) parent selection is realised.  ``'index'`` is a constant
#: XLA gather -- exact, and no arithmetic at all.  ``'matmul'`` is the one-hot
#: contraction at HIGHEST precision, which is also exact but costs three MXU
#: passes.  Set with ``BRAX_FORK_LANE_GATHER``.
GATHER = os.environ.get('BRAX_FORK_LANE_GATHER', 'index')


def gather_parent(comp, sel, fill=0.0):
  """Select parents of a ``(..., n_link)`` component array.

  ``fill`` is the value the appended (root) row carries -- 0 for a position or
  velocity, 1 for a quaternion's w component.
  """
  pad = jp.full(comp.shape[:-1] + (1,), fill, comp.dtype)
  cat = jp.concatenate([comp, pad], axis=-1)
  if GATHER == 'matmul':
    return jp.matmul(cat, sel[1].T, precision=jax.lax.Precision.HIGHEST)
  return cat[..., sel[0]]


def parent_selector(link_parents):
  """Both forms of the selector, so ``GATHER`` can be flipped at runtime."""
  return (parent_gather_index(link_parents),
          parent_gather_matrix(link_parents))


def gather_many(comps, fills, sel):
  """One gather (or one matmul) for SEVERAL components sharing an index.

  In lane form a single ``take`` on an ``(n_link, 3)`` array would become
  three separate gathers, and a gather is exactly the op this whole rewrite is
  trying to avoid.  Concatenating the components along the lane axis first
  turns k gathers into one, whatever k is -- position_update needs 17 of them
  (a parent transform, its inverse inertia and its inverse mass) and gets one.
  """
  k = len(comps)
  n = comps[0].shape[-1]
  parts = [jp.concatenate([c, jp.full(c.shape[:-1] + (1,), f, c.dtype)], -1)
           for c, f in zip(comps, fills)]
  cat = jp.concatenate(parts, axis=-1)
  if GATHER == 'matmul':
    import jax.scipy  # noqa: F401  (keep the import local to the branch)
    big = np.zeros((k * n, k * (n + 1)), np.float32)
    for i in range(k):
      big[i * n:(i + 1) * n, i * (n + 1):(i + 1) * (n + 1)] = sel[1]
    g = jp.matmul(cat, big.T, precision=jax.lax.Precision.HIGHEST)
  else:
    idx = np.concatenate([sel[0] + i * (n + 1) for i in range(k)])
    g = cat[..., idx]
  return tuple(g[..., i * n:(i + 1) * n] for i in range(k))


def gather_parent3(v, sel):
  return tuple(gather_parent(c, sel) for c in v)


def gather_parent_quat(q, sel):
  return (gather_parent(q[0], sel, 1.0),
          gather_parent(q[1], sel),
          gather_parent(q[2], sel),
          gather_parent(q[3], sel))


def segment_matrix(link_parents):
  """Constant ``(n_link, n_link)`` matrix for ``segment_sum(x, parent_idx)``.

  ``s[j, i] = 1`` when link ``i``'s parent is ``j``.  Unlike the gather this
  one really is a sum (a link can have several children), so there is no exact
  index form -- it is contracted at HIGHEST precision for the same reason.
  """
  n = len(link_parents)
  m = np.zeros((n, n), dtype=np.float32)
  for i, p in enumerate(link_parents):
    if p >= 0:
      m[p, i] = 1.0
  return m


def segment_to_parent(comp, mat):
  """``segment_sum(comp, parent_idx, n_link)`` for a ``(..., n_link)`` array."""
  return jp.matmul(comp, mat.T, precision=jax.lax.Precision.HIGHEST)


def segment_to_parent_many(comps, mat):
  """``segment_to_parent`` for several components as one batched matmul."""
  x = jp.stack(comps, axis=-2)
  y = jp.matmul(x, mat.T, precision=jax.lax.Precision.HIGHEST)
  return tuple(y[..., i, :] for i in range(len(comps)))


def onehot_pick(comp, sel):
  """``comp[..., idx]`` for a static index vector ``idx`` (exact)."""
  return comp[..., sel]
