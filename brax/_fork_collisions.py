# explore_bench fork of brax: register convex collision pairs mjx omits.
"""Adds (CYLINDER, BOX) to mjx's collision dispatch table.

brax does not implement contact detection itself -- ``brax/contact.py`` calls
``mjx.collision``, so brax's supported geom pairs ARE mjx's.  mjx dispatches on
an explicit ``_COLLISION_FUNC`` table, and while it carries a generic
``convex_convex`` GJK/EPA routine (used for BOX-MESH and MESH-MESH) and
already handles CYLINDER against SPHERE, CAPSULE, PLANE, ELLIPSOID and
CYLINDER, the CYLINDER-BOX entry is simply absent.  Both shapes are convex
primitives that mjx already converts to ``ConvexInfo``, so the generic routine
applies unchanged; the gap is registration, not geometry.

Without this, `hibachi/l1-retrieve` -- whose spatula and rails are cylinders
resting on box surfaces -- raises
``(mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented``.

Registered on import of ``brax.contact``.  Idempotent, and it never replaces a
pair mjx already defines, so a future mujoco release that adds its own
CYLINDER-BOX implementation silently wins.
"""

import warnings

_APPLIED = False

#: pairs this fork adds -> the mjx routine to use for them. Every one is a
#: convex/convex combination among shapes mjx already converts to ConvexInfo
#: and already collides against OTHER convex shapes; only these specific
#: registrations are absent. Keys follow mjx's ascending-GeomType convention
#: (PLANE 0, HFIELD 1, SPHERE 2, CAPSULE 3, ELLIPSOID 4, CYLINDER 5, BOX 6,
#: MESH 7).
_FORK_PAIRS = (
    ('CYLINDER', 'BOX', 'cylinder_box'),       # hibachi: tools on the griddle
    ('ELLIPSOID', 'BOX', 'ellipsoid_box'),     # hibachi: plates vs gripper pads
)


def _make_box_sdf_colliders():
  """Cylinder-box and ellipsoid-box through mjx's SDF collision path.

  mjx collides primitive pairs (cylinder-cylinder, ellipsoid-cylinder, ...) by
  gradient-descending the CLEARANCE between two analytic signed distance
  functions (`collision_sdf`). It ships SDFs for plane, sphere, capsule,
  ellipsoid and cylinder -- but not for a box, which is why every pair
  involving a box falls back to the mesh-face routine and a primitive/box pair
  has nowhere to go. The box SDF is the standard exact one, so the existing
  machinery covers these pairs once it exists.
  """
  import functools
  import jax
  from jax import numpy as jp
  from mujoco.mjx._src import collision_sdf as sdf
  from mujoco.mjx._src import math as mjx_math

  def _box(pos, size):
    q = jp.abs(pos) - size
    return (jp.linalg.norm(jp.maximum(q, 0.0))
            + jp.minimum(jp.max(q), 0.0))

  @sdf.collider(ncon=4)
  def cylinder_box(c, b):
    # four seeds around the separating direction, as cylinder_cylinder does:
    # a single contact point cannot hold a cylinder resting on a face.
    basis = mjx_math.make_frame(b.pos - c.pos)
    mid = 0.5 * (c.pos + b.pos)
    r = jp.maximum(c.size[0], jp.max(b.size))
    x0 = jp.array([
        mid + r * basis[1],
        mid + r * basis[2],
        mid - r * basis[1],
        mid - r * basis[2],
    ])
    optim_ = functools.partial(sdf._optim, sdf._cylinder, _box, c, b)
    return jax.vmap(optim_)(x0)

  @sdf.collider(ncon=4)
  def ellipsoid_box(e, b):
    basis = mjx_math.make_frame(b.pos - e.pos)
    mid = 0.5 * (e.pos + b.pos)
    r = jp.maximum(jp.max(e.size), jp.max(b.size))
    x0 = jp.array([
        mid + r * basis[1],
        mid + r * basis[2],
        mid - r * basis[1],
        mid - r * basis[2],
    ])
    optim_ = functools.partial(sdf._optim, sdf._ellipsoid, _box, e, b)
    return jax.vmap(optim_)(x0)

  return {'cylinder_box': cylinder_box, 'ellipsoid_box': ellipsoid_box}


def register() -> list:
  """Adds the missing pairs to mjx's table. Returns what was added."""
  global _APPLIED
  if _APPLIED:
    return []
  added = []
  try:
    from mujoco.mjx._src import collision_driver as cd
    from mujoco.mjx._src import collision_convex as cc
    from mujoco.mjx._src.types import GeomType
  except ImportError as e:  # pragma: no cover
    warnings.warn(f'explore_bench fork: cannot patch mjx collisions: {e}')
    return []

  _FORK_IMPLS = _make_box_sdf_colliders()
  for a, b, fname in _FORK_PAIRS:
    key = (getattr(GeomType, a), getattr(GeomType, b))
    if key in cd._COLLISION_FUNC:  # pylint: disable=protected-access
      continue
    fn = _FORK_IMPLS.get(fname) or getattr(cc, fname, None)
    if fn is None:  # pragma: no cover
      warnings.warn(f'explore_bench fork: mjx has no {fname}')
      continue
    cd._COLLISION_FUNC[key] = fn  # pylint: disable=protected-access
    added.append((a, b, fname))
  _APPLIED = True
  return added
