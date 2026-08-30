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

  # mjx's own primitive SDFs all carry a hand-written gradient because the
  # analytic forms are non-differentiable exactly where the optimiser likes to
  # sit (`_cylinder` defines a custom_jvp for precisely this).  The box is no
  # different and the naive form is worse: INSIDE the box `max(q, 0)` is the
  # zero vector, `norm` of which has an undefined gradient, and `jax.grad`
  # returns NaN rather than raising.  `_optim` seeds its descent at
  # `mid +/- r * basis` with r the LARGER geom's extent, so against a big box
  # (a counter, a griddle) most seeds start inside it and every contact the
  # pair produced was NaN -- which is what made the first Brax action
  # non-finite on both egg scenes.
  @jax.custom_jvp
  def _box(pos, size):
    q = jp.abs(pos) - size
    b = jp.maximum(q, 0.0)
    return jp.sqrt(b @ b + 1e-24) + jp.minimum(jp.max(q), 0.0)

  def _box_grad(pos, size):
    """Exact box-SDF gradient, with both singular cases resolved.

    Outside, d/dp = sign(p) * b / |b|; the only place |b| vanishes is the
    interior, where the distance is `max(q)` and the gradient is the signed
    axis attaining it.  Selecting between the two by `jp.where` (not by
    control flow) keeps this a single fused expression under vmap.
    """
    q = jp.abs(pos) - size
    b = jp.maximum(q, 0.0)
    bnorm = jp.sqrt(b @ b)
    sgn = jp.where(pos < 0, -1.0, 1.0)
    # `bnorm == 0` covers the interior AND the boundary itself, which is where
    # a resting contact actually sits; the outside branch degenerates to 0/0
    # there, so both must take the axis form.
    g_in = sgn * (q == jp.max(q))
    g_in = g_in / jp.sqrt(g_in @ g_in + 1e-24)
    g_out = sgn * b / (bnorm + (bnorm == 0.0) * 1e-12)
    return jp.where(bnorm == 0.0, g_in, g_out)

  @_box.defjvp
  def _box_jvp(primals, tangents):
    pos, size = primals
    pos_dot, _ = tangents
    return _box(pos, size), jp.dot(_box_grad(pos, size), pos_dot)

  @sdf.collider(ncon=4)
  def cylinder_box(c, b):
    # four seeds around the separating direction, as cylinder_cylinder does:
    # a single contact point cannot hold a cylinder resting on a face.
    basis = mjx_math.make_frame(b.pos - c.pos)
    mid = 0.5 * (c.pos + b.pos)
    # Spread the seeds by the SMALLER shape's extent, as mjx's own
    # cylinder_cylinder does with the two radii. Using the box's extent put
    # every seed a counter-width away from the contact patch, and 10 gradient
    # steps did not walk back.
    r = jp.minimum(c.size[0], jp.max(b.size))
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
    r = jp.minimum(jp.max(e.size), jp.max(b.size))
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


def collide(sys, d, max_geom_pairs: int = -1, max_contact_points: int = -1,
            by_type=()):
  """mjx's collision pass with the broadphase cull reachable from brax.

  mjx already implements exactly the narrowing this needs -- a `top_k` on
  bounding-sphere clearance before the narrow phase, and a `top_k` on
  penetration depth after it -- but gates both behind `custom` NUMERICS in the
  MjModel (`max_geom_pairs`, `max_contact_points`, read by
  `collision_driver._numeric`). brax hands `mjx.collision` a `System` built
  from an already-compiled MjModel, and a compiled model's arrays cannot grow
  a new numeric, so from brax the feature was unreachable.

  Why it matters here: mjx expands a contact row per candidate geom pair
  STATICALLY. The two-arm scene has 1358 candidate pairs -> 3891 contact rows,
  and a MuJoCo rollout of the same scene never had more than a handful live at
  once -- so >99% of the rows are permanently inactive and are still evaluated,
  differentiated and solved every substep on every world. That is the whole
  accelerator cost of these scenes.

  The cull is a broadphase, not a semantic change: it keeps the CLOSEST pairs,
  which is what MuJoCo's own broadphase does. It is only safe while the cap
  exceeds the number of simultaneously-live contacts, so pick the cap from a
  measurement (`brax_contact_budget` in explore_bench) and never from a guess;
  a cap that binds silently deletes real contacts.

  Args:
    sys: brax System (an mjx.Model)
    d: mjx Data whose geom_xpos/geom_xmat have been set
    max_geom_pairs: per collision-function group, keep this many closest
      candidate pairs. -1 disables.
    max_contact_points: per condim group, keep this many deepest contacts.
      -1 disables.

  Returns:
    d with `contact` replaced.
  """
  import jax
  from jax import numpy as jp
  from mujoco.mjx._src import collision_driver as cd

  if d._impl.ncon == 0:
    return d
  if max_geom_pairs < 0 and max_contact_points < 0 and not by_type:
    from mujoco import mjx
    return mjx.collision(sys, d)

  # Per-geom-type-pair budgets. One global number is the wrong granularity:
  # measured against CPU MuJoCo's own contacts, every group on the arm-pair
  # scenes has its real contacts ranked FIRST by AABB clearance (worst rank 0
  # of 556 for capsule/box), while BOX-BOX reaches rank 528 -- dozens of tiny
  # interlocking gripper boxes overlap simultaneously, so the metric cannot
  # discriminate within that group and culling it drops real contacts.
  overrides = {(int(a), int(b)): int(v) for a, b, v in by_type}

  groups = cd._contact_groups(sys, d)
  for key, contact in groups.items():
    t1, t2 = int(key.types[0]), int(key.types[1])
    budget = overrides.get((min(t1, t2), max(t1, t2)), max_geom_pairs)
    if (budget > -1
        and contact.geom.shape[0] > budget
        and not set(key.types) & cd._GEOM_NO_BROADPHASE):
      g1, g2 = contact.geom.T
      # ORIENTED-AABB clearance, not mjx's bounding-sphere clearance.
      # `geom_rbound` is the radius of a sphere containing the geom, which for
      # a flat counter 1.82 x 1.07 x 0.78 m is 1.13 m in EVERY direction: on
      # the arm-pair scene that made 199 of 556 capsule/box pairs read as
      # "live" at once, and a cull ranked on it barely culls. MuJoCo already
      # stores each geom's local AABB (`geom_aabb`, exact for every geom type
      # including meshes); rotating its half-extents by |R| gives a world-axis
      # box that is tight in the flat direction. Separation along the widest
      # separating axis is a lower bound on the true distance, so a pair with
      # positive clearance genuinely cannot be in contact.
      c1, c2 = sys.geom_aabb[g1, :3], sys.geom_aabb[g2, :3]
      h1, h2 = sys.geom_aabb[g1, 3:], sys.geom_aabb[g2, 3:]
      m1, m2 = d.geom_xmat[g1], d.geom_xmat[g2]
      p1 = d.geom_xpos[g1] + jax.vmap(jp.matmul)(m1, c1)
      p2 = d.geom_xpos[g2] + jax.vmap(jp.matmul)(m2, c2)
      e1 = jax.vmap(jp.matmul)(jp.abs(m1), h1)
      e2 = jax.vmap(jp.matmul)(jp.abs(m2), h2)
      dist = jp.max(jp.abs(p2 - p1) - (e1 + e2), axis=-1)
      _, idx = jax.lax.top_k(-dist, k=budget)
      contact = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], contact)
    func = cd._COLLISION_FUNC[key.types]
    ncon = func.ncon
    dist, pos, frame = func(sys, d, key, contact.geom)
    if ncon > 1:
      contact = jax.tree_util.tree_map(
          lambda x, r=ncon: jp.repeat(x, r, axis=0), contact)
    groups[key] = contact.replace(dist=dist, pos=pos, frame=frame)

  condim_groups = {}
  for key, contact in groups.items():
    condim_groups.setdefault(key.condim, []).append(contact)
  if max_contact_points > -1:
    for key, contacts in condim_groups.items():
      contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)
      if contact.geom.shape[0] > max_contact_points:
        _, idx = jax.lax.top_k(-contact.dist, k=max_contact_points)
        contact = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], contact)
      condim_groups[key] = [contact]

  contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
  contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)
  return d.tree_replace({'_impl.contact': contact})
