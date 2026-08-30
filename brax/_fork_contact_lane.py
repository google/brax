# explore_bench fork of brax: LANE-FORM contact resolution.
"""Component-split (\"lane form\") rewrite of ``positional/collisions.py``.

Why
---
XLA cost analysis of one vmapped substep (``dynmanip/catch-gripper``,
N=1024, TPU v4) attributes ``bytes accessed`` like this:

    full substep                        337.7 GB
      narrowphase (``contact.get``)     138.9 GB   41%
      ``resolve_position`` x2 passes    157.1 GB   46%
      ``resolve_velocity``               41.7 GB   12%
      everything else (joints, kinematics, integrator)
                                          0.5 GB  0.2%

and the substep is bandwidth-bound end to end (337 GB / ~1.2 TB/s = 281 ms
against a measured 308 ms).  So the whole throughput problem is the contact
path, and inside it the contact SOLVE is the larger half.

``resolve_position`` is written per-contact and vmapped, so every intermediate
is ``(n_worlds, n_con, 3)``, ``(n_worlds, n_con, 1)`` or
``(n_worlds, n_con, 3, 3)``.  ``n_con`` is 369 on catch-gripper against 10
links, so these are the biggest arrays in the pipeline by a wide margin, and
they are in the worst possible layout: a TPU vector register is 8 sublanes x
128 lanes tiled from the last two axes, so ``(1024, 369, 3)`` pads to
``(1024, 376, 128)`` -- 49.3M register slots for 1.13M values, **2.3%
occupancy**.  ``(., ., 1)`` is 0.8% and ``(., ., 2, 3, 3)`` is 0.9%.
Three ``(1024, 369)`` component arrays instead pad to ``(1024, 384)`` each:
96% occupancy and **42x fewer slots**.

Second mechanism, the same one ``_fork_lane.py`` and ``_fork_narrowphase.py``
recorded: ``math.rotate``/``jp.dot``/``jp.cross`` over a trailing size-3 axis
each become a ``dot_general`` or a reduce under vmap, which is a fusion
barrier that materialises a buffer of the shape above.  Componentwise the
identical arithmetic is elementwise multiply-add and XLA emits one fused loop.

What this file does NOT change
------------------------------
The physics.  Every routine here reproduces
``collisions.resolve_position_vecform`` / ``resolve_velocity_vecform``
operation for operation, including the ``1e-6`` guards, the NaN scrub, the
constraint-count normalisation and the pseudo-velocity subtraction.  The
inverse-mass formulation and the iteration scheme belong to the contact-energy
agent and are untouched; ``.work/contact_equiv.py`` re-checks that claim
against the in-tree originals on randomised states, and will fail loudly if
either side is edited without the other.

The function BOUNDARY is also unchanged: inputs and outputs are brax's packed
``Transform``/``Motion``/``Contact`` pytrees, because those are PER-LINK
(n_link = 10) and therefore cheap.  Only the per-CONTACT interior is lane
form.

Switches
--------
``BRAX_FORK_LANE_CONTACT=0``   restores ``collisions.py``'s originals.
``BRAX_FORK_LANE_CONTACT_SEG`` ``matmul`` (default) or ``scatter`` for the
                               contact -> link accumulation.
``BRAX_FORK_LANE_SWEEP=0``     restores the packed depth refresh in
                               ``positional/pipeline.py::_contact_sweeps``.
"""

import os

from brax import _fork_lane as lane
from brax import com
from brax.base import Motion, Transform
import jax
from jax import numpy as jp

#: A/B switch.  ``0`` routes ``collisions.resolve_position`` /
#: ``resolve_velocity`` back to the ``*_vecform`` originals kept beside them.
ENABLED = os.environ.get('BRAX_FORK_LANE_CONTACT', '1') not in (
    '0', 'false', 'False', 'off', 'no')

#: How per-contact rows are accumulated onto links.  ``scatter`` is jax's
#: ``segment_sum`` (a scatter-add, what brax uses).  ``matmul`` contracts with
#: a one-hot ``(2 n_con, n_link)`` at HIGHEST precision, which keeps the whole
#: reduction in lane orientation instead of transposing to a size-7 minor axis.
SEG = os.environ.get('BRAX_FORK_LANE_CONTACT_SEG', 'matmul')

#: Lane form for the depth refresh in ``pipeline._contact_sweeps``.
ENABLE_SWEEP = os.environ.get('BRAX_FORK_LANE_SWEEP', '1') not in (
    '0', 'false', 'False', 'off', 'no')


# ------------------------------------------------------------------ helpers --


def split_mat3(m):
  """``(..., n, 3, 3)`` -> 3-tuple of 3-tuples of ``(..., n)``, rows first."""
  return tuple(tuple(m[..., i, j] for j in range(3)) for i in range(3))


def gather_rows(comps, fills, idx):
  """Gather many ``(..., n_link)`` component arrays with ONE gather.

  ``comps`` are per-link lane arrays; ``fills`` is the value each one takes on
  the appended "world body" row (0 for a position, velocity or inverse mass;
  1 for a quaternion's w).  ``idx`` is the per-contact link index with -1
  already remapped to ``n_link``.

  A separate ``take`` per component would be 24 gathers, and a gather is the
  op this rewrite exists to avoid.  Stacking first makes it one, whatever the
  component count -- the same trick ``_fork_lane.gather_many`` uses for the
  static parent selection, except that this index is data-dependent (the
  broadphase cull picks different geom pairs in every world) so the one-hot
  matmul form is not available here.
  """
  parts = [
      jp.concatenate([c, jp.full(c.shape[:-1] + (1,), f, c.dtype)], axis=-1)
      for c, f in zip(comps, fills)
  ]
  cat = jp.stack(parts, axis=-2)                       # (..., k, n_link + 1)
  g = jp.take_along_axis(cat, idx[..., None, :], axis=-1)   # (..., k, n_con)
  return tuple(g[..., i, :] for i in range(len(comps)))


def _accumulate(rows, link_idx, n_link):
  """``segment_sum(row, link_idx, n_link)`` for several ``(..., 2 n_con)`` rows.

  Returns one ``(..., m, n_link)`` array; row ``i`` is component ``i``.
  """
  x = jp.stack(rows, axis=-2)                          # (..., m, 2 n_con)
  if SEG == 'scatter':
    y = jax.ops.segment_sum(
        jp.moveaxis(x, -1, 0), link_idx, n_link)       # (n_link, ..., m)
    return jp.moveaxis(y, 0, -1)
  onehot = (link_idx[..., :, None] == jp.arange(n_link)).astype(x.dtype)
  return jp.matmul(x, onehot, precision=jax.lax.Precision.HIGHEST)


def _side_index(link_idx, n_link):
  """-1 (the world body) -> the appended row, everything else unchanged.

  brax reaches the same rows two different ways -- ``x.concatenate(zero).take``
  with ``mode='wrap'`` sends -1 to the appended zero transform, while
  ``inv_inertia.take(...) * (link_idx > -1)`` wraps onto the LAST link and
  then multiplies by zero.  Appending a zero row and sending -1 to it
  reproduces both exactly (the inertia product is 0 either way).
  """
  return jp.where(link_idx < 0, n_link, link_idx)


# -------------------------------------------------------- resolve_position --


def resolve_position(sys, state, x_i_prev, contact):
  """Lane form of ``collisions.resolve_position``; same inputs and outputs."""
  n_link = sys.num_links()
  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  # computed packed and split: n_link is 10, so the packed (10, 3, 3) array is
  # free, and using the SAME call as the vecform keeps the two bit-comparable
  ii = split_mat3(com.inv_inertia(sys, state.x))

  l0 = jp.asarray(contact.link_idx[0])
  l1 = jp.asarray(contact.link_idx[1])
  i0, i1 = _side_index(l0, n_link), _side_index(l1, n_link)

  comps = (
      state.x_i.pos[..., 0], state.x_i.pos[..., 1], state.x_i.pos[..., 2],
      state.x_i.rot[..., 0], state.x_i.rot[..., 1],
      state.x_i.rot[..., 2], state.x_i.rot[..., 3],
      x_i_prev.pos[..., 0], x_i_prev.pos[..., 1], x_i_prev.pos[..., 2],
      x_i_prev.rot[..., 0], x_i_prev.rot[..., 1],
      x_i_prev.rot[..., 2], x_i_prev.rot[..., 3],
      ii[0][0], ii[0][1], ii[0][2],
      ii[1][0], ii[1][1], ii[1][2],
      ii[2][0], ii[2][1], ii[2][2],
      inv_mass,
  )
  fills = (0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0.)
  a = gather_rows(comps, fills, i0)
  b = gather_rows(comps, fills, i1)

  def unpack(g):
    return dict(
        pos=(g[0], g[1], g[2]),
        rot=(g[3], g[4], g[5], g[6]),
        ppos=(g[7], g[8], g[9]),
        prot=(g[10], g[11], g[12], g[13]),
        ii=((g[14], g[15], g[16]), (g[17], g[18], g[19]), (g[20], g[21], g[22])),
        m=g[23],
    )

  a, b = unpack(a), unpack(b)

  n = (-contact.frame[..., 0, 0], -contact.frame[..., 0, 1],
       -contact.frame[..., 0, 2])
  c = contact.dist
  cp = lane.vec3(contact.pos)

  half = tuple(ni * c / 2.0 for ni in n)
  pos_p = tuple(cp[k] + half[k] - a['pos'][k] for k in range(3))
  pos_c = tuple(cp[k] - half[k] - b['pos'][k] for k in range(3))

  cr1 = lane.cross3(pos_p, n)
  cr2 = lane.cross3(pos_c, n)
  w1 = a['m'] + lane.dot3(cr1, lane.matvec(a['ii'], cr1))
  w2 = b['m'] + lane.dot3(cr2, lane.matvec(b['ii'], cr2))

  dlambda = -c / (w1 + w2 + 1e-6)
  coll_mask = c < 0
  p = tuple((dlambda * n[k]) * coll_mask for k in range(3))

  dp_p_pos = lane.scale3(p, a['m'])
  dp_c_pos = tuple(-p[k] * b['m'] for k in range(3))
  dp_p_rot = lane.vec_quat_mul(
      lane.matvec(a['ii'], lane.cross3(pos_p, p)), a['rot'])
  dp_c_rot = tuple(-v for v in lane.vec_quat_mul(
      lane.matvec(b['ii'], lane.cross3(pos_c, p)), b['rot']))

  # static friction
  r1 = lane.rotate(lane.sub3(cp, a['pos']), lane.qinv(a['rot']))
  r2 = lane.rotate(lane.sub3(cp, b['pos']), lane.qinv(b['rot']))
  p1bar = lane.add3(a['ppos'], lane.rotate(r1, a['prot']))
  p2bar = lane.add3(b['ppos'], lane.rotate(r2, b['prot']))

  deltap = lane.sub3(lane.sub3(cp, p1bar), lane.sub3(cp, p2bar))
  dpn = lane.dot3(deltap, n)
  deltap_t = tuple(deltap[k] - dpn * n[k] for k in range(3))

  pos_p = lane.sub3(cp, a['pos'])
  pos_c = lane.sub3(cp, b['pos'])
  ct = lane.safe_norm3(deltap_t)
  nt = tuple(v / (ct + 1e-6) for v in deltap_t)

  cr1 = lane.cross3(pos_p, nt)
  cr2 = lane.cross3(pos_c, nt)
  w1 = a['m'] + lane.dot3(cr1, lane.matvec(a['ii'], cr1))
  w2 = b['m'] + lane.dot3(cr2, lane.matvec(b['ii'], cr2))

  dlambdat = -ct / (w1 + w2)
  static_mask = jp.where(jp.abs(dlambdat) < jp.abs(dlambda), 1.0, 0.0)
  pt = tuple(((dlambdat * nt[k]) * static_mask) * coll_mask for k in range(3))

  dp_p_pos = tuple(dp_p_pos[k] + pt[k] * a['m'] for k in range(3))
  q = lane.vec_quat_mul(lane.matvec(a['ii'], lane.cross3(pos_p, pt)), a['rot'])
  dp_p_rot = tuple(dp_p_rot[k] + 0.5 * q[k] for k in range(4))
  dp_c_pos = tuple(dp_c_pos[k] - pt[k] * b['m'] for k in range(3))
  q = lane.vec_quat_mul(lane.matvec(b['ii'], lane.cross3(pos_c, pt)), b['rot'])
  dp_c_rot = tuple(dp_c_rot[k] - 0.5 * q[k] for k in range(4))

  cs = sys.collide_scale
  rows = [jp.concatenate([dp_p_pos[k] * cs, dp_c_pos[k] * cs], axis=-1)
          for k in range(3)]
  rows += [jp.concatenate([dp_p_rot[k] * cs, dp_c_rot[k] * cs], axis=-1)
           for k in range(4)]
  rows = [jp.where(jp.isnan(r), 0.0, r) for r in rows]

  link_idx = jp.concatenate([l0, l1], axis=-1)
  on_link = link_idx > -1
  active = jp.asarray(coll_mask, jp.float32)
  rows.append(jp.concatenate([active, active], axis=-1))
  rows = [jp.where(on_link, r, 0.0) for r in rows]

  acc = _accumulate(rows, link_idx, n_link)             # (..., 8, n_link)
  denom = jp.maximum(acc[..., 7, :], 1.0)
  dpos = jp.stack([acc[..., k, :] / denom for k in range(3)], axis=-1)
  drot = jp.stack([acc[..., k, :] / denom for k in range(3, 7)], axis=-1)

  x_i_pre = state.x_i
  x_i = state.x_i + Transform(pos=dpos, rot=drot)
  # per-LINK, so left packed and taken straight from brax
  from brax import math
  x_i = x_i.replace(rot=jax.vmap(math.normalize)(x_i.rot)[0])
  return x_i, (dlambda * coll_mask, x_i_pre)


# -------------------------------------------------------- resolve_velocity --


def resolve_velocity(sys, state, xd_i_prev, contact, dlambda):
  """Lane form of ``collisions.resolve_velocity``; same inputs and outputs."""
  n_link = sys.num_links()
  dlam, x_i_pre = dlambda
  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)
  ii = split_mat3(inv_inertia)

  # the velocity `project_xd` manufactured out of the positional correction
  # alone, computed the same way it was (per LINK -- n_link is 10, so this
  # stays packed)
  from brax import math

  @jax.vmap
  def _push_vel(x, x_pre):
    vel = (x.pos - x_pre.pos) / sys.opt.timestep
    dq = math.relative_quat(x_pre.rot, x.rot)
    ang = 2.0 * dq[1:] / sys.opt.timestep
    ang = jp.where(dq[0] >= 0.0, 1.0, -1.0) * ang
    return Motion(vel=vel, ang=ang)

  xdv_push = _push_vel(state.x_i, x_i_pre)
  xd_eff = state.xd_i - xdv_push

  l0 = jp.asarray(contact.link_idx[0])
  l1 = jp.asarray(contact.link_idx[1])
  i0, i1 = _side_index(l0, n_link), _side_index(l1, n_link)

  comps = (
      state.x_i.pos[..., 0], state.x_i.pos[..., 1], state.x_i.pos[..., 2],
      xd_eff.vel[..., 0], xd_eff.vel[..., 1], xd_eff.vel[..., 2],
      xd_eff.ang[..., 0], xd_eff.ang[..., 1], xd_eff.ang[..., 2],
      xd_i_prev.vel[..., 0], xd_i_prev.vel[..., 1], xd_i_prev.vel[..., 2],
      xd_i_prev.ang[..., 0], xd_i_prev.ang[..., 1], xd_i_prev.ang[..., 2],
      ii[0][0], ii[0][1], ii[0][2],
      ii[1][0], ii[1][1], ii[1][2],
      ii[2][0], ii[2][1], ii[2][2],
      inv_mass,
  )
  fills = (0.,) * 25
  a = gather_rows(comps, fills, i0)
  b = gather_rows(comps, fills, i1)

  def unpack(g):
    return dict(
        pos=(g[0], g[1], g[2]),
        vel=(g[3], g[4], g[5]),
        ang=(g[6], g[7], g[8]),
        pvel=(g[9], g[10], g[11]),
        pang=(g[12], g[13], g[14]),
        ii=((g[15], g[16], g[17]), (g[18], g[19], g[20]),
            (g[21], g[22], g[23])),
        m=g[24],
    )

  a, b = unpack(a), unpack(b)
  cp = lane.vec3(contact.pos)
  n = (-contact.frame[..., 0, 0], -contact.frame[..., 0, 1],
       -contact.frame[..., 0, 2])

  ra = lane.sub3(cp, a['pos'])
  rb = lane.sub3(cp, b['pos'])
  rel_vel = lane.sub3(lane.add3(a['vel'], lane.cross3(a['ang'], ra)),
                      lane.add3(b['vel'], lane.cross3(b['ang'], rb)))
  v_n = lane.dot3(rel_vel, n)
  v_t = tuple(rel_vel[k] - n[k] * v_n for k in range(3))
  v_t_dir, v_t_norm = lane.normalize3(v_t)
  scale = -jp.minimum(
      contact.friction[..., 0] * jp.abs(dlam) / sys.opt.timestep, v_t_norm)
  dvel = lane.scale3(v_t_dir, scale)

  angw_1 = lane.cross3(ra, v_t_dir)
  angw_2 = lane.cross3(rb, v_t_dir)
  w1 = a['m'] + lane.dot3(angw_1, lane.matvec(a['ii'], angw_1))
  w2 = b['m'] + lane.dot3(angw_2, lane.matvec(b['ii'], angw_2))
  wsum = w1 + w2 + 1e-6
  p_dyn = tuple(v / wsum for v in dvel)

  rel_prev = lane.sub3(lane.add3(a['pvel'], lane.cross3(a['pang'], ra)),
                       lane.add3(b['pvel'], lane.cross3(b['pang'], rb)))
  v_n_prev = lane.dot3(rel_prev, n)
  rest = -v_n - jp.minimum(contact.elasticity * v_n_prev, 0)
  dx = lane.scale3(n, rest)

  # pos_p = contact.pos - x.pos[0]
  # pos_c = contact.pos + frame[0] * dist - x.pos[1]
  pos_p = ra
  pos_c = tuple(cp[k] - n[k] * contact.dist - b['pos'][k] for k in range(3))

  c2 = lane.safe_norm3(dx)
  n2 = tuple(v / (c2 + 1e-6) for v in dx)
  cr1 = lane.cross3(pos_p, n2)
  cr2 = lane.cross3(pos_c, n2)
  w1 = a['m'] + lane.dot3(cr1, lane.matvec(a['ii'], cr1))
  w2 = b['m'] + lane.dot3(cr2, lane.matvec(b['ii'], cr2))

  dlambda_rest = c2 / (w1 + w2 + 1e-6)
  penetrating = contact.dist < 0
  sinking = v_n_prev <= 0.0
  pvel = tuple(((dlambda_rest * n2[k]) * sinking + p_dyn[k]) * penetrating
               for k in range(3))

  # Transform.create(pos=rel).do(Force(vel=p, ang=0)):
  #   vel = rotate(p, identity) = p ; ang = 0 + cross(rel, p)
  cat = lambda u, v: jp.concatenate([u, v], axis=-1)
  vel_cat = tuple(cat(pvel[k], -pvel[k]) for k in range(3))
  rel_cat = tuple(cat(ra[k], rb[k]) for k in range(3))
  ang_cat = lane.cross3(rel_cat, vel_cat)

  is_contact = jp.asarray(penetrating, jp.float32)
  rows = list(vel_cat) + list(ang_cat) + [cat(is_contact, is_contact)]
  link_idx = jp.concatenate([l0, l1], axis=-1)
  on_link = link_idx > -1
  rows = [jp.where(on_link, r, 0.0) for r in rows]
  acc = _accumulate(rows, link_idx, n_link)             # (..., 7, n_link)
  ncnt = acc[..., 6, :] + 1e-8
  xp_vel = jp.stack([acc[..., k, :] / ncnt for k in range(3)], axis=-1)
  xp_ang = jp.stack([acc[..., k, :] / ncnt for k in range(3, 6)], axis=-1)

  xdv_i = Motion(
      vel=jax.vmap(lambda x, y: x * y)(inv_mass, xp_vel),
      ang=jax.vmap(lambda x, y: x @ y)(inv_inertia, xp_ang),
  )
  return xdv_i - xdv_push


# ------------------------------------------------------ depth refresh sweep --


def refresh_depth(sys, x, link, r_local, n_w, dist0):
  """Lane form of the per-pass depth refresh in ``pipeline._contact_sweeps``.

  ``link`` is ``(2, n_con)``, ``r_local`` a pair of 3-tuples (the contact
  point in each link's own frame) and ``n_w`` a 3-tuple.
  """
  n_link = sys.num_links()
  comps = (x.pos[..., 0], x.pos[..., 1], x.pos[..., 2],
           x.rot[..., 0], x.rot[..., 1], x.rot[..., 2], x.rot[..., 3])
  fills = (0., 0., 0., 1., 0., 0., 0.)
  out = []
  for s in range(2):
    g = gather_rows(comps, fills, _side_index(link[s], n_link))
    out.append(lane.add3((g[0], g[1], g[2]),
                         lane.rotate(r_local[s], (g[3], g[4], g[5], g[6]))))
  return dist0 + lane.dot3(n_w, lane.sub3(out[1], out[0]))


def contact_local(sys, x, link, cpos):
  """The contact point in each link's own frame, lane form."""
  n_link = sys.num_links()
  comps = (x.pos[..., 0], x.pos[..., 1], x.pos[..., 2],
           x.rot[..., 0], x.rot[..., 1], x.rot[..., 2], x.rot[..., 3])
  fills = (0., 0., 0., 1., 0., 0., 0.)
  out = []
  for s in range(2):
    g = gather_rows(comps, fills, _side_index(link[s], n_link))
    out.append(lane.rotate(lane.sub3(cpos, (g[0], g[1], g[2])),
                           lane.qinv((g[3], g[4], g[5], g[6]))))
  return out
