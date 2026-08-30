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

"""Functions to resolve collisions."""

# pylint:disable=g-multiple-import
from typing import Optional, Tuple

import os

from brax import _fork_contact_lane as _lane_contact
from brax import com
from brax import kinematics
from brax import math
from brax.base import (Contact, Force, Motion, QD_WIDTHS, System, Transform)
from brax.positional.base import State
import jax
from jax import numpy as jp
import numpy as np
from jax.ops import segment_sum


# ---------------------------------------------------------------------------
# explore_bench fork: ARTICULATED contact resolution (opt-in, A/B'd)
#
# `BRAX_FORK_ARTMASS=1` swaps `resolve_position`'s normal push-out for the
# articulated formulation described in `.work/contact_convergence_report.md`.
# Everything is behind the flag; at 0 the original runs untouched.
# ---------------------------------------------------------------------------
def _envflag(name, default='0'):
  return os.environ.get(name, default).lower() not in ('0', '', 'false', 'no')


_ART_CACHE = {}


def _art_tables(sys):
  """Static structure for the articulated contact path.

  Returns ``(anc, own, free_root, n_dof)``:
    anc       (n_link, n_dof) 1.0 where dof i is an ancestor-or-self dof of
              link k -- the same `desc` matrix `joints.drive_update` builds,
              in dof rather than link columns.  Single-DOF joints only; a
              free joint contributes no column (its mobility is the free-body
              term instead).
    own       (n_dof,) the link that owns each dof, for indexing the per-LINK
              joint anchor `a_p`.
    free_root (n_link,) 1.0 when the link's chain root is a FREE joint, i.e.
              when the free-body inverse mass is a real mobility rather than
              a fiction.  A fixed-base arm's fingertip is NOT free: today's
              `1/m_link` treats it as though it were.
  """
  key = id(sys)
  hit = _ART_CACHE.get(key)
  if hit is not None:
    return hit
  n_link = sys.num_links()
  link_dof = np.full(n_link, -1, np.int64)
  is_free = np.zeros(n_link, bool)
  off = 0
  for i, ty in enumerate(sys.link_types):
    if QD_WIDTHS[ty] == 1:
      link_dof[i] = off
    if ty == 'f':
      is_free[i] = True
    off += QD_WIDTHS[ty]
  n_dof = off
  anc = np.zeros((n_link, n_dof), np.float32)
  own = np.zeros(n_dof, np.int64)
  free_root = np.zeros(n_link, np.float32)
  for k in range(n_link):
    a = k
    while a >= 0:
      if link_dof[a] >= 0:
        anc[k, link_dof[a]] = 1.0
        own[link_dof[a]] = a
      if is_free[a]:
        free_root[k] = 1.0
      a = int(sys.link_parents[a])
  out = (anc, own, free_root, n_dof)
  _ART_CACHE[key] = out
  return out


def _art_dofs(sys, state, own, n_dof):
  """Per-dof world axis, anchor and the 6-vector that evaluates J_i . f.

  For a contact impulse ``f`` applied at world point ``p``, the generalised
  force on dof i is ``J_i(p) . f`` with

      revolute :  J_i = axis_ang_i x (p - anchor_i)
      prismatic:  J_i = axis_vel_i

  A brax DoF carries the revolute axis in ``motion.ang`` and the prismatic one
  in ``motion.vel``, with the other zero, so ``J_i = ang_i x (p - anchor_i) +
  vel_i`` covers both with no branch.  PRISMATIC IS NOT OPTIONAL: the Panda's
  and the Robotiq's FINGERS are slide joints, so a revolute-only Jacobian sees
  no resistance from exactly the joints that do the grasping (measured: the
  jaw closes through the ball to +0.00014 m).

  Expanding the triple product,

      J_i(p) . f = ang_i . (p x f) + (vel_i - ang_i x anchor_i) . f
                 = A_i . [p x f ; f]

  so a per-dof 6-vector ``A_i`` turns every Jacobian evaluation into a dot
  product -- and every reduction over contacts into a matmul.
  """
  _, _, a_p, _ = kinematics.world_to_joint(sys, state.x, state.xd)
  own_j = jp.asarray(own)
  rot_own = a_p.rot.take(own_j, axis=0)
  anchor = a_p.pos.take(own_j, axis=0)
  ang = jax.vmap(math.rotate)(sys.dof.motion.ang[:n_dof], rot_own)
  vel = jax.vmap(math.rotate)(sys.dof.motion.vel[:n_dof], rot_own)
  dof_i = sys.dof_inertia[:n_dof]
  winv = jp.where(dof_i > 0, 1.0 / jp.where(dof_i > 0, dof_i, 1.0), 0.0)
  a6 = jp.concatenate([ang, vel - jp.cross(ang, anchor)], axis=-1)
  return ang, vel, anchor, winv, a6


def resolve_position_vecform(
    sys: System,
    state: State,
    x_i_prev: Transform,
    contact: Optional[Contact],
) -> Tuple[Transform, Tuple[jax.Array, Transform]]:
  """Resolves positional collision constraint.

  The update equations follow section 3.5 of Müller et al.'s Extended Position
  Based Dynamics, where we have removed the compliance terms.
  (Müller, Matthias, et al. "Detailed rigid body simulation with extended
  position based dynamics." Computer Graphics Forum. Vol. 39. No. 8. 2020.).

  explore_bench fork -- two changes, both measured:

  * The Jacobi sum over contact rows is **constraint-count normalised**.  mjx
    expands one geom pair into up to four contact rows carrying the SAME
    penetration depth, and each row independently pushed the body out by the
    full depth: an egg 15.8 mm inside hibachi's spatula was moved 20.8 mm in
    one substep by two identical rows, i.e. 1.3x its own overlap.  This is the
    same over-relaxed-Jacobi defect that was fixed for the joint sweeps.
  * ``dlambda`` now also carries the link transform as it stood BEFORE the
    collision correction.  ``pipeline.step`` hands whatever this returns
    straight back to ``resolve_velocity`` and reads nothing from it, so the
    pair of functions use it as a private channel; ``resolve_velocity`` needs
    the pre-correction pose to subtract the velocity that
    ``integrator.project_xd`` manufactures out of the push-out (see there).

  Args:
    sys: System to forward propagate
    state: positional pipeline state
    x_i_prev: center of mass position from previous step
    contact: Contact pytree

  Returns:
    x_i: new position after update
    (dlambda, x_i_pre): normal force information, and the pre-correction pose
  """
  if contact is None:
    return state.x_i, (jp.zeros((1,)), state.x_i)

  # explore_bench fork: opt-in articulated resolution (BRAX_FORK_ARTMASS=1).
  # Original retained below for A/B; see `_resolve_position_art`.
  if _ART_ENABLED:
    return _resolve_position_art(sys, state, x_i_prev, contact)

  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)

  @jax.vmap
  def translate(contact):
    link_idx = jp.array(contact.link_idx)
    x = state.x_i.concatenate(Transform.zero((1,))).take(link_idx)
    x_prev = x_i_prev.concatenate(Transform.zero((1,))).take(link_idx)

    # TODO(cdfreeman): rewrite these updates to use pbd methods
    n = -contact.frame[0]
    c = contact.dist
    pos_p = contact.pos + n * c / 2.0 - x.pos[0]
    pos_c = contact.pos - n * c / 2.0 - x.pos[1]

    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)
    mass_inv = inv_mass.take(link_idx) * (link_idx > -1)

    # only spherical inertia effects
    cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
    w1 = mass_inv[0] + jp.dot(cr1, i_inv[0] @ cr1)
    w2 = mass_inv[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambda = -c / (w1 + w2 + 1e-6)
    coll_mask = c < 0
    p = dlambda * n * coll_mask

    dp_p_pos, dp_c_pos = p * mass_inv[0], -p * mass_inv[1]
    dp_p_rot = math.vec_quat_mul(i_inv[0] @ jp.cross(pos_p, p), x.rot[0])
    dp_c_rot = -math.vec_quat_mul(i_inv[1] @ jp.cross(pos_c, p), x.rot[1])

    # static friction
    q1inv, q2inv = jax.vmap(math.quat_inv)(x.rot)
    r1 = math.rotate(contact.pos - x.pos[0], q1inv)
    r2 = math.rotate(contact.pos - x.pos[1], q2inv)
    p1bar = x_prev.pos[0] + math.rotate(r1, x_prev.rot[0])
    p2bar = x_prev.pos[1] + math.rotate(r2, x_prev.rot[1])
    p0 = contact.pos

    deltap = (p0 - p1bar) - (p0 - p2bar)
    deltap_t = deltap - jp.dot(deltap, n) * n

    pos_p, pos_c = contact.pos - x.pos
    c = math.safe_norm(deltap_t)
    n = deltap_t / (c + 1e-6)

    cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
    w1 = mass_inv[0] + jp.dot(cr1, i_inv[0] @ cr1)
    w2 = mass_inv[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambdat = -c / (w1 + w2)
    static_mask = jp.where(jp.abs(dlambdat) < jp.abs(dlambda), 1.0, 0.0)
    p = dlambdat * n * static_mask * coll_mask

    dp_p_pos += p * mass_inv[0]
    dp_p_rot += 0.5 * math.vec_quat_mul(i_inv[0] @ jp.cross(pos_p, p), x.rot[0])
    dp_p = Transform(pos=dp_p_pos, rot=dp_p_rot) * sys.collide_scale

    dp_c_pos -= p * mass_inv[1]
    dp_c_rot -= 0.5 * math.vec_quat_mul(i_inv[1] @ jp.cross(pos_c, p), x.rot[1])
    dp_c = Transform(pos=dp_c_pos, rot=dp_c_rot) * sys.collide_scale

    return dp_p, dp_c, dlambda * coll_mask, jp.asarray(coll_mask, jp.float32)

  dp_p, dp_c, dlambda, active = translate(contact)
  dp = jax.tree.map(lambda x, y: jp.vstack([x, y]), dp_p, dp_c)
  dp = jax.tree.map(lambda x: jp.where(jp.isnan(x), 0.0, x), dp)
  link_idx = jp.concatenate(contact.link_idx)
  on_link = link_idx > -1
  dp *= on_link.reshape((-1, 1))
  dp = jax.tree.map(
      lambda f: jax.ops.segment_sum(f, link_idx, sys.num_links()), dp
  )
  # explore_bench fork: constraint-count normalisation.  Without it N contact
  # rows quoting the same penetration each push the body out by the whole of
  # it, so the body leaves the surface with N-1 depths of spare clearance.
  n_act = jax.ops.segment_sum(
      jp.concatenate([active, active]) * on_link, link_idx, sys.num_links()
  )
  denom = jp.maximum(n_act, 1.0)
  dp = jax.tree.map(lambda f: f / denom.reshape((-1,) + (1,) * (f.ndim - 1)), dp)

  x_i_pre = state.x_i
  x_i = state.x_i + dp
  x_i = x_i.replace(rot=jax.vmap(math.normalize)(x_i.rot)[0])

  return x_i, (dlambda, x_i_pre)



_ART_ENABLED = _envflag('BRAX_FORK_ARTMASS')
_ART_ALPHA = float(os.environ.get('BRAX_FORK_ARTMASS_ALPHA', '1e-4'))
_ART_CLAMP = float(os.environ.get('BRAX_FORK_ARTMASS_CLAMP', '1.0'))
_ART_KAPPA = _envflag('BRAX_FORK_ARTMASS_KAPPA', '1')


def _rownorm(v):
  """row-wise 2-norm with a smooth floor (`math.safe_norm` ignores `axis`)."""
  return jp.sqrt(jp.sum(v * v, axis=-1) + 1e-24)



def _art_distribute(com_pos, anc, ang, vel, anchor, winv, a6,
                    li, cpos, imp, r, minv, iinv, fr, n_link):
  """Map contact impulses to per-link linear/angular response.

  The response is LINEAR in the generalised coordinate, so the reduction over
  contacts happens in DOF space FIRST: a ``(n_link, 6)`` segment_sum plus one
  ``(n_dof, n_link)`` contraction, never an ``(ncon, n_link, n_dof)`` tensor
  (which at the juggle budget would be ~368k elements per pass on the hottest
  path in the substep).

  The returned pair is (linear, angular) per link.  For the POSITION pass they
  are a displacement and a rotation vector; for the VELOCITY pass the very
  same numbers are a delta-velocity and a delta-angular-velocity, because an
  articulated inverse mass applied to an impulse is exactly that in both.
  """
  torque = jp.cross(jp.broadcast_to(cpos[:, None, :], imp.shape), imp)
  wrench = jp.concatenate([torque, imp], axis=-1)          # (ncon, 2, 6)
  w_link = segment_sum(wrench.reshape(-1, 6), li, n_link)
  d_theta = winv * jp.sum(a6 * (anc.T @ w_link), axis=-1)  # (n_dof,)

  dq_ang, dq_vel = ang * d_theta[:, None], vel * d_theta[:, None]
  angv = anc @ dq_ang
  # bilinear factoring, as in joints.drive_update: never materialise
  # (n_link, n_dof, 3) on a bandwidth-bound solver.
  lin = (anc @ dq_vel + jp.cross(angv, com_pos)
         - anc @ jp.cross(dq_ang, anchor))

  # free-body share, only where the chain root is a free joint
  imp_f = imp * fr[..., None]
  lin = lin + segment_sum((imp_f * minv[..., None]).reshape(-1, 3), li, n_link)
  angv = angv + segment_sum(
      jp.einsum('ksab,ksb->ksa', iinv, jp.cross(r, imp_f)).reshape(-1, 3),
      li, n_link)
  return lin, angv


def _resolve_position_art(sys, state, x_i_prev, contact):
  """Articulated, opposition-aware positional contact resolution.

  Two defects in ``resolve_position`` above, and they have to be fixed
  together (measured: fixing either alone is a regression, not a partial
  improvement).

  1. SIZING AND APPLICATION.  The shipped routine sizes ``dlambda = -c /
     (w1 + w2)`` with each link's OWN free-body inverse mass and then applies
     the push-out to that link alone.  A Robotiq or Panda finger is a few
     grams at the tip of a nine-link arm, so it is shoved out by nearly the
     full penetration, the joint sweep hauls it back, and
     ``integrator.project_xd`` turns each undone push into real velocity.
     Here ``w`` is the ARTICULATED inverse mass at the contact point,
     ``sum_i (J_i . n)^2 / dof_inertia_i`` over the link's ancestor dofs, and
     the impulse is DISTRIBUTED along the chain as a subtree rotation
     (revolute) or translation (prismatic).  The free-body term survives only
     where the chain root really is free (the ball, Spot's trunk).

  2. OPPOSING CONTACTS.  ``w_art`` is 100-1000x SMALLER than ``w_free`` at
     real contacts, so on a finger-vs-ball row nearly the whole correction is
     assigned to the BALL -- physically right.  In a two-sided SQUEEZE the
     ball cannot move: the opposing rows cancel on it, and stopping the
     fingers requires the two rows to communicate THROUGH the ball.  For the
     2-contact system A = J W J^T that is

         A = [[w_f + w_b, -w_b], [-w_b, w_f + w_b]]

     whose exact solution is ``lambda = -c / w_f`` -- the ball's mobility
     CANCELS -- while Jacobi (and Gauss-Seidel, whose rate here is the square)
     converges at ``w_b / (w_f + w_b) ~ 0.997`` per pass, i.e. ~1000 passes.
     No iteration scheme fixes a spectral radius of 0.997 in two passes, so
     the OPERATOR has to change, not the sweep.

     ``kappa_L = |sum_k s_k| / n_k`` over the active rows on link L (``s_k``
     the direction L is pushed) is exactly that: 1 for a body pushed one way,
     0 for a body pinned between opposing rows or inside a three-finger
     grasp.  Using ``kappa_L * w_L`` in the DENOMINATOR reproduces the exact
     coupled answer for the squeeze in a single pass and is inert (kappa = 1)
     wherever there is no opposition.

  Two guards keep it bounded, and both are inert in the uncoupled case:
    * the denominator is floored at ``ALPHA * (w1 + w2)``, capping the
      amplification over Jacobi at ``1/ALPHA``;
    * each body's per-row displacement is clamped to ``CLAMP * |c|``.  In the
      exact single-contact solution no body ever moves more than the
      penetration depth, so this never binds there; in the squeeze it clips
      the ball's nominal 300 ``|c|`` (which cancels anyway) and leaves the
      finger's exactly ``|c|`` untouched.
  """
  anc_np, own, free_np, n_dof = _art_tables(sys)
  n_link = sys.num_links()
  anc = jp.asarray(anc_np)
  ang, vel, anchor, winv, a6 = _art_dofs(sys, state, own, n_dof)

  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)
  # index -1 is brax's world body: pad every per-link table with a zero row so
  # a plain `take` picks up a body with no mobility at all.
  minv_p = jp.concatenate([inv_mass, jp.zeros((1,))])
  iinv_p = jp.concatenate([inv_inertia, jp.zeros((1, 3, 3))])
  free_p = jp.concatenate([jp.asarray(free_np), jp.zeros((1,))])
  anc_p = jp.concatenate([anc, jp.zeros((1, n_dof))], axis=0)

  link = jp.array(contact.link_idx).T                     # (ncon, 2)
  n_w = -contact.frame[:, 0]
  cpos = contact.pos
  c = contact.dist
  coll = c < 0
  onl = link > -1
  li = link.reshape(-1)

  xpos = state.x_i.concatenate(Transform.zero((1,))).pos.take(link, axis=0)
  xrot = state.x_i.concatenate(Transform.zero((1,))).rot.take(link, axis=0)
  half = n_w[:, None, :] * (c[:, None, None] / 2.0) * jp.array([1.0, -1.0])[
      None, :, None]
  r = cpos[:, None, :] + half - xpos                      # (ncon, 2, 3)

  minv = minv_p.take(link)
  iinv = iinv_p.take(link, axis=0)
  fr = free_p.take(link)
  ancL = anc_p.take(link, axis=0)                         # (ncon, 2, n_dof)

  def _w(direc):
    """articulated + (where the root is free) free-body inverse mass."""
    b6 = jp.concatenate([jp.cross(cpos, direc), direc], axis=-1)
    g = b6 @ a6.T                                         # (ncon, n_dof)
    w_art = jp.einsum('kd,ksd->ks', (g * g) * winv, ancL)
    cr = jp.cross(r, direc[:, None, :])
    w_free = minv + jp.einsum('ksa,ksab,ksb->ks', cr, iinv, cr)
    return w_art + fr * w_free, g

  w, _ = _w(n_w)

  # --- opposition-aware preconditioner -------------------------------------
  act = (coll[:, None] & onl).astype(jp.float32)
  push = jp.stack([n_w, -n_w], axis=1) * act[..., None]
  net = segment_sum(push.reshape(-1, 3), li, n_link)
  nact = segment_sum(act.reshape(-1), li, n_link)
  kappa_l = _rownorm(net) / jp.maximum(nact, 1.0)
  kappa_l = jp.clip(kappa_l, 0.0, 1.0)
  if not _ART_KAPPA:
    kappa_l = jp.ones_like(kappa_l)
  kappa = jp.concatenate([kappa_l, jp.zeros((1,))]).take(link)
  nact_c = jp.maximum(jp.concatenate(
      [nact, jp.zeros((1,))]).take(link), 1.0)

  wsum = w[:, 0] + w[:, 1]
  denom = jp.maximum(kappa[:, 0] * w[:, 0] + kappa[:, 1] * w[:, 1],
                     _ART_ALPHA * wsum) + 1e-9
  dlambda = (-c / denom) * coll

  def _impulse(dlam, direc, cap):
    """per-side impulse: displacement clamp, then constraint-count share."""
    disp = w * dlam[:, None]
    scale = jp.minimum(1.0, cap[:, None] / (jp.abs(disp) + 1e-12)) / nact_c
    sgn = jp.array([1.0, -1.0])
    return ((dlam[:, None] * sgn)[..., None] * direc[:, None, :]
            * scale[..., None] * onl[..., None])

  imp = _impulse(dlambda, n_w, _ART_CLAMP * jp.abs(c))

  # --- static friction, same articulated treatment -------------------------
  xp = x_i_prev.concatenate(Transform.zero((1,)))
  xp_pos = xp.pos.take(link, axis=0)
  xp_rot = xp.rot.take(link, axis=0)
  qinv = jax.vmap(jax.vmap(math.quat_inv))(xrot)
  rr = jax.vmap(jax.vmap(math.rotate))(cpos[:, None, :] - xpos, qinv)
  pbar = xp_pos + jax.vmap(jax.vmap(math.rotate))(rr, xp_rot)
  deltap = pbar[:, 1] - pbar[:, 0]
  deltap_t = deltap - jax.vmap(jp.dot)(deltap, n_w)[:, None] * n_w
  ct = _rownorm(deltap_t)
  nt = deltap_t / (ct[:, None] + 1e-6)
  wt, _ = _w(nt)
  dlambdat = -ct / (wt[:, 0] + wt[:, 1] + 1e-6)
  static = (jp.abs(dlambdat) < jp.abs(dlambda)) * coll
  imp = imp + _impulse(dlambdat * static, nt, ct)

  lin, angv = _art_distribute(state.x_i.pos, anc, ang, vel, anchor, winv, a6,
                              li, cpos, imp, r, minv, iinv, fr, n_link)

  # 0.5 * omega (x) q is the quaternion derivative, and `project_xd` reads the
  # angular velocity back as 2 dq / h.  The routine above omits the 0.5 on the
  # normal term, i.e. rotates twice as far as its own `w` was sized for.
  drot = 0.5 * jax.vmap(math.vec_quat_mul)(angv, state.x_i.rot)
  dp = Transform(pos=lin, rot=drot) * sys.collide_scale
  dp = jax.tree.map(lambda v: jp.where(jp.isnan(v), 0.0, v), dp)

  x_i_pre = state.x_i
  x_i = state.x_i + dp
  x_i = x_i.replace(rot=jax.vmap(math.normalize)(x_i.rot)[0])
  return x_i, (dlambda, x_i_pre)



def _resolve_velocity_art(sys, state, xd_i_prev, contact, dlambda_pack):
  """Velocity-level contact response against the ARTICULATED inverse mass.

  The same defect as the position pass, and on this path it is what loses the
  object.  brax sizes the friction impulse as ``p = dvel / (w1 + w2)`` with
  each link's own free-body ``w``.  For a Panda finger that ``w`` is ~451,
  almost all of it the link's SPURIOUS ROTATIONAL mobility -- the finger is on
  a SLIDE joint and cannot rotate at all -- against ~9.5 for the articulated
  value and ~18 for the ball it is holding.  The friction impulse therefore
  comes out ~25x too small, and the measured consequence is exactly that:
  under a held command the ball slides down out of the jaw at ~0.7 m/s where
  MuJoCo holds it stationary (ball z 0.6364 -> 0.5789 m over 80 ms; MuJoCo
  0.6364 -> 0.6330).  Friction cannot beat gravity when its budget is divided
  by a mobility the joint does not have.

  Everything else -- the Coulomb bound, the restitution term, the `sinking`
  gate, and the pseudo-velocity subtraction that makes the positional
  push-out momentum-neutral -- is unchanged from the routine below.
  """
  dlambda, x_i_pre = dlambda_pack
  anc_np, own, free_np, n_dof = _art_tables(sys)
  n_link = sys.num_links()
  anc = jp.asarray(anc_np)
  ang, vel, anchor, winv, a6 = _art_dofs(sys, state, own, n_dof)

  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)
  minv_p = jp.concatenate([inv_mass, jp.zeros((1,))])
  iinv_p = jp.concatenate([inv_inertia, jp.zeros((1, 3, 3))])
  free_p = jp.concatenate([jp.asarray(free_np), jp.zeros((1,))])
  anc_p = jp.concatenate([anc, jp.zeros((1, n_dof))], axis=0)

  # the velocity `project_xd` manufactured out of the positional correction
  xdv_push = jax.vmap(_push_vel_one, in_axes=(0, 0, None))(
      state.x_i, x_i_pre, sys.opt.timestep)
  xd_eff = state.xd_i - xdv_push

  link = jp.array(contact.link_idx).T
  li = link.reshape(-1)
  n_w = -contact.frame[:, 0]
  cpos = contact.pos
  c = contact.dist
  pen = c < 0
  onl = link > -1

  xz = state.x_i.concatenate(Transform.zero((1,)))
  xpos = xz.pos.take(link, axis=0)
  minv = minv_p.take(link)
  iinv = iinv_p.take(link, axis=0)
  fr = free_p.take(link)
  ancL = anc_p.take(link, axis=0)
  r = cpos[:, None, :] - xpos

  def _w(direc):
    b6 = jp.concatenate([jp.cross(cpos, direc), direc], axis=-1)
    g = b6 @ a6.T
    w_art = jp.einsum('kd,ksd->ks', (g * g) * winv, ancL)
    cr = jp.cross(r, direc[:, None, :])
    w_free = minv + jp.einsum('ksa,ksab,ksb->ks', cr, iinv, cr)
    return w_art + fr * w_free

  def _pt_vel(motion):
    m = motion.concatenate(Motion.zero((1,))).take(link)
    return m.vel + jp.cross(m.ang, r)

  pv = _pt_vel(xd_eff)
  rel = pv[:, 0] - pv[:, 1]
  v_n = jax.vmap(jp.dot)(rel, n_w)
  v_t = rel - n_w * v_n[:, None]
  v_t_norm = _rownorm(v_t)
  v_t_dir = v_t / jp.where(v_t_norm > 1e-9, v_t_norm, 1.0)[:, None]

  dvel = -jp.minimum(contact.friction[:, 0] * jp.abs(dlambda) / sys.opt.timestep,
                     v_t_norm)
  wt = _w(v_t_dir)
  p_dyn = (dvel / (wt[:, 0] + wt[:, 1] + 1e-6))[:, None] * v_t_dir

  pvp = _pt_vel(xd_i_prev)
  v_n_prev = jax.vmap(jp.dot)(pvp[:, 0] - pvp[:, 1], n_w)
  dv_rest = n_w * (-v_n - jp.minimum(contact.elasticity * v_n_prev, 0))[:, None]
  cr_ = _rownorm(dv_rest)
  nr = dv_rest / (cr_ + 1e-6)[:, None]
  wr = _w(nr)
  dlambda_rest = cr_ / (wr[:, 0] + wr[:, 1] + 1e-6)
  sinking = v_n_prev <= 0.0

  pimp = (dlambda_rest[:, None] * nr * sinking[:, None] + p_dyn) * pen[:, None]

  act = (pen[:, None] & onl).astype(jp.float32)
  nact = segment_sum(act.reshape(-1), li, n_link)
  share = 1.0 / (jp.concatenate([nact, jp.zeros((1,))]).take(link) + 1e-8)
  imp = (pimp[:, None, :] * jp.array([1.0, -1.0])[None, :, None]
         * share[..., None] * onl[..., None])

  lin, angv = _art_distribute(state.x_i.pos, anc, ang, vel, anchor, winv, a6,
                              li, cpos, imp, r, minv, iinv, fr, n_link)
  return Motion(vel=lin, ang=angv) - xdv_push


def _push_vel_one(x, x_pre, h):
  vel = (x.pos - x_pre.pos) / h
  dq = math.relative_quat(x_pre.rot, x.rot)
  ang = 2.0 * dq[1:] / h
  ang = jp.where(dq[0] >= 0.0, 1.0, -1.0) * ang
  return Motion(vel=vel, ang=ang)


def resolve_velocity_vecform(
    sys: System,
    state: State,
    xd_i_prev: Motion,
    contact: Contact,
    dlambda: jax.Array,
) -> Motion:
  """Velocity-level collision update for position based dynamics.

  The update equations here follow section 3.6 of Müller et al.'s Extended
  Position Based Dynamics. (Müller, Matthias, et al. "Detailed rigid body
  simulation with extended position based dynamics." Computer Graphics Forum.
  Vol. 39. No. 8. 2020.).

  explore_bench fork: the positional contact correction is applied as a
  **pseudo-velocity** (Bullet's "split impulse"), i.e. it moves the bodies
  apart without changing their momentum, and the momentum transfer is left
  entirely to the impulse below.

  Why.  ``integrator.project_xd`` sets ``v = (x - x_prev)/h`` over the whole
  substep, so every metre of contact push-out becomes ``1/h`` metres per
  second of real velocity -- 1 mm at h = 2 ms is 0.5 m/s manufactured.  This
  pass is meant to take it back out again, and cannot: it can only remove the
  relative NORMAL velocity at the contact point, so an off-centre push leaves
  the centre of mass translating and the body spinning.  Measured on a pinned
  drop of hibachi's egg onto the 1.2 mm spatula blade, with ``sys.elasticity``
  already ZERO, in the one substep in which the contact appears:

      penetration reported   15.8 mm      (mjx's ellipsoid-capsule collider
                                            reports NO contact until this
                                            deep, so it cannot be avoided here)
      position correction    20.8 mm
      project_xd velocity    +9.46 m/s    manufactured out of that correction
      cancelled by this pass -7.90 m/s
      left over              +1.57 m/s    against an impact speed of 0.94 m/s

  i.e. a rebound coefficient of 1.66 at zero elasticity -- energy created by
  the contact.  Subtracting the push velocity here makes the correction exactly
  momentum-neutral, so no elasticity value below zero is ever needed.  Contact
  momentum transfer is not lost with it: the restitution term already drives
  the relative normal velocity to ``-e * v_n_prev``, which for e = 0 is the
  inelastic collision response and moves a struck body with its striker.

  Args:
    sys: System to update
    state: positional pipeline state
    xd_i_prev: velocity immediately preceding PBD velocity projection
    contact: Contact information for collision
    dlambda: ``(dlambda, x_i_pre)`` as returned by ``resolve_position`` --
      the normal force of contact times time step squared, and the link
      transforms as they stood before the positional contact correction

  Returns:
    Velocity level update for system state.
  """
  if contact is None:
    return Motion.zero((sys.num_links(),))

  # explore_bench fork: opt-in articulated response (BRAX_FORK_ARTMASS=1).
  if _ART_ENABLED:
    return _resolve_velocity_art(sys, state, xd_i_prev, contact, dlambda)

  dlambda, x_i_pre = dlambda

  x_i = state.x_i.concatenate(Transform.zero((1,)))
  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)

  # the velocity `project_xd` manufactured out of the positional contact
  # correction alone, computed the same way it was
  @jax.vmap
  def _push_vel(x, x_pre):
    vel = (x.pos - x_pre.pos) / sys.opt.timestep
    dq = math.relative_quat(x_pre.rot, x.rot)
    ang = 2.0 * dq[1:] / sys.opt.timestep
    ang = jp.where(dq[0] >= 0.0, 1.0, -1.0) * ang
    return Motion(vel=vel, ang=ang)

  xdv_push = _push_vel(state.x_i, x_i_pre)
  xd_eff = state.xd_i - xdv_push

  @jax.vmap
  def impulse(contact, dlambda):
    link_idx = jp.array(contact.link_idx)
    x = x_i.take(link_idx)
    xd = xd_eff.concatenate(Motion.zero((1,))).take(link_idx)
    xd_prev = xd_i_prev.concatenate(Motion.zero((1,))).take(link_idx)
    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)
    mass_inv = inv_mass.take(link_idx) * (link_idx > -1)

    n = -contact.frame[0]
    rel_vel = (
        xd.vel[0]
        + jp.cross(xd.ang[0], contact.pos - x.pos[0])
        - (xd.vel[1] + jp.cross(xd.ang[1], contact.pos - x.pos[1]))
    )
    v_n = jp.dot(rel_vel, n)
    v_t = rel_vel - n * v_n
    v_t_dir, v_t_norm = math.normalize(v_t)
    dvel = -v_t_dir * jp.minimum(
        contact.friction[0] * jp.abs(dlambda) / sys.opt.timestep, v_t_norm
    )

    angw_1 = jp.cross((contact.pos - x.pos[0]), v_t_dir)
    angw_2 = jp.cross((contact.pos - x.pos[1]), v_t_dir)
    w1 = mass_inv[0] + jp.dot(angw_1, i_inv[0] @ angw_1)
    w2 = mass_inv[1] + jp.dot(angw_2, i_inv[1] @ angw_2)

    p_dyn = dvel / (w1 + w2 + 1e-6)

    # restitution
    rel_vel_prev = (
        xd_prev.vel[0] + jp.cross(xd_prev.ang[0], contact.pos - x.pos[0])
    ) - (xd_prev.vel[1] + jp.cross(xd_prev.ang[1], contact.pos - x.pos[1]))
    v_n_prev = jp.dot(rel_vel_prev, n)
    dv_rest = n * (-v_n - jp.minimum(contact.elasticity * v_n_prev, 0))

    pos_p = contact.pos
    pos_c = contact.pos + contact.frame[0] * contact.dist
    dx = dv_rest
    pos_p, pos_c = pos_p - x.pos[0], pos_c - x.pos[1]

    c = math.safe_norm(dx)
    n = dx / (c + 1e-6)

    # ignoring inertial effects for now
    cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
    w1 = mass_inv[0] + jp.dot(cr1, i_inv[0] @ cr1)
    w2 = mass_inv[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambda_rest = c / (w1 + w2 + 1e-6)
    penetrating = contact.dist < 0
    sinking = v_n_prev <= 0.0

    p = Force.create(vel=(dlambda_rest * n * sinking + p_dyn) * penetrating)

    return p, jp.asarray(penetrating, dtype=jp.float32)

  p, is_contact = impulse(contact, dlambda)

  # calculate the impulse to each link center of mass
  p = jax.tree.map(lambda x: jp.concatenate((x, -x)), p)
  pos = jp.tile(contact.pos, (2, 1))
  link_idx = jp.concatenate(contact.link_idx)
  xp_i = Transform.create(pos=pos - x_i.take(link_idx).pos).vmap().do(p)
  xp_i = jax.tree.map(lambda x: segment_sum(x, link_idx, sys.num_links()), xp_i)

  # average the impulse across multiple contacts
  num_contacts = segment_sum(jp.tile(is_contact, 2), link_idx, sys.num_links())
  xp_i = xp_i / (num_contacts.reshape((-1, 1)) + 1e-8)

  # convert impulse to delta-velocity
  xdv_i = Motion(
      vel=jax.vmap(lambda x, y: x * y)(inv_mass, xp_i.vel),
      ang=jax.vmap(lambda x, y: x @ y)(inv_inertia, xp_i.ang),
  )

  # explore_bench fork: and take back the velocity the positional correction
  # manufactured, so the push-out is a pseudo-velocity and carries no momentum
  return xdv_i - xdv_push


# explore_bench fork: LAYOUT-ONLY A/B in front of the two routines above.
#
# The originals are kept in-tree as `*_vecform` (the same convention
# `_fork_narrowphase.box_box_vecform` and `_fork_lane`'s rewrites use) and are
# the correctness reference: `BRAX_FORK_LANE_CONTACT=0` restores them.
#
# The rewrite is in `brax/_fork_contact_lane.py` and changes NOTHING about the
# physics -- not the inverse mass, not the constraint-count normalisation, not
# the iteration scheme.  It changes only how the per-CONTACT intermediates are
# laid out, from `(n_worlds, n_con, 3)` (2.3% TPU register occupancy at
# n_con = 369) to component `(n_worlds, n_con)` arrays.  `bytes accessed` for
# one vmapped substep is the metric; see `.work/contact_layout_report.md`.
#
# IF YOU CHANGE THE PHYSICS IN EITHER FUNCTION ABOVE, change it in
# `_fork_contact_lane.py` too.  `.work/contact_equiv.py` compares the two on
# randomised states and will fail loudly if they drift apart.
#
# `BRAX_FORK_ARTMASS=1` (the articulated resolution) is a DIFFERENT physics and
# has no lane form, so it takes precedence and routes back to the vecform.


def resolve_position(sys, state, x_i_prev, contact):
  """Resolves positional collision constraint (see `resolve_position_vecform`)."""
  if contact is None or _ART_ENABLED or not _lane_contact.ENABLED:
    return resolve_position_vecform(sys, state, x_i_prev, contact)
  return _lane_contact.resolve_position(sys, state, x_i_prev, contact)


def resolve_velocity(sys, state, xd_i_prev, contact, dlambda):
  """Velocity-level collision update (see `resolve_velocity_vecform`)."""
  if contact is None or _ART_ENABLED or not _lane_contact.ENABLED:
    return resolve_velocity_vecform(sys, state, xd_i_prev, contact, dlambda)
  return _lane_contact.resolve_velocity(sys, state, xd_i_prev, contact, dlambda)
