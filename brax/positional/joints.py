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

"""Joint definition and apply functions."""

# pylint:disable=g-multiple-import
import os as _os
from typing import Tuple

from brax import _fork_lane as lane
from brax import com
from brax import kinematics
from brax import math
from brax import scan
from brax.base import (DoF, Force, Link, Motion, Q_WIDTHS, QD_WIDTHS,
                       System, Transform)
from brax.positional.base import State
import jax
from jax import numpy as jp
import numpy as np
from jax.ops import segment_sum


def acceleration_update(sys: System, state: State, tau: jax.Array) -> Force:
  """Calculates forces to apply to links resulting from joint constraints.

  Args:
    sys: System defining kinematic tree of joints
    state: positional pipeline state
    tau: joint force vector

  Returns:
    xf_i: force to apply to link center of mass in world frame
  """

  def _free_joint(*_) -> Force:
    return Force(vel=jp.zeros(3), ang=jp.zeros(3))

  def _damp(link: Link, jd: Motion, dof: DoF, tau: jax.Array):
    vel = jp.sum(jax.vmap(jp.multiply)(tau, dof.motion.vel), axis=0)
    ang = jp.sum(jax.vmap(jp.multiply)(tau, dof.motion.ang), axis=0)

    # damp the angular and linear motion
    ang -= link.constraint_ang_damping * jd.ang
    vel -= link.constraint_vel_damping * jd.vel

    return Force(ang=ang, vel=vel)

  def j_fn(typ, link, jd, dof, tau):
    # change dof-shape variables into link-shape
    reshape_fn = lambda x: x.reshape((jd.ang.shape[0], -1) + x.shape[1:])
    tau, dof = jax.tree.map(reshape_fn, (tau, dof))
    j_fn_map = {
        'f': _free_joint,
        '1': _damp,
        '2': _damp,
        '3': _damp,
    }

    return jax.vmap(j_fn_map[typ])(link, jd, dof, tau)

  # calculate forces in joint frame, then convert to world frame
  link, jd, dof = sys.link, state.jd, sys.dof
  jf = scan.link_types(sys, j_fn, 'lldd', 'l', link, jd, dof, tau)
  xf = Transform.create(rot=state.a_p.rot).vmap().do(jf)
  # move force to center of mass offset
  fc = Transform.create(pos=state.a_c.pos - state.x_i.pos).vmap().do(xf)
  # also add opposite force to parent link at center of mass
  parent_idx = jp.array(sys.link_parents)
  x_i_parent = state.x_i.take(parent_idx)
  fp = Transform.create(pos=state.a_p.pos - x_i_parent.pos).vmap().do(xf)
  fp = jax.tree.map(lambda x: segment_sum(x, parent_idx, sys.num_links()), fp)
  xf_i = fc - fp
  return xf_i


def position_update_vecform(sys: System, state: State) -> Transform:
  """brax's original.  Calculates position-level joint updates in CoM coordinates for joints.

  Args:
    sys: System defining kinematic tree of joints
    state: positional pipeline state

  Returns:
    x_i: new position after update
  """

  p_idx = jp.array(sys.link_parents)
  xi_p = state.x_i.concatenate(Transform.zero((1,))).take(p_idx)
  j, _, a_p, a_c = kinematics.world_to_joint(sys, state.x, state.xd)

  # pad sys and dof data withs 0s along inactive axes
  d_j = jax.vmap(_three_dof_joint_update)(j, *_sphericalize(sys, j))
  free_mask = jp.array([l != 'f' for l in sys.link_types])
  d_j = jax.tree.map(lambda x: jax.vmap(jp.multiply)(x, free_mask), d_j)
  d_w = jax.tree.map(lambda x: jax.vmap(math.rotate)(x, a_p.rot), d_j)

  i_inv = com.inv_inertia(sys, state.x)
  i_inv_p = jax.vmap(jp.multiply)(i_inv[p_idx], p_idx > -1)
  mass_inv = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  mass_inv_p = mass_inv[p_idx] * (p_idx > -1)
  dp_p_pos, dp_c_pos = jax.vmap(_translation_update)(
      a_p, xi_p, i_inv_p, mass_inv_p, a_c, state.x_i, i_inv, mass_inv, -d_w.pos
  )
  dp_p_ang, dp_c_ang = jax.vmap(_rotation_update)(
      xi_p, i_inv_p, state.x_i, i_inv, d_w.rot
  )

  dp_c = dp_c_pos * sys.joint_scale_pos + dp_c_ang * sys.joint_scale_ang
  dp_p = dp_p_pos * sys.joint_scale_pos + dp_p_ang * sys.joint_scale_ang
  dp_p = jax.tree.map(lambda f: segment_sum(f, p_idx, sys.num_links()), dp_p)

  # explore_bench fork: JACOBI CONSTRAINT NORMALISATION.
  #
  # Every joint is projected against the configuration at the START of the
  # sweep, so a link touched by k joints (its own, plus one per child)
  # receives k corrections that each assume the other k-1 did not happen.
  # Summing them applies roughly k times the needed displacement. With one
  # sweep and the shipped under-relaxation (joint_scale_pos 0.5,
  # joint_scale_ang 0.2) that merely made a serial arm sag; iterating it
  # DIVERGED -- measured on Spot, 4 and 16 sweeps both reach NaN where 1 sweep
  # reaches 4.79 rad of hold error, which is the signature of an over-relaxed
  # Jacobi iteration rather than a slowly-converging one.
  #
  # Dividing each link's accumulated correction by the number of constraints
  # acting on it is the standard fix (the "mass splitting" of Macklin et al.,
  # Small Steps in Physics Simulation) and is what makes the iteration a
  # contraction. It is computed from `link_parents`, which is static, so it
  # costs one precomputed vector and no runtime branching.
  n_con = np.ones(sys.num_links(), dtype=np.float32)
  for parent in sys.link_parents:
    if parent >= 0:
      n_con[parent] += 1.0
  # a link whose own joint is free has no parent constraint of its own
  for i, typ in enumerate(sys.link_types):
    if typ == 'f':
      n_con[i] -= 1.0
  inv_n = jp.asarray(1.0 / np.maximum(n_con, 1.0)).reshape(-1, 1)
  dp = jax.tree.map(lambda a, b: (a + b) * inv_n, dp_c, dp_p)

  return state.x_i + dp


def _rotation_update_lane(xi_p_rot, i_p, xi_c_rot, i_c, dq):
  """``_rotation_update`` with every quantity in lane form."""
  n, th = lane.normalize3(dq)
  w1 = lane.dot3(n, lane.matvec(i_p, n))
  w2 = lane.dot3(n, lane.matvec(i_c, n))
  dlambda = -th / (w1 + w2 + 1e-6)
  p = lane.neg3(lane.scale3(n, dlambda))
  rot_p = lane.scale4(lane.vec_quat_mul(lane.matvec(i_p, p), xi_p_rot), -0.5)
  rot_c = lane.scale4(lane.vec_quat_mul(lane.matvec(i_c, p), xi_c_rot), 0.5)
  return rot_p, rot_c


def _translation_update_lane(pos_p, xi_p_pos, xi_p_rot, i_p, mass_inv_p,
                             pos_c, xi_c_pos, xi_c_rot, i_c, mass_inv_c, dx):
  """``_translation_update`` with every quantity in lane form."""
  pos_p = lane.sub3(pos_p, xi_p_pos)
  pos_c = lane.sub3(pos_c, xi_c_pos)
  n, c = lane.normalize3(dx)
  cr1, cr2 = lane.cross3(pos_p, n), lane.cross3(pos_c, n)
  w1 = mass_inv_p + lane.dot3(cr1, lane.matvec(i_p, cr1))
  w2 = mass_inv_c + lane.dot3(cr2, lane.matvec(i_c, cr2))
  dlambda = -c / (w1 + w2 + 1e-6)
  p = lane.scale3(n, dlambda)
  rot_p = lane.scale4(
      lane.vec_quat_mul(lane.matvec(i_p, lane.cross3(pos_p, p)), xi_p_rot), -0.5)
  rot_c = lane.scale4(
      lane.vec_quat_mul(lane.matvec(i_c, lane.cross3(pos_c, p)), xi_c_rot), 0.5)
  return ((lane.neg3(lane.scale3(p, mass_inv_p)), rot_p),
          (lane.scale3(p, mass_inv_c), rot_c))


def _n_con(sys):
  """Jacobi constraint-count normaliser -- see position_update_vecform."""
  n_con = np.ones(sys.num_links(), dtype=np.float32)
  for parent in sys.link_parents:
    if parent >= 0:
      n_con[parent] += 1.0
  for i, typ in enumerate(sys.link_types):
    if typ == 'f':
      n_con[i] -= 1.0
  return 1.0 / np.maximum(n_con, 1.0)


def position_update(sys: System, state: State) -> Transform:
  """Calculates position-level joint updates in CoM coordinates for joints.

  explore_bench fork: a pure LAYOUT rewrite of ``position_update_vecform``.
  Every 3-vector, quaternion and 3x3 is carried as separate ``(..., n_link)``
  component arrays, the parent ``take`` becomes a constant index gather and
  the ``segment_sum`` a constant contraction.  ``_three_dof_joint_update`` and
  ``_sphericalize`` are unchanged -- they are the one part of this function
  whose per-link 3x3 frame algebra does not lower cleanly to lane form, and
  they were measured to be the smaller half of it.

  Two things change on TPU, both measured:
  * ``(n_worlds, n_link, 3)`` intermediates occupy 1.8% of a vector register;
    ``(n_worlds, n_link)`` components occupy 19.5%.
  * ``i_inv @ v`` and the ``jp.dot``s inside ``math.rotate`` are
    ``dot_general``s, i.e. DEFAULT-precision (bfloat16) matmuls -- see
    ``lane.matvec``.
  """
  if not (lane.ENABLED and lane.ENABLE_POS):
    return position_update_vecform(sys, state)
  sel = kinematics.parent_mat(sys)
  xi_pos, xi_rot = lane.vec3(state.x_i.pos), lane.quat(state.x_i.rot)

  (jp_pos, jp_rot, _, _, ap_pos, ap_rot,
   ac_pos, _) = kinematics.world_to_joint_lane(
       sys, lane.vec3(state.x.pos), lane.quat(state.x.rot),
       lane.vec3(state.xd.ang), lane.vec3(state.xd.vel))
  j = Transform(pos=lane.stack3(jp_pos), rot=lane.stack4(jp_rot))

  d_j = jax.vmap(_three_dof_joint_update)(j, *_sphericalize(sys, j))
  free_mask = jp.asarray(
      np.array([l != 'f' for l in sys.link_types], dtype=np.float32))
  d_w_pos = lane.rotate(lane.scale3(lane.vec3(d_j.pos), free_mask), ap_rot)
  d_w_rot = lane.rotate(lane.scale3(lane.vec3(d_j.rot), free_mask), ap_rot)

  i_inv = com.inv_inertia_lane(sys, lane.quat(state.x.rot))
  mass_inv = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  # ONE gather for all seventeen parent components: the parent CoM transform
  # (3 + 4), its world inverse inertia (9) and its inverse mass.
  flat = i_inv[0] + i_inv[1] + i_inv[2]
  g = lane.gather_many(
      xi_pos + xi_rot + flat + (mass_inv,),
      (0., 0., 0., 1., 0., 0., 0.) + (0.,) * 10, sel)
  xip_pos, xip_rot = g[0:3], g[3:7]
  i_inv_p = (g[7:10], g[10:13], g[13:16])
  mass_inv_p = g[16]

  (pp_pos, pp_rot), (pc_pos, pc_rot) = _translation_update_lane(
      ap_pos, xip_pos, xip_rot, i_inv_p, mass_inv_p,
      ac_pos, xi_pos, xi_rot, i_inv, mass_inv, lane.neg3(d_w_pos))
  ap_rot_p, ap_rot_c = _rotation_update_lane(
      xip_rot, i_inv_p, xi_rot, i_inv, d_w_rot)

  ksp, ksa = sys.joint_scale_pos, sys.joint_scale_ang
  # `_rotation_update` returns a zero translation, so the angular family
  # contributes nothing to pos (0 * joint_scale_ang is exactly 0).
  dpc_pos = lane.scale3(pc_pos, ksp)
  dpp_pos = lane.scale3(pp_pos, ksp)
  dpc_rot = lane.add4(lane.scale4(pc_rot, ksp), lane.scale4(ap_rot_c, ksa))
  dpp_rot = lane.add4(lane.scale4(pp_rot, ksp), lane.scale4(ap_rot_p, ksa))

  seg = _segment_mat(sys)
  acc = lane.segment_to_parent_many(dpp_pos + dpp_rot, seg)
  dpp_pos, dpp_rot = acc[0:3], acc[3:7]

  inv_n = jp.asarray(_n_con(sys))
  pos = lane.scale3(lane.add3(dpc_pos, dpp_pos), inv_n)
  rot = lane.scale4(lane.add4(dpc_rot, dpp_rot), inv_n)
  return Transform(pos=state.x_i.pos + lane.stack3(pos),
                   rot=state.x_i.rot + lane.stack4(rot))


_SEG_MAT = {}


def _segment_mat(sys):
  key = tuple(sys.link_parents)
  m = _SEG_MAT.get(key)
  if m is None:
    m = _SEG_MAT[key] = lane.segment_matrix(key)
  return m


def _translation_update(
    pos_p: Transform,
    xi_p: Transform,
    i_inv_p: jax.Array,
    mass_inv_p: jax.Array,
    pos_c: Transform,
    xi_c: Transform,
    i_inv_c: jax.Array,
    mass_inv_c: jax.Array,
    dx: jax.Array,
) -> Tuple[Transform, Transform]:
  """Calculates a position based translational update."""

  pos_p, pos_c = pos_p.pos - xi_p.pos, pos_c.pos - xi_c.pos
  n, c = math.normalize(dx)

  cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
  w1 = mass_inv_p + jp.dot(cr1, i_inv_p @ cr1)
  w2 = mass_inv_c + jp.dot(cr2, i_inv_c @ cr2)
  dlambda = -c / (w1 + w2 + 1e-6)
  p = dlambda * n

  rot_p = -0.5 * math.vec_quat_mul(i_inv_p @ jp.cross(pos_p, p), xi_p.rot)
  rot_c = 0.5 * math.vec_quat_mul(i_inv_c @ jp.cross(pos_c, p), xi_c.rot)
  pos_p, pos_c = -p * mass_inv_p, p * mass_inv_c

  return Transform(pos=pos_p, rot=rot_p), Transform(pos=pos_c, rot=rot_c)


def _rotation_update(
    xi_p: Transform,
    i_inv_p: jax.Array,
    xi_c: Transform,
    i_inv_c: jax.Array,
    dq: jax.Array,
) -> Tuple[Transform, Transform]:
  """Calculates a position based rotational update."""

  n, th = math.normalize(dq)

  # ignoring inertial effects for now
  w1, w2 = jp.dot(n, i_inv_p @ n), jp.dot(n, i_inv_c @ n)
  dlambda = -th / (w1 + w2 + 1e-6)
  p = -dlambda * n

  rot_p = -0.5 * math.vec_quat_mul(i_inv_p @ p, xi_p.rot)
  rot_c = 0.5 * math.vec_quat_mul(i_inv_c @ p, xi_c.rot)

  return Transform.create(rot=rot_p), Transform.create(rot=rot_c)


def _rotation_update_about(
    xi_p: Transform,
    i_inv_p: jax.Array,
    xi_c: Transform,
    i_inv_c: jax.Array,
    dq: jax.Array,
    pivot: jax.Array,
) -> Tuple[Transform, Transform]:
  """`_rotation_update`, but about a PIVOT rather than about each body's CoM.

  A hinge rotates its child about the joint ANCHOR. `_rotation_update` applies
  a pure rotation to the CoM frame and emits no translation, so the anchor
  swings out on an arc of radius |com - anchor| -- and `position_update`'s
  translation constraint then hauls it back, cancelling most of the rotation
  that put it there. That is fine for the joint constraint itself, which
  applies both halves together and converges. It is NOT fine for the DRIVE:
  the drive gets ONE force-limited budget per substep (its Lagrange multiplier
  is clamped at `drive_lam_max`) while the joint projection re-runs every
  sweep unopposed, so the cancellation is one-sided and MORE sweeps make the
  servo weaker. Measured on the Panda: 0.33x MuJoCo's step response at 8
  sweeps, 0.25x at 64, against 1.15x with the joint constraint disabled
  entirely.

  Rotating a rigid body by a small rotation vector `w` about its own CoM moves
  the material point at `pivot` by `w x (pivot - com)`. Translating the CoM by
  `w x (com - pivot)` cancels exactly that, leaving the pivot fixed -- which is
  what a hinge does, and what leaves the translation constraint nothing to
  undo.
  """
  n, th = math.normalize(dq)
  w1, w2 = jp.dot(n, i_inv_p @ n), jp.dot(n, i_inv_c @ n)
  dlambda = -th / (w1 + w2 + 1e-6)
  p = -dlambda * n

  v_p = -(i_inv_p @ p)
  v_c = i_inv_c @ p
  rot_p = 0.5 * math.vec_quat_mul(v_p, xi_p.rot)
  rot_c = 0.5 * math.vec_quat_mul(v_c, xi_c.rot)
  pos_p = jp.cross(v_p, xi_p.pos - pivot)
  pos_c = jp.cross(v_c, xi_c.pos - pivot)
  return Transform(pos=pos_p, rot=rot_p), Transform(pos=pos_c, rot=rot_c)


def _sphericalize_vecform(sys, j):
  """Transforms system state into an all-3-dof version of the system."""

  def pad_free(_):
    # create dummy data for free links
    inf = jp.array([jp.inf, jp.inf, jp.inf])
    return (
        (-inf, inf),
        Motion(ang=jp.eye(3), vel=jp.eye(3)),
    )

  def pad_x_dof(dof, x):
    if dof.limit:
      stack_fn = lambda a: jp.concatenate((a, jp.zeros(3 - x)))
      limit = jax.tree.map(stack_fn, dof.limit)
    else:
      inf = jp.array([jp.inf, jp.inf, jp.inf])
      limit = (-inf, inf)
    padded_motion = dof.motion.concatenate(Motion.zero((3 - x,)))
    return limit, padded_motion

  def j_fn(typ, j, dof):
    # change dof-shape variables into link-shape
    reshape_fn = lambda x: x.reshape((j.pos.shape[0], -1) + x.shape[1:])
    dof = jax.tree.map(reshape_fn, dof)
    j_fn_map = {
        'f': pad_free,
        '1': lambda x: pad_x_dof(x, 1),
        '2': lambda x: pad_x_dof(x, 2),
        '3': lambda x: pad_x_dof(x, 3),
    }
    limit, padded_motion = jax.vmap(j_fn_map[typ])(dof)

    if typ == 'f':
      joint_frame_fn = lambda x: (Motion(vel=jp.eye(3), ang=jp.eye(3)), 1)
    else:
      joint_frame_fn = kinematics.link_to_joint_frame
    joint_frame, parity = jax.vmap(joint_frame_fn)(dof.motion)

    return limit, padded_motion, joint_frame, parity

  result = scan.link_types(sys, j_fn, 'ld', 'l', j, sys.dof)

  return result


#: `_sphericalize` result, cached on the identity of the `DoF` it is built
#: from.  The DoF object is held in the value so the key cannot be recycled by
#: the garbage collector onto a different object.
_SPH_CACHE = {}

#: A/B switch, in the style of `BRAX_FORK_LANE` / `BRAX_FORK_NARROWPHASE`:
#: `BRAX_FORK_SPH_CONST=0` restores brax's in-graph recomputation, which is
#: what reproduces the TPU core halt described below.
_SPH_CONST = _os.environ.get(
    'BRAX_FORK_SPH_CONST', '1').lower() not in ('0', 'false', 'off', 'no')


def _sphericalize(sys, j):
  """`_sphericalize_vecform`, evaluated ONCE per model instead of per sweep.

  **This exists to fix a TPU core halt**, not as an optimisation.

  The result depends on `sys.dof` alone: `j` is read only for
  `j.pos.shape[0]`, the size of the type group being scanned.  So it is a
  MODEL CONSTANT that brax was rebuilding inside the joint-sweep scan, inside
  the substep scan, inside the vmap over worlds.

  That constant subgraph contains `jax.vmap` of functions whose output does
  not depend on their input -- `pad_free` returns `jp.eye(3)` motions and
  +/-inf limits, and the free branch of `joint_frame_fn` returns `jp.eye(3)`
  -- so with n FREE links it lowers to a `broadcast_in_dim` of a 3x3 constant
  to `f32[n,3,3]`, which XLA places in CMEM and then concatenates (as a `pad`)
  with the hinge group's rows.  For n >= 2 and a world batch > 1, XLA:TPU
  emits a `dma.cmem_to_vmem` for that broadcast whose window walks off the end
  of its own 3-granule base and fails the hardware bounds check, halting the
  core:

      RuntimeUnexpectedCoreHalt ... BoundsCheck 0 [deref of %s20]
      for %23 = dma.cmem_to_vmem ... element_size_in_bytes: 2048
      base_bounds: (3, 1)   window_bounds: (2, 1)   pad_high: (1, 0)
      hlo: broadcast_in_dim...clone.2, while.<inner>, while.<outer>

  matching, in the optimised HLO,

      %broadcast_in_dim.43.clone.2 = f32[2,3,3]{0,2,1:T(4,128)S(3)}
          broadcast(f32[3,3] ...), dimensions={1,2}
          op_name="jit(rollout)/vmap()/while/body/closed_call/vmap()/
                   broadcast_in_dim"

  n >= 2 is necessary but not sufficient -- it also needs XLA to choose that
  DMA form, which depends on the rest of the graph.  Measured on a synthetic
  model of n free bodies plus m hinge links at 256 worlds: (2, 2), (2, 4),
  (2, 23), (2, 24), (3, 24) halt; (1, 24), (1, 25), (2, 0), (2, 8) do not, and
  neither does (2, 24) at ONE world.  In this repo it halted
  `dynmanip/hibachi-egg-catch` (24 hinges + 2 free) while leaving
  `dynmanip/spot-catch` and `dynmanip/drone-catch` -- which also have two free
  links -- working, i.e. those two carried the same latent hazard.

  Materialising the constant on the host removes the broadcast, so the pattern
  is never emitted for any model.  The values are bit-identical: 0.0e+00
  against the in-graph result over all eight dynmanip tasks
  (`.work/hibachi_tpu_crash.md`).  `BRAX_FORK_SPH_CONST=0` restores the old
  path for an A/B.
  """
  if not _SPH_CONST:
    return _sphericalize_vecform(sys, j)
  dof = sys.dof
  if any(isinstance(x, jax.core.Tracer)
         for x in jax.tree_util.tree_leaves(dof)):
    return _sphericalize_vecform(sys, j)   # sys is being traced: not a constant
  key = (id(dof), tuple(sys.link_types))
  hit = _SPH_CACHE.get(key)
  if hit is not None:
    return hit[1]
  # `ensure_compile_time_eval` is required, not decorative: this is normally
  # first reached from INSIDE the substep scan's trace, and under a trace a
  # fresh `jp.zeros` is a tracer rather than a value, so the result would be
  # built into the graph exactly as before.
  with jax.ensure_compile_time_eval():
    out = _sphericalize_vecform(sys, Transform.zero((sys.num_links(),)))
    leaves, treedef = jax.tree_util.tree_flatten(out)
    if any(isinstance(x, jax.core.Tracer) for x in leaves):
      return _sphericalize_vecform(sys, j)  # could not be folded; unchanged
    out = jax.tree_util.tree_unflatten(
        treedef, [jp.asarray(np.asarray(x)) for x in leaves])
  _SPH_CACHE[key] = (dof, out)
  return out


def _three_dof_joint_update(
    x: Transform,
    limit: Tuple[jax.Array, jax.Array],
    motion: Motion,
    joint_frame: Motion,
    parity: float,
) -> Transform:
  """Returns position-level displacements in joint frame for spherical joint."""
  (
      axis_c,
      (_, _, _),
      (line_of_nodes, axis_1_p_in_xz_c),
  ) = kinematics.axis_angle_ang(x, joint_frame, parity)

  axis_p = joint_frame.ang

  axis_1_p_in_xz_c = (
      jp.dot(axis_p[0], axis_c[0]) * axis_c[0]
      + jp.dot(axis_p[0], axis_c[1]) * axis_c[1]
  )

  axis_1_p_in_xz_c, _ = math.normalize(axis_1_p_in_xz_c)
  axis_2_normal, _ = math.normalize(jp.cross(axis_1_p_in_xz_c, axis_p[0]))
  limit_axes = jp.array([
      axis_p[0],
      -axis_2_normal * jp.sign(jp.dot(axis_p[0], axis_c[2])),
      axis_c[2],
  ])

  ref_axis_1 = jp.array([axis_p[1], axis_p[0], line_of_nodes])
  ref_axis_2 = jp.array([line_of_nodes, axis_1_p_in_xz_c, axis_c[1]])

  def limit_angle(n, n_1, n_2, motion, limit, ang_limit):
    ph = math.signed_angle(n, n_1, n_2)
    ph = jp.clip(ph, ang_limit[0], ang_limit[1])
    fixrot = math.quat_rot_axis(n, ph)
    n1 = math.rotate(n_1, fixrot)
    dq = jp.cross(n1, n_2)

    active_axis = motion.vel.any()
    xp = motion.vel @ x.pos
    dx = xp - jp.clip(xp, limit[0], limit[1])
    dx = motion.vel * dx * active_axis

    return dq, dx

  # positional constraints
  dx = -x.pos

  # remove components of update along free prismatic axes
  dx *= 1 - motion.vel.any(axis=0)

  # limit constraints
  if limit:
    # for freezing angular dofs on prismatic axes
    padded_ang_limit = jp.where(
        motion.vel.any(axis=1),
        jp.zeros((2, 3)),
        jp.array(limit),
    ).transpose()

    dq, dx_lim = jax.vmap(limit_angle)(
        limit_axes,
        ref_axis_1,
        ref_axis_2,
        motion,
        limit,
        padded_ang_limit,
    )
    dq = -1.0 * jp.sum(dq, axis=0)
    dx -= jp.sum(dx_lim, axis=0)
  else:
    dq = jp.zeros_like(x.pos)

  return Transform(pos=dx, rot=dq)

def drive_update(sys: System, state: State, act: jax.Array,
                 lam: jax.Array = None, q_now: jax.Array = None):
  """Position servos resolved as POSITIONAL drives, not as forces.

  brax applies a joint torque to the two adjacent LINKS and divides by each
  link's own inertia. For a high-gain position servo on a light link that is
  unconditionally unstable -- a Spot finger has ~1e-6 kg m^2 against kp = 500,
  i.e. a servo natural frequency of ~22000 rad/s at a 2 ms step -- and no
  treatment of the DAMPING term repairs it, because the stiffness term alone
  is the divergence. MuJoCo is fine because in generalized coordinates the
  motor acts against the articulated inertia of the whole distal subtree.

  A position-based solver already has the right primitive for this: a servo is
  a driven joint constraint. Projecting the joint angle a fraction of the way
  to its target is unconditionally stable regardless of link inertia, and the
  projection propagates through the chain exactly like the joint constraints
  it sits beside. The fraction is the XPBD compliance weighting for compliance
  1/kp, precomputed in `io/mjcf.py` as
      drive_alpha = kp h^2 / (kp h^2 + I_axis),
  which is ~1 for a stiff servo on a light link (it tracks) and small for the
  same servo on a heavy arm (it moves a few percent per substep) -- the
  behaviour a real kp produces, rather than a tuned constant.

  Returns the CoM-frame correction to add to `state.x_i`.
  """
  zero = Transform(pos=jp.zeros_like(state.x_i.pos),
                   rot=jp.zeros_like(state.x_i.rot))
  if sys.drive_act is None:
    return zero, lam
  drive_act = np.asarray(sys.drive_act)
  if not (drive_act >= 0).any():
    return zero, lam

  # Walk link_types (a static string) to get each link's q and qd offsets.
  # `sys.q_idx`/`qd_idx` would do this too but return device arrays, which
  # cannot index a numpy table under trace -- and this IS structure, not data.
  n_link = sys.num_links()
  link_dof = np.full(n_link, -1, dtype=np.int64)
  link_q = np.zeros(n_link, dtype=np.int64)
  q_off = qd_off = 0
  for i, typ in enumerate(sys.link_types):
    qw, dw = Q_WIDTHS[typ], QD_WIDTHS[typ]
    if dw == 1 and drive_act[qd_off] >= 0:
      link_dof[i] = qd_off
      link_q[i] = q_off
    q_off += qw
    qd_off += dw
  active = link_dof >= 0
  if not active.any():
    return zero, lam
  safe = np.where(active, link_dof, 0)

  safe_j = jp.asarray(safe)
  a_idx = jp.asarray(drive_act[safe])
  gear = sys.drive_gear[safe_j]
  lo, hi = sys.drive_lo[safe_j], sys.drive_hi[safe_j]

  ctrl = jp.clip(act, sys.actuator.ctrl_range[:, 0],
                 sys.actuator.ctrl_range[:, 1])
  target = jp.clip(ctrl[a_idx] / gear, lo, hi)
  # The angle must be RE-DERIVED at the current configuration on every sweep.
  # Reading `state.q` (the substep's starting angle) makes each sweep command
  # the same correction, so n sweeps apply n times the intended displacement
  # and diverge; recomputing makes the sweep a contraction toward the target,
  # which is what lets the drive be iterated at all.
  j, jd, a_p, _ = kinematics.world_to_joint(sys, state.x, state.xd)
  q_all, qd_all = kinematics.inverse(sys, j, jd)
  if q_now is None:
    q_now = q_all
  theta = q_now[jp.asarray(link_q)]
  theta_dot = qd_all[safe_j]

  # XPBD, with the Lagrange multiplier ACCUMULATED across sweeps.
  #
  #     dlambda = (-C - alpha_tilde * lambda) / (w + alpha_tilde)
  #     d_theta = w * dlambda
  #
  # Iterating WITHOUT the multiplier drives C to zero, i.e. converges to a
  # rigid constraint that teleports the joint onto its target inside a single
  # substep -- and `integrator.project_xd` then reads that displacement as
  # velocity, so a stiff servo manufactures enormous speed. With it the
  # iteration converges to `C = -alpha_tilde * lambda`, the compliant response
  # a real kp produces, and the answer stops depending on the sweep count.
  # The multiplier is then clamped to the actuator's own force limit, because
  # a servo cannot apply more torque than it has.
  w = sys.drive_w[safe_j]
  at = sys.drive_at[safe_j]
  lam_max = sys.drive_lam_max[safe_j]
  if lam is None:
    lam = jp.zeros_like(w)
  # Mask BEFORE the arithmetic, not after: an undriven link has w = 0, and
  # multiplying a masked-out `inf` or `nan` by zero does not clear it.
  live = jp.asarray(active, dtype=float)
  # MuJoCo's position servo is tau = kp (q* - q) - kv qd. The drive models kp
  # as the XPBD compliance `at`, so folding (kv/kp) qd into the constraint
  # ERROR reproduces the kv half exactly -- kp (C + (kv/kp) qd) = kp C + kv qd
  # -- and it goes through the SAME force clamp, as a real servo's does.
  # Without it the drive is a pure P servo: it rises correctly and then
  # overshoots and rings, which is what the Panda did once the subtree fix
  # gave it its authority back (1.156x MuJoCo at 90 ms, and worse tracking
  # under a fast-changing command stream than before the fix).
  # NO kv TERM HERE. MuJoCo's position servo is tau = kp (q* - q) - kv qd, and
  # it is tempting to fold (kv/kp) qd into this constraint's error. Doing so
  # DOUBLE-COUNTS it: `io/mjcf.py` already adds the actuator's kv into
  # `dof_damping_total` (`_act_damp = max(-bias_qd gear^2, 0)`), which
  # `_fork_joint_dynamics.velocity_relaxation` applies implicitly ONCE per
  # substep. Adding it again inside the sweep loop applies the duplicate once
  # per SWEEP: measured on Spot, where w reaches 1231, the extra term removes
  # ~97% of the joint velocity per sweep and eight sweeps drive it to a growing
  # oscillation. Isolated on the gate path -- kv on 4/4 non-finite, kv off 0/4
  # non-finite at reward -200.46. `drive_kvkp` is still built and carried on
  # System for anyone who needs the ratio; it is deliberately not used here.
  c_err = (theta - target) * live
  denom = jp.where(jp.asarray(active), w + at, 1.0)
  dlam = (-c_err - at * lam) / denom
  lam_new = jp.where(jp.asarray(active),
                     jp.clip(lam + dlam, -lam_max, lam_max), 0.0)
  d_theta = w * (lam_new - lam) * live
  axis_j = sys.dof.motion.ang[safe_j]
  axis_w = jax.vmap(math.rotate)(axis_j, a_p.rot)
  dq = axis_w * d_theta[:, None]

  # A joint drive moves the ENTIRE DISTAL SUBTREE, not the child link alone.
  #
  # `drive_w` is 1 / I_articulated -- the inertia of everything the joint has
  # to swing -- so `d_theta` is a subtree-sized rotation. Handing it to
  # `_rotation_update`, which rotates the CHILD LINK using that link's own
  # inverse inertia, applies a subtree-sized angle to one light link and
  # leaves the seven distal links behind. `position_update` then spends its
  # sweeps dragging them along, and in doing so pulls the child back by
  # roughly I_link / I_articulated.
  #
  # That loss is ONE-SIDED, because the drive's Lagrange multiplier is clamped
  # at `drive_lam_max` and gets a single force-limited budget per substep
  # while the joint projection re-runs every sweep. Hence the fingerprint that
  # located this: MORE sweeps made the servo WEAKER (Panda step response 0.33x
  # MuJoCo at 8 sweeps, 0.25x at 64, 1.14x with the angular joint constraint
  # switched off entirely).
  #
  # Rotating the whole subtree rigidly about the anchor is the pairing that
  # matches `drive_w`: the realised response is tau h^2 / I_articulated, and
  # no distal joint is violated, so the projection has nothing to undo.
  # `desc` is built from `link_parents`, which is static, and the whole thing
  # is a constant matmul rather than a gather.
  desc = np.zeros((n_link, n_link), np.float32)
  for k in range(n_link):
    anc = k
    while anc >= 0:
      desc[k, anc] = 1.0
      anc = int(sys.link_parents[anc])
  # only DRIVEN columns contribute
  desc = desc * active.astype(np.float32)[None, :]
  desc_j = jp.asarray(desc)

  rot_vec = desc_j @ dq
  # keep each driving anchor fixed: a body rotated by `w` about its CoM moves
  # the point at `anchor` by `w x (anchor - com)`; translating the CoM by
  # `w x (com - anchor)` cancels it exactly.
  #
  # Written WITHOUT the (n_link, n_link, 3) intermediate the obvious form
  # builds. The cross product is bilinear, so
  #     sum_i desc[k,i] * (dq_i x (x_k - anchor_i))
  #   = (sum_i desc[k,i] * dq_i) x x_k  -  sum_i desc[k,i] * (dq_i x anchor_i)
  #   = rot_vec_k x x_k                 -  (desc @ cross(dq, anchor))_k
  # i.e. one elementwise cross plus one matmul over arrays that are already
  # being formed, instead of materialising an (n_link, n_link, 3) tensor per
  # world. Exactly the same value; the solver is bandwidth-bound, so the
  # intermediate was pure cost (flagged in `.work/solver_layout_report.md` as a
  # suspect for juggle-gripper's 436 sps against 586 recorded earlier).
  pos_vec = (jp.cross(rot_vec, state.x_i.pos)
             - desc_j @ jp.cross(dq, a_p.pos))

  rot_q = 0.5 * jax.vmap(math.vec_quat_mul)(rot_vec, state.x_i.rot)
  # No parent reaction term: for the three fixed-base arms the driven chain's
  # root has no parent and the world absorbs it exactly. Spot's floating trunk
  # does not recoil from its own servos here, which is an approximation --
  # measured below in the regression, and the trunk is 'planted' in this cell.
  return Transform(pos=pos_vec, rot=rot_q), lam_new

def _dof_tables(sys):
  """Static per-link (dof index, q index) for single-DOF links, and the
  inverse maps. Walking `link_types` is the only safe route: `sys.q_idx` and
  `sys.qd_idx` return device arrays, which cannot index a numpy table under
  trace, and this is structure rather than data."""
  n_link = sys.num_links()
  link_dof = np.full(n_link, -1, np.int64)
  link_q = np.zeros(n_link, np.int64)
  dof_link = {}
  q_off = qd_off = 0
  for i, typ in enumerate(sys.link_types):
    qw, dw = Q_WIDTHS[typ], QD_WIDTHS[typ]
    if dw == 1:
      link_dof[i] = qd_off
      link_q[i] = q_off
      dof_link[qd_off] = i
    q_off += qw
    qd_off += dw
  return link_dof, link_q, dof_link


def equality_update(sys: System, state: State, q_now: jax.Array) -> Transform:
  """mjEQ_JOINT couplings, projected.

  brax models no equality constraints at all, and a gripper is where that
  hurts: a Robotiq 2F-85's six finger joints are one mechanism held together
  by four loop closures and two joint couplings, and a Panda's two fingers are
  coupled by exactly one. Dropped, the fingers move independently and the grip
  is not a grip.

  MuJoCo's joint equality is `q1 = poly(q2)`. Measured on these models the
  coupling is a pure mimic -- the Robotiq linkage tracks its driver to 1.3e-4
  rad over the full stroke, the Panda's fingers to 9.3e-10 m -- so projecting
  the pair onto the polynomial is EXACT here, not an approximation. The
  correction is split between the two joints by inverse inertia so neither is
  treated as infinitely massive, and applied as an angular projection about
  each joint's own axis, exactly like the joint constraints beside it.

  Returns the CoM-frame correction to add to `state.x_i`.
  """
  zero = Transform(pos=jp.zeros_like(state.x_i.pos),
                   rot=jp.zeros_like(state.x_i.rot))
  if not sys.eq_q1:
    return zero
  link_dof, link_q, dof_link = _dof_tables(sys)
  l1 = np.array([dof_link.get(int(d), -1) for d in sys.eq_dof1])
  l2 = np.array([dof_link.get(int(d), -1) for d in sys.eq_dof2])
  keep = (l1 >= 0) & (l2 >= 0)
  if not keep.any():
    return zero
  l1, l2 = l1[keep], l2[keep]
  a1 = np.array(sys.eq_q1)[keep]
  a2 = np.array(sys.eq_q2)[keep]
  d1 = np.array(sys.eq_dof1)[keep]
  d2 = np.array(sys.eq_dof2)[keep]
  poly = sys.eq_poly[jp.asarray(np.where(keep)[0])]

  q1 = q_now[jp.asarray(a1)]
  y = q_now[jp.asarray(a2)]
  # MuJoCo: q1 = a0 + a1 y + a2 y^2 + a3 y^3 + a4 y^4
  tgt = (poly[:, 0] + poly[:, 1] * y + poly[:, 2] * y ** 2
         + poly[:, 3] * y ** 3 + poly[:, 4] * y ** 4)
  dtgt = (poly[:, 1] + 2 * poly[:, 2] * y + 3 * poly[:, 3] * y ** 2
          + 4 * poly[:, 4] * y ** 3)
  c_err = q1 - tgt

  w1 = 1.0 / sys.dof_inertia[jp.asarray(d1)]
  w2 = 1.0 / sys.dof_inertia[jp.asarray(d2)] * dtgt ** 2
  denom = w1 + w2 + 1e-12
  relax = sys.eq_relax
  dq1 = -c_err * w1 / denom * relax
  dq2 = (c_err * w2 / denom
         / jp.where(jp.abs(dtgt) < 1e-9, 1e-9, dtgt)) * relax

  # scatter the per-equality rotations onto links
  n_link = sys.num_links()
  d_theta = jp.zeros(n_link).at[jp.asarray(l1)].add(dq1)
  d_theta = d_theta.at[jp.asarray(l2)].add(dq2)

  safe = np.where(link_dof >= 0, link_dof, 0)
  _, _, a_p, a_c = kinematics.world_to_joint(sys, state.x, state.xd)
  slide = sys.eq_is_slide[jp.asarray(np.where(keep)[0])]
  touched = np.zeros(n_link)
  touched[l1] = 1.0
  touched[l2] = 1.0
  touch = jp.asarray(touched)

  ang_w = jax.vmap(math.rotate)(sys.dof.motion.ang[jp.asarray(safe)], a_p.rot)
  vel_w = jax.vmap(math.rotate)(sys.dof.motion.vel[jp.asarray(safe)], a_p.rot)

  # split the per-equality corrections into the hinge and slide halves
  ang_amt = jp.zeros(n_link).at[jp.asarray(l1)].add(dq1 * (1.0 - slide))
  ang_amt = ang_amt.at[jp.asarray(l2)].add(dq2 * (1.0 - slide))
  lin_amt = jp.zeros(n_link).at[jp.asarray(l1)].add(dq1 * slide)
  lin_amt = lin_amt.at[jp.asarray(l2)].add(dq2 * slide)

  p_idx = jp.array(sys.link_parents)
  xi_p = state.x_i.concatenate(Transform.zero((1,))).take(p_idx)
  i_inv = com.inv_inertia(sys, state.x)
  i_inv_p = jax.vmap(jp.multiply)(i_inv[p_idx], p_idx > -1)

  dq = ang_w * (ang_amt * touch)[:, None]
  dp_p, dp_c = jax.vmap(_rotation_update)(xi_p, i_inv_p, state.x_i, i_inv, dq)

  # a SLIDE coupling is a translation along the joint axis, not a rotation
  mass_inv = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  mass_inv_p = mass_inv[p_idx] * (p_idx > -1)
  dx = vel_w * (lin_amt * touch)[:, None]
  tp, tc = jax.vmap(_translation_update)(
      a_p, xi_p, i_inv_p, mass_inv_p, a_c, state.x_i, i_inv, mass_inv, -dx)
  dp_p, dp_c = dp_p + tp, dp_c + tc

  dp_p = jax.tree.map(lambda f: segment_sum(f, p_idx, n_link), dp_p)
  return dp_c + dp_p
