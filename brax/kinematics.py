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

# pylint:disable=g-multiple-import
"""Functions for forward and inverse kinematics."""

import functools
from typing import Any, Tuple

from brax import _fork_lane as lane
from brax import base
from brax import math
from brax import scan
from brax.base import Motion
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
import numpy as np


def forward(
    sys: System, q: jax.Array, qd: jax.Array
) -> Tuple[Transform, Motion]:
  """Converts joint position/velocity to transform/motion in world frame.

  Args:
    sys: system defining the kinematic tree and other properties
    q: joint position vector
    qd: joint velocity vector

  Returns:
    x: transform in world frame
    xd: motion in world frame
  """

  # convert joint position/velocity to transform/motion in joint frame
  def jcalc(typ, q, qd, motion):
    if typ == 'f':
      q, qd = q.reshape((-1, 7)), qd.reshape((-1, 6))
      j = Transform(pos=q[:, 0:3], rot=q[:, 3:7])
      jd = Motion(ang=qd[:, 3:6], vel=qd[:, 0:3])
    else:
      # create joint transforms and motions:
      # - rotation/velocity about axis for revolute (motion.ang)
      # - translation/velocity along axis for prismatic (motion.vel)
      rot_fn = lambda ang, q: math.normalize(math.quat_rot_axis(ang, q))[0]
      j = Transform.create(
          rot=jax.vmap(rot_fn)(motion.ang, q),
          pos=jax.vmap(jp.multiply)(motion.vel, q),
      )
      jd = jax.vmap(lambda a, b: a * b)(motion, qd)

      # then group them by link, so each link has num_dofs joints
      num_links, num_dofs = qd.shape[0] // int(typ), int(typ)
      s = (num_links, num_dofs, -1)
      j_stack, jd_stack = j.reshape(s), jd.reshape(s)

      # accumulate j and jd one dof at a time
      j, jd = j_stack.take(0, axis=1), jd_stack.take(0, axis=1)
      for i in range(1, num_dofs):
        j_i, jd_i = j_stack.take(i, axis=1), jd_stack.take(i, axis=1)
        j = j.vmap().do(j_i)
        # TODO(brax-team): fix qd->jd calculation for stacked/offset joints
        jd = jd + Motion(
            ang=jax.vmap(math.rotate)(jd_i.ang, j_i.rot),
            vel=jax.vmap(math.rotate)(
                jd_i.vel + jax.vmap(jp.cross)(j_i.pos, jd_i.ang), j_i.rot
            ),
        )

    return j, jd

  j, jd = scan.link_types(sys, jcalc, 'qdd', 'l', q, qd, sys.dof.motion)

  anchor = Transform.create(rot=j.rot).vmap().do(sys.link.joint)
  j = j.replace(pos=j.pos + sys.link.joint.pos - anchor.pos)  # joint pos offset
  j = sys.link.transform.vmap().do(j)  # link transform

  def world(parent, j, jd):
    """Convert transform/motion from joint frame to world frame."""
    if parent is None:
      jd = jd.replace(ang=jax.vmap(math.rotate)(jd.ang, j.rot))
      return j, jd
    x_p, xd_p = parent
    x = x_p.vmap().do(j)
    # get the linear velocity at the tip of the parent
    vel = xd_p.vel + jax.vmap(jp.cross)(xd_p.ang, x.pos - x_p.pos)
    # add in the child linear velocity in the world frame
    vel += jax.vmap(math.rotate)(jd.vel, x_p.rot)
    ang = xd_p.ang + jax.vmap(math.rotate)(jd.ang, x.rot)
    xd = Motion(vel=vel, ang=ang)
    return x, xd

  x, xd = scan.tree(sys, world, 'll', j, jd)

  x = x.replace(rot=jax.vmap(math.normalize)(x.rot)[0])

  return x, xd


_PARENT_MAT = {}


def parent_mat(sys: System):
  """Cached constant parent selector for this topology."""
  key = tuple(sys.link_parents)
  m = _PARENT_MAT.get(key)
  if m is None:
    m = _PARENT_MAT[key] = lane.parent_selector(key)
  return m


def world_to_joint_lane(sys: System, pos, rot, ang, vel):
  """``world_to_joint`` with every quantity in lane form.

  explore_bench fork: this is a pure LAYOUT rewrite of the function below --
  the same operations in the same order, with each 3- or 4-vector carried as
  separate ``(..., n_link)`` component arrays instead of a trailing size-3/4
  axis.  Two things change for TPU:

  * ``math.rotate`` and ``quat_mul`` contain ``jp.dot``/``jp.cross`` over the
    trailing axis, and under vmap each of those is a ``dot_general`` -- a
    fusion barrier that materialises a ``(n_worlds, n_link, 3)`` buffer.
    Componentwise the whole chain is elementwise multiply-add and fuses into
    one loop.
  * ``(256, 25, 3)`` tiles to ``(256, 32, 128)`` on an 8x128 vector register:
    1.8% occupancy.  ``(256, 25)`` tiles to ``(256, 128)``, 10.7x fewer slots
    for the same three components.

  The parent ``take`` also becomes a constant one-hot matmul: ``link_parents``
  is static, and a dynamic gather is the most TPU-hostile op in this code.

  Args and returns are 3-/4-tuples of ``(..., n_link)`` arrays:
  ``(j_pos, j_rot, jd_ang, jd_vel, ap_pos, ap_rot, ac_pos, ac_rot)``.
  """
  mat = parent_mat(sys)
  # one gather for all thirteen parent components, not thirteen
  g = lane.gather_many(pos + rot + ang + vel,
                       (0., 0., 0., 1., 0., 0., 0.) + (0.,) * 6, mat)
  xp_pos, xp_rot, xdp_ang, xdp_vel = g[0:3], g[3:7], g[7:10], g[10:13]

  lt_pos, lt_rot = lane.const3(sys.link.transform.pos), lane.quat(sys.link.transform.rot)
  lj_pos, lj_rot = lane.const3(sys.link.joint.pos), lane.quat(sys.link.joint.rot)

  # a_p = x_p . link.transform . link.joint
  t_pos = lane.add3(xp_pos, lane.rotate(lt_pos, xp_rot))
  t_rot = lane.qmul(xp_rot, lt_rot)
  ap_pos = lane.add3(t_pos, lane.rotate(lj_pos, t_rot))
  ap_rot = lane.qmul(t_rot, lj_rot)

  # a_c = x . link.joint
  ac_pos = lane.add3(pos, lane.rotate(lj_pos, rot))
  ac_rot = lane.qmul(rot, lj_rot)

  # j = a_c.to_local(a_p)
  ap_inv = lane.qinv(ap_rot)
  j_pos = lane.rotate(lane.sub3(ac_pos, ap_pos), ap_inv)
  j_rot = lane.qmul(ap_inv, ac_rot)

  # world velocity of the joint location point on the parent, then into joint
  # coordinates
  d = lane.sub3(xp_pos, ap_pos)
  wj_vel = lane.sub3(xdp_vel, lane.cross3(d, xdp_ang))
  jd_ang = lane.rotate(lane.sub3(ang, xdp_ang), ap_inv)
  jd_vel = lane.rotate(lane.sub3(vel, wj_vel), ap_inv)

  return j_pos, j_rot, jd_ang, jd_vel, ap_pos, ap_rot, ac_pos, ac_rot


def world_to_joint_vecform(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion, Transform, Transform]:
  """brax's original, kept as the reference implementation and for A/B."""
  parent_idx = jp.array(sys.link_parents)
  x_p = x.concatenate(Transform.zero((1,))).take(parent_idx)
  xd_p = xd.concatenate(Motion.zero((1,))).take(parent_idx)
  a_p = x_p.vmap().do(sys.link.transform).vmap().do(sys.link.joint)
  a_c = x.vmap().do(sys.link.joint)
  j = a_c.vmap().to_local(a_p)
  xd_wj = Transform.create(pos=x_p.pos - a_p.pos).vmap().do(xd_p)
  xd_joint = xd - xd_wj
  inv_rotate = jax.vmap(math.inv_rotate)
  jd = jax.tree.map(lambda y: inv_rotate(y, a_p.rot), xd_joint)
  return j, jd, a_p, a_c


def world_to_joint(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion, Transform, Transform]:
  """Moves into the joint frame of a maximal coordinates kinematic tree."""
  if not lane.ENABLED:
    return world_to_joint_vecform(sys, x, xd)
  (j_pos, j_rot, jd_ang, jd_vel, ap_pos, ap_rot,
   ac_pos, ac_rot) = world_to_joint_lane(
       sys, lane.vec3(x.pos), lane.quat(x.rot),
       lane.vec3(xd.ang), lane.vec3(xd.vel))
  j = Transform(pos=lane.stack3(j_pos), rot=lane.stack4(j_rot))
  jd = Motion(ang=lane.stack3(jd_ang), vel=lane.stack3(jd_vel))
  a_p = Transform(pos=lane.stack3(ap_pos), rot=lane.stack4(ap_rot))
  a_c = Transform(pos=lane.stack3(ac_pos), rot=lane.stack4(ac_rot))
  return j, jd, a_p, a_c


def link_to_joint_frame(motion: Motion) -> Tuple[Motion, float]:
  """Calculates 3-dof frames for joints corresponding to a given link motion.

  Args:
    motion: Motion dof with leaves of shape (QD_WIDTH, 3)

  Returns:
    Frame with all three dofs for this joint, and the handedness of the joint

  Joint axes are not always aligned along the local frame of a link. For
  example, a joint might be along the local z-axis, and we would need to
  find the rotation that takes local-x to local-z, or:
  from_to([1., 0., 0.], [0., 0., 1.]).

  For joints with more than 1 axis (e.g., universal, spherical), there's one
  additional degree of freedom in the orientation of the second axis, which is
  set by performing one final rotation of the link-frame y-axis so that it
  aligns with the joint-frame y-axis.

  Combined, these two rotations move the identity link-frame into the joint
  frame, where the new local-x and local-y point along the directions specified
  in the system dof.

  We also need translational components because the prismatic components of a
  joint might not be aligned with the rotational components of the joint.
  """
  if motion.ang.shape[0] > 3 or motion.ang.shape[0] == 0:
    raise AssertionError(
        f'Motion shape must be in (0, 3], got {motion.ang.shape[0]}'
    )

  # 1-dof
  if motion.ang.shape[0] == 1:
    ortho_ang = math.orthogonals(motion.ang[0])
    ang_frame = jp.array([motion.ang[0], ortho_ang[0], ortho_ang[1]])
    ang_frame = jp.where(motion.ang[0].any(), ang_frame, jp.eye(3))
    ortho_vel = math.orthogonals(motion.vel[0])
    vel_frame = jp.array([motion.vel[0], ortho_vel[0], ortho_vel[1]])
    vel_frame = jp.where(motion.vel[0].any(), vel_frame, jp.eye(3))
    parity = 1

    return Motion(ang=ang_frame, vel=vel_frame), parity

  # orthogonal frames for different possible joint configurations
  ortho_ang = jax.vmap(math.orthogonals)(motion.ang)
  ortho_vel = jax.vmap(math.orthogonals)(motion.vel)

  # 2-dof
  if motion.ang.shape[0] == 2:
    # logic for fully translational or fully rotational
    is_translational = motion.vel.any()
    ang = jp.array([
        motion.ang[0],
        motion.ang[1],
        jp.cross(motion.ang[0], motion.ang[1]),
    ])
    ang = jp.where(is_translational, jp.eye(3), ang)
    vel = jp.array([
        motion.vel[0],
        motion.vel[1],
        jp.cross(motion.vel[0], motion.vel[1]),
    ])
    vel = jp.where(is_translational, vel, jp.eye(3))
    ang_frame = ang
    vel_frame = vel
    parity = 1.0

    # logic for rp / pr
    axis_r_1 = jp.where(motion.ang[0].any(), motion.ang[0], ortho_ang[1][1])
    axis_r_2 = jp.where(motion.ang[1].any(), motion.ang[1], ortho_ang[0][0])
    axis_r_3 = jp.cross(axis_r_1, axis_r_2)

    axis_p_1 = jp.where(motion.vel[0].any(), motion.vel[0], ortho_vel[1][1])
    axis_p_2 = jp.where(motion.vel[1].any(), motion.vel[1], ortho_vel[0][0])
    axis_p_3 = jp.cross(axis_p_1, axis_p_2)

  # 3-dof
  if motion.ang.shape[0] == 3:
    # pure-rotational or pure-translational logic
    ang_frame = jp.array([
        motion.ang[0],
        motion.ang[1],
        jp.cross(motion.ang[0], motion.ang[1]),
    ])
    vel_frame = jp.array([
        motion.vel[0],
        motion.vel[1],
        jp.cross(motion.vel[0], motion.vel[1]),
    ])
    parity = jp.dot(jp.cross(motion.ang[0], motion.ang[1]), motion.ang[2])

    # logic for rpp, prp, ppr
    axis_r_1 = jp.where(
        motion.ang[0].any(),
        motion.ang[0],
        jp.where(motion.ang[1].any(), ortho_ang[1][1], ortho_ang[0][2]),
    )
    axis_r_2 = jp.where(
        motion.ang[1].any(),
        motion.ang[1],
        jp.where(motion.ang[0].any(), ortho_ang[0][0], ortho_ang[1][2]),
    )
    axis_r_3 = jp.cross(axis_r_1, axis_r_2)

    axis_p_1 = jp.where(motion.vel[0].any(), motion.vel[0], ortho_vel[1][1])
    axis_p_2 = jp.where(motion.vel[1].any(), motion.vel[1], ortho_vel[0][0])
    axis_p_3 = jp.cross(axis_p_1, axis_p_2)

  # branch based on whether the joint has both `p` and `r` types
  ang_frame_p = jp.array([axis_r_1, axis_r_2, axis_r_3])
  vel_frame_p = jp.array([axis_p_1, axis_p_2, axis_p_3])

  is_both = jp.logical_and(motion.ang.any(), motion.vel.any())

  ang_frame = jp.where(
      is_both,
      ang_frame_p,
      ang_frame,
  )
  vel_frame = jp.where(
      is_both,
      vel_frame_p,
      vel_frame,
  )
  parity = jp.where(is_both, 1, parity)

  return Motion(ang=ang_frame, vel=vel_frame), parity  # pytype: disable=bad-return-type  # jnp-type


def axis_angle_ang(
    j: Transform,
    joint_motion: Motion,
    parity: float = 1.0,
) -> Tuple[Any, Any, Any]:
  """Returns axes, torque axes, angles, and angular velocities of a joint.

    This function calculates intrinsic euler angles in the x-y'-z'' convention
    between the parent and child.  It uses the line of nodes construction
    described in section 3.2.3.2 here:
    https://www.sedris.org/wg8home/Documents/WG80485.pdf

  Args:
    j: The transform for this joint.
    joint_motion: motion degrees of freedom for this joint
    parity: handedness of this joint

  Returns:
    Joint frame axis, angles, and auxiliary axes
  """

  v_rot = jax.vmap(math.rotate, in_axes=[0, None])
  child_frame = v_rot(joint_motion.ang, j.rot)

  line_of_nodes = jp.cross(child_frame[2], joint_motion.ang[0])
  line_of_nodes, _ = math.normalize(line_of_nodes)
  y_n_normal = joint_motion.ang[0]
  psi = math.signed_angle(y_n_normal, joint_motion.ang[1], line_of_nodes)
  axis_1_p_in_xz_c = (
      jp.dot(joint_motion.ang[0], child_frame[0]) * child_frame[0]
      + jp.dot(joint_motion.ang[0], child_frame[1]) * child_frame[1]
  )
  axis_1_p_in_xz_c, _ = math.normalize(axis_1_p_in_xz_c)
  ang_between_1_p_xz_c = jp.dot(axis_1_p_in_xz_c, joint_motion.ang[0])
  theta = math.safe_arccos(jp.clip(ang_between_1_p_xz_c, -1, 1)) * jp.sign(
      jp.dot(joint_motion.ang[0], child_frame[2])
  )
  yc_n_normal = -child_frame[2] * parity
  phi = math.signed_angle(yc_n_normal, child_frame[1], line_of_nodes)

  axis = (child_frame[0], child_frame[1], child_frame[2] * parity)
  angle = (psi, theta, phi)
  aux_axes = (line_of_nodes, axis_1_p_in_xz_c)

  return axis, angle, aux_axes


def axis_slide_vel(
    x: Transform, xd: Motion, motion: Motion
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Returns axes and slide dofs for a joint.

  Args:
    x: (3,) The transform for this joint type.
    xd: (3,) The motion for this joint type.
    motion: Motion degrees of freedom for this joint

  Returns:
    Joint frame axis, positions, and velocities.
  """

  coords = motion.vel @ x.pos
  velocities = motion.vel @ xd.vel

  return motion.vel, coords, velocities


_INV_TABLE = {}


def _inverse_tables(sys: System):
  """Static tables for the lane-form ``inverse``.

  ``first_dof``  the dof index of each link's first dof
  ``root``       (n_link,) 1.0 where the link has no parent
  ``pick_q``     (q_size,) index into the flattened candidate block
  ``pick_qd``    (qd_size,) likewise
  ``ok``         whether every link is 1-dof or free (the lane path's domain)

  The candidate block is 8 rows of ``(n_link,)`` for q -- the 1-dof joint
  coordinate, then a free joint's ``pos`` (3) and ``rot`` (4) -- and 7 rows for
  qd -- the 1-dof velocity, then ``vel`` (3) and body-frame ``ang`` (3).  Every
  entry of q is exactly one of those candidates, so the assembly is a static
  index rather than a scatter or a one-hot contraction (which on TPU would be
  a bfloat16 matmul and would throw away three decimal digits of a pure data
  movement).
  """
  key = (tuple(sys.link_types), tuple(sys.link_parents))
  t = _INV_TABLE.get(key)
  if t is not None:
    return t
  nl = len(sys.link_types)
  first_dof, root = [], []
  d = 0
  for i, typ in enumerate(sys.link_types):
    first_dof.append(d)
    d += base.QD_WIDTHS[typ]
    root.append(1.0 if sys.link_parents[i] < 0 else 0.0)
  ok = all(typ in '1f' for typ in sys.link_types)
  pick_q, pick_qd = [], []
  for i, typ in enumerate(sys.link_types):
    if typ == '1':
      pick_q.append(0 * nl + i)
      pick_qd.append(0 * nl + i)
    elif typ == 'f':
      pick_q.extend((1 + k) * nl + i for k in range(7))
      pick_qd.extend((1 + k) * nl + i for k in range(6))
  t = _INV_TABLE[key] = (
      np.array(first_dof), np.array(root, dtype=np.float32),
      np.array(pick_q, dtype=np.int32), np.array(pick_qd, dtype=np.int32), ok)
  return t


def inverse_lane(sys: System, j_pos, j_rot, jd_ang, jd_vel):
  """``inverse`` in lane form, plus the per-link 1-dof joint coordinate.

  Returns ``(q, qd, q1)`` where ``q1`` is the ``(..., n_link)`` array of
  single-dof joint coordinates before assembly -- ``positional/joints.py``
  wants exactly that and had been reassembling the whole ``q`` vector to read
  a handful of entries out of it.

  Only 1-dof and free links are handled; ``_inverse_tables`` reports whether
  that covers this model, and ``inverse`` falls back to brax's scan otherwise.

  This replaces ``scan.link_types``, which slices the system into one group per
  joint type and vmaps a per-link routine over each.  Every intermediate there
  carries a trailing size-3 axis and every ``jp.dot``/``jp.cross`` inside
  ``axis_angle_ang`` is a fusion barrier; here the whole thing is elementwise
  arithmetic on ``(..., n_link)`` arrays.
  """
  first_dof, root, pick_q, pick_qd, _ = _inverse_tables(sys)
  nl = len(sys.link_types)

  # --- static joint frames, exactly link_to_joint_frame's 1-dof branch ------
  a0 = sys.dof.motion.ang[first_dof]
  v0 = sys.dof.motion.vel[first_dof]
  oa = jax.vmap(math.orthogonals)(a0)
  ov = jax.vmap(math.orthogonals)(v0)
  eye = jp.eye(3)
  ang_frame = jp.where(
      a0.any(axis=-1)[:, None, None], jp.stack([a0, oa[0], oa[1]], axis=1), eye)
  del ov, v0
  # transposed to lane form: component arrays of length n_link
  f0 = (ang_frame[:, 0, 0], ang_frame[:, 0, 1], ang_frame[:, 0, 2])
  f1 = (ang_frame[:, 1, 0], ang_frame[:, 1, 1], ang_frame[:, 1, 2])
  f2 = (ang_frame[:, 2, 0], ang_frame[:, 2, 1], ang_frame[:, 2, 2])
  dof_ang = (a0[:, 0], a0[:, 1], a0[:, 2])
  dof_vel = lane.const3(sys.dof.motion.vel[first_dof])
  has_ang = jp.asarray(np.asarray(sys.dof.motion.ang)[first_dof]).any(axis=-1) \
      if False else (dof_ang[0] != 0) | (dof_ang[1] != 0) | (dof_ang[2] != 0)

  # --- 1-dof joint coordinate ----------------------------------------------
  # jd.ang is taken into the joint frame only for a ROOT link; for a link with
  # a parent brax substitutes the identity quaternion, which leaves jd.ang
  # untouched.
  ident = (jp.ones(()) * 1.0,)
  rq = (jp.where(root > 0, j_rot[0], 1.0), jp.where(root > 0, j_rot[1], 0.0),
        jp.where(root > 0, j_rot[2], 0.0), jp.where(root > 0, j_rot[3], 0.0))
  del ident
  jd_ang_l = lane.inv_rotate(jd_ang, rq)

  cf0 = lane.rotate(f0, j_rot)
  cf2 = lane.rotate(f2, j_rot)
  lon, _ = lane.normalize3(lane.cross3(cf2, f0))
  psi = jp.arctan2(lane.dot3(lane.cross3(f1, lon), f0), lane.dot3(f1, lon))
  slide = lane.dot3(dof_vel, j_pos)
  q1 = jp.where(has_ang, psi, slide)
  qd1 = jp.where(has_ang, lane.dot3(cf0, jd_ang_l), lane.dot3(dof_vel, jd_vel))

  # --- free joint ----------------------------------------------------------
  free_ang = lane.inv_rotate(jd_ang, j_rot)

  cand_q = jp.concatenate(
      [q1[..., None, :], jp.stack(j_pos + j_rot, axis=-2)], axis=-2)
  cand_qd = jp.concatenate(
      [qd1[..., None, :], jp.stack(jd_vel + free_ang, axis=-2)], axis=-2)
  q = cand_q.reshape(cand_q.shape[:-2] + (8 * nl,))[..., pick_q]
  qd = cand_qd.reshape(cand_qd.shape[:-2] + (7 * nl,))[..., pick_qd]
  return q, qd, q1


def inverse_vecform(
    sys: System, j: Transform, jd: Motion
) -> Tuple[jax.Array, jax.Array]:
  """brax's original, kept as the reference implementation and for A/B."""

  def free(x, xd, *_):
    ang = math.inv_rotate(xd.ang, x.rot)
    return jp.concatenate([x.pos, x.rot]), jp.concatenate([xd.vel, ang])

  def x_dof(j, jd, parent_idx, motion, x):
    j_rot = jp.where(parent_idx == -1, j.rot, jp.array([1.0, 0.0, 0.0, 0.0]))
    jd = jd.replace(ang=math.inv_rotate(jd.ang, j_rot))
    joint_frame, parity = link_to_joint_frame(motion)
    axis, angles, _ = axis_angle_ang(j, joint_frame, parity)
    angle_vels = jax.tree.map(lambda x: jp.dot(x, jd.ang), axis)
    _, slides, slide_vels = axis_slide_vel(j, jd, motion)
    # TODO(cdfreeman): investigate removing this `where`
    q = jp.where(
        motion.ang.any(axis=1), jp.array(angles[:x]), jp.array(slides[:x])
    )
    qd = jp.where(
        motion.ang.any(axis=1),
        jp.array(angle_vels[:x]),
        jp.array(slide_vels[:x]),
    )
    return q, qd

  def q_fn(typ, j, jd, parent_idx, motion):
    motion = jax.tree.map(
        lambda y: y.reshape((-1, base.QD_WIDTHS[typ], 3)), motion
    )
    q_fn_map = {
        'f': free,
        '1': functools.partial(x_dof, x=1),
        '2': functools.partial(x_dof, x=2),
        '3': functools.partial(x_dof, x=3),
    }

    q, qd = jax.vmap(q_fn_map[typ])(j, jd, parent_idx, motion)

    # transposed to preserve order of outputs
    return jp.array(q).reshape(-1), jp.array(qd).reshape(-1)

  parent_idx = jp.array(sys.link_parents)
  q, qd = scan.link_types(
      sys, q_fn, 'llld', 'qd', j, jd, parent_idx, sys.dof.motion
  )
  return q, qd


def inverse(
    sys: System, j: Transform, jd: Motion
) -> Tuple[jax.Array, jax.Array]:
  """Translates maximal coordinates into reduced coordinates."""
  if not lane.ENABLED or not _inverse_tables(sys)[4]:
    return inverse_vecform(sys, j, jd)
  q, qd, _ = inverse_lane(sys, lane.vec3(j.pos), lane.quat(j.rot),
                          lane.vec3(jd.ang), lane.vec3(jd.vel))
  return q, qd
