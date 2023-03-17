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

"""Joint definition and apply functions."""
# pylint:disable=g-multiple-import
import functools
from typing import Tuple

from brax.v2 import kinematics
from brax.v2 import math
from brax.v2 import scan
from brax.v2.base import DoF, Force, Link, Motion, System, Transform
from brax.v2.positional.base import State
import jax
from jax import numpy as jp
from jax.ops import segment_sum


def acceleration_update(sys: System, state: State, tau: jp.ndarray) -> Motion:
  """Calculates forces to apply to links resulting from joint constraints.

  Args:
    sys: System defining kinematic tree of joints
    state: spring pipeline state
    tau: joint force vector

  Returns:
    xdd_i: acceleration to apply to link center of mass in world frame
  """

  def _free_joint(*_) -> Force:
    return Force(vel=jp.zeros(3), ang=jp.zeros(3))

  def _damp(link: Link, jd: Motion, dof: DoF, tau: jp.ndarray):
    vel = jp.sum(jax.vmap(jp.multiply)(tau, dof.motion.vel), axis=0)
    ang = jp.sum(jax.vmap(jp.multiply)(tau, dof.motion.ang), axis=0)

    # damp the angular motion
    ang += -1.0 * link.constraint_ang_damping * jd.ang

    return Force(ang=ang, vel=vel)

  def j_fn(typ, link, jd, dof, tau):
    # change dof-shape variables into link-shape
    reshape_fn = lambda x: x.reshape((jd.ang.shape[0], -1) + x.shape[1:])
    tau, dof = jax.tree_map(reshape_fn, (tau, dof))
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
  fp = jax.tree_map(lambda x: segment_sum(x, parent_idx, sys.num_links()), fp)
  xf_i = fc - fp

  # convert to acceleration
  xdd_i = Motion(
      ang=jax.vmap(lambda x, y: x @ y)(state.i_inv, xf_i.ang),
      vel=jax.vmap(lambda x, y: x / y)(xf_i.vel, state.mass),
  )

  return xdd_i


def position_update(
    sys: System,
    x: Transform,
    xd: Motion,
    xi: Transform,
    inv_inertia: jp.ndarray,
    mass: jp.ndarray,
) -> Transform:
  """Calculates position-level joint updates in CoM coordinates for joints.

  Args:
    sys: System defining kinematic tree of joints
    x: link transform in world frame
    xd: link motion in world frame
    xi: link transform in world CoM frame
    inv_inertia: inverse of inertia matrix in world CoM frame
    mass: link mass

  Returns:
    dp: position level update Transform for links
  """

  p_idx = jp.array(sys.link_parents)
  c_idx = jp.array(range(sys.num_links()))

  x_w_pad = jax.tree_map(
      lambda x, y: jp.vstack((x, y)), xi, Transform.zero((1,))
  )
  x_p_com = x_w_pad.take(p_idx)
  x_c_com = xi

  j, _, a_p, a_c = kinematics.world_to_joint(sys, x, xd)

  # pad sys and dof data withs 0s along inactive axes
  d_j = jax.vmap(_three_dof_joint_update)(j, *_sphericalize(sys, j))
  free_mask = jp.array([l != 'f' for l in sys.link_types])
  d_j = jax.tree_map(lambda x: jax.vmap(jp.multiply)(x, free_mask), d_j)

  d_w = jax.tree_map(lambda x: jax.vmap(math.rotate)(x, a_p.rot), d_j)

  inv_mass = jp.concatenate([1.0 / mass, jp.array([0.0])])
  inv_inertia = jp.concatenate([inv_inertia, jp.zeros((1, 3, 3))])

  # TODO: refactor this
  dp_p_pos, dp_c_pos = jax.vmap(_translation_update)(
      a_p,
      x_p_com,
      inv_inertia[p_idx],
      inv_mass[p_idx],
      a_c,
      x_c_com,
      inv_inertia[c_idx],
      inv_mass[c_idx],
      -d_w.pos,
  )

  dp_p_ang, dp_c_ang = jax.vmap(_rotation_update)(
      x_p_com, inv_inertia[p_idx], xi, inv_inertia[c_idx], d_w.rot
  )

  dp = jax.tree_map(
      lambda x, y: jp.vstack([x, y]),
      dp_p_pos * sys.joint_scale_pos + dp_p_ang * sys.joint_scale_ang,
      dp_c_pos * sys.joint_scale_pos + dp_c_ang * sys.joint_scale_ang,
  )

  link_idx = jp.hstack((p_idx, c_idx))
  dp *= link_idx.reshape((-1, 1)) != -1

  dp = jax.tree_map(
      lambda f: segment_sum(f, link_idx, xi.pos.shape[0]), dp  # pytype: disable=attribute-error
  )

  return dp


def _rotation_update(
    com_p: Transform,
    inv_inertia_p: jp.ndarray,
    com_c: Transform,
    inv_inertia_c: jp.ndarray,
    dq: jp.ndarray,
) -> Tuple[Transform, Transform]:
  """Calculates a position based rotational update."""

  n, th = math.normalize(dq)

  # ignoring inertial effects for now
  w1 = jp.dot(n, inv_inertia_p @ n)
  w2 = jp.dot(n, inv_inertia_c @ n)

  dlambda = -th / (w1 + w2 + 1e-6)
  p = -dlambda * n

  dq_pos_p = -jp.zeros_like(p)
  dq_rot_p = -0.5 * math.vec_quat_mul(inv_inertia_p @ p, com_p.rot)

  dq_pos_c = jp.zeros_like(p)
  dq_rot_c = 0.5 * math.vec_quat_mul(inv_inertia_c @ p, com_c.rot)

  return Transform(pos=dq_pos_p, rot=dq_rot_p), Transform(
      pos=dq_pos_c, rot=dq_rot_c
  )


def _translation_update(
    pos_p: Transform,
    com_p: Transform,
    inv_inertia_p: jp.ndarray,
    inv_mass_p: jp.ndarray,
    pos_c: Transform,
    com_c: Transform,
    inv_inertia_c: jp.ndarray,
    inv_mass_c: jp.ndarray,
    dx: jp.ndarray,
) -> Tuple[Transform, Transform]:
  """Calculates a position based translational update."""

  pos_p = pos_p.pos - com_p.pos
  pos_c = pos_c.pos - com_c.pos

  n, c = math.normalize(dx)

  cr1 = jp.cross(pos_p, n)
  w1 = inv_mass_p + jp.dot(cr1, inv_inertia_p @ cr1)

  cr2 = jp.cross(pos_c, n)
  w2 = inv_mass_c + jp.dot(cr2, inv_inertia_c @ cr2)

  dlambda = -c / (w1 + w2 + 1e-6)
  p = dlambda * n

  dq_pos_p = -p * inv_mass_p
  dq_rot_p = -0.5 * math.vec_quat_mul(
      inv_inertia_p @ jp.cross(pos_p, p), com_p.rot
  )

  dq_pos_c = p * inv_mass_c
  dq_rot_c = 0.5 * math.vec_quat_mul(
      inv_inertia_c @ jp.cross(pos_c, p), com_c.rot
  )

  return Transform(pos=dq_pos_p, rot=dq_rot_p), Transform(
      pos=dq_pos_c, rot=dq_rot_c
  )


def _sphericalize(sys, j):
  """Transforms system state into an all-3-dof version of the system."""

  def pad_free(_):
    # create dummy data for free links
    limit = (
        jp.array([-jp.inf, -jp.inf, -jp.inf]),
        jp.array([jp.inf, jp.inf, jp.inf]),
    )
    return (
        limit,
        Motion(vel=jp.eye(3), ang=jp.eye(3)),
    )

  def pad_x_dof(dof, x):
    if dof.limit:
      limit = (
          jp.concatenate([dof.limit[0], jp.zeros(3 - x)]),
          jp.concatenate([dof.limit[1], jp.zeros(3 - x)]),
      )
    else:
      limit = (
          jp.array([-jp.inf, -jp.inf, -jp.inf]),
          jp.array([jp.inf, jp.inf, jp.inf]),
      )
    padded_motion = Motion(
        vel=jp.concatenate([dof.motion.vel, jp.zeros((3 - x, 3))]),
        ang=jp.concatenate([dof.motion.ang, jp.zeros((3 - x, 3))]),
    )
    return limit, padded_motion

  def j_fn(typ, j, dof):
    # change dof-shape variables into link-shape
    reshape_fn = lambda x: x.reshape((j.pos.shape[0], -1) + x.shape[1:])
    dof = jax.tree_map(reshape_fn, dof)
    j_fn_map = {
        'f': pad_free,
        '1': functools.partial(pad_x_dof, x=1),
        '2': functools.partial(pad_x_dof, x=2),
        '3': functools.partial(pad_x_dof, x=3),
    }
    limit, padded_motion = jax.vmap(j_fn_map[typ])(dof)

    if typ == 'f':
      joint_frame_fn = lambda x: (Motion(vel=jp.eye(3), ang=jp.eye(3)), 1)
    else:
      joint_frame_fn = kinematics.link_to_joint_frame
    joint_motion, parity = jax.vmap(joint_frame_fn)(dof.motion)

    return limit, padded_motion, joint_motion, parity

  result = scan.link_types(sys, j_fn, 'ld', 'l', j, sys.dof)

  return result


def _three_dof_joint_update(
    x: Transform,
    limit: Tuple[jp.ndarray, jp.ndarray],
    motion: Motion,
    joint_motion: Motion,
    parity: float,
) -> Transform:
  """Returns position-level displacements in joint frame for spherical joint."""
  (
      axis_c,
      (_, _, _),
      (line_of_nodes, axis_1_p_in_xz_c),
  ) = kinematics.axis_angle_ang(x, joint_motion, parity)

  axis_p = joint_motion.ang

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

  is_translational = motion.vel.any()
  # remove components of update along free prismatic axes
  dx -= jp.sum(motion.vel * dx, axis=0) * is_translational

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
