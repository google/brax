# Copyright 2024 The Brax Authors.
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
from typing import Tuple

from brax import com
from brax import kinematics
from brax import math
from brax import scan
from brax.base import DoF, Force, Link, Motion, System, Transform
from brax.positional.base import State
import jax
from jax import numpy as jp
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


def position_update(sys: System, state: State) -> Transform:
  """Calculates position-level joint updates in CoM coordinates for joints.

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
      a_p, xi_p, i_inv_p, mass_inv_p, a_c, state.x_i, i_inv, mass_inv, -d_w.pos)
  dp_p_ang, dp_c_ang = jax.vmap(_rotation_update)(
      xi_p, i_inv_p, state.x_i, i_inv, d_w.rot
  )

  dp_c = dp_c_pos * sys.joint_scale_pos + dp_c_ang * sys.joint_scale_ang
  dp_p = dp_p_pos * sys.joint_scale_pos + dp_p_ang * sys.joint_scale_ang
  dp_p = jax.tree.map(lambda f: segment_sum(f, p_idx, sys.num_links()), dp_p)

  return state.x_i + dp_c + dp_p


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


def _sphericalize(sys, j):
  """Transforms system state into an all-3-dof version of the system."""

  def pad_free(_):
    # create dummy data for free links
    inf = jp.array([jp.inf, jp.inf, jp.inf])
    return (-inf, inf), Motion(ang=jp.eye(3), vel=jp.eye(3)),

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
