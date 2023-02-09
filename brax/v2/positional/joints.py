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
from typing import Tuple

from brax.v2 import base
from brax.v2 import kinematics
from brax.v2 import math
from brax.v2 import scan
from brax.v2.base import DoF, Link, Motion, System, Transform, Force
from brax.v2.positional.base import State
import jax
from jax import numpy as jp
from jax.ops import segment_sum


def _revolute(x: Transform, xd: Motion, dof: DoF) -> Transform:
  """Returns position-level displacements in joint frame for revolute joint."""

  joint_motion, _ = kinematics.link_to_joint_motion(dof.motion)
  axis_c, _, (psi, _, _), _ = kinematics.axis_angle_ang(x, xd, dof.motion)
  axis_p = joint_motion.ang

  # positional constraints
  dx = -x.pos

  # angular constraints
  dq = -jp.cross(axis_p[0], axis_c[0])

  # limit constraints
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    ph = jp.clip(psi, limit_min, limit_max)
    fixrot = math.quat_rot_axis(axis_p[0], ph).reshape((4))
    n1 = math.rotate(axis_p[2], fixrot)
    dq -= jp.cross(n1, axis_c[2])

  return Transform(pos=dx, rot=dq)


def _free(*_) -> Transform:
  """Returns force resulting from free constraint in joint frame."""

  return Transform(pos=jp.zeros(3), rot=jp.zeros(3))


def _damp(
    link: Link, j: Transform, jd: Motion, dof: DoF, tau: jp.ndarray
) -> Force:
  """Returns acceleration level joint damping in joint frame.

  Args:
    link: link in joint frame
    j: joint transform
    jd: joint motion
    dof: dofs for this link
    tau: actuator forces

  Returns:
    Force in joint frame
  """
  del j

  joint_motion, _ = kinematics.link_to_joint_motion(dof.motion)
  vel = tau * joint_motion.vel[0]
  ang = tau * joint_motion.ang[0]

  # damp the angular motion
  ang += -1.0 * link.constraint_ang_damping * jd.ang

  return Force(ang=ang, vel=vel)


def resolve_damping(sys: System, state: State, tau: jp.ndarray) -> Motion:
  """Calculates forces to apply to links resulting from joint constraints.

  Args:
    sys: System defining kinematic tree of joints
    state: spring pipeline state
    tau: joint force vector

  Returns:
    xdd_i: acceleration to apply to link center of mass in world frame
  """

  def _free_joint(*_) -> Force:
    """Returns force resulting from free constraint in joint frame."""
    return Force(vel=jp.zeros(3), ang=jp.zeros(3))

  def j_fn(typ, link, j, jd, dof, tau):
    # change dof-shape variables into link-shape
    reshape_fn = lambda x: x.reshape((j.pos.shape[0], -1) + x.shape[1:])
    tau, dof = jax.tree_map(reshape_fn, (tau, dof))
    j_fn_map = {
        'f': _free_joint,
        '1': _damp,
    }

    return jax.vmap(j_fn_map[typ])(link, j, jd, dof, tau)

  # calculate forces in joint frame, then convert to world frame
  link, j, jd, dof = sys.link, state.j, state.jd, sys.dof
  jf = scan.link_types(sys, j_fn, 'llldd', 'l', link, j, jd, dof, tau)
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


def resolve_displacement(
    sys: System,
    x: Transform,
    xd: Motion,
    xi: Transform,
    inv_inertia: jp.ndarray,
    mass: jp.ndarray,
) -> Tuple[Transform]:
  """Calculates position-level joint updates in CoM coordinates for joints.

  Args:
    sys: System defining kinematic tree of joints
    x: link transform in world frame
    xd: link motion in world frame
    xi: link transform in world CoM frame
    inv_inertia: inverse of inertia matrix in world CoM frame
    mass: link mass

  Returns:
    Tuple of
    displacement: world space displacements to apply to each link
    positions: location in world space to apply each displacement
    idxs: link to which displacement is applied
  """

  def j_fn(typ, x_j, xd_j, dof):
    dof = jax.tree_map(lambda x: x.reshape((x_j.pos.shape[0], -1)), dof)
    dof = dof.replace(
        motion=jax.tree_map(
            lambda x: x.reshape((-1, base.QD_WIDTHS[typ], 3)), dof.motion
        )
    )
    j_fn_map = {
        'f': _free,
        '1': _revolute,
        # TODO: support prismatic for pbd
        # '2': _universal,
        # '3': _spherical,
    }

    return jax.vmap(j_fn_map[typ])(x_j, xd_j, dof)

  p_idx = jp.array(sys.link_parents)
  c_idx = jp.array(range(sys.num_links()))

  x_pad = jax.tree_map(lambda x, y: jp.vstack((x, y)), x, Transform.zero((1,)))
  x_p = x_pad.take(p_idx)

  x_w_pad = jax.tree_map(
      lambda x, y: jp.vstack((x, y)), xi, Transform.zero((1,))
  )
  x_p_com = x_w_pad.take(p_idx)
  x_c_com = xi

  x_c = x.vmap().do(sys.link.joint)
  x_joint = x_p.vmap().do(sys.link.transform).vmap().do(sys.link.joint)

  j, jd, _, _ = kinematics.world_to_joint(sys, x, xd)
  d_j = scan.link_types(sys, j_fn, 'lld', 'l', j, jd, sys.dof)

  d_w = jax.tree_map(lambda x: jax.vmap(math.rotate)(x, x_joint.rot), d_j)

  dp_p_pos, dp_c_pos = position_update(
      x_joint,
      x_p_com,
      inv_inertia[p_idx],
      mass[p_idx],
      x_c,
      x_c_com,
      inv_inertia[c_idx],
      mass[c_idx],
      -d_w.pos,
      scale=sys.joint_scale,
  )

  dp_p_ang, dp_c_ang = angle_update(
      x_p_com,
      inv_inertia[p_idx],
      xi,
      inv_inertia[c_idx],
      d_w.rot,
      scale=sys.joint_scale,
  )

  dp = jax.tree_map(
      lambda x, y: jp.vstack([x, y]), dp_p_pos + dp_p_ang, dp_c_pos + dp_c_ang
  )

  link_idx = jp.hstack((p_idx, c_idx))
  dp *= link_idx.reshape((-1, 1)) != -1

  dp = jax.tree_map(
      lambda f: jax.ops.segment_sum(f, link_idx, xi.pos.shape[0]), dp  # pytype: disable=attribute-error
  )

  return dp


def angle_update(
    com_p: Transform,
    inertia_p: jp.ndarray,
    com_c: Transform,
    inertia_c: jp.ndarray,
    dq: jp.ndarray,
    scale: float,
) -> Tuple[Transform, Transform]:
  """Calculates a position based angular update."""

  @jax.vmap
  def _ang_update(
      com_p: Transform,
      inertia_p: jp.ndarray,
      com_c: Transform,
      inertia_c: jp.ndarray,
      dq: jp.ndarray,
  ):
    th = math.safe_norm(dq)
    n = dq / (th + 1e-6)

    # ignoring inertial effects for now
    w1 = jp.dot(n, inertia_p @ n)
    w2 = jp.dot(n, inertia_c @ n)

    dlambda = -th / (w1 + w2 + 1e-6)
    p = -dlambda * n

    dq_pos_p = -jp.zeros_like(p)
    dq_rot_p = -0.5 * math.vec_quat_mul(inertia_p @ p, com_p.rot)

    dq_pos_c = jp.zeros_like(p)
    dq_rot_c = 0.5 * math.vec_quat_mul(inertia_c @ p, com_c.rot)

    return Transform(pos=scale * dq_pos_p, rot=scale * dq_rot_p), Transform(
        pos=scale * dq_pos_c, rot=scale * dq_rot_c
    )

  return _ang_update(com_p, inertia_p, com_c, inertia_c, dq)


def position_update(
    pos_p: Transform,
    com_p: Transform,
    inertia_p: jp.ndarray,
    mass_p: jp.ndarray,
    pos_c: Transform,
    com_c: Transform,
    inertia_c: jp.ndarray,
    mass_c: jp.ndarray,
    dx: jp.ndarray,
    scale: float,
) -> Tuple[Transform, Transform]:
  """Calculates a position based positional update."""

  @jax.vmap
  def _pos_update(
      pos_p: Transform,
      com_p: Transform,
      inertia_p: jp.ndarray,
      mass_p: jp.ndarray,
      pos_c: Transform,
      com_c: Transform,
      inertia_c: jp.ndarray,
      mass_c: jp.ndarray,
      dx: jp.ndarray,
  ):
    pos_p = pos_p.pos - com_p.pos
    pos_c = pos_c.pos - com_c.pos

    c = math.safe_norm(dx)
    n = dx / (c + 1e-6)
    n = n / math.safe_norm(n, axis=0)
    n = jp.where(jp.isnan(n), 0.0, n)

    # only treating spherical inertias
    cr1 = jp.cross(pos_p, n)
    w1 = (1.0 / mass_p) + jp.dot(cr1, inertia_p @ cr1)

    cr2 = jp.cross(pos_c, n)
    w2 = (1.0 / mass_c) + jp.dot(cr2, inertia_c @ cr2)

    dlambda = -c / (w1 + w2 + 1e-6)
    p = dlambda * n

    dq_pos_p = -p / mass_p
    dq_rot_p = -0.5 * math.vec_quat_mul(
        inertia_p @ jp.cross(pos_p, p), com_p.rot
    )

    dq_pos_c = p / mass_c
    dq_rot_c = 0.5 * math.vec_quat_mul(
        inertia_c @ jp.cross(pos_c, p), com_c.rot
    )
    # import pdb; pdb.set_trace()

    return Transform(pos=scale * dq_pos_p, rot=scale * dq_rot_p), Transform(
        pos=scale * dq_pos_c, rot=scale * dq_rot_c
    )

  return _pos_update(
      pos_p, com_p, inertia_p, mass_p, pos_c, com_c, inertia_c, mass_c, dx
  )
