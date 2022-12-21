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
from brax.v2.base import DoF, Link, Motion, System, Transform
import jax
from jax import numpy as jp


def _free(*_) -> Motion:
  """Returns force resulting from free constraint in joint frame."""

  return Motion(vel=jp.zeros(3), ang=jp.zeros(3))


def _one_dof(link: Link, x: Transform, xd: Motion, dof: DoF) -> Motion:
  """Returns force resulting from a 1-dof constraint in joint frame.

  Args:
    link: link in joint frame
    x: transform in joint frame
    xd: motion in joint frame
    dof: dofs for joint

  Returns:
    Force in joint frame
  """

  joint_motion, _ = kinematics.link_to_joint_motion(dof.motion)

  # push the link to zero offset
  vel = -x.pos * link.constraint_stiffness
  # if prismatic, don't pin along free axis
  is_prismatic = (dof.motion.vel > 0).any()
  vel = (
      vel
      - jp.dot(joint_motion.vel[0], vel) * joint_motion.vel[0] * is_prismatic
  )

  # linear damp
  damp = -xd.vel * link.constraint_damping
  # if prismatic, don't damp along free axis
  vel += (
      damp
      - jp.dot(joint_motion.vel[0], damp) * joint_motion.vel[0] * is_prismatic
  )

  axis_c_x = math.rotate(joint_motion.ang[0], x.rot)
  axis_c_y = math.rotate(joint_motion.ang[1], x.rot)
  _, _, (psi, _, _), _ = kinematics.axis_angle_ang(x, xd, dof.motion)

  # torque to align to axis
  ang = -1 * link.constraint_stiffness * jp.cross(joint_motion.ang[0], axis_c_x)
  # remove second free rotational dof if prismatic
  ang -= (
      link.constraint_stiffness
      * jp.cross(joint_motion.ang[1], axis_c_y)
      * is_prismatic
  )

  # angular damp
  ang -= link.constraint_ang_damping * xd.ang

  # stay within angle limit
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    dang = jp.where(psi < limit_min, psi - limit_min, 0)
    dang = jp.where(psi > limit_max, psi - limit_max, dang)
    ang -= (
        link.constraint_limit_stiffness
        * joint_motion.ang[0]
        * dang
        * (1 - is_prismatic)
    )

    xp = jp.dot(x.pos, joint_motion.vel[0])
    dvel = jp.where(xp < limit_min, xp - limit_min, 0)
    dvel = jp.where(xp > limit_max, xp - limit_max, dvel)
    vel -= (
        link.constraint_limit_stiffness
        * joint_motion.vel[0]
        * dvel
        * (is_prismatic)
    )

  return Motion(ang=ang, vel=vel)


def _universal(link: Link, x: Transform, xd: Motion, dof: DoF) -> Motion:
  """Returns force resulting from universal constraint in joint frame.

  Args:
    link: link in joint frame
    x: transform in joint frame
    xd: motion in joint frame
    dof: dofs for joint

  Returns:
    Force in joint frame
  """

  # push the link to zero offset
  vel = -x.pos * link.constraint_stiffness

  # linear damp
  vel += -xd.vel * link.constraint_damping

  # torque the bodies to align to a joint plane
  _, (axis_1, axis_2, _), angles, _ = kinematics.axis_angle_ang(
      x, xd, dof.motion
  )
  axis_c_proj = axis_2 - jp.dot(axis_2, axis_1) * axis_1
  axis_c_proj = axis_c_proj / math.safe_norm(axis_c_proj)
  ang = -1.0 * link.constraint_limit_stiffness * jp.cross(axis_c_proj, axis_2)

  # torque the bodies to stay within angle limits
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    axis, angle = jp.array((axis_1, axis_2)), jp.array(angles)[:2]
    dang = jp.where(angle < limit_min, angle - limit_min, 0)
    dang = jp.where(angle > limit_max, angle - limit_max, dang)
    ang -= link.constraint_limit_stiffness * jp.sum(
        jax.vmap(jp.multiply)(axis, dang), 0
    )

  # damp the angular motion
  ang -= link.constraint_ang_damping * xd.ang

  return Motion(ang=ang, vel=vel)


def _spherical(link: Link, x: Transform, xd: Motion, dof: DoF) -> Motion:
  """Returns force resulting from spherical constraint in joint frame.

  Args:
    link: link in joint frame
    x: transform in joint frame
    xd: motion in joint frame
    dof: dofs for joint

  Returns:
    Force in joint frame
  """

  # push the link to zero offset
  vel = -x.pos * link.constraint_stiffness

  # linear damp
  vel += -xd.vel * link.constraint_damping

  # damp the angular motion
  ang = -1.0 * link.constraint_ang_damping * xd.ang

  # torque the bodies to stay within angle limits
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    _, axes, angles, _ = kinematics.axis_angle_ang(x, xd, dof.motion)
    axis, angle = jp.array(axes), jp.array(angles)
    dang = jp.where(angle < limit_min, angle - limit_min, 0)
    dang = jp.where(angle > limit_max, angle - limit_max, dang)
    ang -= link.constraint_limit_stiffness * jp.sum(
        jax.vmap(jp.multiply)(axis, dang), 0
    )

  return Motion(ang=ang, vel=vel)


def resolve(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Motion, jp.ndarray, jp.ndarray]:
  """Calculates springy updates in center of mass coordinates for joints.

  Args:
    sys: System defining kinematic tree of joints
    x: link transform in world frame
    xd: link motion in world frame

  Returns:
    Tuple of
    forces: world space forces to apply to each link
    positions: location in world space to apply each force
    idxs: link to which force is applied
  """

  def j_fn(typ, link, x_j, xd_j, dof):
    dof = jax.tree_map(lambda x: x.reshape((x_j.pos.shape[0], -1)), dof)
    dof = dof.replace(
        motion=jax.tree_map(
            lambda x: x.reshape((-1, base.QD_WIDTHS[typ], 3)), dof.motion
        )
    )
    j_fn_map = {
        'f': _free,
        '1': _one_dof,
        # TODO: support prismatic for 2-dof, 3-dof
        '2': _universal,
        '3': _spherical,
    }

    return jax.vmap(j_fn_map[typ])(link, x_j, xd_j, dof)

  p_idx = jp.array(sys.link_parents)
  c_idx = jp.array(range(sys.num_links()))

  x_pad = jax.tree_map(lambda x, y: jp.vstack((x, y)), x, Transform.zero((1,)))
  x_p = x_pad.take(p_idx)
  x_c = x.vmap().do(sys.link.joint)
  x_joint = x_p.vmap().do(sys.link.transform).vmap().do(sys.link.joint)

  j, jd = kinematics.world_to_joint_frame(sys, x, xd)
  f_j = scan.link_types(sys, j_fn, 'llld', 'l', sys.link, j, jd, sys.dof)

  f_w = jax.tree_map(lambda x: jax.vmap(math.rotate)(x, x_joint.rot), f_j)

  f = jax.tree_map(lambda x, y: jp.vstack([x, y]), f_w, -f_w)
  pos = jp.vstack((x_c.pos, x_joint.pos))
  link_idx = jp.hstack((c_idx, p_idx))
  f *= link_idx.reshape((-1, 1)) != -1

  return f, pos, link_idx
