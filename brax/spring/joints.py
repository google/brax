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
from brax import kinematics
from brax import math
from brax import scan
from brax.base import DoF, Force, Link, Motion, System, Transform
from brax.spring.base import State
import jax
from jax import numpy as jp
from jax.ops import segment_sum


def _free(*_) -> Force:
  """Returns force resulting from free constraint in joint frame."""

  return Force(vel=jp.zeros(3), ang=jp.zeros(3))


def _one_dof(
    link: Link, j: Transform, jd: Motion, dof: DoF, tau: jax.Array
) -> Force:
  """Returns force resulting from a 1-dof constraint in joint frame.

  Args:
    link: link in joint frame
    j: transform in joint frame
    jd: motion in joint frame
    dof: dofs for joint
    tau: joint force

  Returns:
    force in joint frame
  """
  joint_frame, parity = kinematics.link_to_joint_frame(dof.motion)

  # push the link to zero offset
  vel = -j.pos * link.constraint_stiffness
  # if prismatic, don't pin along free axis
  is_translational = dof.motion.vel.any()
  is_rotational = dof.motion.ang.any()
  vel = (
      vel
      - jp.dot(joint_frame.vel[0], vel)
      * joint_frame.vel[0]
      * is_translational
  )

  # add in force
  vel += tau * joint_frame.vel[0] * is_translational
  # linear damp
  damp = -jd.vel * link.constraint_vel_damping
  # if prismatic, don't damp along free axis
  vel += (
      damp
      - jp.dot(joint_frame.vel[0], damp)
      * joint_frame.vel[0]
      * is_translational
  )

  axis_c_x = math.rotate(joint_frame.ang[0], j.rot)
  axis_c_y = math.rotate(joint_frame.ang[1], j.rot)
  (
      _,
      (psi, _, _),
      _,
  ) = kinematics.axis_angle_ang(j, joint_frame, parity)

  # torque to align to axis
  ang = -1 * link.constraint_stiffness * jp.cross(joint_frame.ang[0], axis_c_x)
  # remove second free rotational dof if prismatic and not revolute
  ang -= (
      link.constraint_stiffness
      * jp.cross(joint_frame.ang[1], axis_c_y)
      * is_translational
      * (1 - is_rotational)
  )
  # add in force
  ang += tau * joint_frame.ang[0] * is_rotational
  # angular damp
  ang -= link.constraint_ang_damping * jd.ang

  # stay within angle limit
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    dang = jp.where(psi < limit_min, psi - limit_min, 0)
    dang = jp.where(psi > limit_max, psi - limit_max, dang)
    ang -= (
        link.constraint_limit_stiffness
        * joint_frame.ang[0]
        * dang
        * (1 - is_translational)
    )

    xp = jp.dot(j.pos, joint_frame.vel[0])
    dvel = jp.where(xp < limit_min, xp - limit_min, 0)
    dvel = jp.where(xp > limit_max, xp - limit_max, dvel)
    vel -= (
        link.constraint_limit_stiffness
        * joint_frame.vel[0]
        * dvel
        * (is_translational)
    )

  return Force(ang=ang, vel=vel)


def _two_dof(
    link: Link, j: Transform, jd: Motion, dof: DoF, tau: jax.Array
) -> Force:
  """Returns force resulting from universal constraint in joint frame.

  Args:
    link: link in joint frame
    j: transform in joint frame
    jd: motion in joint frame
    dof: dofs for joint
    tau: joint force

  Returns:
    force in joint frame
  """
  is_translational = dof.motion.vel.any()
  is_universal = dof.motion.ang.any()
  joint_frame, parity = kinematics.link_to_joint_frame(dof.motion)

  # push the link to zero offset
  vel = -j.pos * link.constraint_stiffness

  # linear damp
  vel += -jd.vel * link.constraint_vel_damping

  # remove components of vel along prismatic axes
  vel -= (
      jp.dot(vel, dof.motion.vel[0]) * dof.motion.vel[0]
      + jp.dot(vel, dof.motion.vel[1]) * dof.motion.vel[1]
  ) * is_translational

  # torque the bodies to align to a joint plane
  (
      axis_c,
      angles,
      _,
  ) = kinematics.axis_angle_ang(j, joint_frame, parity)
  axis_1, axis_2 = joint_frame.ang[0], axis_c[1]
  axis_c_proj = axis_2 - jp.dot(axis_2, axis_1) * axis_1
  axis_c_proj = axis_c_proj / math.safe_norm(axis_c_proj)
  # if prismatic and revolute components, need to impose revolute ang update
  axis_c_x = math.rotate(joint_frame.ang[0], j.rot)
  axis_c_y = math.rotate(joint_frame.ang[1], j.rot)
  axis_c_cand = jp.where(
      dof.motion.ang[0].any(), joint_frame.ang[0], joint_frame.ang[1]
  )
  torque_axis_2 = jp.where(dof.motion.ang[0].any(), axis_c_x, axis_c_y)

  # if only one angular dof, update the torque axes
  axis_c_proj = jp.where(is_translational, axis_c_cand, axis_c_proj)
  torque_axis_2 = jp.where(is_translational, torque_axis_2, axis_2)

  ang = (
      -1.0
      * link.constraint_limit_stiffness
      * jp.cross(axis_c_proj, torque_axis_2)
  )

  # add in force
  ang_axis = jp.array(
      [axis_1 * dof.motion.ang[0].any(), axis_2 * dof.motion.ang[1].any()]
  )
  ang += jp.sum(ang_axis * tau[:, None], axis=0)
  vel_axis = dof.motion.vel[0:2]
  vel += jp.sum(vel_axis * tau[:, None], axis=0)

  # if no rotational dofs, pin rotational axes
  axis_c_z = math.rotate(joint_frame.ang[2], j.rot)
  ang -= (
      link.constraint_stiffness
      * jp.cross(joint_frame.ang[2], axis_c_z)
      * is_translational
      * (1 - is_universal)
  )

  # torque the bodies to stay within angle limits
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    angle = jp.array(angles)[:2]
    dang = jp.where(angle < limit_min, angle - limit_min, 0)
    dang = jp.where(angle > limit_max, angle - limit_max, dang)
    ang -= (
        link.constraint_limit_stiffness
        * jp.sum(jax.vmap(jp.multiply)(ang_axis, dang), 0)
        * (is_universal)
    )

    xp = jp.array(
        [jp.dot(j.pos, dof.motion.vel[0]), jp.dot(j.pos, dof.motion.vel[1])]
    )
    dvel = jp.where(xp < limit_min, xp - limit_min, 0)
    dvel = jp.where(xp > limit_max, xp - limit_max, dvel)
    vel -= (
        link.constraint_limit_stiffness
        * jp.sum(jax.vmap(jp.multiply)(vel_axis, dvel), 0)
        * (is_translational)
    )

  # damp the angular motion
  ang -= link.constraint_ang_damping * jd.ang

  return Force(ang=ang, vel=vel)


def _three_dof(
    link: Link, j: Transform, jd: Motion, dof: DoF, tau: jax.Array
) -> Force:
  """Returns force resulting from spherical constraint in joint frame.

  Args:
    link: link in joint frame
    j: transform in joint frame
    jd: motion in joint frame
    dof: dofs for joint
    tau: joint force

  Returns:
    force in joint frame
  """
  is_translational = dof.motion.vel.any()
  is_rotational = dof.motion.ang.any()

  # push the link to zero offset
  vel = -j.pos * link.constraint_stiffness

  # linear damp
  vel += -jd.vel * link.constraint_vel_damping

  # remove vel components along prismatic axes
  vel -= (
      jp.dot(dof.motion.vel[0], vel) * dof.motion.vel[0]
      + jp.dot(dof.motion.vel[1], vel) * dof.motion.vel[1]
      + jp.dot(dof.motion.vel[2], vel) * dof.motion.vel[2]
  ) * is_translational

  # damp the angular motion
  ang = -1.0 * link.constraint_ang_damping * jd.ang

  # add in force
  joint_frame, parity = kinematics.link_to_joint_frame(dof.motion)
  axis_c, angles, _ = kinematics.axis_angle_ang(j, joint_frame, parity)
  axes = (joint_frame.ang[0], axis_c[1], axis_c[2])

  ang_axis, angle = jp.array(axes), jp.array(angles)
  ang_axis = jax.vmap(jp.multiply)(ang_axis, dof.motion.ang.any(axis=1))
  ang += jp.sum(ang_axis * tau[:, None], axis=0)
  vel_axis = dof.motion.vel
  vel += jp.sum(vel_axis * tau[:, None], axis=0)

  # if angular components, add constraint torque to align to axis
  # (for any single rotational axis). we assume there is only one rotational
  # dof present in dof.motion.ang.
  ang += (
      (
          -1
          * link.constraint_stiffness
          * jp.cross(
              jp.sum(dof.motion.ang, axis=0),
              jp.sum(
                  jax.vmap(math.rotate, in_axes=[0, None])(
                      dof.motion.ang, j.rot
                  ),
                  axis=0,
              ),
          )
      )
      * is_translational
      * is_rotational
  )

  # torque the bodies to stay within angle limits
  if dof.limit is not None:
    limit_min, limit_max = dof.limit
    dang = jp.where(angle < limit_min, angle - limit_min, 0)
    dang = jp.where(angle > limit_max, angle - limit_max, dang)
    ang -= link.constraint_limit_stiffness * jp.sum(
        jax.vmap(jp.multiply)(ang_axis, dang), 0
    )

    xp = dof.motion.vel @ j.pos
    dvel = jp.where(xp < limit_min, xp - limit_min, 0)
    dvel = jp.where(xp > limit_max, xp - limit_max, dvel)
    vel -= (
        link.constraint_limit_stiffness
        * jp.sum(jax.vmap(jp.multiply)(vel_axis, dvel), 0)
        * (is_translational)
    )

  return Force(ang=ang, vel=vel)


def resolve(sys: System, state: State, tau: jax.Array) -> Force:
  """Calculates forces to apply to links resulting from joint constraints.

  Args:
    sys: System defining kinematic tree of joints
    state: spring pipeline state
    tau: joint force vector

  Returns:
    xf_i: force to apply to link center of mass in world frame
  """

  def j_fn(typ, link, j, jd, dof, tau):
    # change dof-shape variables into link-shape
    reshape_fn = lambda x: x.reshape((j.pos.shape[0], -1) + x.shape[1:])
    tau, dof = jax.tree.map(reshape_fn, (tau, dof))
    j_fn_map = {
        'f': _free,
        '1': _one_dof,
        '2': _two_dof,
        '3': _three_dof,
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
  fp = jax.tree.map(lambda x: segment_sum(x, parent_idx, sys.num_links()), fp)
  xf_i = fc - fp
  return xf_i
