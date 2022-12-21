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

# pylint:disable=g-multiple-import
"""Functions for forward and inverse kinematics."""
from typing import Tuple

from brax.v2 import base
from brax.v2 import math
from brax.v2 import scan
from brax.v2.base import Motion, System, Transform
import jax
from jax import numpy as jp


def forward(
    sys: System, q: jp.ndarray, qd: jp.ndarray
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
      return j, jd
    x, xd = parent
    # TODO: determine why the motion `do` is inverted
    x = x.vmap().do(j)
    xd = xd + Motion(
        ang=jax.vmap(math.rotate)(jd.ang, x.rot),
        vel=jax.vmap(math.rotate)(
            jd.vel + jax.vmap(jp.cross)(x.pos, jd.ang), x.rot
        ),
    )
    return x, xd

  x, xd = scan.tree(sys, world, 'll', j, jd)

  x = x.replace(rot=jax.vmap(math.normalize)(x.rot)[0])

  return x, xd


def world_to_joint_frame(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion]:
  """Moves into the joint frame of a maximal coordinates kinematic tree."""

  # pad a world transform onto the end
  x_pad = x.concatenate(Transform.zero((1,)))
  xd_pad = xd.concatenate(Motion.zero((1,)))

  p_idx = jp.array(sys.link_parents)

  x_p, xd_p = x_pad.take(p_idx), xd_pad.take(p_idx)
  x_c, xd_c = x, xd

  # move x and xd into joint frame
  x_joint = x_p.vmap().do(sys.link.transform).vmap().do(sys.link.joint)
  x_c = x_c.vmap().do(sys.link.joint)
  j = x_c.vmap().to_local(x_joint)

  # find world velocity of joint location point on parent
  xd_wj = Transform.create(pos=x_p.pos - x_joint.pos).vmap().do(xd_p)

  # move into joint coordinates
  xd_joint = xd_c - xd_wj
  inv_rotate = jax.vmap(lambda x, y: math.rotate(x, math.quat_inv(y)))
  jd = jax.tree_map(lambda x: inv_rotate(x, x_joint.rot), xd_joint)

  return j, jd


def link_to_joint_motion(motion: Motion) -> Tuple[Motion, float]:
  """Calculates 3-dof motions for joints corresponding to a given link motion.

  Args:
    motion: Motion dof with leaves of shape (QD_WIDTH, 3)

  Returns:
    Motion with all three dofs for this joint, and the handedness of the joint

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

  # if this is already a 3-dof motion, then we're done, up to handedness
  if motion.ang.shape[0] == 3:
    return Motion(
        ang=jp.array([
            motion.ang[0],
            motion.ang[1],
            jp.cross(motion.ang[0], motion.ang[1]),
        ]),
        vel=jp.array([
            motion.vel[0],
            motion.vel[1],
            jp.cross(motion.vel[0], motion.vel[1]),
        ]),
    ), jp.dot(jp.cross(motion.ang[0], motion.ang[1]), motion.ang[2])

  # if motion is 1 or 2-dof, then we need to reconstruct the 3-dof frame that
  # completely defines its joint frame
  def j_axes(axis):
    frame = jp.eye(3)
    rot = math.from_to(frame[0], axis[0])
    y_temp = math.rotate(frame[1], rot)
    second_axis = axis[1] if axis.shape[0] > 1 else y_temp
    second_angle = math.signed_angle(axis[0], y_temp, second_axis)
    second_rot = math.quat_rot_axis(axis[0], second_angle)
    # if second axis is all zeros, then is an identity op
    rot = math.quat_mul(second_rot, rot)
    return rot

  link_rot_ang, link_rot_vel = j_axes(motion.ang), j_axes(motion.vel)

  def rotate_frame(axis_rotation):
    return jax.vmap(math.rotate, in_axes=(0, None))(jp.eye(3), axis_rotation)

  link_frame_ang = rotate_frame(link_rot_ang)
  link_frame_vel = rotate_frame(link_rot_vel)

  return Motion(ang=link_frame_ang, vel=link_frame_vel), 1.0


def axis_angle_ang(
    j: Transform, jd: Motion, motion: Motion
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
  """Returns axes, torque axes, angles, and angular velocities of a joint.

  Args:
    j: The transform for this joint.
    jd: The motion for this joint.
    motion: motion degrees of freedom for this joint

  Returns:
    Joint frame axis, torque axes, angles, and angular velocities.
  """
  joint_motion, parity = link_to_joint_motion(motion)

  v_rot = jax.vmap(math.rotate, in_axes=[0, None])
  child_frame = v_rot(joint_motion.ang, j.rot)

  line_of_nodes = jp.cross(child_frame[2], joint_motion.ang[0])
  line_of_nodes = line_of_nodes / (1e-10 + math.safe_norm(line_of_nodes))
  y_n_normal = joint_motion.ang[0]
  psi = math.signed_angle(y_n_normal, joint_motion.ang[1], line_of_nodes)
  axis_1_p_in_xz_c = (
      jp.dot(joint_motion.ang[0], child_frame[0]) * child_frame[0]
      + jp.dot(joint_motion.ang[0], child_frame[1]) * child_frame[1]
  )
  axis_1_p_in_xz_c = axis_1_p_in_xz_c / (
      1e-10 + math.safe_norm(axis_1_p_in_xz_c)
  )
  ang_between_1_p_xz_c = jp.dot(axis_1_p_in_xz_c, joint_motion.ang[0])
  theta = math.safe_arccos(jp.clip(ang_between_1_p_xz_c, -1, 1)) * jp.sign(
      jp.dot(joint_motion.ang[0], child_frame[2])
  )
  yc_n_normal = -child_frame[2] * parity
  phi = math.signed_angle(yc_n_normal, child_frame[1], line_of_nodes)

  axis = (child_frame[0], child_frame[1], child_frame[2] * parity)
  torque_axis = (joint_motion.ang[0], child_frame[1], child_frame[2] * parity)

  angle = (psi, theta, phi)
  vel = jax.tree_map(lambda x: jp.dot(x, jd.ang), axis)

  return axis, torque_axis, angle, vel


def axis_slide_vel(
    x: Transform, xd: Motion, motion: Motion
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
  """Returns axes and slide dofs for a joint.

  Args:
    x: (3,) The transform for this joint type.
    xd: (3,) The motion for this joint type.
    motion: Motion degrees of freedom for this joint

  Returns:
    Joint frame axis, positions, and velocities.
  """

  joint_motion, _ = link_to_joint_motion(motion)

  coords = joint_motion.vel @ x.pos
  velocities = joint_motion.vel @ xd.vel

  return joint_motion.vel, coords, velocities


def inverse(
    sys: System, x: Transform, xd: Motion
) -> Tuple[jp.ndarray, jp.ndarray]:
  """Translates maximal coordinates into reduced coordinates."""

  j, jd = world_to_joint_frame(sys, x, xd)

  def one_dof(j, jd, motion):
    _, _, (angle, _, _), (ang_vel, _, _) = axis_angle_ang(j, jd, motion)
    _, (slide_x, _, _), (vel, _, _) = axis_slide_vel(j, jd, motion)
    # TODO: investigate removing this `where``
    q = jp.where(motion.ang.any(), angle, slide_x)
    qd = jp.where(motion.ang.any(), ang_vel, vel)
    return q, qd

  def two_dof(j, jd, motion):
    _, _, angles, vels = axis_angle_ang(j, jd, motion)
    return jp.array(angles[0:2]), jp.array(vels[0:2])

  def three_dof(j, jd, motion):
    _, _, angles, vels = axis_angle_ang(j, jd, motion)
    return jp.array(angles[0:3]), jp.array(vels[0:3])

  def free(x, xd, _):
    return jp.concatenate([x.pos, x.rot]), jp.concatenate([xd.vel, xd.ang])

  def q_fn(typ, j, jd, motion):
    motion = jax.tree_map(
        lambda y: y.reshape((-1, base.QD_WIDTHS[typ], 3)), motion
    )
    q_fn_map = {
        'f': free,
        '1': one_dof,
        '2': two_dof,
        '3': three_dof,
    }

    q, qd = jax.vmap(q_fn_map[typ])(j, jd, motion)

    # transposed to preserve order of outputs
    return jp.array(q).reshape(-1), jp.array(qd).reshape(-1)

  q, qd = scan.link_types(sys, q_fn, 'lld', 'qd', j, jd, sys.dof.motion)
  return q, qd
