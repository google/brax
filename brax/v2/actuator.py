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
"""Functions for applying actuators to a physics pipeline."""

from typing import Tuple

from brax.v2 import kinematics
from brax.v2 import scan
from brax.v2.base import math
from brax.v2.base import Motion, System, Transform
import jax
from jax import numpy as jp


def to_tau(sys: System, act: jp.ndarray, q: jp.ndarray) -> jp.ndarray:
  """Convert actuator to a joint force tau.

  Args:
    sys: system defining the kinematic tree and other properties
    act: (act_size,) actuator force input vector
    q: joint position vector

  Returns:
    tau: (qd_size,) vector of joint forces
  """
  if sys.act_size() == 0:
    return jp.zeros(sys.qd_size())

  def act_fn(act_type, act, actuator, q, qd_idx):
    if act_type not in ('p', 'm'):
      raise RuntimeError(f'unrecognized act type: {act_type}')

    force = jp.clip(act, actuator.ctrl_range[:, 0], actuator.ctrl_range[:, 1])
    if act_type == 'p':
      force -= q  # positional actuators have a bias
    tau = actuator.gear * force

    return tau, qd_idx

  qd_idx = jp.arange(sys.qd_size())
  tau, qd_idx = scan.actuator_types(
      sys, act_fn, 'aaqd', 'a', act, sys.actuator, q, qd_idx
  )
  tau = jp.zeros(sys.qd_size()).at[qd_idx].add(tau)

  return tau


def to_tau_world(
    sys: System, q: jp.ndarray, tau: jp.ndarray
) -> Tuple[Motion, jp.ndarray, jp.ndarray]:
  """Converts joint force tau to world frame.

  Args:
    sys: system defining the kinematic tree and other properties
    q: joint position vector
    tau: joint force

  Returns:
    f_a: joint force from actuator in world frame
    pos: position in world space the actuator is being applied
    f_idxs: indices where actuators are acting
  """

  # convert joint position/velocity to transform/motion in joint frame
  def jcalc(typ, q, tau, motion):
    if typ == 'f':
      q, tau = q.reshape((-1, 7)), jp.zeros((q.shape[0], 6))
      j = Transform(pos=q[:, 0:3], rot=q[:, 3:7])
      # no actuators for free joints
      tau = Motion(ang=jp.zeros_like(j.pos), vel=jp.zeros_like(j.pos))
    else:
      # create joint transforms and motions:
      # - rotation/velocity about axis for revolute (motion.ang)
      # - translation/velocity along axis for prismatic (motion.vel)
      rot_fn = lambda ang, q: math.normalize(math.quat_rot_axis(ang, q))[0]
      j = Transform.create(
          rot=jax.vmap(rot_fn)(motion.ang, q),
          pos=jax.vmap(jp.multiply)(motion.vel, q),
      )
      tau = jax.vmap(lambda a, b: a * b)(motion, tau)

      # then group them by link, so each link has num_dofs joints
      num_links, num_dofs = tau.ang.shape[0] // int(typ), int(typ)
      s = (num_links, num_dofs, -1)
      j_stack, tau_stack = j.reshape(s), tau.reshape(s)

      # accumulate j and tau one dof at a time
      j, tau = j_stack.take(0, axis=1), tau_stack.take(0, axis=1)
      for i in range(1, num_dofs):
        j_i, tau_i = j_stack.take(i, axis=1), tau_stack.take(i, axis=1)
        j = j.vmap().do(j_i)
        tau = tau + Motion(
            ang=jax.vmap(math.rotate)(tau_i.ang, j_i.rot),
            vel=jax.vmap(math.rotate)(tau_i.vel, j_i.rot),
        )

    return tau

  p_idx = jp.array(sys.link_parents)
  c_idx = jp.array(range(sys.num_links()))

  x, _ = kinematics.forward(sys, q, jp.zeros(sys.qd_size()))
  x_pad = jax.tree_map(lambda x, y: jp.vstack((x, y)), x, Transform.zero((1,)))
  x_p = x_pad.take(p_idx)
  x_c = x.vmap().do(sys.link.joint)
  x_joint = x_p.vmap().do(sys.link.transform).vmap().do(sys.link.joint)

  f_a = scan.link_types(sys, jcalc, 'qdd', 'l', q, tau, sys.dof.motion)

  f_a = jax.tree_map(lambda x, y: jp.vstack([x, y]), f_a, -f_a)
  pos = jp.vstack((x_c.pos, x_joint.pos))
  link_idx = jp.hstack((c_idx, p_idx))
  f_a *= link_idx.reshape((-1, 1)) != -1

  return f_a, pos, link_idx
