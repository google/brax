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

# pylint:disable=g-multiple-import, g-importing-member
"""Functions for smooth forward and inverse dynamics."""
from brax import fluid
from brax import math
from brax import scan
from brax.base import Motion, System, Transform
from brax.generalized.base import State
from brax.generalized.constraint import point_jacobian
import jax
from jax import numpy as jp


def transform_com(sys: System, state: State) -> State:
  """Transforms inertia, dof, and link velocity into center of mass frame.

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    state: generalized state with com, cinr, cd, cdof, cdofd updated
  """
  x_i = state.x.vmap().do(sys.link.inertia.transform)
  root_fn = lambda i, p=sys.link_parents: i if p[i] < 0 else root_fn(p[i])
  root = jp.array([root_fn(i) for i in range(sys.num_links())])
  mass_xi = jax.vmap(jp.multiply)(sys.link.inertia.mass, x_i.pos)
  mass_xi_sum = jax.ops.segment_sum(mass_xi, root, sys.num_links())
  mass_sum = jax.ops.segment_sum(sys.link.inertia.mass, root, sys.num_links())
  root_com = jax.vmap(jp.divide)(mass_xi_sum[root], mass_sum[root])
  cinr = x_i.replace(pos=x_i.pos - root_com).vmap().do(sys.link.inertia)

  # motion dofs to global frame centered at subtree-CoM
  parent_idx = jp.array(
      [
          i if t == 'f' else p
          for i, (t, p) in enumerate(zip(sys.link_types, sys.link_parents))
      ]
  )
  parent = state.x.concatenate(Transform.zero(shape=(1,))).take(parent_idx)
  j = parent.vmap().do(sys.link.transform).vmap().do(sys.link.joint)

  # propagate motion through stacked joints
  def cdof_fn(typ, q, motion):
    if typ == 'f':
      return motion

    # create joint transforms and motions:
    # - rotation/velocity about axis for revolute (motion.ang)
    # - translation/velocity along axis for prismatic (motion.vel)
    rot_fn = lambda ang, q: math.normalize(math.quat_rot_axis(ang, q))[0]
    j = Transform.create(
        rot=jax.vmap(rot_fn)(motion.ang, q),
        pos=jax.vmap(jp.multiply)(motion.vel, q),
    )
    jd = motion

    # then group them by link, so each has num_dofs joints
    num_links, num_dofs = motion.ang.shape[0] // int(typ), int(typ)
    s = (num_links, num_dofs, -1)
    j_stack, jd_stack = j.reshape(s), jd.reshape(s)

    # accumulate jds as successive j transforms around joint
    j, jds = Transform.zero(shape=(num_links,)), []
    for i in range(0, num_dofs):
      jds.append(j.vmap().inv_do(jd_stack.take(i, axis=1)))
      j = j.vmap().do(j_stack.take(i, axis=1))

    # interleave jds back together to match joint stack order
    motion = jax.tree.map(lambda *x: jp.column_stack(x), *jds).reshape((-1, 3))

    return motion

  cdof = scan.link_types(sys, cdof_fn, 'qd', 'd', state.q, sys.dof.motion)
  ang = jax.vmap(math.rotate)(cdof.ang, j.take(sys.dof_link()).rot)
  cdof = cdof.replace(ang=ang)
  off = Transform.create(pos=root_com - j.pos)
  cdof = off.take(sys.dof_link()).vmap().do(cdof)
  cdof_qd = jax.vmap(lambda x, y: x * y)(cdof, state.qd)

  # forward scan down tree: accumulate link center of mass velocity
  def cd_fn(cd_parent, cdof_qd, dof_idx):
    if cd_parent is None:
      num_roots = len([p for p in sys.link_parents if p == -1])
      cd_parent = Motion.zero(shape=(num_roots,))

    # cd = cd[parent] + map-sum(cdof * qd)
    cd = cd_parent.index_sum(dof_idx, cdof_qd)

    return cd

  cd = scan.tree(sys, cd_fn, 'dd', cdof_qd, sys.dof_link(depth=True))

  # propagate cd through stacked joints to calculate cdofd
  def cdofd_fn(typ, cd, cdof, cdof_qd):
    if typ == 'f':
      cdof_qd = cdof_qd.reshape((-1, 6, 3))
      cd = jax.tree.map(lambda x: jp.sum(x[:, 0:3], axis=1), cdof_qd)
      cdofd = cd.vmap().vmap(in_axes=(None, 0)).cross(cdof.reshape((-1, 6, 3)))
      cdofd = jax.tree.map(lambda x: x.at[:, 0:3].set(jp.zeros(3)), cdofd)
      return cdofd.reshape((-1, 3))

    # group cdof_qd by link, so each has num_dofs joints
    num_dofs = int(typ)
    cdof_qd = cdof_qd.reshape((cd.ang.shape[0], num_dofs, -1))
    cds = [cd]

    # accumulate cd as successive cdof velocities
    for i in range(0, num_dofs - 1):
      cds.append(cds[-1] + cdof_qd.take(i, axis=1))

    # interleave cds back together to match joint stack order
    cd = jax.tree.map(lambda *x: jp.column_stack(x), *cds).reshape((-1, 3))
    cdofd = cd.vmap().cross(cdof)

    return cdofd

  cd_p = cd.concatenate(Motion.zero(shape=(1,))).take(parent_idx)
  cdofd = scan.link_types(sys, cdofd_fn, 'ldd', 'd', cd_p, cdof, cdof_qd)

  return state.replace(
      root_com=root_com, cinr=cinr, cd=cd, cdof=cdof, cdofd=cdofd
  )


def inverse(sys: System, state: State) -> jax.Array:
  """Calculates the system's forces given input motions.

  This function computes inverse dynamics using the Newton-Euler algorithm:

  https://scaron.info/robot-locomotion/recursive-newton-euler-algorithm.html

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    tau: generalized forces resulting from joint positions and velocities
  """
  # forward scan over tree: accumulate link center of mass acceleration
  def cdd_fn(cdd_parent, cdofd, qd, dof_idx):
    if cdd_parent is None:
      num_roots = len([p for p in sys.link_parents if p == -1])
      cdd_parent = Motion.create(vel=-jp.tile(sys.gravity, (num_roots, 1)))

    # cdd = cdd[parent] + map-sum(cdofd * qd)
    cdd = cdd_parent.index_sum(dof_idx, jax.vmap(lambda x, y: x * y)(cdofd, qd))

    return cdd

  cdd = scan.tree(
      sys, cdd_fn, 'ddd', state.cdofd, state.qd, sys.dof_link(depth=True)
  )

  # cfrc_flat = cinr * cdd + cd x (cinr * cd)
  def frc(cinr, cdd, cd):
    return cinr.mul(cdd) + cd.cross(cinr.mul(cd))

  cfrc_flat = jax.vmap(frc)(state.cinr, cdd, state.cd)

  # backward scan up tree: accumulate link center of mass forces
  def cfrc_fn(cfrc_child, cfrc):
    if cfrc_child is not None:
      cfrc += cfrc_child
    return cfrc

  cfrc = scan.tree(sys, cfrc_fn, 'l', cfrc_flat, reverse=True)

  # tau = cdof * cfrc[dof_link]
  tau = jax.vmap(lambda x, y: x.dot(y))(state.cdof, cfrc.take(sys.dof_link()))

  return tau


def _passive(sys: System, state: State) -> jax.Array:
  """Calculates the system's passive forces given input motion and position."""
  def stiffness_fn(typ, q, dof):
    if typ in 'fb':
      return jp.zeros_like(dof.stiffness)
    return -q * dof.stiffness

  frc = scan.link_types(sys, stiffness_fn, 'qd', 'd', state.q, sys.dof)
  frc -= sys.dof.damping * state.qd

  if sys.enable_fluid:
    fluid_frc = fluid.force(
        sys,
        state.x,
        state.cd,
        sys.link.inertia.mass,
        sys.link.inertia.i,
        state.root_com,
    )
    link_idx = jp.arange(sys.num_links())
    x_i = state.x.vmap().do(sys.link.inertia.transform)
    jac_fn = jax.vmap(point_jacobian, in_axes=(None, None, None, 0, 0))
    jac = jac_fn(sys, state.root_com, state.cdof, x_i.pos, link_idx)
    frc += jax.vmap(lambda x, y: x.dot(y))(jac, fluid_frc).sum(axis=0)

  return frc


def forward(sys: System, state: State, tau: jax.Array) -> jax.Array:
  """Calculates resulting joint forces given input forces.

  This method builds and solves the linear system: M @ qdd = -C + tau

  where M is the joint space inertia matrix, "mass matrix" and
        C is the bias force calculated by inverse()

  Args:
    sys: a brax system
    state: generalized state
    tau: joint force input vector

  Returns:
    qfrc: joint force vector
  """
  qfrc_passive = _passive(sys, state)
  qfrc_bias = inverse(sys, state)
  qfrc = qfrc_passive - qfrc_bias + tau

  return qfrc
