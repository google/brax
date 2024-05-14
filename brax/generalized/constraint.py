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
"""Functions for constraint satisfaction."""
from typing import Tuple

from brax import contact
from brax import math
from brax import scan
from brax.base import Motion, System, Transform
from brax.generalized.base import State
import jax
from jax import numpy as jp
import jaxopt


def _imp_aref(
    params: jax.Array, pos: jax.Array, vel: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Calculates impedance and offset acceleration in constraint frame.

  Args:
    params: solver params
    pos: position in constraint frame
    vel: velocity in constraint frame

  Returns:
    imp: constraint impedance
    Aref: offset acceleration in constraint frame
  """
  # this formulation corresponds to the parameterization described here:
  # https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  timeconst, dampratio, dmin, dmax, width, mid, power = params

  imp_x = jp.abs(pos) / width
  imp_a = (1.0 / jp.power(mid, power - 1)) * jp.power(imp_x, power)
  imp_b = 1 - (1.0 / jp.power(1 - mid, power - 1)) * jp.power(1 - imp_x, power)
  imp_y = jp.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = jp.clip(imp, dmin, dmax)
  imp = jp.where(imp_x > 1.0, dmax, imp)

  b = 2 / (dmax * timeconst)
  k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

  # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  stiffness, damping = params[:2]
  b = jp.where(damping <= 0, -damping / dmax, b)
  k = jp.where(stiffness <= 0, -stiffness / (dmax * dmax), k)

  aref = -b * vel - k * imp * pos

  return imp, aref


def point_jacobian(
    sys: System,
    com: jax.Array,
    cdof: Motion,
    pos: jax.Array,
    link_idx: jax.Array,
) -> Motion:
  """Calculates the jacobian of a point on a link.

  Args:
    sys: a brax system
    com: center of mass position
    cdof: dofs in com frame
    pos: position in world frame to calculate the jacobian
    link_idx: index of link frame to transform point jacobian

  Returns:
    pt: point jacobian
  """
  # backward scan up tree: build the link mask corresponding to link_idx
  def mask_fn(mask_child, link):
    mask = link == link_idx
    if mask_child is not None:
      mask += mask_child
    return mask

  mask = scan.tree(sys, mask_fn, 'l', jp.arange(sys.num_links()), reverse=True)
  cdof = jax.vmap(lambda a, b: a * b)(cdof, jp.take(mask, sys.dof_link()))
  off = Transform.create(pos=pos - com[link_idx])
  return off.vmap(in_axes=(None, 0)).do(cdof)


def jac_limit(
    sys: System, state: State
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Calculates the jacobian for angle limits in dof frame.

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    jac: the angle limit jacobian
    pos: angle in constraint frame
    diag: approximate diagonal of A matrix
  """
  if sys.dof.limit is None:
    return jp.zeros((0, sys.qd_size())), jp.zeros((0,)), jp.zeros((0,))

  # determine q and qd indices for non-free joints
  q_idx, qd_idx = sys.q_idx('123'), sys.qd_idx('123')

  pos_min = state.q[q_idx] - sys.dof.limit[0][qd_idx]
  pos_max = sys.dof.limit[1][qd_idx] - state.q[q_idx]
  pos = jp.minimum(jp.minimum(pos_min, pos_max), 0)

  side = ((pos_min < pos_max) * 2 - 1) * (pos < 0)
  jac = jax.vmap(jp.multiply)(jp.eye(sys.qd_size())[qd_idx], side)
  params = sys.dof.solver_params[qd_idx]
  imp, aref = jax.vmap(_imp_aref)(params, pos, jac @ state.qd)
  diag = sys.dof.invweight[qd_idx] * (pos < 0) * (1 - imp) / (imp + 1e-8)
  aref = jax.vmap(lambda x, y: x * y)(aref, (pos < 0))

  return jac, diag, aref


def jac_contact(
    sys: System, state: State
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Calculates the jacobian for contact constraints.

  Args:
    sys: the brax system
    state: generalized state

  Returns:
    jac: the contact jacobian
    pos: contact position in constraint frame
    diag: approximate diagonal of A matrix
  """
  c = contact.get(sys, state.x)

  if c is None:
    return jp.zeros((0, sys.qd_size())), jp.zeros((0,)), jp.zeros((0,))

  def row_fn(c):
    link_a, link_b = c.link_idx
    a = point_jacobian(sys, state.root_com, state.cdof, c.pos, link_a)
    b = point_jacobian(sys, state.root_com, state.cdof, c.pos, link_b)
    diff = b.vel - a.vel

    # 4 pyramidal friction directions
    jac = []
    for d in -c.frame[1:]:
      for f in [-c.friction[0], c.friction[0]]:
        jac.append(diff @ (d * f + c.frame[0]))

    jac = jp.stack(jac)
    pos = jp.tile(c.dist, 4)
    solver_params = jp.concatenate([c.solref, c.solimp])
    imp, aref = _imp_aref(solver_params, pos, jac @ state.qd)
    t = sys.link.invweight[link_a] * (link_a > -1) + sys.link.invweight[link_b]
    diag = jp.tile(t + c.friction[0] * c.friction[0] * t, 4)
    diag *= 2 * c.friction[0] * c.friction[0] * (1 - imp) / (imp + 1e-8)

    return jax.tree.map(lambda x: x * (c.dist < 0), (jac, diag, aref))

  return jax.tree.map(jp.concatenate, jax.vmap(row_fn)(c))


def jacobian(sys: System, state: State) -> State:
  """Calculates the constraint jacobian, position, and A matrix diagonal.

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    state: generalized state with jac, pos, diag, aref updated
  """
  jpds = jac_contact(sys, state), jac_limit(sys, state)
  jac, diag, aref = jax.tree.map(lambda *x: jp.concatenate(x), *jpds)
  return state.replace(con_jac=jac, con_diag=diag, con_aref=aref)


def force(sys: System, state: State) -> jax.Array:
  """Calculates forces that satisfy joint, collision constraints.

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    qf_constraint: (qd_size,) constraint force
  """
  if state.con_jac.shape[0] == 0:
    return jp.zeros(sys.qd_size())

  # calculate A matrix and b vector
  a = state.con_jac @ state.mass_mx_inv @ state.con_jac.T
  a = a + jp.diag(state.con_diag)
  b = state.con_jac @ state.mass_mx_inv @ state.qf_smooth - state.con_aref

  # solve for forces in constraint frame, Ax + b = 0 s.t. x >= 0
  def objective(x):
    residual = a @ x + b
    return jp.sum(0.5 * residual**2)

  # profiling a physics step, most of the time is spent running this solver:
  #
  # there might still be some opportunities to speed this up.  consider that
  # the A matrix is positive definite.  could possibly use conjugate gradient?
  # we made a jax version of this: https://github.com/david-cortes/nonneg_cg
  # but it was still not as fast as the projected gradient solver below.
  # this is possibly due to the line search method, which is a big part
  # of the cost.  jaxopt uses FISTA which we did not implement
  #
  # another avenue worth pursuing is that these A matrices are often
  # fairly sparse.  perhaps worth trying some kind of random or
  # learned projection to solve a smaller dense matrix at each step
  pg = jaxopt.ProjectedGradient(
      objective,
      jaxopt.projection.projection_non_negative,
      maxiter=sys.solver_iterations,
      implicit_diff=False,
      maxls=sys.solver_maxls,
  )

  # solve and convert back to q coordinates
  qf_constraint = state.con_jac.T @ pg.run(jp.zeros_like(b)).params

  return qf_constraint
