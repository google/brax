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
"""Functions for constraint satisfaction."""
from typing import Tuple

from brax.v2 import math
from brax.v2 import scan
from brax.v2.base import Motion, System, Transform
from brax.v2.generalized.base import State
import jax
from jax import numpy as jp
import jaxopt


def _pt_jac(
    sys: System,
    com: jp.ndarray,
    cdof: Motion,
    pos: jp.ndarray,
    link_idx: jp.ndarray,
) -> jp.ndarray:
  """Calculates the point jacobian.

  Args:
    sys: a brax system
    com: center of mass position
    cdof: dofs in com frame
    pos: position in world frame
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
  pt = Transform.create(pos=pos - com).vmap(in_axes=(None, 0)).do(cdof).vel

  return pt


def _imp_aref(pos: jp.ndarray, vel: jp.ndarray) -> jp.ndarray:
  """Calculates impedance and offset acceleration in constraint frame.

  Args:
    pos: position in constraint frame
    vel: velocity in constraint frame

  Returns:
    imp: constraint impedance
    Aref: offset acceleration in constraint frame
  """
  # this formulation corresponds to the parameterization described here:
  # https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  # TODO: support custom solimp, solref
  timeconst, dampratio = 0.02, 1.0
  dmin, dmax, width, mid, power = 0.9, 0.95, 0.001, 0.5, 2.0

  imp_x = jp.abs(pos) / width
  imp_a = (1.0 / jp.power(mid, power - 1)) * jp.power(imp_x, power)
  imp_b = 1 - (1.0 / jp.power(1 - mid, power - 1)) * jp.power(1 - imp_x, power)
  imp_y = jp.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = jp.clip(imp, dmin, dmax)
  imp = jp.where(imp_x > 1.0, dmax, imp)

  b = 2 / (dmax * timeconst)
  k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

  aref = -b * vel - k * imp * pos

  return imp, aref


def jac_limit(
    sys: System, state: State
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
  """Calculates the jacobian for angle limits in dof frame.

  Args:
    sys: the brax system
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
  diag = sys.dof.invweight[qd_idx] * (pos < 0)

  return jac, pos, diag


def jac_contact(
    sys: System, state: State
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
  """Calculates the jacobian for contact constraints.

  Args:
    sys: the brax system
    state: generalized state

  Returns:
    jac: the contact jacobian
    pos: contact position in constraint frame
    diag: approximate diagonal of A matrix
  """
  if state.contact is None:
    return jp.zeros((0, sys.qd_size())), jp.zeros((0,)), jp.zeros((0,))

  def row_fn(contact):
    link_a, link_b = contact.link_idx
    a = _pt_jac(sys, state.com, state.cdof, contact.pos, link_a)
    b = _pt_jac(sys, state.com, state.cdof, contact.pos, link_b)
    diff = b - a

    # 4 pyramidal friction directions
    jac = []
    for d in math.orthogonals(contact.normal):
      for f in [-contact.friction, contact.friction]:
        jac.append(diff @ (d * f - contact.normal))

    jac = jp.stack(jac)
    pos = -jp.tile(contact.penetration, 4)
    t = sys.link.invweight[link_a] + sys.link.invweight[link_b] * (link_b > -1)
    diag = jp.tile(t + contact.friction * contact.friction * t, 4)
    diag = 2 * contact.friction * contact.friction * diag

    return jax.tree_map(
        lambda x: x * (contact.penetration > 0), (jac, pos, diag)
    )

  return jax.tree_map(jp.concatenate, jax.vmap(row_fn)(state.contact))


def jacobian(sys: System, state: State) -> State:
  """Calculates the constraint jacobian, position, and A matrix diagonal.

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    state: generalized state with jac, pos, diag updated
  """
  jpds = jac_contact(sys, state), jac_limit(sys, state)
  jac, pos, diag = jax.tree_map(lambda *x: jp.concatenate(x), *jpds)
  return state.replace(con_jac=jac, con_pos=pos, con_diag=diag)


def force(sys: System, state: State) -> jp.ndarray:
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
  imp, aref = _imp_aref(state.con_pos, state.con_jac @ state.qd)
  a = state.con_jac @ state.mass_mx_inv @ state.con_jac.T
  a = a + jp.diag(state.con_diag * (1 - imp) / imp)
  b = state.con_jac @ state.mass_mx_inv @ state.qf_smooth - aref

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
      maxls=5,
  )

  # solve and convert back to q coordinates
  qf_constraint = state.con_jac.T @ pg.run(jp.zeros_like(b)).params

  return qf_constraint
