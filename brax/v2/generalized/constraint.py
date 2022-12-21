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
from typing import Optional, Tuple

from brax.v2 import math
from brax.v2 import scan
from brax.v2.base import Contact, Motion, System, Transform
import jax
from jax import numpy as jp
import jaxopt
from jaxopt.projection import projection_non_negative


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
    sys: System, q: jp.ndarray
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
  """Calculates the jacobian for angle limits in dof frame.

  Args:
    sys: the brax system
    q: joint angle vector

  Returns:
    jac: the angle limit jacobian
    pos: angle in constraint frame
    diag: approximate diagonal of A matrix
  """
  if sys.dof.limit is None:
    return jp.zeros((0, sys.qd_size())), jp.zeros((0,)), jp.zeros((0,))

  # determine q and qd indices for non-free joints
  q_idx, qd_idx = sys.q_idx('123'), sys.qd_idx('123')

  pos_min = q[q_idx] - sys.dof.limit[0][qd_idx]
  pos_max = sys.dof.limit[1][qd_idx] - q[q_idx]
  pos = jp.minimum(jp.minimum(pos_min, pos_max), 0)

  side = ((pos_min < pos_max) * 2 - 1) * (pos < 0)
  jac = jax.vmap(jp.multiply)(jp.eye(sys.qd_size())[qd_idx], side)
  diag = sys.dof.invweight[qd_idx] * (pos < 0)

  return jac, pos, diag


def jac_contact(
    sys: System, com: jp.ndarray, cdof: Motion, contact: Optional[Contact]
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
  """Calculates the jacobian for contact constraints.

  Args:
    sys: the brax system
    com: center of mass position
    cdof: dofs in com frame
    contact: contacts computed for link geometries

  Returns:
    jac: the contact jacobian
    pos: contact position in constraint frame
    diag: approximate diagonal of A matrix
  """
  if contact is None:
    return jp.zeros((0, sys.qd_size())), jp.zeros((0,)), jp.zeros((0,))

  def row_fn(contact):
    link_a, link_b = contact.link_idx
    a = _pt_jac(sys, com, cdof, contact.pos, link_a)
    b = _pt_jac(sys, com, cdof, contact.pos, link_b)
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

  return jax.tree_map(jp.concatenate, jax.vmap(row_fn)(contact))


def jacobian(
    sys: System,
    q: jp.ndarray,
    com: jp.ndarray,
    cdof: Motion,
    contact: Optional[Contact],
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
  """Calculates the full constraint jacobian and constraint position.

  Args:
    sys: a brax system
    q: joint position vector
    com: center of mass position
    cdof: dofs in com frame
    contact: contacts computed for link geometries

  Returns:
    jac: the constraint jacobian
    pos: position in constraint frame
    diag: approximate diagonal of A matrix
  """
  jpds = jac_contact(sys, com, cdof, contact), jac_limit(sys, q)

  return jax.tree_map(lambda *x: jp.concatenate(x), *jpds)


def force(
    sys: System,
    qd: jp.ndarray,
    qf_smooth: jp.ndarray,
    mass_mx_inv: jp.ndarray,
    jac: jp.ndarray,
    pos: jp.ndarray,
    diag: jp.ndarray,
) -> jp.ndarray:
  """Calculates forces that satisfy joint, collision constraints.

  Args:
    sys: a brax system
    qd: joint velocity vector
    qf_smooth: joint force vector for smooth dynamics
    mass_mx_inv: inverse mass matrix
    jac: the constraint jacobian
    pos: position in constraint frame
    diag: approximate diagonal of A matrix

  Returns:
    qf_constraint: (qd_size,) constraint force
  """
  if jac.shape[0] == 0:
    return jp.zeros_like(qd)

  # calculate A matrix and b vector
  imp, aref = _imp_aref(pos, jac @ qd)
  a = jac @ mass_mx_inv @ jac.T
  a = a + jp.diag(diag * (1 - imp) / imp)
  b = jac @ mass_mx_inv @ qf_smooth - aref

  # solve for forces in constraint frame, Ax + b = 0 s.t. x >= 0
  def objective(x):
    residual = a @ x + b
    return jp.sum(0.5 * residual**2)

  # profiling a physics step, most of the time is spent running this solver:
  pg = jaxopt.ProjectedGradient(
      objective,
      projection_non_negative,
      maxiter=sys.solver_iterations,
      implicit_diff=False,
      maxls=5,
  )

  # solve and convert back to q coordinates
  qf_constraint = jac.T @ pg.run(jp.zeros_like(b)).params

  return qf_constraint
