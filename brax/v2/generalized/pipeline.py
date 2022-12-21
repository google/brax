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
"""Physics pipeline for generalized coordinates engine."""

from typing import Optional

from brax.v2 import actuator
from brax.v2 import base
from brax.v2 import geometry
from brax.v2 import kinematics
from brax.v2 import math
from brax.v2.base import Contact, Inertia, Motion, System
from brax.v2.generalized import constraint
from brax.v2.generalized import dynamics
from brax.v2.generalized import integrator
from brax.v2.generalized import mass
from flax import struct
import jax
from jax import numpy as jp


@struct.dataclass
class State(base.State):
  """Dynamic state that changes after every step.

  Attributes:
    com: center of mass position
    cinr: inertia in com frame
    cd: body velocities in com frame
    cdof: dofs in com frame
    cdofd: cdof velocity
    mass_mx: (qd_size, qd_size) mass matrix
    mass_mx_inv: (qd_size, qd_size) inverse mass matrix
    contact: calculated contacts
    con_jac: constraint jacobian
    con_pos: constraint position
    con_diag: constraint A diagonal
    qf_smooth: smooth dynamics force
    qf_constraint: (qd_size,) force from constraints (collision etc)
    qdd: (qd_size,) joint acceleration vector
  """

  # position/velocity based terms are updated at the end of each step:
  com: jp.ndarray
  cinr: Inertia
  cd: Motion
  cdof: Motion
  cdofd: Motion
  mass_mx: jp.ndarray
  mass_mx_inv: jp.ndarray
  contact: Optional[Contact]
  con_jac: jp.ndarray
  con_pos: jp.ndarray
  con_diag: jp.ndarray
  # acceleration based terms are calculated using terms from the previous step:
  qf_smooth: jp.ndarray
  qf_constraint: jp.ndarray
  qdd: jp.ndarray


def init(sys: System, q: jp.ndarray, qd: jp.ndarray) -> State:
  """Initializes physics state.

  Args:
    sys: a brax system
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector

  Returns:
    state: initial physics state
  """
  x, xd = kinematics.forward(sys, q, qd)
  com, cinr, cd, cdof, cdofd = dynamics.transform_com(sys, q, qd, x)
  mass_mx = mass.matrix(sys, cinr, cdof)
  one = jp.eye(sys.qd_size())
  mass_mx_inv = jax.scipy.linalg.solve(mass_mx, one, assume_a='pos')
  contact = geometry.contact(sys, x)
  con_jac, con_pos, con_diag = constraint.jacobian(sys, q, com, cdof, contact)
  qf_smooth, qf_constraint, qdd = jp.zeros((3, sys.qd_size()))

  return State(
      q,
      qd,
      x,
      xd,
      com,
      cinr,
      cd,
      cdof,
      cdofd,
      mass_mx,
      mass_mx_inv,
      contact,
      con_jac,
      con_pos,
      con_diag,
      qf_smooth,
      qf_constraint,
      qdd,
  )


def step(sys: System, state: State, act: jp.ndarray) -> State:
  """Performs a physics step.

  Args:
    sys: a brax system
    state: physics state prior to step
    act: (act_size,) actuator input vector

  Returns:
    state: physics state after step
  """
  tau = actuator.to_tau(sys, act, state.q)

  # calculate acceleration terms
  qf_smooth = dynamics.forward(
      sys, state.q, state.qd, state.cinr, state.cd, state.cdof, state.cdofd, tau
  )
  qf_constraint = constraint.force(
      sys,
      state.qd,
      qf_smooth,
      state.mass_mx_inv,
      state.con_jac,
      state.con_pos,
      state.con_diag,
  )
  # add dof damping to the mass matrix
  # because we already have M^-1, we use the derivative of the inverse:
  # (A +  εX)^-1 = A^-1 - εA^-1 @ X @ A^-1 + O(ε^2)
  mx_inv = state.mass_mx_inv
  mx_inv_damp = mx_inv - mx_inv @ (jp.diag(sys.dof.damping) * sys.dt) @ mx_inv
  qdd = mx_inv_damp @ (qf_smooth + qf_constraint)

  # update position/velocity level terms
  q, qd = integrator.integrate(sys, state.q, state.qd, qdd)
  x, xd = kinematics.forward(sys, q, qd)
  com, cinr, cd, cdof, cdofd = dynamics.transform_com(sys, q, qd, x)
  mass_mx = mass.matrix(sys, cinr, cdof)
  mass_mx_inv = math.inv_approximate(mass_mx, state.mass_mx_inv)
  contact = geometry.contact(sys, x)
  con_jac, con_pos, con_diag = constraint.jacobian(sys, q, com, cdof, contact)

  return State(
      q,
      qd,
      x,
      xd,
      com,
      cinr,
      cd,
      cdof,
      cdofd,
      mass_mx,
      mass_mx_inv,
      contact,
      con_jac,
      con_pos,
      con_diag,
      qf_smooth,
      qf_constraint,
      qdd,
  )
