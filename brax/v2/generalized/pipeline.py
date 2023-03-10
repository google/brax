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

from brax.v2 import actuator
from brax.v2 import geometry
from brax.v2 import kinematics
from brax.v2.base import System
from brax.v2.generalized import constraint
from brax.v2.generalized import dynamics
from brax.v2.generalized import integrator
from brax.v2.generalized import mass
from brax.v2.generalized.base import State
from jax import numpy as jp


def init(
    sys: System, q: jp.ndarray, qd: jp.ndarray, debug: bool = False
) -> State:
  """Initializes physics state.

  Args:
    sys: a brax system
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    debug_contact: if True, adds contact to the state for debugging

  Returns:
    state: initial physics state
  """
  x, xd = kinematics.forward(sys, q, qd)
  state = State.init(q, qd, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
  state = dynamics.transform_com(sys, state)
  state = mass.matrix_inv(sys, state)
  state = constraint.jacobian(sys, state)
  if debug:
    state = state.replace(contact=geometry.contact(sys, state.x))

  return state


def step(
    sys: System, state: State, act: jp.ndarray, debug: bool = False
) -> State:
  """Performs a physics step.

  Args:
    sys: a brax system
    state: physics state prior to step
    act: (act_size,) actuator input vector
    debug: if True, adds contact to the state for debugging

  Returns:
    state: physics state after step
  """
  # calculate acceleration terms
  tau = actuator.to_tau(sys, act, state.q)
  state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
  state = state.replace(qf_constraint=constraint.force(sys, state))
  qdd = state.mass_mx_inv @ (state.qf_smooth + state.qf_constraint)
  state = state.replace(qdd=qdd)

  # update position/velocity level terms
  q, qd = integrator.integrate(sys, state.q, state.qd, qdd)
  x, xd = kinematics.forward(sys, q, qd)
  state = state.replace(q=q, qd=qd, x=x, xd=xd)
  state = dynamics.transform_com(sys, state)
  state = mass.matrix_inv(sys, state, approximate=True)
  state = constraint.jacobian(sys, state)

  if debug:
    state = state.replace(contact=geometry.contact(sys, state.x))

  return state
