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

# pylint:disable=g-multiple-import
"""Integrator functions."""

from brax import math
from brax import scan
from brax.base import System
from brax.generalized.base import State
import jax
from jax import numpy as jp


def _integrate_q_axis(sys: System, q: jax.Array, qd: jax.Array) -> jax.Array:
  """Integrates next q for revolute/prismatic joints."""
  return q + qd * sys.opt.timestep


def _integrate_q_free(sys: System, q: jax.Array, qd: jax.Array) -> jax.Array:
  """Integrates next q for free joints."""
  rot, ang = q[3:7], qd[3:6]
  ang_norm = jp.linalg.norm(ang) + 1e-8
  axis = ang / ang_norm
  angle = sys.opt.timestep * ang_norm
  qrot = math.quat_rot_axis(axis, angle)
  rot = math.quat_mul(rot, qrot)
  rot = rot / jp.linalg.norm(rot)
  pos, vel = q[0:3], qd[0:3]
  pos += vel * sys.opt.timestep

  return jp.concatenate([pos, rot])


def integrate(sys: System, state: State) -> State:
  """Semi-implicit Euler integration.

  Args:
    sys: system defining the kinematic tree and other properties
    state: generalized state

  Returns:
    state: state with q, qd, and qdd updated
  """
  # integrate joint damping implicitly to increase stability when we are not
  # using approximate inverse
  if sys.matrix_inv_iterations == 0:
    mx = state.mass_mx + jp.diag(sys.dof.damping) * sys.opt.timestep
    mx_inv = jax.scipy.linalg.solve(mx, jp.eye(sys.qd_size()), assume_a='pos')
  else:
    mx_inv = state.mass_mx_inv
  qdd = mx_inv @ (state.qf_smooth + state.qf_constraint)
  qd = state.qd + qdd * sys.opt.timestep

  def q_fn(typ, link, q, qd):
    q = q.reshape(link.transform.pos.shape[0], -1)
    qd = qd.reshape(link.transform.pos.shape[0], -1)
    fun = jax.vmap(
        {
            'f': _integrate_q_free,
            '1': _integrate_q_axis,
            '2': _integrate_q_axis,
            '3': _integrate_q_axis,
        }[typ],
        in_axes=(None, 0, 0),
    )
    q_s = fun(sys, q, qd).reshape(-1)

    return q_s

  q = scan.link_types(sys, q_fn, 'lqd', 'q', sys.link, state.q, qd)

  return state.replace(q=q, qd=qd, qdd=qdd)
