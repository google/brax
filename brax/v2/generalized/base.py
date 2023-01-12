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
"""Base types for generalized pipeline."""

from brax.v2 import base
from brax.v2.base import Inertia, Motion, System, Transform
from flax import struct
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
  con_jac: jp.ndarray
  con_pos: jp.ndarray
  con_diag: jp.ndarray
  # acceleration based terms are calculated using terms from the previous step:
  qf_smooth: jp.ndarray
  qf_constraint: jp.ndarray
  qdd: jp.ndarray

  @classmethod
  def zero(cls, sys: System) -> 'State':
    """Returns an initial State given a brax system."""
    return State(
        q=jp.zeros(sys.q_size()),
        qd=jp.zeros(sys.qd_size()),
        x=Transform.zero((sys.num_links(),)),
        xd=Motion.zero((sys.num_links(),)),
        contact=None,
        com=jp.zeros((sys.num_links(), 3)),
        cinr=Inertia(
            Transform.zero((sys.num_links(),)),
            jp.zeros((sys.num_links(), 3, 3)),
            jp.zeros((sys.num_links(),)),
        ),
        cd=Motion.zero((sys.num_links(),)),
        cdof=Motion.zero((sys.num_links(),)),
        cdofd=Motion.zero((sys.num_links(),)),
        mass_mx=jp.zeros((sys.num_links(), sys.num_links())),
        mass_mx_inv=jp.zeros((sys.num_links(), sys.num_links())),
        con_jac=jp.zeros(()),
        con_pos=jp.zeros(()),
        con_diag=jp.zeros(()),
        qf_smooth=jp.zeros((sys.qd_size(),)),
        qf_constraint=jp.zeros((sys.qd_size(),)),
        qdd=jp.zeros(sys.qd_size()),
    )
