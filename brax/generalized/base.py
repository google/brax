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
"""Base types for generalized pipeline."""

from brax import base
from brax.base import Inertia, Motion, Transform
from flax import struct
import jax
from jax import numpy as jp


@struct.dataclass
class State(base.State):
  """Dynamic state that changes after every step.

  Attributes:
    root_com: (num_links,) center of mass position of link root kinematic tree
    cinr: (num_links,) inertia in com frame
    cd: (num_links,) link velocities in com frame
    cdof: (qd_size,) dofs in com frame
    cdofd: (qd_size,) cdof velocity
    mass_mx: (qd_size, qd_size) mass matrix
    mass_mx_inv: (qd_size, qd_size) inverse mass matrix
    contact: calculated contacts
    con_jac: constraint jacobian
    con_diag: constraint A diagonal
    con_aref: constraint reference acceleration
    qf_smooth: (qd_size,) smooth dynamics force
    qf_constraint: (qd_size,) force from constraints (collision etc)
    qdd: (qd_size,) joint acceleration vector
  """

  # position/velocity based terms are updated at the end of each step:
  root_com: jax.Array
  cinr: Inertia
  cd: Motion
  cdof: Motion
  cdofd: Motion
  mass_mx: jax.Array
  mass_mx_inv: jax.Array
  con_jac: jax.Array
  con_diag: jax.Array
  con_aref: jax.Array
  # acceleration based terms are calculated using terms from the previous step:
  qf_smooth: jax.Array
  qf_constraint: jax.Array
  qdd: jax.Array

  @classmethod
  def init(
      cls, q: jax.Array, qd: jax.Array, x: Transform, xd: Motion
  ) -> 'State':
    """Returns an initial State given a brax system."""
    num_links = x.pos.shape[0]
    qd_size = qd.shape[0]
    return State(
        q=q,
        qd=qd,
        x=x,
        xd=xd,
        contact=None,
        root_com=jp.zeros(3),
        cinr=Inertia(
            Transform.zero((num_links,)),
            jp.zeros((num_links, 3, 3)),
            jp.zeros((num_links,)),
        ),
        cd=Motion.zero((num_links,)),
        cdof=Motion.zero((num_links,)),
        cdofd=Motion.zero((num_links,)),
        mass_mx=jp.zeros((qd_size, qd_size)),
        mass_mx_inv=jp.zeros((qd_size, qd_size)),
        con_jac=jp.zeros(()),
        con_diag=jp.zeros(()),
        con_aref=jp.zeros(()),
        qf_smooth=jp.zeros_like(qd),
        qf_constraint=jp.zeros_like(qd),
        qdd=jp.zeros_like(qd),
    )
