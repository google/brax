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
"""Functions for calculating the mass matrix."""
import itertools

from brax.v2 import scan
from brax.v2.base import System
import jax
from jax import numpy as jp


def matrix(sys: System, cinr: jp.ndarray, cdof: jp.ndarray) -> jp.ndarray:
  """Calculates the mass matrix for the system given joint position.

  This function uses the Composite-Rigid-Body Algorithm as described here:

  https://users.dimi.uniud.it/~antonio.dangelo/Robotica/2019/helper/Handbook-dynamics.pdf

  Args:
    sys: system defining the kinematic tree and other properties
    cinr: inertia in com frame
    cdof: dofs in com frame

  Returns:
    a symmetric positive matrix (qd_size, qd_size) representing the generalized
    mass and inertia of the system
  """
  # backward scan up tree: accumulate composite link inertias
  def crb_fn(crb_child, crb):
    if crb_child is not None:
      crb += crb_child
    return crb

  crb = scan.tree(sys, crb_fn, 'l', cinr, reverse=True)

  # expand composite inertias to a matrix: M[i,j] = cdof_j * crb[i] * cdof_i
  @jax.vmap
  def mx_row(dof_link, cdof_i):
    f = crb.take(dof_link).mul(cdof_i)

    @jax.vmap
    def mx_col(cdof_j):
      return cdof_j.dot(f)

    return mx_col(cdof)

  mx = mx_row(sys.dof_link(), cdof)

  # mask out empty parts of the matrix
  si, sj = [], []
  dof_ranges = sys.dof_ranges()
  for i in range(len(sys.link_parents)):
    j = i
    while j > -1:
      for dof_i, dof_j in itertools.product(dof_ranges[i], dof_ranges[j]):
        si, sj = si + [dof_i], sj + [dof_j]
      j = sys.link_parents[j]

  mx = mx * jp.zeros_like(mx).at[(jp.array(si), jp.array(sj))].set(1.0)

  # we mask i, j<=i, which is the lower triangular portion of the matrix
  # mirror it onto the upper triangular
  mx = jp.tril(mx) + jp.tril(mx, -1).T

  # add the armature inertia for rotors
  mx = mx + jp.diag(sys.dof.armature)

  return mx
