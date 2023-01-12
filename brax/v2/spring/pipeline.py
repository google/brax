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
"""Physics pipeline for fully articulated dynamics and collisiion."""

from brax.v2 import actuator
from brax.v2 import base
from brax.v2 import geometry
from brax.v2 import kinematics
from brax.v2.base import System, Transform
from brax.v2.spring import collisions
from brax.v2.spring import integrator
from brax.v2.spring import joints
from brax.v2.spring import maximal
from flax import struct

import jax
from jax import numpy as jp


@struct.dataclass
class State(base.State):
  """Dynamic state that changes after every step."""


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
  contact = geometry.contact(sys, x)
  return State(q, qd, x, xd, contact)


def step(sys: System, state: State, act: jp.ndarray) -> State:
  """Performs a single physics step using spring-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    sys: system defining the kinematic tree and other properties
    state: physics state prior to step
    act: (act_size,) actuator input vector

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  x, xd, q, qd = state.x, state.xd, state.q, state.qd

  # calculate forces arising from different components
  # TODO: consider a segment_sum in joints.resolve, so that tau and
  # f_j can be combined here
  f_j, pos_j, link_idx_j = joints.resolve(sys, x, xd)

  tau_local = actuator.to_tau(sys, act, state.q)
  f_a, pos_a, link_idx_a = actuator.to_tau_world(sys, state.q, tau_local)

  # move to center of mass
  xi, xdi = maximal.maximal_to_com(sys, x, xd)
  coord_transform = Transform(pos=xi.pos - x.pos, rot=x.rot)
  inv_inertia = maximal.com_inv_inertia(sys, x)

  f, pos, link_idxs = (
      jax.tree_map(lambda x, y: jp.vstack([x, y]), f_j, f_a),
      jp.concatenate([pos_j, pos_a]),
      jp.concatenate([link_idx_j, link_idx_a]),
  )

  # update state with forces
  xdi = integrator.forward(
      sys,
      xi,
      xdi,
      inv_inertia,
      f=f,
      pos=pos,
      link_idx=link_idxs,
  )

  # resolve collisions
  contact = geometry.contact(sys, x)
  p_c, pos_c, link_idx_c = collisions.resolve(
      sys, xi, xdi, inv_inertia, contact
  )
  xi, xdi = integrator.forward_c(
      sys, xi, xdi, inv_inertia, p=p_c, pos=pos_c, link_idx=link_idx_c
  )

  # move back to world frame
  x, xd = maximal.com_to_maximal(xi, xdi, coord_transform)

  q, qd = kinematics.inverse(sys, x, xd)
  return State(q, qd, x, xd, contact)
