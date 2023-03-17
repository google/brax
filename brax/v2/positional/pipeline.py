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

"""Physics pipeline for fully articulated dynamics and collisiion."""
# pylint:disable=g-multiple-import
from brax.v2 import actuator
from brax.v2 import geometry
from brax.v2 import kinematics
from brax.v2.base import Motion, System
from brax.v2.positional import collisions
from brax.v2.positional import com
from brax.v2.positional import integrator
from brax.v2.positional import joints
from brax.v2.positional.base import State
from jax import numpy as jp


def init(
    sys: System, q: jp.ndarray, qd: jp.ndarray, debug: bool = False
) -> State:
  """Initializes physics state.

  Args:
    sys: a brax system
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    debug: if True, adds contact to the state for debugging

  Returns:
    state: initial physics state
  """
  # position/velocity level terms
  x, xd = kinematics.forward(sys, q, qd)
  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  x_i, xd_i = com.from_world(sys, x, xd)
  i_inv = com.inv_inertia(sys, x)
  mass = sys.link.inertia.mass ** (1 - sys.spring_mass_scale)

  return State(
      q=q,
      qd=qd,
      x=x,
      xd=xd,
      contact=geometry.contact(sys, x) if debug else None,
      x_i=x_i,
      xd_i=xd_i,
      j=j,
      jd=jd,
      a_p=a_p,
      a_c=a_c,
      i_inv=i_inv,
      mass=mass,
  )


def step(
    sys: System, state: State, act: jp.ndarray, debug: bool = False
) -> State:
  """Performs a single physics step using position-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    sys: system defining the kinematic tree and other properties
    state: physics state prior to step
    act: (act_size,) actuator input vector
    debug: if True, adds contact to the state for debugging

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  x, xd, q, qd, x_i, xd_i = (
      state.x,
      state.xd,
      state.q,
      state.qd,
      state.x_i,
      state.xd_i,
  )

  x_i_prev = x_i

  # calculate acceleration level updates
  tau = actuator.to_tau(sys, act, q)
  xdd_i = joints.acceleration_update(sys, state, tau) + Motion.create(
      vel=sys.gravity
  )

  xd_i = xd_i + xdd_i * sys.dt

  # now integrate and update position/velocity-level terms
  x_i, xd_i = integrator.integrate(
      sys,
      x_i,
      xd_i,
      Motion(vel=jp.zeros_like(xd_i.vel), ang=jp.zeros_like(xd_i.ang)),
  )

  # TODO: consolidate coordinate transformation and inv_inertia calls
  x, xd = com.to_world(sys, x_i, xd_i)
  inv_inertia = com.inv_inertia(sys, x)

  # perform position level joint updates
  p_j = joints.position_update(sys, x, xd, x_i, inv_inertia, state.mass)

  # apply position level joint updates
  x_i += p_j

  x, xd = com.to_world(sys, x_i, xd_i)
  inv_inertia = com.inv_inertia(sys, x)

  # apply position level contact updates
  contact = geometry.contact(sys, x)
  p_c, dlambda = collisions.resolve_position(
      sys, x_i, x_i_prev, inv_inertia, state.mass, contact=contact
  )
  x_i += p_c
  x_i_before, xd_i_before = x_i, xd_i

  # pbd velocity projection step
  x_i, xd_i = integrator.velocity_projection(sys, x_i, x_i_prev)

  x, xd = com.to_world(sys, x_i, xd_i)
  inv_inertia = com.inv_inertia(sys, x)

  # apply velocity level collision updates
  xdv_i = collisions.resolve_velocity(
      sys,
      x_i,
      xd_i,
      x_i_before,
      xd_i_before,
      inv_inertia,
      state.mass,
      contact,
      dlambda,
  )

  x_i, xd_i = integrator.integrate(sys, x_i, xd_i, xdv_i, include_kinetic=False)

  x, xd = com.to_world(sys, x_i, xd_i)

  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  q, qd = kinematics.inverse(sys, j, jd)
  state = state.replace(
      q=q,
      qd=qd,
      a_p=a_p,
      a_c=a_c,
      j=j,
      jd=jd,
      x_i=x_i,
      xd_i=xd_i,
      x=x,
      xd=xd,
      contact=geometry.contact(sys, x) if debug else None,
  )
  return state
