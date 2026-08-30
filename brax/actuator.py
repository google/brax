# Copyright 2026 The Brax Authors.
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
"""Functions for applying actuators to a physics pipeline."""

from brax.base import System
import jax
from jax import numpy as jp


def to_tau(
    sys: System, act: jax.Array, q: jax.Array, qd: jax.Array
) -> jax.Array:
  """Convert actuator to a joint force tau.

  Args:
    sys: system defining the kinematic tree and other properties
    act: (act_size,) actuator force input vector
    q: joint position vector
    qd: joint velocity vector

  Returns:
    tau: (qd_size,) vector of joint forces
  """
  if sys.act_size() == 0:
    return jp.zeros(sys.qd_size())

  ctrl_range = sys.actuator.ctrl_range
  force_range = sys.actuator.force_range

  # explore_bench fork: transmission through a MOMENT MATRIX. For a joint
  # transmission the row is one-hot and this is identical to the previous
  # q[q_id] / tau.at[qd_id].add(...) formulation. For a FIXED TENDON the row
  # holds the wrap coefficients, so the transmission coordinate is the tendon
  # LENGTH sum(coef_i * q_i) -- which is what a tendon position servo
  # (biastype 1) reads, and what per-joint indices could not express.
  moment_q, moment_qd = sys.actuator.moment_q, sys.actuator.moment_qd
  if moment_q is None:
    q, qd = q[sys.actuator.q_id], qd[sys.actuator.qd_id]
  else:
    q, qd = moment_q @ q, moment_qd @ qd
  act = jp.clip(act, ctrl_range[:, 0], ctrl_range[:, 1])
  # See https://github.com/deepmind/mujoco/discussions/754 for why gear is
  # used for the bias term.
  bias = sys.actuator.gear * (
      q * sys.actuator.bias_q + qd * sys.actuator.bias_qd
  )

  force = sys.actuator.gain * act + bias
  force = jp.clip(force, force_range[:, 0], force_range[:, 1])
  # explore_bench fork: a position servo resolved as a positional DRIVE
  # (positional/joints.drive_update) must not also be applied here.
  if sys.drive_force_mask is not None:
    force = force * sys.drive_force_mask

  force *= sys.actuator.gear
  if moment_qd is None:
    tau = jp.zeros(sys.qd_size()).at[sys.actuator.qd_id].add(force)
  else:
    tau = moment_qd.T @ force

  return tau


def site_force(sys: System, act: jax.Array, x, x_i):
  """explore_bench fork: per-link wrench from SITE-transmission actuators.

  A site actuator (a quadrotor rotor, say) applies a force and torque at a
  FRAME rather than about a joint, so it has no joint-space equivalent and no
  model-level substitution that leaves the force where the task's rules read
  it. brax's positional pipeline is maximal-coordinate, so no Jacobian is
  needed: rotate the geared wrench into the world, then move the force to the
  link's centre of mass, adding the r x F transport torque.

  Args:
    sys: system, carrying the fork's site_act_* arrays
    act: (act_size,) actuator input vector
    x: link transforms in world frame
    x_i: link centre-of-mass transforms in world frame

  Returns:
    Force with one row per link.
  """
  from brax import math
  from brax.base import Force
  from jax.ops import segment_sum

  link = sys.site_act_link
  n_link = sys.num_links()
  gear = sys.site_act_gear
  is_site = link >= 0
  idx = jp.where(is_site, link, 0)

  act = jp.clip(act, sys.actuator.ctrl_range[:, 0], sys.actuator.ctrl_range[:, 1])
  f = sys.actuator.gain * act
  f = jp.clip(f, sys.actuator.force_range[:, 0], sys.actuator.force_range[:, 1])
  f = jp.where(is_site, f, 0.0)

  # site frame in world: link transform composed with the site's link-relative
  # pose (both recorded by io/_fork_links)
  link_rot = x.rot[idx]
  site_rot = jax.vmap(math.quat_mul)(link_rot, sys.site_act_quat)
  site_pos = x.pos[idx] + jax.vmap(math.rotate)(sys.site_act_pos, link_rot)

  vel = jax.vmap(math.rotate)(gear[:, 0:3] * f[:, None], site_rot)
  ang = jax.vmap(math.rotate)(gear[:, 3:6] * f[:, None], site_rot)
  # transport the force from the site to the link centre of mass
  ang = ang + jp.cross(site_pos - x_i.pos[idx], vel)

  return Force(
      vel=segment_sum(vel, idx, n_link),
      ang=segment_sum(ang, idx, n_link),
  )
