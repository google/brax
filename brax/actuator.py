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

  force *= sys.actuator.gear
  if moment_qd is None:
    tau = jp.zeros(sys.qd_size()).at[sys.actuator.qd_id].add(force)
  else:
    tau = moment_qd.T @ force

  return tau
