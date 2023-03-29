# Copyright 2023 The Brax Authors.
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

from brax import scan
from brax.base import System
from jax import numpy as jp


def to_tau(sys: System, act: jp.ndarray, q: jp.ndarray) -> jp.ndarray:
  """Convert actuator to a joint force tau.

  Args:
    sys: system defining the kinematic tree and other properties
    act: (act_size,) actuator force input vector
    q: joint position vector

  Returns:
    tau: (qd_size,) vector of joint forces
  """
  if sys.act_size() == 0:
    return jp.zeros(sys.qd_size())

  def act_fn(act_type, act, actuator, q, qd_idx):
    if act_type not in ('p', 'm'):
      raise RuntimeError(f'unrecognized act type: {act_type}')

    force = jp.clip(act, actuator.ctrl_range[:, 0], actuator.ctrl_range[:, 1])
    if act_type == 'p':
      force -= q  # positional actuators have a bias
    tau = actuator.gear * force

    return tau, qd_idx

  qd_idx = jp.arange(sys.qd_size())
  tau, qd_idx = scan.actuator_types(
      sys, act_fn, 'aaqd', 'a', act, sys.actuator, q, qd_idx
  )
  tau = jp.zeros(sys.qd_size()).at[qd_idx].add(tau)

  return tau
