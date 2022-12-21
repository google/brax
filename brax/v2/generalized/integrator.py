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
"""Integrator functions."""

from typing import Tuple
from brax.v2 import math
from brax.v2 import scan
from brax.v2.base import System
import jax
from jax import numpy as jp


def _integrate_q_axis(sys: System, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
  """Integrates next q for revolute/prismatic joints."""
  return q + qd * sys.dt


def _integrate_q_free(sys: System, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
  """Integrates next q for free joints."""
  rot, ang = q[3:7], qd[3:6]
  ang_norm = jp.linalg.norm(ang) + 1e-8
  axis = ang / ang_norm
  angle = sys.dt * ang_norm
  qrot = math.quat_rot_axis(axis, angle)
  rot = math.quat_mul(rot, qrot)
  rot = rot / jp.linalg.norm(rot)
  pos, vel = q[0:3], qd[0:3]
  pos += vel * sys.dt

  return jp.concatenate([pos, rot])


def integrate(
    sys: System, q: jp.ndarray, qd: jp.ndarray, qdd: jp.ndarray
) -> Tuple[jp.ndarray, jp.ndarray]:
  """Semi-implicit euler integration.

  Args:
    sys: system defining the kinematic tree and other properties
    q: joint angle vector. quaternions for free/spherical, scalar otherwise
    qd: joint velocity vector
    qdd: joint acceleration vector

  Returns:
    q: updated joint angle vector
    qd: updated joint velocity vector
  """

  qd += qdd * sys.dt

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

  q = scan.link_types(sys, q_fn, 'lqd', 'q', sys.link, q, qd)

  return q, qd
