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
"""Calculations for generating contacts."""

from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx


def get(sys: System, x: Transform) -> Optional[Contact]:
  """Calculates contacts.

  Args:
    sys: system defining the kinematic tree and other properties
    x: link transforms in world frame

  Returns:
    Contact pytree
  """
  d = mjx.make_data(sys)
  if d.ncon == 0:
    return None

  @jax.vmap
  def local_to_global(pos1, quat1, pos2, quat2):
    pos = pos1 + math.rotate(pos2, quat1)
    mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
    return pos, mat

  # explore_bench fork: resolve a geom's link through the explicit map, and
  # place it with its LINK-relative pose. `geom_bodyid - 1` was only valid
  # while every body was a link (see io/_fork_links).
  x = x.concatenate(Transform.zero((1,)))
  g_link = sys.geom_link_idx
  geom_xpos, geom_xmat = local_to_global(
      x.pos[g_link], x.rot[g_link], sys.geom_link_pos, sys.geom_link_quat
  )

  # pytype: disable=wrong-arg-types
  d = d.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
  d = mjx.collision(sys, d)
  # pytype: enable=wrong-arg-types

  c = d.contact
  elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5

  link_idx = (jp.array(g_link)[c.geom1], jp.array(g_link)[c.geom2])

  return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
