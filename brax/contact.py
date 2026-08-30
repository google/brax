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

# explore_bench fork: brax's supported geom pairs are mjx's, and mjx omits
# CYLINDER-BOX despite carrying the generic convex routine. See
# brax/_fork_collisions.py.
from brax import _fork_collisions as _fork_collisions
_fork_collisions.register()

# explore_bench fork: the narrowphase A/B switch.  mjx's `box_box` is 87% of
# the whole narrowphase and the narrowphase is 90.6% of a substep, so BOX-BOX
# is routed to `brax/_fork_narrowphase.py`.  mjx's path is retained unchanged
# as the correctness reference: set `BRAX_FORK_NARROWPHASE=0` (or
# `contact.USE_FORK_NARROWPHASE = False` before the first trace) to get it back
# for an A/B.  Which geom-type pairs the fork claims is
# `_fork_narrowphase.ENABLED`; everything else goes to mjx either way.
import os as _os
from brax import _fork_narrowphase as _fork_narrowphase

USE_FORK_NARROWPHASE = _os.environ.get(
    'BRAX_FORK_NARROWPHASE', '1').lower() not in ('0', 'false', 'off', 'no')


#: explore_bench fork: `mjx.make_data(sys)` rebuilt an ENTIRE mjx Data on every
#: call to `get`, i.e. once per physics substep, inside the vmap over worlds --
#: qpos/qvel/qfrc/efc/contact buffers and all, none of which brax uses except
#: as a container to hand `geom_xpos`/`geom_xmat` to `mjx.collision`. It
#: depends only on the model, so it is built once per System and reused.
#: Keyed on the model's identity, which is a non-pytree field and therefore
#: stable across traces.
_DATA_CACHE = {}


def _template_data(sys: System):
  key = id(sys.mj_model) if sys.mj_model is not None else id(sys)
  d = _DATA_CACHE.get(key)
  if d is None:
    d = mjx.make_data(sys)
    _DATA_CACHE[key] = d
  return d


def get(sys: System, x: Transform) -> Optional[Contact]:
  """Calculates contacts.

  Args:
    sys: system defining the kinematic tree and other properties
    x: link transforms in world frame

  Returns:
    Contact pytree
  """
  d = _template_data(sys)
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
  # explore_bench fork: route through the broadphase cull. Defaults (-1, -1)
  # are exactly mjx.collision, so this is a no-op unless a budget is set.
  _collide = (_fork_narrowphase.collide if USE_FORK_NARROWPHASE
              else _fork_collisions.collide)
  d = _collide(
      sys, d,
      max_geom_pairs=int(sys.max_geom_pairs),
      max_contact_points=int(sys.max_contact_points),
      by_type=sys.max_geom_pairs_by_type)
  # pytype: enable=wrong-arg-types

  c = d.contact
  elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5

  link_idx = (jp.array(g_link)[c.geom1], jp.array(g_link)[c.geom2])

  return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
