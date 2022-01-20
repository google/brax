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

"""Common component functions."""
import brax
from brax import math
from brax.experimental.braxlines.common import sim_utils
from jax import numpy as jnp


def upright_term_fn(done, sys, qp: brax.QP, info: brax.Info, component):
  """Terminate when it falls."""
  del info
  # upright termination
  index = sim_utils.names2indices(sys.config, component['root'], 'body')[0][0]
  rot = qp.rot[index]
  up = jnp.array([0., 0., 1.])
  torso_up = math.rotate(up, rot)
  torso_is_up = jnp.dot(torso_up, up)
  done = jnp.where(torso_is_up < 0.0, x=1.0, y=done)
  return done


def height_term_fn(done,
                   sys,
                   qp: brax.QP,
                   info: brax.Info,
                   component,
                   max_height: float = 1.0,
                   min_height: float = 0.2):
  """Terminate when it flips or jumps too high."""
  del info
  # height termination
  z_offset = component.get('term_params', {}).get('z_offset', 0.0)
  index = sim_utils.names2indices(sys.config, component['root'], 'body')[0][0]
  z = qp.pos[index][2]
  done = jnp.where(z < min_height + z_offset, x=1.0, y=done)
  done = jnp.where(z > max_height + z_offset, x=1.0, y=done)
  return done
