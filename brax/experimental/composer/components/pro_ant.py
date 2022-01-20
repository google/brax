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

"""Procedural ant."""
from brax.experimental.composer.components.ant import DEFAULT_OBSERVERS
from brax.experimental.composer.components.ant import ROOT
from brax.experimental.composer.components.ant import term_fn
import numpy as np


def generate_ant_config_with_n_legs(n):
  """Generate info for n-legged ant."""

  def template_leg(theta, ind):
    tmp = f"""
      bodies {{
      name: "Aux 1_{str(ind)}"
      colliders {{
        rotation {{ x: 90 y: -90 }}
        capsule {{
          radius: 0.08
          length: 0.4428427219390869
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: 1
    }}
    bodies {{
      name: "$ Body 4_{str(ind)}"
      colliders {{
        rotation {{ x: 90 y: -90 }}
        capsule {{
          radius: 0.08
          length: 0.7256854176521301
          end: -1
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: 1
    }}
      joints {{
        name: "{ROOT}_Aux 1_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*np.cos(theta)} y: {((0.4428427219390869/2.)+.08)*np.sin(theta)} }}
        child_offset {{ }}
        parent: "{ROOT}"
        child: "Aux 1_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {theta*180/np.pi} }}
      }}
      joints {{
      name: "Aux 1_$ Body 4_{str(ind)}"
      parent_offset {{ x: {0.4428427219390869/2. - .08}  }}
      child_offset {{ x:{-0.7256854176521301/2. + .08}  }}
      parent: "Aux 1_{str(ind)}"
      child: "$ Body 4_{str(ind)}"
      stiffness: 5000.0
      angular_damping: 35
      rotation: {{ z: 90 }}
      angle_limit {{
        min: 30.0
        max: 70.0
      }}
    }}
    actuators {{
      name: "{ROOT}_Aux 1_{str(ind)}"
      joint: "{ROOT}_Aux 1_{str(ind)}"
      strength: 350.0
      torque {{}}
    }}
    actuators {{
      name: "Aux 1_$ Body 4_{str(ind)}"
      joint: "Aux 1_$ Body 4_{str(ind)}"
      strength: 350.0
      torque {{}}
    }}
    """
    collides = (f'Aux 1_{str(ind)}', f'$ Body 4_{str(ind)}')
    return tmp, collides

  base_config = f"""
    bodies {{
      name: "{ROOT}"
      colliders {{
        capsule {{
          radius: 0.25
          length: 0.5
          end: 1
        }}
      }}
      inertia {{ x: 1.0 y: 1.0 z: 1.0 }}
      mass: 10
    }}
    """
  collides = (ROOT,)
  for i in range(n):
    config_i, collides_i = template_leg((1. * i / n) * 2 * np.pi, i)
    base_config += config_i
    collides += collides_i

  return base_config, collides


def get_specs(num_legs: int = 10):
  message_str, collides = generate_ant_config_with_n_legs(num_legs)
  return dict(
      message_str=message_str,
      collides=collides,
      root=ROOT,
      term_fn=term_fn,
      observers=DEFAULT_OBSERVERS)
