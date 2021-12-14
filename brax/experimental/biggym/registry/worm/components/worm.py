# Copyright 2021 The Brax Authors.
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

"""Procedural worm."""
# pylint:disable=unused-import
from brax.experimental.composer.components.ant import DEFAULT_OBSERVERS
from brax.experimental.composer.components.pro_ant import get_specs

def generate_worm_config_with_n_torso(n):
  assert n >= 2
  """Generate info for n-torso worm."""

  def template_torso(ind):
    tmp = f"""
      bodies {{
        name: "torso_{str(ind)}"
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
    collides = (f'torso_{str(ind)}',)
    return tmp, collides

  def template_joint(ind):
    tmp = f"""
      joints {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}"
        parent_offset {{ x: 0.25 z: 0.0 }}
        child_offset {{ x: -0.25 z: -0.0 }}
        parent: "torso_{str(ind)}"
        child: "torso_{str(ind+1)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -60.0 max: 60.0 }}
        rotation {{ y: -90 z: 0.0 }}
        reference_rotation {{ y: 0.0 }}
      }}
      actuators {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}"
        joint: "torso_{str(ind)}_torso_{str(ind+1)}"
        strength: 3000.0
        torque {{}}
      }}
      joints {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}_updown"
        parent_offset {{ x: 0.25 z: 0.0 }}
        child_offset {{ x: -0.25 z: -0.0 }}
        parent: "torso_{str(ind)}"
        child: "torso_{str(ind+1)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -60.0 max: 60.0 }}
        rotation {{ y: 0.0 z: 90.0 }}
        reference_rotation {{ y: 0.0 }}
      }}
      actuators {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}_updown"
        joint: "torso_{str(ind)}_torso_{str(ind+1)}_updown"
        strength: 3000.0
        torque {{}}
      }}
      """
    collides = tuple()
    return tmp, collides

  base_config = f""""""
  collides = tuple()
  for i in range(n):
    config_i, collides_i = template_torso(i)
    base_config += config_i
    collides += collides_i
    if i < n - 1:
      joint_i, _ = template_joint(i)
      base_config += joint_i
  return base_config, collides

def get_specs(num_torso: int = 6):
  message_str, collides = generate_worm_config_with_n_torso(num_torso)
  return dict(
      message_str=message_str,
      collides=collides,
      root='torso_0',
      term_fn=None,
      observers=DEFAULT_OBSERVERS)
