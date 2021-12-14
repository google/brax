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

"""Procedural centipade."""
from brax.experimental.composer.components.ant import DEFAULT_OBSERVERS
from brax.experimental.composer.components.ant import term_fn
from jax import numpy as jnp

def generate_centipede_config_with_n_torso(n):
  assert n >= 2
  assert n == 3  # TODO: debug n > 3
  """Generate info for n-torso centipede."""

  def template_torso(theta, ind):
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
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*jnp.cos(theta)} y: {((0.4428427219390869/2.)+.08)*jnp.sin(theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 1_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {theta*180/jnp.pi} }}
      }}
      joints {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
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
        name: "torso_{str(ind)}_Aux 1_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 1_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        joint: "Aux 1_{str(ind)}_$ Body 4_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      bodies {{
        name: "Aux 2_{str(ind)}"
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
        name: "$ Body 5_{str(ind)}"
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
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        parent_offset {{ x: {((0.4428427219390869/2.)+.08)*jnp.cos(-theta)} y: {((0.4428427219390869/2.)+.08)*jnp.sin(-theta)} }}
        child_offset {{ }}
        parent: "torso_{str(ind)}"
        child: "Aux 2_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        angle_limit {{ min: -30.0 max: 30.0 }}
        rotation {{ y: -90 }}
        reference_rotation {{ z: {-theta*180/jnp.pi} }}
      }}
      joints {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        parent_offset {{ x: 0.14142136096954344  }}
        child_offset {{ x:-0.28284270882606505  }}
        parent: "Aux 2_{str(ind)}"
        child: "$ Body 5_{str(ind)}"
        stiffness: 5000.0
        angular_damping: 35
        rotation: {{ z: 90 }}
        angle_limit {{
          min: 30.0
          max: 70.0
        }}
      }}
      actuators {{
        name: "torso_{str(ind)}_Aux 2_{str(ind)}"
        joint: "torso_{str(ind)}_Aux 2_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      actuators {{
        name: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        joint: "Aux 2_{str(ind)}_$ Body 5_{str(ind)}"
        strength: 350.0
        torque {{}}
      }}
      """
    collides = (
      f'torso_{str(ind)}',
      f'Aux 1_{str(ind)}',
      f'$ Body 4_{str(ind)}',
      f'Aux 2_{str(ind)}',
      f'$ Body 5_{str(ind)}')

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
        angle_limit {{ min: -20.0 max: 20.0 }}
        rotation {{ y: -90 z: 60 }}
        reference_rotation {{ y: 0.0 }}
      }}
      actuators {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}"
        joint: "torso_{str(ind)}_torso_{str(ind+1)}"
        strength: 300.0
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
        angle_limit {{ min: -10.0 max: 30.0 }}
        rotation {{ y: 0.0 z: 90.0 }}
        reference_rotation {{ y: 0.0 }}
      }}
      actuators {{
        name: "torso_{str(ind)}_torso_{str(ind+1)}_updown"
        joint: "torso_{str(ind)}_torso_{str(ind+1)}_updown"
        strength: 300.0
        torque {{}}
      }}
      """
    collides = tuple()
    return tmp, collides

  base_config = f""""""
  collides = tuple()
  for i in range(n):
    theta = jnp.pi / 2
    config_i, collides_i = template_torso(theta, i)
    base_config += config_i
    collides += collides_i
    if i < n - 1:
      joint_i, _ = template_joint(i)
      base_config += joint_i
  return base_config, collides

def get_specs(num_torso: int = 3):
  message_str, collides = generate_centipede_config_with_n_torso(num_torso)
  return dict(
      message_str=message_str,
      collides=collides,
      root='torso_0',
      term_fn=term_fn,
      observers=DEFAULT_OBSERVERS)
