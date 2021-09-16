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

"""Ant."""
import brax
from brax.experimental.braxlines.common import sim_utils
from brax.experimental.composer import component_editor
from jax import numpy as jnp

COLLIDES = ('torso', '$ Body 4', '$ Body 7', '$ Body 10', '$ Body 13')

ROOT = 'torso'

DEFAULT_OBSERVERS = ('root_z_joints', 'cfrc')


def term_fn(done, sys, qp: brax.QP, info: brax.Info, suffix: str):
  """Terminate when it flips or jumps too high."""
  del info
  index = sim_utils.names2indices(sys.config,
                                  component_editor.add_suffix(ROOT, suffix),
                                  'body')[0][0]
  z = qp.pos[index][2]
  done = jnp.where(z < 0.2, x=1.0, y=done)
  done = jnp.where(z > 1.0, x=1.0, y=done)
  return done


TERM_FN = term_fn

SYSTEM_CONFIG = """
bodies {
  name: "torso"
  colliders {
    capsule {
      radius: 0.25
      length: 0.5
      end: 1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 10
}
bodies {
  name: "Aux 1"
  colliders {
    rotation { x: 90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 4"
  colliders {
    rotation { x: 90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 2"
  colliders {
    rotation { x: 90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 7"
  colliders {
    rotation { x: 90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 3"
  colliders {
    rotation { x: -90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 10"
  colliders {
    rotation { x: -90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 4"
  colliders {
    rotation { x: -90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 13"
  colliders {
    rotation { x: -90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
joints {
  name: "torso_Aux 1"
  parent_offset { x: 0.2 y: 0.2 }
  child_offset { x: -0.1 y: -0.1 }
  parent: "torso"
  child: "Aux 1"
  stiffness: 5000.0
  angular_damping: 35
  angle_limit { min: -30.0 max: 30.0 }
  rotation { y: -90 }
}
joints {
  name: "Aux 1_$ Body 4"
  parent_offset { x: 0.1 y: 0.1 }
  child_offset { x: -0.2 y: -0.2 }
  parent: "Aux 1"
  child: "$ Body 4"
  stiffness: 5000.0
  angular_damping: 35
  rotation: { z: 135 }
  angle_limit {
    min: 30.0
    max: 70.0
  }
}
joints {
  name: "torso_Aux 2"
  parent_offset { x: -0.2 y: 0.2 }
  child_offset { x: 0.1 y: -0.1 }
  parent: "torso"
  child: "Aux 2"
  stiffness: 5000.0
  angular_damping: 35
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 2_$ Body 7"
  parent_offset { x: -0.1 y: 0.1 }
  child_offset { x: 0.2 y: -0.2 }
  parent: "Aux 2"
  child: "$ Body 7"
  stiffness: 5000.0
  angular_damping: 35
  rotation { z: 45 }
  angle_limit { min: -70.0 max: -30.0 }
}
joints {
  name: "torso_Aux 3"
  parent_offset { x: -0.2 y: -0.2 }
  child_offset { x: 0.1 y: 0.1 }
  parent: "torso"
  child: "Aux 3"
  stiffness: 5000.0
  angular_damping: 35
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 3_$ Body 10"
  parent_offset { x: -0.1 y: -0.1 }
  child_offset {
    x: 0.2
    y: 0.2
  }
  parent: "Aux 3"
  child: "$ Body 10"
  stiffness: 5000.0
  angular_damping: 35
  rotation { z: 135 }
  angle_limit { min: -70.0 max: -30.0 }
}
joints {
  name: "torso_Aux 4"
  parent_offset { x: 0.2 y: -0.2 }
  child_offset { x: -0.1 y: 0.1 }
  parent: "torso"
  child: "Aux 4"
  stiffness: 5000.0
  angular_damping: 35
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 4_$ Body 13"
  parent_offset { x: 0.1 y: -0.1 }
  child_offset { x: -0.2 y: 0.2 }
  parent: "Aux 4"
  child: "$ Body 13"
  stiffness: 5000.0
  angular_damping: 35
  rotation { z: 45 }
  angle_limit { min: 30.0 max: 70.0 }
}
actuators {
  name: "torso_Aux 1"
  joint: "torso_Aux 1"
  strength: 300.0
  torque {}
}
actuators {
  name: "Aux 1_$ Body 4"
  joint: "Aux 1_$ Body 4"
  strength: 300.0
  torque {}
}
actuators {
  name: "torso_Aux 2"
  joint: "torso_Aux 2"
  strength: 300.0
  torque {}
}
actuators {
  name: "Aux 2_$ Body 7"
  joint: "Aux 2_$ Body 7"
  strength: 300.0
  torque {}
}
actuators {
  name: "torso_Aux 3"
  joint: "torso_Aux 3"
  strength: 300.0
  torque {}
}
actuators {
  name: "Aux 3_$ Body 10"
  joint: "Aux 3_$ Body 10"
  strength: 300.0
  torque {}
}
actuators {
  name: "torso_Aux 4"
  joint: "torso_Aux 4"
  strength: 300.0
  torque {}
}
actuators {
  name: "Aux 4_$ Body 13"
  joint: "Aux 4_$ Body 13"
  strength: 300.0
  torque {}
}
"""


def get_specs():
  return dict(
      message_str=SYSTEM_CONFIG,
      collides=COLLIDES,
      root=ROOT,
      term_fn=TERM_FN,
      observers=DEFAULT_OBSERVERS)
