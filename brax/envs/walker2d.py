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

"""Trains a 2D walker to run in the +x direction."""

from typing import Tuple

from brax.envs import hopper


class Walker2d(hopper.Hopper):
  """Trains a 2D walker to run in the +x direction.

  This is similar to the Walker2d-V3 Mujoco environment in OpenAI Gym, which is
  a variant of Hopper with two legs. The two legs do not collide with each
  other.
  """

  def __init__(self,
               *argv,
               healthy_z_range: Tuple[float, float] = (0.7, 2.0),
               **kwargs):
    super().__init__(
        *argv,
        **kwargs,
        healthy_z_range=healthy_z_range,
        system_config=_SYSTEM_CONFIG)


_SYSTEM_CONFIG = """
bodies {
  name: "torso"
  colliders {
    position {}
    rotation {}
    capsule {
      radius: 0.05
      length: 0.5
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 3.6651914
}
bodies {
  name: "thigh"
  colliders {
    position { z: -0.225 }
    rotation {}
    capsule {
      radius: 0.05
      length: 0.55
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 4.0578904
}
bodies {
  name: "leg"
  colliders {
    position {}
    rotation {}
    capsule {
      radius: 0.04
      length: 0.58
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 2.7813568
}
bodies {
  name: "foot"
  colliders {
    position {
      x: -0.1
      y: -0.2
      z: -0.1
    }
    rotation { y: 90.0 }
    capsule {
      radius: 0.06
      length: 0.32
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 3.1667254
}
bodies {
  name: "thigh_left"
  colliders {
    position { z: -0.225 }
    rotation {}
    capsule {
      radius: 0.05
      length: 0.55
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 4.0578904
}
bodies {
  name: "leg_left"
  colliders {
    position {}
    rotation {}
    capsule {
      radius: 0.04
      length: 0.58
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 2.7813568
}
bodies {
  name: "foot_left"
  colliders {
    position {
      x: -0.1
      y: -0.2
      z: -0.1
    }
    rotation { y: 90.0 }
    capsule {
      radius: 0.06
      length: 0.32
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 3.1667254
}
bodies {
  name: "floor"
  colliders {
    plane {}
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  frozen { all: true }
}
joints {
  name: "thigh_joint"
  stiffness: 10000.0
  parent: "torso"
  child: "thigh"
  parent_offset { z: -0.2 }
  rotation { z: -90.0 }
  angle_limit { min: -150.0 }
  angular_damping: 20.0
}
joints {
  name: "leg_joint"
  stiffness: 10000.0
  parent: "thigh"
  child: "leg"
  parent_offset { z: -0.45 }
  child_offset { z: 0.25 }
  rotation { z: -90.0 }
  angle_limit { min: -150.0 }
  angular_damping: 20.0
}
joints {
  name: "foot_joint"
  stiffness: 10000.0
  parent: "leg"
  child: "foot"
  parent_offset { z: -0.25 }
  child_offset {
    x: -0.2
    y: -0.2
    z: -0.1
  }
  rotation { z: -90.0 }
  angle_limit { min: -45.0 max: 45.0 }
  angular_damping: 20.0
}
joints {
  name: "thigh_left_joint"
  stiffness: 10000.0
  parent: "torso"
  child: "thigh_left"
  parent_offset { z: -0.2 }
  rotation { z: -90.0 }
  angle_limit { min: -150.0 }
  angular_damping: 20.0
}
joints {
  name: "leg_left_joint"
  stiffness: 10000.0
  parent: "thigh_left"
  child: "leg_left"
  parent_offset { z: -0.45 }
  child_offset { z: 0.25 }
  rotation { z: -90.0 }
  angle_limit { min: -150.0 }
  angular_damping: 20.0
}
joints {
  name: "foot_left_joint"
  stiffness: 10000.0
  parent: "leg_left"
  child: "foot_left"
  parent_offset { z: -0.25 }
  child_offset {
    x: -0.2
    y: -0.2
    z: -0.1
  }
  rotation { z: -90.0 }
  angle_limit { min: -45.0 max: 45.0 }
  angular_damping: 20.0
}
actuators {
  name: "thigh_joint"
  joint: "thigh_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "leg_joint"
  joint: "leg_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "foot_joint"
  joint: "foot_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "thigh_left_joint"
  joint: "thigh_left_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "leg_left_joint"
  joint: "leg_left_joint"
  strength: 100.0
  torque {}
}
actuators {
  name: "foot_left_joint"
  joint: "foot_left_joint"
  strength: 100.0
  torque {}
}
friction: 0.9
gravity { z: -9.81 }
velocity_damping: 1.0
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "floor"
  second: "torso"
}
collide_include {
  first: "floor"
  second: "thigh"
}
collide_include {
  first: "floor"
  second: "leg"
}
collide_include {
  first: "floor"
  second: "foot"
}
collide_include {
  first: "floor"
  second: "thigh_left"
}
collide_include {
  first: "floor"
  second: "leg_left"
}
collide_include {
  first: "floor"
  second: "foot_left"
}
dt: 0.02
substeps: 4
frozen {
  position { y: 1.0 }
  rotation { x: 1.0 z: 1.0 }
}
defaults {
  qps { name: "torso" pos { z: 1.19 } }
  angles { name: "thigh_joint" angle {} }
  angles { name: "leg_joint" angle {} }
  angles { name: "thigh_left_joint" angle {} }
  angles { name: "leg_left_joint" angle {} }
}
"""
