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

"""Halfcheetah."""

COLLIDES = ('torso', 'bfoot', 'ffoot')

ROOT = 'torso'

DEFAULT_OBSERVERS = ('root_z_joints',)

TERM_FN = None

SYSTEM_CONFIG = """
bodies {
  name: "torso"
  colliders {
    rotation {
      y: 90.0
    }
    capsule {
      radius: 0.04600000008940697
      length: 1.0920000076293945
    }
  }
  colliders {
    position {
      x: 0.6000000238418579
      z: 0.10000000149011612
    }
    rotation {
      y: 49.847328186035156
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.3919999897480011
    }
  }
  inertia {
    x: 0.9447969794273376
    y: 0.9447969794273376
    z: 0.9447969794273376
  }
  mass: 9.457332611083984
}
bodies {
  name: "bthigh"
  colliders {
    position {
      x: 0.10000000149011612
      z: -0.12999999523162842
    }
    rotation {
      x: -180.0
      y: 37.723960876464844
      z: -180.0
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.38199999928474426
    }
  }
  inertia {
    x: 0.029636280611157417
    y: 0.029636280611157417
    z: 0.029636280611157417
  }
  mass: 2.335526943206787
}
bodies {
  name: "bshin"
  colliders {
    position {
      x: -0.14000000059604645
      z: -0.07000000029802322
    }
    rotation {
      x: 180.0
      y: -63.68956756591797
      z: 180.0
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.3919999897480011
    }
  }
  inertia {
    x: 0.032029107213020325
    y: 0.032029107213020325
    z: 0.032029107213020325
  }
  mass: 2.402003049850464
}
bodies {
  name: "bfoot"
  colliders {
    position {
      x: 0.029999999329447746
      z: -0.09700000286102295
    }
    rotation {
      y: -15.469860076904297
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.2800000011920929
    }
  }
  inertia {
    x: 0.0117056118324399
    y: 0.0117056118324399
    z: 0.0117056118324399
  }
  mass: 1.6574708223342896
}
bodies {
  name: "fthigh"
  colliders {
    position {
      x: -0.07000000029802322
      z: -0.11999999731779099
    }
    rotation {
      y: 29.793806076049805
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.3580000102519989
    }
  }
  inertia {
    x: 0.024391336366534233
    y: 0.024391336366534233
    z: 0.024391336366534233
  }
  mass: 2.1759843826293945
}
bodies {
  name: "fshin"
  colliders {
    position {
      x: 0.06499999761581421
      z: -0.09000000357627869
    }
    rotation {
      y: -34.37746810913086
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.30399999022483826
    }
  }
  inertia {
    x: 0.014954624697566032
    y: 0.014954624697566032
    z: 0.014954624697566032
  }
  mass: 1.8170133829116821
}
bodies {
  name: "ffoot"
  colliders {
    position {
      x: 0.04500000178813934
      z: -0.07000000029802322
    }
    rotation {
      y: -34.37746810913086
    }
    capsule {
      radius: 0.04600000008940697
      length: 0.23199999332427979
    }
  }
  inertia {
    x: 0.006711110472679138
    y: 0.006711110472679138
    z: 0.006711110472679138
  }
  mass: 1.3383854627609253
}
joints {
  name: "bthigh"
  stiffness: 8000.0
  parent: "torso"
  child: "bthigh"
  parent_offset {
    x: -0.5
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  spring_damping: 100.0
  angle_limit {
    min: -29.793806076049805
    max: 60.16056823730469
  }
  limit_strength: 2000.0
}
joints {
  name: "bshin"
  stiffness: 5000.0
  parent: "bthigh"
  child: "bshin"
  parent_offset {
    x: 0.1599999964237213
    z: -0.25
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  spring_damping: 100.0
  angle_limit {
    min: -44.97718811035156
    max: 44.97718811035156
  }
  limit_strength: 1200.0
}
joints {
  name: "bfoot"
  stiffness: 5000.0
  parent: "bshin"
  child: "bfoot"
  parent_offset {
    x: -0.2800000011920929
    z: -0.14000000059604645
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  spring_damping: 100.0
  angle_limit {
    min: -22.918312072753906
    max: 44.97718811035156
  }
  limit_strength: 400.0
}
joints {
  name: "fthigh"
  stiffness: 8000.0
  parent: "torso"
  child: "fthigh"
  parent_offset {
    x: 0.5
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  spring_damping: 100.0
  angle_limit {
    min: -57.295780181884766
    max: 40.1070442199707
  }
  limit_strength: 2000.0
}
joints {
  name: "fshin"
  stiffness: 5000.0
  parent: "fthigh"
  child: "fshin"
  parent_offset {
    x: -0.14000000059604645
    z: -0.23999999463558197
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  spring_damping: 80.0
  angle_limit {
    min: -68.75493621826172
    max: 49.847328186035156
  }
  limit_strength: 400.0
}
joints {
  name: "ffoot"
  stiffness: 3500.0
  parent: "fshin"
  child: "ffoot"
  parent_offset {
    x: 0.12999999523162842
    z: -0.18000000715255737
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  spring_damping: 50.0
  angle_limit {
    min: -28.647890090942383
    max: 28.647890090942383
  }
  limit_strength: 200.0
}
actuators {
  name: "bthigh"
  joint: "bthigh"
  strength: 120.0
  torque {
  }
}
actuators {
  name: "bshin"
  joint: "bshin"
  strength: 90.0
  torque {
  }
}
actuators {
  name: "bfoot"
  joint: "bfoot"
  strength: 60.0
  torque {
  }
}
actuators {
  name: "fthigh"
  joint: "fthigh"
  strength: 120.0
  torque {
  }
}
actuators {
  name: "fshin"
  joint: "fshin"
  strength: 60.0
  torque {
  }
}
actuators {
  name: "ffoot"
  joint: "ffoot"
  strength: 30.0
  torque {
  }
}
"""


def get_specs():
  return dict(
      message_str=SYSTEM_CONFIG,
      collides=COLLIDES,
      root=ROOT,
      term_fn=TERM_FN,
      observers=DEFAULT_OBSERVERS)
