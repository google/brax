# Copyright 2024 The Brax Authors.
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

"""Octopus."""
from brax.v1.experimental.composer.components.common import upright_term_fn

ROOT = 'octopus'

SYSTEM_CONFIG = """
bodies {
  name: "octopus"
  colliders {
    position {
      z: 0.009999999776482582
    }
    sphere {
      radius: 0.009999999776482582
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_body"
  colliders {
    position {
      z: 0.009999999776482582
    }
    sphere {
      radius: 0.009999999776482582
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_B"
  colliders {
    position {
      x: -0.03700000047683716
      z: 0.11810000240802765
    }
    rotation {
      y: -45.00010681152344
    }
    capsule {
      radius: 0.06199999898672104
      length: 0.23399999737739563
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_0_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_0_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_1_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_1_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_1_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_2_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_2_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_2_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_3_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_3_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_3_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_4_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_4_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_4_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_5_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_5_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_5_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_6_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_6_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_6_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_7_B"
  colliders {
    position {
      y: 0.06830000132322311
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.057999998331069946
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_7_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
bodies {
  name: "octopus_0_7_0_0_B"
  colliders {
    position {
      y: 0.03700000047683716
    }
    rotation {
      x: 89.95437622070312
    }
    capsule {
      radius: 0.02199999988079071
      length: 0.11800000071525574
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
    }
    rotation {
    }
  }
}
joints {
  name: "$octopus.octopus_body"
  stiffness: 7500.0
  parent: "octopus"
  child: "octopus_body"
  rotation {
    y: -90.0
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_body"
  child: "octopus_0_B"
  parent_offset {
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_0_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_0_B"
  child: "octopus_0_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_0_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_0_0_B"
  child: "octopus_0_0_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_1_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_1_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: 45.0
  }
}
joints {
  name: "octopus_0_1_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_1_B"
  child: "octopus_0_1_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_1_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_1_0_B"
  child: "octopus_0_1_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_2_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_2_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: 90.0
  }
}
joints {
  name: "octopus_0_2_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_2_B"
  child: "octopus_0_2_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_2_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_2_0_B"
  child: "octopus_0_2_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_3_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_3_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: 135.0
  }
}
joints {
  name: "octopus_0_3_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_3_B"
  child: "octopus_0_3_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_3_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_3_0_B"
  child: "octopus_0_3_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_4_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_4_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: 180.0
  }
}
joints {
  name: "octopus_0_4_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_4_B"
  child: "octopus_0_4_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_4_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_4_0_B"
  child: "octopus_0_4_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_5_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_5_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: -135.0
  }
}
joints {
  name: "octopus_0_5_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_5_B"
  child: "octopus_0_5_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_5_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_5_0_B"
  child: "octopus_0_5_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_6_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_6_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: -90.0
  }
}
joints {
  name: "octopus_0_6_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_6_B"
  child: "octopus_0_6_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_6_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_6_0_B"
  child: "octopus_0_6_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_7_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_B"
  child: "octopus_0_7_B"
  parent_offset {
    z: 0.014999999664723873
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
    z: -45.0
  }
}
joints {
  name: "octopus_0_7_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_7_B"
  child: "octopus_0_7_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
joints {
  name: "octopus_0_7_0_0_J_R100"
  stiffness: 7500.0
  parent: "octopus_0_7_0_B"
  child: "octopus_0_7_0_0_B"
  parent_offset {
    y: 0.07400000095367432
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 10.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 400.0
  spring_damping: 50.0
  reference_rotation {
  }
}
actuators {
  name: "$octopus.octopus_body"
  joint: "$octopus.octopus_body"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_J_R100"
  joint: "octopus_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_0_J_R100"
  joint: "octopus_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_0_0_J_R100"
  joint: "octopus_0_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_0_0_0_J_R100"
  joint: "octopus_0_0_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_1_J_R100"
  joint: "octopus_0_1_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_1_0_J_R100"
  joint: "octopus_0_1_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_1_0_0_J_R100"
  joint: "octopus_0_1_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_2_J_R100"
  joint: "octopus_0_2_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_2_0_J_R100"
  joint: "octopus_0_2_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_2_0_0_J_R100"
  joint: "octopus_0_2_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_3_J_R100"
  joint: "octopus_0_3_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_3_0_J_R100"
  joint: "octopus_0_3_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_3_0_0_J_R100"
  joint: "octopus_0_3_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_4_J_R100"
  joint: "octopus_0_4_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_4_0_J_R100"
  joint: "octopus_0_4_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_4_0_0_J_R100"
  joint: "octopus_0_4_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_5_J_R100"
  joint: "octopus_0_5_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_5_0_J_R100"
  joint: "octopus_0_5_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_5_0_0_J_R100"
  joint: "octopus_0_5_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_6_J_R100"
  joint: "octopus_0_6_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_6_0_J_R100"
  joint: "octopus_0_6_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_6_0_0_J_R100"
  joint: "octopus_0_6_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_7_J_R100"
  joint: "octopus_0_7_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_7_0_J_R100"
  joint: "octopus_0_7_0_J_R100"
  strength: 150.0
  torque {
  }
}
actuators {
  name: "octopus_0_7_0_0_J_R100"
  joint: "octopus_0_7_0_0_J_R100"
  strength: 150.0
  torque {
  }
}
"""

COLLIDES = ('octopus', 'octopus_body', 'octopus_0_B')
for i in range(8):
  COLLIDES = (
      f'octopus_0_{i}_B',
      f'octopus_0_{i}_0_B',
      f'octopus_0_{i}_0_0_B',
  )

DEFAULT_OBSERVERS = ('root_z_joints',)


def get_specs():
  return dict(
      message_str=SYSTEM_CONFIG,
      collides=COLLIDES,
      root=ROOT,
      term_fn=upright_term_fn,
      observers=DEFAULT_OBSERVERS)
