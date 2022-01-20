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

"""Tests for the URDF converter."""

from absl.testing import absltest
from brax.tools import urdf


_TEST_XML = """
<robot name="test robot">
	<joint name="test_joint" type="revolute">
		<parent link="parent_link" />
		<child link="child_link" />
		<dynamics damping="1.0" friction="0.0001" />
		<origin rpy="1.57080 0.0 1.57080" xyz="0.1 0.2 -0.3" />
		<axis xyz="1.00000 0.00000 0.00000" />
	</joint>
  <link name="parent_link">
      <inertial>
        <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000" />
        <mass value="1.00000" />
        <inertia ixx="0.00100" ixy="0" ixz="0" iyy="0.00100" iyz="0" izz="0.00100" />
      </inertial>
      <visual>
        <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000" />
        <geometry>
          <sphere radius="0.05000" />
        </geometry>
      </visual>
    </link>
    <link name="child_link">
      <inertial>
        <origin rpy="0.00000 -0.00000 0.00000" xyz="0.0 0.0 -0.0" />
        <mass value="2.0" />
        <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1" />
      </inertial>
      <visual>
        <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0" />
        <geometry>
          <cylinder length="0.5" radius="0.1" />
        </geometry>
      </visual>
        <collision>
          <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0" />
            <geometry>
              <cylinder length="0.5" radius="0.1" />
            </geometry>
        </collision>
      </link>
</robot>
"""


class UrdfTest(absltest.TestCase):

  def test_build(self):
    m = urdf.UrdfConverter(_TEST_XML, add_collision_pairs=True)
    # Sanity check.
    config = m.config
    self.assertEqual(len(config.bodies), 2)
    self.assertEqual(config.bodies[0].name, 'parent_link')
    self.assertEqual(config.bodies[1].name, 'child_link')
    self.assertEqual(len(config.joints), 1)
    self.assertEqual(config.joints[0].name, 'test_joint')
    self.assertEqual(len(config.actuators), 1)
    self.assertEqual(config.actuators[0].name, 'test_joint')


if __name__ == '__main__':
  absltest.main()
