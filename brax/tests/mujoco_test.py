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

"""Tests for Mujoco converter."""

from absl.testing import absltest
from brax.tools import mujoco


_TEST_XML = """
<mujoco model="mjmodel">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="parent" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="parent_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="child_1" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="child_2" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="child_2_joint" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="child_2_geom" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="anon_joint" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="child_2_joint" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="anon_joint" gear="150"/>
  </actuator>
</mujoco>
"""


class MujocoTest(absltest.TestCase):

  def test_build(self):
    m = mujoco.MujocoConverter(_TEST_XML, add_collision_pairs=True)
    # Sanity check.
    config = m.config
    self.assertTrue(config.bodies)
    self.assertTrue(config.joints)
    self.assertTrue(config.actuators)
    self.assertTrue(config.collide_include)


if __name__ == '__main__':
  absltest.main()
