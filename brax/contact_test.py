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

"""Tests for contacts."""

from absl.testing import absltest
from brax import contact
from brax import kinematics
from brax import test_utils
from brax.io import mjcf
from etils import epath
from jax import numpy as jp
import numpy as np


class SphereTest(absltest.TestCase):

  def test_sphere_plane(self):
    sys = test_utils.load_fixture('ant.xml')
    # all 4 feet are just barely contacting the floor
    q = jp.array([0, 0, 0.556008, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    x, _ = kinematics.forward(sys, q, jp.zeros(sys.qd_size()))
    c = contact.get(sys, x)
    self.assertEqual(c.pos.shape[0], 4)  # 4 contacts

    # front right foot is in +x, +y quadrant and is just resting on the surface
    c = c.take(0)
    np.testing.assert_array_almost_equal(c.pos, jp.array([0.61612, 0.61612, 0]))
    np.testing.assert_array_almost_equal(c.frame[0], jp.array([0, 0, 1]))
    np.testing.assert_array_almost_equal(c.dist, 0)
    np.testing.assert_array_almost_equal(c.friction[0], 1)
    np.testing.assert_array_almost_equal(c.elasticity, 0)
    self.assertEqual(c.link_idx, (-1, 2))

  _SPHERE_SPHERE = """
    <mujoco model="two_spheres">
      <custom>
        <numeric data="0.2" name="elasticity"/>
        <tuple name="elasticity">
          <element objtype="geom" objname="sphere1" prm="0.1"/>
        </tuple>
      </custom>
      <worldbody>
        <body name="body1" pos="0 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="sphere1" pos="0 0 0" size="0.2" type="sphere"/>
        </body>
        <body name="body2" pos="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="sphere2" pos="0 0.3 0" size="0.11" type="sphere"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_sphere_sphere(self):
    sys = mjcf.loads(self._SPHERE_SPHERE)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = contact.get(sys, x).take(0)

    np.testing.assert_array_almost_equal(c.dist, -0.01)
    np.testing.assert_array_almost_equal(c.pos, jp.array([0.0, 0.195, 0.0]))
    np.testing.assert_array_almost_equal(c.frame[0], jp.array([0, 1.0, 0.0]))
    np.testing.assert_array_almost_equal(c.elasticity, jp.array([0.15]))


class ConvexTest(absltest.TestCase):

  _CONVEX_CONVEX = """
    <mujoco model="convex_convex">
      <asset>
        <mesh name="tetrahedron" file="meshes/tetrahedron.stl" scale="0.1 0.1 0.1" />
        <mesh name="dodecahedron" file="meshes/dodecahedron.stl" scale="0.01 0.01 0.01" />
      </asset>
      <worldbody>
        <body name="obj1" pos="0.0 2.0 0.096">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="obj1" pos="0 0 0" size="0.2 0.2 0.2" type="mesh" mesh="tetrahedron"/>
        </body>
        <body name="obj2" pos="0.0 2.0 0.289" euler="0.1 -0.1 45">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="obj2" pos="0 0 0" size="0.1 0.1 0.1" type="mesh" mesh="dodecahedron"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_convex_convex(self):
    """Tests generic convex-convex collision."""
    sys = mjcf.loads(
        self._CONVEX_CONVEX,
        asset_path=epath.resource_path('brax') / 'test_data/',
    )
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = contact.get(sys, x)
    # Only one contact point for an edge contact.
    self.assertLess(c.dist[0], 0)
    self.assertTrue((c.dist[1:] > 0).all())
    np.testing.assert_array_almost_equal(c.frame[0, 0], jp.array([0, 0, 1]))


class GeomPairTest(absltest.TestCase):
  """Tests for colliding geom pairs."""

  def test_no_world_self_collision(self):
    sys = test_utils.load_fixture('world_self_collision.xml')
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = contact.get(sys, x)
    self.assertEqual(c.pos.shape[0], 13)

  def test_collision_with_world_geom(self):
    sys = test_utils.load_fixture('world_fromto.xml')
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = contact.get(sys, x)
    self.assertEqual(c.pos.shape[0], 1)


if __name__ == '__main__':
  absltest.main()
