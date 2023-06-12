# Copyright 2023 The Brax Authors.
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
from brax import geometry
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
    c = geometry.contact(sys, x)
    self.assertEqual(c.pos.shape[0], 4)  # 4 contacts

    # front right foot is in +x, +y quadrant and is just resting on the surface
    c = c.take(0)
    np.testing.assert_array_almost_equal(c.pos, jp.array([0.61612, 0.61612, 0]))
    np.testing.assert_array_almost_equal(c.normal, jp.array([0, 0, 1]))
    np.testing.assert_array_almost_equal(c.penetration, 0)
    np.testing.assert_array_almost_equal(c.friction, 1)
    np.testing.assert_array_almost_equal(c.elasticity, 0)
    self.assertEqual(c.link_idx, (2, -1))

  _SPHERE_SPHERE = """
    <mujoco model="two_spheres">
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
    c = geometry.contact(sys, x).take(0)

    np.testing.assert_array_almost_equal(c.penetration, 0.01)
    np.testing.assert_array_almost_equal(c.pos, jp.array([0.0, 0.195, 0.0]))
    np.testing.assert_array_almost_equal(c.normal, jp.array([0, -1.0, 0.0]))

  _SPHERE_CAP = """
    <mujoco model="sphere_cap">
      <worldbody>
        <body name="body1" pos="0 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="sphere1" pos="0 0.3 0" size="0.05" type="sphere"/>
        </body>
        <body name="body2" pos="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="cap2" fromto="0.0 -0.5 0.14 0.0 0.5 0.14" pos="0 0 0" size="0.1" type="capsule"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_sphere_capsule(self):
    sys = mjcf.loads(self._SPHERE_CAP)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x).take(0)

    np.testing.assert_array_almost_equal(c.penetration, 0.01)
    np.testing.assert_array_almost_equal(c.pos, jp.array([0.0, 0.3, 0.045]))
    np.testing.assert_array_almost_equal(c.normal, jp.array([0, 0.0, -1.0]))

  _SPHERE_CYLINDER = """
    <mujoco model="sphere_cylinder">
      <worldbody>
        <body name="body1" pos="0 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="sphere1" pos="0 0.04 0" size="0.05" type="sphere"/>
        </body>
        <body name="body2" pos="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="cyl" pos="0 0 0" size="0.1 0.001" type="cylinder"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_sphere_cylinder(self):
    sys = mjcf.loads(self._SPHERE_CYLINDER)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x).take(0)

    np.testing.assert_array_almost_equal(c.penetration, 0.01)

  _SPHERE_CONVEX = """
    <mujoco model="sphere_convex">
      <worldbody>
        <body name="body1" pos="0.52 0 0.52">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="sphere1" pos="0 0 0" size="0.05" type="sphere"/>
        </body>
        <body name="body2" pos="0 0 0" euler="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_sphere_convex(self):
    sys = mjcf.loads(self._SPHERE_CONVEX)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    # contact occurs on the edge of the box
    self.assertGreater(c.penetration, 0.02)
    normal = jp.array([[1 / jp.sqrt(2), 0.0, 1 / jp.sqrt(2)]])
    np.testing.assert_array_almost_equal(c.normal, normal, 4)
    pos = jp.array([0.52, 0, 0.52]) + normal * (c.penetration / 2.0 - 0.05)
    np.testing.assert_array_almost_equal(c.pos, pos)

  _SPHERE_MESH = """
    <mujoco model="sphere_mesh">
      <custom>
        <tuple name="convex">
          <element objtype="geom" objname="box" prm="0"/>
        </tuple>
      </custom>
      <worldbody>
        <body name="body1" pos="0 0 0.54">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="sphere1" pos="0 0 0" size="0.05" type="sphere"/>
        </body>
        <body name="body2" pos="0 0 0" euler="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_sphere_mesh(self):
    sys = mjcf.loads(self._SPHERE_MESH)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    np.testing.assert_array_almost_equal((c.penetration > 0).sum(), 2)
    c = c.take(c.penetration.argsort()[-2:])
    np.testing.assert_array_almost_equal(c.penetration, jp.repeat(0.01, 2))
    np.testing.assert_array_almost_equal(
        c.pos, jp.array([[0.0, 0.0, 0.495]] * 2)
    )
    np.testing.assert_array_almost_equal(
        c.normal, jp.array([[0, 0.0, 1.0]] * 2), 5
    )


class CapsuleTest(absltest.TestCase):
  """Tests the capsule contact functions."""

  _CAP_CAP = """
    <mujoco model="two_capsules">
      <worldbody>
        <body name="body1" pos="0 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="cap1" fromto="0.62235904  0.58846647 0.651046 1.5330081 0.33564585 0.977849" pos="0 0 0" size="0.05" type="capsule"/>
        </body>
        <body name="body2" pos="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="cap2" fromto="0.5505271 0.60345304 0.476661 1.3900293 0.30709633 0.932082" pos="0 0 0" size="0.05" type="capsule"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_capsule_capsule(self):
    """Tests that two capsules are colliding."""
    sys = mjcf.loads(self._CAP_CAP)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x).take(0)
    self.assertGreaterEqual(c.penetration, 0)

  _PARALLEL_CAP = """
    <mujoco model="parallel_capsules">
      <worldbody>
        <body name="body1" pos="0 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="cap1" fromto="-0.5 0.1 0.25 0.5 0.1 0.25" pos="0 0 0" size="0.1" type="capsule"/>
        </body>
        <body name="body2" pos="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="cap2" fromto="-0.5 0.1 0.1 0.5 0.1 0.1" pos="0 0 0" size="0.1" type="capsule"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_parallel_capsules(self):
    """Tests that two parallel capsules are colliding at the midpoint."""
    sys = mjcf.loads(self._PARALLEL_CAP)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x).take(0)

    np.testing.assert_array_almost_equal(c.penetration, 0.05)
    np.testing.assert_array_almost_equal(
        c.pos, jp.array([0.0, 0.1, (0.15 + 0.2) / 2.0])
    )
    np.testing.assert_array_almost_equal(c.normal, jp.array([0, 0.0, 1.0]), 5)

  _CAP_BOX = """
    <mujoco model="cap_box">
      <worldbody>
        <body name="body1" pos="0 0 0.54">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="cap1" fromto="-0.6 0 0 0.6 0 0" pos="0 0 0" size="0.05" type="capsule"/>
        </body>
        <body name="body2" pos="0 0 0" euler="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_capsule_hull(self):
    """Tests a convex-capsule collision for a face contact."""
    sys = mjcf.loads(self._CAP_BOX)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    self.assertEqual(c.pos.shape[0], 2)
    np.testing.assert_array_almost_equal(c.penetration, jp.repeat(0.01, 2))
    np.testing.assert_array_almost_equal(c.pos[:, 2], jp.repeat(0.495, 2))
    np.testing.assert_array_almost_equal(
        c.normal, jp.array([[0, 0.0, 1.0]] * 2), 4
    )

  _CAP_EDGE_BOX = """
    <mujoco model="cap_edge_box">
      <worldbody>
        <body name="body1" pos="0.5 0 0.55" euler="0 30 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="cap1" fromto="-0.6 0 0 0.6 0 0" pos="0 0 0" size="0.05" type="capsule"/>
        </body>
        <body name="body2" pos="0 0 0" euler="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_capsule_edge_hull(self):
    """Tests a convex-capsule collision for an edge contact."""
    sys = mjcf.loads(self._CAP_EDGE_BOX)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    self.assertEqual(c.pos.shape[0], 2)
    np.testing.assert_array_almost_equal(
        c.penetration, jp.array([0.006699, -0.3])
    )

    # The contact point is in the correct box quadrant.
    pos_r = round(c.pos[0], 5)
    self.assertTrue(((pos_r <= 0.5) & (pos_r >= 0)).all())
    # Normal points in +z and +x.
    self.assertGreater(c.normal[0, 0], 0)
    np.testing.assert_array_almost_equal(c.normal[0, 1], 0, 4)
    self.assertGreater(c.normal[0, 2], 0)

  _CAP_MESH = """
    <mujoco model="cap_mesh">
      <custom>
        <tuple name="convex">
          <element objtype="geom" objname="box" prm="0"/>
        </tuple>
      </custom>
      <worldbody>
        <body name="body1" pos="0 0 0.54">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="cap1" fromto="-0.5 0 0 0.5 0 0" pos="0 0 0" size="0.05" type="capsule"/>
        </body>
        <body name="body2" pos="0 0 0" euler="0 0 0">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_capsule_mesh(self):
    sys = mjcf.loads(self._CAP_MESH)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    self.assertEqual(c.pos.shape[0], 12)  # one contact per triangle
    np.testing.assert_array_almost_equal((c.penetration > 0).sum(), 4)
    c = c.take(c.penetration.argsort()[-4:])
    np.testing.assert_array_almost_equal(c.penetration, jp.repeat(0.01, 4))
    np.testing.assert_array_almost_equal(c.pos[:, 2], jp.repeat(0.495, 4))
    np.testing.assert_array_almost_equal(
        c.normal, jp.array([[0, 0.0, 1.0]] * 4), 4
    )


class ConvexTest(absltest.TestCase):
  """Tests the convex contact functions."""

  _BOX_PLANE = """
    <mujoco model="box_plane">
      <worldbody>
        <geom name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
        <body name="body1" pos="0 0 0.7" euler="45 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_box_plane(self):
    """Tests a plane contact for a box convex collision."""
    sys = mjcf.loads(self._BOX_PLANE)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    self.assertEqual(c.pos.shape[0], 4)  # 4 potential contacts with a plane
    self.assertEqual((c.penetration > 0.0).sum(), 2)  # 2 corners penetrating
    np.testing.assert_array_almost_equal(
        c.pos[c.penetration > 0.0].sort(axis=0),
        jp.array([[-0.5, 0.0, -0.00710678], [0.5, 0.0, -0.00710678]]),
    )
    self.assertTrue((c.normal == jp.array([0.0, 0.0, 1.0])).all())

  _BOX_BOX = """
    <mujoco model="box_box">
      <worldbody>
        <body name="body1" pos="0.0 1.0 0.2">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="box1" pos="0 0 0" size="0.2 0.2 0.2" type="box"/>
        </body>
        <body name="body2" pos="0.1 1.0 0.499" euler="0.1 -0.1 45">
          <joint axis="1 0 0" name="free2" pos="0 0 0" type="free"/>
          <geom name="box2" pos="0 0 0" size="0.1 0.1 0.1" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_box_box(self):
    """Tests a face contact for a box-box convex collision."""
    sys = mjcf.loads(self._BOX_BOX)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)
    self.assertEqual(c.pos.shape[0], 4)
    self.assertTrue((c.penetration > 0).all())
    np.testing.assert_array_almost_equal(c.pos[:, 2], jp.array([0.39] * 4), 2)
    np.testing.assert_array_almost_equal(
        c.normal, jp.array([[0.0, 0.0, -1.0]] * 4)
    )

  _BOX_BOX_EDGE = """
    <mujoco model="box_box">
      <worldbody>
        <body name="box15" pos="-1.0 -1.0 0.2">
          <joint axis="1 0 0" name="free15" pos="0 0 0" type="free"/>
          <geom name="box15" pos="0 0 0" size="0.2 0.2 0.2" type="box"/>
        </body>
        <body name="box16" pos="-1.0 -1.2 0.55" euler="0 45 30">
          <joint axis="1 0 0" name="free16" pos="0 0 0" type="free"/>
          <geom name="box16" pos="0 0 0" size="0.1 0.1 0.1" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_box_box_edge(self):
    """Tests an edge contact for a box-box convex collision."""
    sys = mjcf.loads(self._BOX_BOX_EDGE)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)
    # Only one contact point.
    self.assertGreater(c.penetration[0], 0)
    self.assertTrue((c.penetration[1:] < 0).all())
    # The normal is pointing in the edge-edge axis direction.
    np.testing.assert_array_almost_equal(c.normal[0, 0], 0)
    self.assertGreater(c.normal[0, 1], 0)
    self.assertLess(c.normal[0, 2], 0)

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

  def test_hull_hull(self):
    """Tests generic convex-convex collision."""
    sys = mjcf.loads(
        self._CONVEX_CONVEX,
        asset_path=epath.resource_path('brax') / 'test_data/',
    )
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)
    # Only one contact point for an edge contact.
    self.assertGreater(c.penetration[0], 0)
    self.assertTrue((c.penetration[1:] < 0).all())
    np.testing.assert_array_almost_equal(c.normal[0], jp.array([0, 0, -1]))


class MeshTest(absltest.TestCase):
  """Tests the mesh contact functions."""

  _MESH_PLANE = """
    <mujoco model="mesh_plane">
      <custom>
        <tuple name="convex">
          <element objtype="geom" objname="box" prm="0"/>
        </tuple>
      </custom>
      <worldbody>
        <geom name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
        <body name="body1" pos="0 0 0.7" euler="45 0 0">
          <joint axis="1 0 0" name="free1" pos="0 0 0" type="free"/>
          <geom name="box" pos="0 0 0" size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_mesh_plane(self):
    sys = mjcf.loads(self._MESH_PLANE)
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)

    self.assertEqual(c.pos.shape[0], 8)  # 8 potential contacts for each corner
    self.assertEqual((c.penetration > 0.0).sum(), 2)  # 2 corners penetrating
    np.testing.assert_array_almost_equal(
        c.pos[c.penetration > 0.0].sort(axis=0),
        jp.array([[-0.5, 0.0, -0.00710678], [0.5, 0.0, -0.00710678]]),
    )
    self.assertTrue((c.normal == jp.array([0.0, 0.0, 1.0])).all())


class GeomPairTest(absltest.TestCase):
  """Tests for colliding geom pairs."""

  def test_no_world_self_collision(self):
    sys = test_utils.load_fixture('world_self_collision.xml')
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)
    self.assertEqual(c.pos.shape[0], 13)

  def test_collision_with_world_geom(self):
    sys = test_utils.load_fixture('world_fromto.xml')
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    c = geometry.contact(sys, x)
    self.assertEqual(c.pos.shape[0], 1)

if __name__ == '__main__':
  absltest.main()
