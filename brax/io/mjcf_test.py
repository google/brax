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

# pylint:disable=g-multiple-import
"""Tests for mjcf."""

from absl.testing import absltest
from brax import test_utils
from brax.base import Box, Convex, Plane, Sphere, Mesh
import numpy as np

assert_almost_equal = np.testing.assert_array_almost_equal


class MjcfTest(absltest.TestCase):

  def test_load_pendulum(self):
    sys = test_utils.load_fixture('triple_pendulum.xml')

    assert_almost_equal(sys.gravity, np.array([0, 0, -9.81]))

    # check links
    self.assertSequenceEqual(sys.link_names, ['body1', 'body2', 'body3'])
    assert_almost_equal(
        sys.link.transform.pos, np.array([[0, 0, 0], [0, 0.5, 0], [0, 0.5, 0]])
    )
    assert_almost_equal(
        sys.link.transform.rot,
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
    )
    assert_almost_equal(
        sys.link.inertia.i, np.tile(np.eye(3), (3, 1, 1)) * 0.009
    )
    assert_almost_equal(
        sys.link.inertia.transform.pos,
        np.array([[0, 0.5, 0], [0, 0.5, 0], [0, 0.5, 0]]),
    )
    assert_almost_equal(sys.link.inertia.mass, np.array([1, 1, 1]))

    # check static fields: link types, parents
    self.assertEqual(sys.link_types, '111')
    self.assertEqual(sys.link_parents, (-1, 0, 1))

  def test_load_ant(self):
    sys = test_utils.load_fixture('ant.xml')

    # check links
    self.assertSequenceEqual(
        sys.link_names,
        ['torso', 'aux_1', '', 'aux_2', '', 'aux_3', '', 'aux_4', ''],
    )

    # check static fields: link types, parents
    self.assertEqual(sys.link_types, 'f11111111')
    self.assertEqual(sys.link_parents, (-1, 0, 1, 0, 3, 0, 5, 0, 7))

    # check contacts
    plane, _, _, sphere = sys.geoms
    self.assertIsInstance(sphere, Sphere)
    self.assertIsInstance(plane, Plane)
    assert_almost_equal(
        sphere.transform.pos,
        np.array(([
            [0.4, 0.4, 0.0],
            [-0.4, 0.4, 0.0],
            [-0.4, -0.4, 0.0],
            [0.4, -0.4, 0.0],
        ])),
    )
    assert_almost_equal(plane.transform.pos, np.array([[0, 0, 0]]))
    assert_almost_equal(sphere.link_idx, np.array([2, 4, 6, 8]))
    self.assertIsNone(plane.link_idx)

  def test_load_humanoid(self):
    sys = test_utils.load_fixture('humanoid.xml')

    # check links
    self.assertSequenceEqual(
        sys.link_names,
        [
            'torso',
            'lwaist',
            'pelvis',
            'right_thigh',
            'right_shin',
            'left_thigh',
            'left_shin',
            'right_upper_arm',
            'right_lower_arm',
            'left_upper_arm',
            'left_lower_arm',
        ],
    )
    self.assertEqual(sys.link_types, 'f2131312121')

    # check contacts
    plane, _, _, sphere = sys.geoms
    self.assertIsInstance(sphere, Sphere)
    self.assertIsInstance(plane, Plane)
    assert_almost_equal(
        sphere.transform.pos, np.array(([[0, 0, -0.35], [0, 0, -0.35]]))
    )
    assert_almost_equal(plane.transform.pos, np.array([[0, 0, 0]]))
    assert_almost_equal(sphere.link_idx, np.array([4, 6]))
    self.assertIsNone(plane.link_idx)

  def test_load_mesh_and_box(self):
    sys = test_utils.load_fixture('convex_convex.xml')
    n_box, n_convex, n_mesh = 0, 0, 0
    for g in sys.geoms:
      if isinstance(g, Box):
        n_box += 1
      elif isinstance(g, Convex):
        n_convex += 1
      elif isinstance(g, Mesh):
        n_mesh += 1
    self.assertEqual(n_box, 1)
    self.assertEqual(n_convex, 4)
    self.assertEqual(n_mesh, 3)

  def test_load_urdf(self):
    sys = test_utils.load_fixture('laikago/laikago_toes_zup.urdf')
    n_meshes = sum(
        g.friction.shape[0] if isinstance(g, Mesh) else 0 for g in sys.geoms
    )
    n_convex = sum(
        g.friction.shape[0] if isinstance(g, Convex) else 0 for g in sys.geoms
    )
    self.assertEqual(n_meshes, 26)
    self.assertEqual(n_convex, 26)

  def test_custom(self):
    sys = test_utils.load_fixture('capsule.xml')
    self.assertSequenceEqual([g.elasticity for g in sys.geoms], [0.2, 0.1])

  def test_joint_ref_check(self):
    with self.assertRaisesRegex(NotImplementedError, '`ref` attribute'):
      test_utils.load_fixture('nonzero_joint_ref.xml')

  def test_rgba(self):
    sys = test_utils.load_fixture('colour_objects.xml')
    # non default colour in plane
    assert_almost_equal(
        sys.geoms[0].rgba.squeeze(), np.array([1, 0, 0.8, 1]), 6
    )
    # other geometris have default colour
    for g in sys.geoms[1:]:
      assert_almost_equal(g.rgba.squeeze(), np.array([0.8, 0.6, 0.4, 1.0]), 6)

  def test_world_body_transform(self):
    sys = test_utils.load_fixture('world_body_transform.xml')
    # world body is in the right position/orientation
    r = 0.70710677
    assert_almost_equal(sys.geoms[1].transform.pos, np.array([[1.0, 0.0, 0.0]]))
    assert_almost_equal(
        sys.geoms[1].transform.rot, np.array([[r, 0.0, r, 0.0]]), 5
    )
    # child body is transformed wrt world body
    assert_almost_equal(
        sys.init_q, np.array([1.245, 0.0, 0.0, 0.5, 0.5, 0.5, -0.5])
    )

  def test_load_flat_cylinder(self):
    sys = test_utils.load_fixture('flat_cylinder.xml')
    self.assertEqual(sys.geoms[1].radius, 0.25)
    self.assertEqual(sys.geoms[1].length, 0.002)

  def test_load_fat_cylinder(self):
    with self.assertRaisesRegex(
        NotImplementedError, 'Cylinders of half-length'
    ):
      test_utils.load_fixture('fat_cylinder.xml')

  def test_load_fluid_box(self):
    sys = test_utils.load_fixture('fluid_box.xml')
    assert_almost_equal(sys.density, 1.2)
    assert_almost_equal(sys.viscosity, 0.15)

  def test_load_fluid_ellipsoid(self):
    with self.assertRaisesRegex(
        NotImplementedError, 'Ellipsoid fluid model not implemented'
    ):
      test_utils.load_fixture('fluid_ellipsoid.xml')

  def test_load_wind(self):
    with self.assertRaisesRegex(
        NotImplementedError, 'option.wind is not implemented'
    ):
      test_utils.load_fixture('fluid_wind.xml')

  def test_world_fromto(self):
    """Tests that a world element with fromto does not break mjcf.load."""
    test_utils.load_fixture('world_fromto.xml')


if __name__ == '__main__':
  absltest.main()
