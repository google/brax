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

# pylint:disable=g-multiple-import
"""Tests for mjcf."""

from absl.testing import absltest
from brax.v2 import test_utils
from brax.v2.base import Box, Plane, Sphere, Mesh
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

    # check contacts
    self.assertEmpty(sys.contacts)

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
    self.assertLen(sys.contacts, 1)
    sphere, plane = sys.contacts[0]
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
    assert_almost_equal(
        plane.transform.pos,
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    )
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
    self.assertLen(sys.contacts, 1)
    sphere, plane = sys.contacts[0]
    self.assertIsInstance(sphere, Sphere)
    self.assertIsInstance(plane, Plane)
    assert_almost_equal(
        sphere.transform.pos, np.array(([[0, 0, -0.35], [0, 0, -0.35]]))
    )
    assert_almost_equal(plane.transform.pos, np.array([[0, 0, 0], [0, 0, 0]]))
    assert_almost_equal(sphere.link_idx, np.array([4, 6]))
    self.assertIsNone(plane.link_idx)

  def test_load_mesh_and_box(self):
    sys = test_utils.load_fixture('ur5e/robot.xml')
    n_meshes = sum(isinstance(g, Mesh) for g in sys.geoms)
    self.assertEqual(n_meshes, 14)
    n_boxes = sum(isinstance(g, Box) for g in sys.geoms)
    self.assertEqual(n_boxes, 1)

  def test_load_urdf(self):
    sys = test_utils.load_fixture('laikago/laikago_toes_zup.urdf')
    n_meshes = sum(isinstance(g, Mesh) for g in sys.geoms)
    self.assertEqual(n_meshes, 26)

  def test_custom(self):
    sys = test_utils.load_fixture('capsule.xml')
    self.assertSequenceEqual([g.elasticity for g in sys.geoms], [0.2, 0.1])


if __name__ == '__main__':
  absltest.main()
