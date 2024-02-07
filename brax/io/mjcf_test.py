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

# pylint:disable=g-multiple-import
"""Tests for mjcf."""

from absl.testing import absltest
from brax import test_utils
from brax.io import mjcf
import mujoco
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

  def test_custom(self):
    sys = test_utils.load_fixture('capsule.xml')
    assert_almost_equal(sys.elasticity, [0.2, 0.1])

  def test_joint_ref_check(self):
    with self.assertRaisesRegex(NotImplementedError, '`ref` attribute'):
      sys = test_utils.load_fixture('nonzero_joint_ref.xml')
      mjcf.validate_model(sys.mj_model)

  def test_world_body_transform(self):
    sys = test_utils.load_fixture('world_body_transform.xml')
    # world body is in the right position/orientation
    r = 0.70710677
    assert_almost_equal(sys.geom_pos[1], np.array([1.0, 0.0, 0.0]))
    assert_almost_equal(sys.geom_quat[1], np.array([r, 0.0, r, 0.0]), 5)
    # child body is transformed wrt world body
    assert_almost_equal(
        sys.init_q, np.array([1.245, 0.0, 0.0, 0.5, 0.5, 0.5, -0.5])
    )

  def test_load_fluid_box(self):
    sys = test_utils.load_fixture('fluid_box.xml')
    assert_almost_equal(sys.density, 1.2)
    assert_almost_equal(sys.viscosity, 0.15)

  def test_load_fluid_ellipsoid(self):
    with self.assertRaisesRegex(
        NotImplementedError, 'Ellipsoid fluid model not implemented'
    ):
      sys = test_utils.load_fixture('fluid_ellipsoid.xml')
      mjcf.validate_model(sys.mj_model)

  def test_load_wind(self):
    with self.assertRaisesRegex(
        NotImplementedError, 'option.wind is not implemented'
    ):
      sys = test_utils.load_fixture('fluid_wind.xml')
      mjcf.validate_model(sys.mj_model)

  def test_world_fromto(self):
    """Tests that a world element with fromto does not break mjcf.load."""
    sys = test_utils.load_fixture('world_fromto.xml')
    mjcf.validate_model(sys.mj_model)

  def test_loads_different_transmission(self):
    """Tests that the brax model loads with different transmission types."""
    mj = test_utils.load_fixture_mujoco('ant.xml')
    mj.actuator_trntype[0] = mujoco.mjtTrn.mjTRN_SITE
    mjcf.load_model(mj)  # loads without raising an error

    with self.assertRaisesRegex(NotImplementedError, 'transmission types'):
      mjcf.validate_model(mj)  # raises an error

if __name__ == '__main__':
  absltest.main()
