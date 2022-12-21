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
"""Tests for kinematics."""

from absl.testing import absltest
from absl.testing import parameterized
from brax.v2 import base
from brax.v2 import kinematics
from brax.v2 import scan
from brax.v2 import test_utils
import jax
import jax.numpy as jp
import numpy as np


class KinematicsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ant.xml',),
      ('triple_pendulum.xml',),
      ('humanoid.xml',),
      ('halfcheetah.xml',),
  )
  def test_forward_q(self, xml_file):
    """Test dynamics forward q."""
    sys = test_utils.load_fixture(xml_file)
    for mj_prev, mj_next in test_utils.sample_mujoco_states(xml_file):
      x, _ = jax.jit(kinematics.forward)(sys, mj_prev.qpos, mj_prev.qvel)
      np.testing.assert_almost_equal(x.pos, mj_next.xpos[1:], 3)
      np.testing.assert_almost_equal(x.rot, mj_next.xquat[1:], 3)

  def test_init_q(self):
    sys = test_utils.load_fixture('ant.xml')
    np.testing.assert_almost_equal(
        sys.init_q,
        np.array([0, 0, 0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]),
        7,
    )

  @parameterized.parameters(('ant.xml',), ('humanoid.xml',))
  def test_inverse(self, xml_file):
    sys = test_utils.load_fixture(xml_file)
    # # test at random init
    rand_q = np.random.rand(sys.init_q.shape[0])
    # normalize quaternion part of init_q
    rand_q[3:7] = rand_q[3:7] / np.linalg.norm(rand_q[3:7])
    rand_q = jp.array(rand_q)
    rand_qd = jp.array(np.random.rand(sys.qd_size())) * 0.1

    x, xd = kinematics.forward(sys, rand_q, rand_qd)
    q, qd = kinematics.inverse(sys, x, xd)
    np.testing.assert_array_almost_equal(q, rand_q, decimal=5)
    np.testing.assert_array_almost_equal(qd, rand_qd)

  def test_joint_axes(self):
    sys = test_utils.load_fixture('triple_pendulum.xml')

    def _collect_frame(typ, motion):
      shape = base.QD_WIDTHS[typ]
      motion = jax.tree_map(lambda y: y.reshape((-1, shape, 3)), motion)
      return jax.vmap(kinematics.link_to_joint_motion)(motion)[0]

    # default setup
    joint_motion = scan.link_types(
        sys, _collect_frame, 'd', 'l', sys.dof.motion
    )
    np.testing.assert_array_almost_equal(
        joint_motion.ang[:, 0], sys.dof.motion.ang
    )

    # reorient joint axes
    sys = sys.replace(
        dof=base.DoF(
            motion=base.Motion(
                ang=np.array(
                    [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                vel=np.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
            ),
            armature=np.array([0.0, 0.0, 0.0, 0.0]),
            invweight=np.array([0.0, 0.0, 0.0, 0.0]),
            stiffness=np.array([0.0, 0.0, 0.0, 0.0]),
            damping=np.array([0.0, 0.0, 0.0, 0.0]),
            limit=None,
        )
    )

    joint_motion = scan.link_types(
        sys, _collect_frame, 'd', 'l', sys.dof.motion
    )
    # joint x-axes should correspond to dof angular degrees of freedom
    np.testing.assert_array_almost_equal(
        joint_motion.ang[:, 0], sys.dof.motion.ang
    )

    # test system with multiple types of joint
    sys = sys.replace(link_types='211')
    # reorient joint axes
    sys = sys.replace(
        dof=base.DoF(
            motion=base.Motion(
                ang=np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]),
                vel=np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]),
            ),
            armature=np.array([0.0, 0.0, 0.0, 0.0]),
            invweight=np.array([0.0, 0.0, 0.0, 0.0]),
            stiffness=np.array([0.0, 0.0, 0.0, 0.0]),
            damping=np.array([0.0, 0.0, 0.0, 0.0]),
            limit=None,
        )
    )

    joint_motion = scan.link_types(
        sys, _collect_frame, 'd', 'l', sys.dof.motion
    )
    # check universal joint axes
    np.testing.assert_array_almost_equal(
        joint_motion.ang[0, :2], sys.dof.motion.ang[0:2]
    )

    # check revolute axes
    np.testing.assert_array_almost_equal(
        joint_motion.ang[1:, 0], sys.dof.motion.ang[2:]
    )


if __name__ == '__main__':
  absltest.main()
