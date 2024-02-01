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

"""Tests for joints."""

from absl.testing import absltest
from brax import test_utils
from brax.positional import joints
from brax.positional import pipeline
from jax import numpy as jp
import numpy as np


class JointsTest(absltest.TestCase):

  def test_sphericalize_humanoid(self):
    sys = test_utils.load_fixture('humanoid.xml')
    qd = jp.zeros(sys.qd_size())
    state = pipeline.init(sys, sys.init_q, qd)

    sys_data = joints._sphericalize(sys, state.x)

    # check lower limits
    np.testing.assert_array_almost_equal(
        sys_data[0][0],
        jp.array([
            [-jp.inf, -jp.inf, -jp.inf],
            [-0.7853982, -1.3089969, 0.0],
            [-0.61086524, 0.0, 0.0],
            [-0.43633232, -1.0471976, -1.9198622],
            [-2.7925267, 0.0, 0.0],
            [-0.43633232, -1.0471976, -1.9198622],
            [-2.7925267, 0.0, 0.0],
            [-1.4835298, -1.4835298, 0.0],
            [-1.5707964, 0.0, 0.0],
            [-1.0471976, -1.0471976, 0.0],
            [-1.5707964, 0.0, 0.0],
        ]),
    )

    # check upper limits
    np.testing.assert_array_almost_equal(
        sys_data[0][1],
        jp.array([
            [jp.inf, jp.inf, jp.inf],
            [0.7853982, 0.5235988, 0.0],
            [0.61086524, 0.0, 0.0],
            [0.08726646, 0.61086524, 0.34906584],
            [-0.03490658, 0.0, 0.0],
            [0.08726646, 0.61086524, 0.34906584],
            [-0.03490658, 0.0, 0.0],
            [1.0471976, 1.0471976, 0.0],
            [0.87266463, 0.0, 0.0],
            [1.4835298, 1.4835298, 0.0],
            [0.87266463, 0.0, 0.0],
        ]),
    )

    # check that a 1-dof link motion is padded correctly
    # check that base dof is the same
    np.testing.assert_array_almost_equal(
        sys_data[1].ang[2][0], sys.dof.motion.ang[8]
    )

    # check padding
    np.testing.assert_array_almost_equal(
        sys_data[1].ang[2],
        jp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    # check that a 2-dof link motion is padded correctly
    # check base dof is the same
    np.testing.assert_array_almost_equal(
        sys_data[1].ang[1][0:2], sys.dof.motion.ang[6:8]
    )

    # check padding
    np.testing.assert_array_almost_equal(
        sys_data[1].ang[1],
        jp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    # check link parities
    np.testing.assert_array_almost_equal(
        sys_data[3], [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    # spot check parity for first 3-dof link (corresponds to fourth idx above)
    three_dof_link = sys.dof.motion.ang[sys.qd_idx('3')[0:3]]
    np.testing.assert_equal(
        np.dot(
            np.cross(three_dof_link[0], three_dof_link[1]), three_dof_link[2]
        ),
        -1.0,
    )


if __name__ == '__main__':
  absltest.main()
