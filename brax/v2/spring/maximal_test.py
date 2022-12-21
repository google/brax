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

"""Tests for maximal."""
# pylint:disable=g-multiple-import
from absl.testing import absltest
from brax.v2 import kinematics
from brax.v2 import math
from brax.v2 import test_utils
from brax.v2.base import Transform
from brax.v2.spring import maximal
import jax
from jax import numpy as jp
import numpy as np


class MaximalTest(absltest.TestCase):

  def test_transform(self):
    sys = test_utils.load_fixture('capsule.xml')
    sys = sys.replace(
        link=sys.link.replace(
            inertia=sys.link.inertia.replace(
                transform=sys.link.inertia.transform.replace(
                    pos=jp.array([[0.0, 0.0, -0.1]])
                )
            )
        )
    )

    x, xd = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    x = x.replace(
        pos=jp.array([[1.0, -1.0, 0.3]]),
        rot=jp.array([[0.976442, 0.16639178, -0.13593051, 0.01994515]]),
    )
    xd = xd.replace(
        vel=jp.array([[5.0, 1.0, -1.0]]), ang=jp.array([[1.0, 2.0, -3.0]])
    )

    xi, xdi = maximal.maximal_to_com(sys, x, xd)
    self.assertNotAlmostEqual(jp.abs(xi.pos - x.pos).sum(), 0)
    np.testing.assert_array_almost_equal(xi.rot, x.rot)
    self.assertNotAlmostEqual(jp.abs(xdi.vel - xd.vel).sum(), 0)
    np.testing.assert_array_almost_equal(xdi.ang, xd.ang)

    coord_transform = Transform(pos=xi.pos - x.pos, rot=x.rot)
    xp, xdp = maximal.com_to_maximal(xi, xdi, coord_transform)
    np.testing.assert_array_almost_equal(x.pos, xp.pos)
    np.testing.assert_array_almost_equal(x.rot, xp.rot)
    np.testing.assert_array_almost_equal(xd.vel, xdp.vel)
    np.testing.assert_array_almost_equal(xd.ang, xdp.ang)

  def test_inv_inertia(self):
    sys = test_utils.load_fixture('capsule.xml')
    sys = sys.replace(
        link=sys.link.replace(
            transform=sys.link.transform.replace(
                rot=math.euler_to_quat(jp.array([0.0, 0.0, 45.0])).reshape(
                    1, -1
                )
            )
        )
    )
    x, _ = kinematics.forward(sys, sys.init_q, jp.zeros(sys.qd_size()))
    x = x.replace(
        rot=math.euler_to_quat(jp.array([45.0, 0.0, 0.0])).reshape(1, -1)
    )

    r_inv = jax.vmap(math.quat_inv)(sys.link.inertia.transform.rot)
    ri = jax.vmap(lambda x, y: math.quat_to_3x3(math.quat_mul(x, y)))(
        r_inv, x.rot
    )
    expected = jax.vmap(lambda r, i: math.inv_3x3(r @ i @ r.T))(
        ri, sys.link.inertia.i
    )

    inv_i = maximal.com_inv_inertia(sys, x)
    np.testing.assert_array_almost_equal(expected, inv_i, 1e-6)


if __name__ == '__main__':
  absltest.main()
