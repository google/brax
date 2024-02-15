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
"""Tests for scan functions."""

from absl.testing import absltest
from brax import scan
from brax import test_utils
import numpy as np


class ScanTest(absltest.TestCase):

  def test_tree(self):
    """Test scanning down 3 tree levels of ant."""
    sys = test_utils.load_fixture('ant.xml')

    qs, qds = [], []

    def f(parent, link, q, qd):
      if parent is not None:
        link += parent
      qs.append(q)
      qds.append(qd)
      return link

    x = scan.tree(
        sys,
        f,
        'lqd',
        np.arange(sys.num_links()),
        np.arange(sys.q_size()),
        np.arange(sys.qd_size()),
    )

    # our scan carry is the sum of index + ancestry indices for each link
    np.testing.assert_array_equal(
        x,
        np.array([
            0,  # 0
            1,  # 1 + 0
            3,  # 2 + 1 + 0
            3,  # 3 + 0
            7,  # 4 + 3 + 0
            5,  # 5 + 0
            11,  # 6 + 5 + 0
            7,  # 7 + 0
            15,  # 8 + 7 + 0
        ]),
    )

    self.assertLen(qs, 3)  # three levels
    np.testing.assert_array_equal(qs[0], np.array([0, 1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(qs[1], np.array([7, 9, 11, 13]))
    np.testing.assert_array_equal(qs[2], np.array([8, 10, 12, 14]))

    self.assertLen(qds, 3)  # three levels
    np.testing.assert_array_equal(qds[0], np.array([0, 1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(qds[1], np.array([6, 8, 10, 12]))
    np.testing.assert_array_equal(qds[2], np.array([7, 9, 11, 13]))

  def test_tree_reverse(self):
    """Test scanning up 3 tree levels of ant."""
    sys = test_utils.load_fixture('ant.xml')

    qs, qds = [], []

    def f(parent, link, q, qd):
      if parent is not None:
        link += parent
      qs.append(q)
      qds.append(qd)
      return link

    x = scan.tree(
        sys,
        f,
        'lqd',
        np.arange(sys.num_links()),
        np.arange(sys.q_size()),
        np.arange(sys.qd_size()),
        reverse=True,
    )

    # our scan carry is the sum of index + ancestry indices for each link
    np.testing.assert_array_equal(
        x,
        np.array([
            36,  # 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8
            3,  # 1 + 2
            2,  # 2
            7,  # 3 + 4
            4,  # 4
            11,  # 5 + 6
            6,  # 6
            15,  # 7 + 8
            8,  # 8
        ]),
    )

    self.assertLen(qs, 3)  # three levels
    np.testing.assert_array_equal(qs[0], np.array([8, 10, 12, 14]))
    np.testing.assert_array_equal(qs[1], np.array([7, 9, 11, 13]))
    np.testing.assert_array_equal(qs[2], np.array([0, 1, 2, 3, 4, 5, 6]))

    self.assertLen(qds, 3)  # three levels
    np.testing.assert_array_equal(qds[0], np.array([7, 9, 11, 13]))
    np.testing.assert_array_equal(qds[1], np.array([6, 8, 10, 12]))
    np.testing.assert_array_equal(qds[2], np.array([0, 1, 2, 3, 4, 5]))

  def test_scan_link_types(self):
    """Test scanning 2 links types of ant."""
    sys = test_utils.load_fixture('ant.xml')

    typs, qs, qds = [], [], []

    def f(typ, link, q, qd):
      typs.append(typ)
      qs.append(q)
      qds.append(qd)
      return link

    x = scan.link_types(
        sys,
        f,
        'lqd',
        'l',
        np.arange(sys.num_links()),
        np.arange(sys.q_size()),
        np.arange(sys.qd_size()),
    )

    self.assertSequenceEqual(typs, ['f', '1'])
    np.testing.assert_array_equal(x, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    self.assertLen(qs, 2)
    np.testing.assert_array_equal(qs[0], np.arange(0, 7))
    np.testing.assert_array_equal(qs[1], np.arange(7, 15))

    self.assertLen(qds, 2)
    np.testing.assert_array_equal(qds[0], np.arange(0, 6))
    np.testing.assert_array_equal(qds[1], np.arange(6, 14))


if __name__ == '__main__':
  absltest.main()
