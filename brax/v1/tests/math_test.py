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

"""Tests for math."""

from absl.testing import absltest
from brax.v1 import jumpy as jp
from brax.v1 import math


class RotateTest(absltest.TestCase):
  """Tests math.rotate."""

  def test_rotate(self):
    vec = jp.array([0, 0, 1])
    quat = math.euler_to_quat(jp.array([0, 45, 0]))
    self.assertSequenceAlmostEqual(math.rotate(vec, quat),
                                   [1. / jp.sqrt(2), 0, 1. / jp.sqrt(2)])

  def test_rotate_identity(self):
    vec = jp.ones(3)
    quat = jp.array([1, 0, 0, 0])
    self.assertSequenceAlmostEqual(math.rotate(vec, quat), vec)

  def test_rotate_bad_input_shape(self):
    vec = jp.ones((3, 3))
    quat = jp.array([1, 0, 0, 0])
    with self.assertRaises(AssertionError):
      math.rotate(vec, quat)


if __name__ == '__main__':
  absltest.main()
