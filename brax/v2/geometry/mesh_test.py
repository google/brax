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

"""Tests for mesh.py."""

from absl.testing import absltest
from brax.v2.base import Box
from brax.v2.geometry import mesh
import numpy as np


class BoxMeshTest(absltest.TestCase):

  def test_box(self):
    b = Box(
        halfsize=np.repeat(0.5, 6).reshape(2, 3),
        link_idx=None,
        transform=None,
        friction=0.42,
        elasticity=1,
    )
    m = mesh.box(b)
    self.assertSequenceEqual(m.vert.shape, (2, 8, 3))  # eight box corners
    self.assertEqual(np.unique(np.abs(m.vert)), 0.5)
    self.assertSequenceEqual(m.face.shape, (2, 12, 3))  # two triangles per face
    self.assertEqual(m.friction, 0.42)


if __name__ == '__main__':
  absltest.main()
