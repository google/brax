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

"""Tests for image."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import test_utils
from brax.io import image
from brax.spring import pipeline
import jax
import jax.numpy as jp


class ImageTest(parameterized.TestCase):

  @parameterized.parameters([('ant.xml',), ('convex_convex.xml',)])
  def test_render_array(self, xml):
    sys = test_utils.load_fixture(xml)
    state = jax.jit(pipeline.init)(sys, sys.init_q, jp.zeros(sys.qd_size()))
    im = image.render_array(sys, state, 32, 32)
    self.assertEqual(im.shape, (32, 32, 3))

if __name__ == '__main__':
  absltest.main()
