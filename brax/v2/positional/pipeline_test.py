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
"""Tests for positional physics pipeline."""

from absl.testing import absltest
from brax.v2 import test_utils
from brax.v2.base import Motion, Transform
from brax.v2.positional import pipeline
from jax import numpy as jp


class PipelineTest(absltest.TestCase):

  def test_ant_fall(self):
    sys = test_utils.load_fixture('ant.xml')
    x = Transform.zero((sys.num_links(),))
    xd = Motion.zero((sys.num_links(),))
    tau = jp.zeros(sys.qd_size())

    x, xd = pipeline.step(sys, x, xd, tau)

    # TODO: implement

if __name__ == '__main__':
  absltest.main()
