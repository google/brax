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
"""PBD perf tests."""

from absl.testing import absltest
from brax import test_utils
from brax.positional import pipeline
import jax
from jax import numpy as jp


class PerfTest(absltest.TestCase):

  def test_pipeline_ant(self):
    sys = test_utils.load_fixture('ant.xml')

    def init_fn(rng):
      rng1, rng2 = jax.random.split(rng, 2)
      q = jax.random.uniform(rng1, (sys.q_size(),), minval=-0.1, maxval=0.1)
      qd = 0.1 * jax.random.normal(rng2, (sys.qd_size(),))
      return pipeline.init(sys, q, qd)

    def step_fn(state):
      return pipeline.step(sys, state, jp.zeros(sys.act_size()))

    test_utils.benchmark('pbd pipeline ant', init_fn, step_fn)


if __name__ == '__main__':
  absltest.main()
