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

"""Tests for domain randomization."""

from absl.testing import absltest
from brax.v1.envs import ant
from brax.v1.experimental.tracing import randomizers
from brax.v1.experimental.tracing import wrappers
import brax.v1.jumpy as jp
import jax


class WrappersTest(absltest.TestCase):

  def test_randomize_ant_friction(self):

    test_ant_fn = ant.Ant

    # generate random friction values
    random_friction = jp.array(
        jax.random.uniform(jax.random.PRNGKey(42), (128,)))

    # build traceable config trees
    friction_tree, friction_axes = randomizers.friction_randomizer(
        test_ant_fn(), random_friction)

    random_friction_test_ant = wrappers.DomainRandomizationWrapper(
        test_ant_fn, friction_tree, friction_axes)

    # test reset
    out_state = jax.jit(random_friction_test_ant.reset)(
        jax.random.split(jax.random.PRNGKey(42), 128))
    self.assertEqual(out_state.qp.pos.shape[0], 128)

    # test step
    next_state = jax.jit(random_friction_test_ant.step)(out_state,
                                                        jp.zeros((128, 10)))
    self.assertEqual(next_state.qp.pos.shape[0], 128)


if __name__ == '__main__':
  absltest.main()
