# Copyright 2026 The Brax Authors.
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

"""Tests for pmap utilities."""

from absl.testing import absltest
from brax.training import pmap
import jax
import jax.numpy as jnp


class PmapTest(absltest.TestCase):

  def testBcastLocalDevicesUsesRequestedAxisName(self):
    value = pmap.bcast_local_devices(
        jnp.arange(4), local_devices_to_use=1, axis_name='i'
    )

    self.assertEqual(value.sharding.mesh.axis_names, ('i',))
    self.assertEqual(value.sharding.spec, jax.P('i'))


if __name__ == '__main__':
  absltest.main()
