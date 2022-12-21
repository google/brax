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

"""Tests for json."""

import json
from absl.testing import absltest
from brax.v2 import test_utils
from brax.v2.io import json as bjson


class JsonTest(absltest.TestCase):

  def test_dumps(self):
    sys = test_utils.load_fixture('ur5e/robot.xml')
    res = bjson.dumps(sys, [])
    res = json.loads(res)

    self.assertIsInstance(res['geoms'], dict)
    self.assertSequenceEqual(
        sorted(res['geoms'].keys()),
        [
            'forearm_link',
            'shoulder_link',
            'upper_arm_link',
            'world',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
        ],
    )
    self.assertLen(res['geoms']['world'], 2)


if __name__ == '__main__':
  absltest.main()
