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


"""Tests for brax.inertia."""

from absl.testing import absltest

import brax
from brax.physics.base import vec_to_arr
from brax.physics.inertia import infer_inertia
from brax import jumpy as jp
import jax

from google.protobuf import text_format


class InertiaInferTest(absltest.TestCase):
  def test_inertia_infer(self):
    """Test inferring inertia from geometry."""
    config = text_format.Parse("""
    bodies {
        name: "left_thigh"
        colliders {
            position {
                y: -0.005
                z: -0.17
            }
            rotation {
                x: 178.31532
            }
            capsule {
                radius: 0.06
                length: 0.46014702
            }
        }
    }
    """, brax.Config())

    # Ground truth (use only diagonal elements)
    # Mass 4.751750683148922
    # Inertia
    # [[ 0.21239604,  0.        ,  0.        ],
    #  [ 0.        ,  0.21221958, -0.00599976],
    #  [ 0.        , -0.00599976,  0.00840389]]

    gt_mass    = 4.751750683148922
    gt_inertia = jp.array([0.21239604, 0.21221958, 0.00840389])

    config = infer_inertia(config)
    self.assertAlmostEqual(config.bodies[0].mass, gt_mass, 5)
    self.assertSequenceAlmostEqual(vec_to_arr(config.bodies[0].inertia), gt_inertia, 5)


if __name__ == '__main__':
  absltest.main()
