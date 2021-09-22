# Copyright 2021 The Brax Authors.
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

"""Composer tests."""

import collections
import functools
from absl.testing import absltest
from absl.testing import parameterized
from brax.experimental.composer import composer
from brax.experimental.composer import observers
from jax import numpy as jnp


class ComposerTest(parameterized.TestCase):
  """Tests for Composer module."""

  @parameterized.parameters('ant_push', 'ant_chase')
  def testEnvCreation(self, env_name):
    composer.create(env_name=env_name)

  def testObservationConcatSplit(self):
    leading_dims = (5, 4)
    obs_shapes = ((3, 4), (12,), (2, 2))
    obs_sizes = [
        functools.reduce(lambda x, y: x * y, shape) for shape in obs_shapes
    ]
    obs_vec_size = sum(obs_sizes)
    obs_dict = collections.OrderedDict([(i, jnp.zeros(leading_dims + shape))
                                        for i, shape in enumerate(obs_shapes)])
    # get observer_shapes
    obs_shapes_from_data = observers.get_obs_dict_shape(
        obs_dict, batch_shape=leading_dims)
    for s1, s2 in zip(obs_shapes, obs_shapes_from_data.values()):
      s2 = s2['shape']
      self.assertEqual(s1, s2, f'{s1} != {s2}')

    # concat
    obs = composer.concat_obs(obs_dict, obs_shapes_from_data)
    self.assertEqual(obs.shape, leading_dims + (obs_vec_size,),
                     f'{obs.shape} != {leading_dims} + ({obs_vec_size},)')

    # split again
    obs_dict_2 = composer.split_obs(obs, obs_shapes_from_data)
    for s1, s2 in zip(obs_dict_2.values(), obs_shapes_from_data.values()):
      s1 = s1.shape
      s2 = s2['shape']
      self.assertEqual(s1, leading_dims + s2, f'{s1} != {leading_dims} + {s2}')


if __name__ == '__main__':
  absltest.main()
