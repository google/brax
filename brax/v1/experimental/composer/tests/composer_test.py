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

"""Composer tests."""

import collections
import functools
from absl.testing import absltest
from absl.testing import parameterized
from brax.v1.experimental.composer import composer
from brax.v1.experimental.composer import data_utils
import jax
from jax import numpy as jnp


class ComposerTest(parameterized.TestCase):
  """Tests for Composer module."""

  @parameterized.parameters('ant_push', 'ant_chase', 'pro_ant_run')
  def testEnvCreation(self, env_name):
    env = composer.create(env_name=env_name)
    env.reset(rng=jax.random.PRNGKey(0))

  @parameterized.parameters('chase', 'follow')
  def testMultiAgentEnvCreation(self, env_name):
    env = composer.create(env_name=env_name)
    state = env.reset(rng=jax.random.PRNGKey(0))
    assert state.reward.ndim > 0, state.reward

  def testInspect(self):
    env_params, support_kwargs = composer.inspect_env('pro_ant_run')
    assert tuple(sorted(env_params)) == ('num_legs',), env_params
    assert not support_kwargs, support_kwargs

  def testActionConcatSplit(self):
    env = composer.create(env_name='humanoid')
    action_shapes = composer.get_action_shapes(env.sys)
    leading_dims = (5,)
    actions = jnp.zeros(leading_dims + (env.action_size,))
    actions_dict = data_utils.split_array(actions, action_shapes)
    actions2 = data_utils.concat_array(actions_dict, action_shapes)
    assert actions.shape == actions2.shape, f'{actions.shape} != {actions2.shape}'

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
    obs_shapes_from_data = data_utils.get_array_shapes(
        obs_dict, batch_shape=leading_dims)
    for s1, s2 in zip(obs_shapes, obs_shapes_from_data.values()):
      s2 = s2['shape']
      self.assertEqual(s1, s2, f'{s1} != {s2}')

    # concat
    obs = data_utils.concat_array(obs_dict, obs_shapes_from_data)
    self.assertEqual(obs.shape, leading_dims + (obs_vec_size,),
                     f'{obs.shape} != {leading_dims} + ({obs_vec_size},)')

    # split again
    obs_dict_2 = data_utils.split_array(obs, obs_shapes_from_data)
    for s1, s2 in zip(obs_dict_2.values(), obs_shapes_from_data.values()):
      s1 = s1.shape
      s2 = s2['shape']
      self.assertEqual(s1, leading_dims + s2, f'{s1} != {leading_dims} + {s2}')


if __name__ == '__main__':
  absltest.main()
