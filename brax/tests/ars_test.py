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

"""ARS training tests."""
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import ars
import jax


class ARSTest(parameterized.TestCase):
  """Tests for ARS module."""


  @parameterized.parameters(True, False)
  def testModelEncoding(self, normalize_observations):
    env_fn = envs.create_fn('fast')
    _, params, _ = ars.train(env_fn, num_timesteps=128, episode_length=128)
    env = env_fn()
    inference = ars.make_inference_fn(env.observation_size, env.action_size,
                                      normalize_observations)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    action = inference(decoded_params, state.obs, jax.random.PRNGKey(0))
    env.step(state, action)


if __name__ == '__main__':
  absltest.main()
