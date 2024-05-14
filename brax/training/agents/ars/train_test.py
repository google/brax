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

"""Augmented Random Search training tests."""
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.ars import networks as ars_networks
from brax.training.agents.ars import train as ars
import jax


class ARSTest(parameterized.TestCase):
  """Tests for ARS module."""

  @parameterized.parameters(True, False)
  def testModelEncoding(self, normalize_observations):
    env = envs.get_environment('fast')
    _, params, _ = ars.train(
        env,
        num_timesteps=128,
        episode_length=128,
        normalize_observations=normalize_observations)
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    ars_network = ars_networks.make_policy_network(env.observation_size,
                                                   env.action_size,
                                                   normalize_fn)
    inference = ars_networks.make_inference_fn(ars_network)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    action = inference(decoded_params)(state.obs, jax.random.PRNGKey(0))[0]
    env.step(state, action)

  def testTrainDomainRandomize(self):
    """Test with domain randomization."""

    def rand_fn(sys, rng):
      @jax.vmap
      def get_offset(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

      sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
      in_axes = jax.tree.map(lambda x: None, sys)
      in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
      return sys_v, in_axes

    _, _, _ = ars.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=128,
        episode_length=128,
        normalize_observations=True,
        randomization_fn=rand_fn,
    )


if __name__ == '__main__':
  absltest.main()
