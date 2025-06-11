# Copyright 2025 The Brax Authors.
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

"""Test PPO checkpointing."""

import functools

from absl import flags
from absl.testing import absltest
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
import jax
from jax import numpy as jp


class CheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()

  def test_ppo_params_config(self):
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(16, 21, 13),
    )
    config = checkpoint.network_config(
        action_size=3,
        observation_size=1,
        normalize_observations=True,
        network_factory=network_factory,
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()["policy_hidden_layer_sizes"],
        (16, 21, 13),
    )
    self.assertEqual(config.action_size, 3)
    self.assertEqual(config.observation_size, 1)

  def test_save_and_load_checkpoint(self):
    path = self.create_tempdir("test")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(16, 21, 13),
    )
    config = checkpoint.network_config(
        observation_size=1,
        action_size=3,
        normalize_observations=True,
        network_factory=network_factory,
    )

    # Generate network params for saving a dummy checkpoint.
    normalize = lambda x, y: x
    if config.normalize_observations:
      normalize = running_statistics.normalize
    ppo_network = network_factory(
        config.observation_size,
        config.action_size,
        preprocess_observations_fn=normalize,
        **config.network_factory_kwargs,
    )
    dummy_key = jax.random.PRNGKey(0)
    network_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(dummy_key),
        value=ppo_network.value_network.init(dummy_key),
    )
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(jp.zeros, config.observation_size)
    )
    params = (normalizer_params, network_params.policy, network_params.value)

    # Save and load a checkpoint.
    checkpoint.save(
        path.full_path,
        step=1,
        params=params,
        config=config,
    )

    policy_fn = checkpoint.load_policy(
        epath.Path(path.full_path) / "000000000001",
    )
    out = policy_fn(jp.zeros(1), jax.random.PRNGKey(0))
    self.assertEqual(out[0].shape, (3,))


if __name__ == "__main__":
  absltest.main()
