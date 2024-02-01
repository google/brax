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

"""MI-Max Sweep."""
# pylint:disable=g-complex-comprehension
from brax.v1.experimental.braxlines.experiments import defaults

ENV_NAMES = ('ant', 'halfcheetah', 'humanoid', 'hopper', 'walker2d')
AGENT_MODULE = 'brax.experimental.braxlines.vgcrl.train'
CONFIG = [
    dict(
        env_name=[env_name],
        obs_indices='vel',
        algo_name=['gcrl', 'diayn', 'cdiayn', 'diayn_full'],
        obs_scale=5.0,
        seed=list(range(10)),
        normalize_obs_for_disc=False,
        evaluate_mi=True,
        evaluate_lgr=True,
        env_reward_multiplier=0.0,
        spectral_norm=True,
        ppo_params=defaults.get_ppo_params(env_name, 2))
    for env_name in ENV_NAMES
]
