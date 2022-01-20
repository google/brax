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

"""D-min Sweep."""
# pylint:disable=g-complex-comprehension
from brax.experimental.braxlines.experiments import defaults

ENV_NAMES = ('ant', 'halfcheetah', 'humanoid', 'hopper', 'walker2d')
AGENT_MODULE = 'brax.experimental.braxlines.irl_smm.train'
CONFIG = [
    dict(
        env_name=[env_name],
        obs_indices='vel',
        target_num_modes=2,
        obs_scale=8,
        reward_type=['gail', 'fairl', 'gail2', 'mle', 'airl'],
        seed=list(range(10)),
        normalize_obs_for_disc=False,
        evaluate_dist=True,
        env_reward_multiplier=0.0,
        spectral_norm=False,
        gradient_penalty_weight=0.0,
        ppo_params=defaults.get_ppo_params(env_name, 2))
    for env_name in ENV_NAMES
]
