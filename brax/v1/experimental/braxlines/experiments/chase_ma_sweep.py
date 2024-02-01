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

"""composer Sweep."""
# pylint:disable=g-complex-comprehension
import itertools
from brax.v1.experimental.braxlines.experiments import defaults

seed = 0
comps = [('ant', {}), ('pro_ant', dict(num_legs=[2, 6]))]

AGENT_MODULE = 'brax.experimental.composer.train'
CONFIG = [
    dict(
        env_name='chase',
        env_params=dict(
            main_agent=[comp1[0]],
            main_agent_params=comp1[1],
            other_agent=[comp2[0]],
            other_agent_params=comp2[1],
            num_agents=list(range(2, 5)),
        ),
        seed=seed,
        ppo_params=defaults.get_ppo_params(comp1[0], 3, default='ant'))
    for comp1, comp2 in itertools.product(comps, comps)
]
