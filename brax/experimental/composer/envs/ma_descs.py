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

"""multi-agent ants environments."""

ENV_DESCS = {
    'ant_chase_ma':
        dict(
            agent_groups=dict(
                agent1=dict(reward_names=(('dist', 'agent1', 'agent2'),)),
                agent2=dict(reward_agents=('agent2',)),
            ),
            components=dict(
                agent1=dict(component='ant', pos=(0, 0, 0)),
                agent2=dict(
                    component='ant',
                    pos=(0, 2, 0),
                    reward_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            offset=5,
                            target_goal=(4, 0)),),
                ),
            ),
            edges=dict(
                agent1__agent2=dict(
                    extra_observers=[
                        dict(observer_type='root_vec', indices=(0, 1)),
                    ],
                    reward_fns=dict(
                        dist=dict(
                            reward_type='root_dist', min_dist=1, offset=5)),
                ),)),
}
