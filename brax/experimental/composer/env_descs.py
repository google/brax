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

"""Environment descriptions."""
from brax.experimental.composer.components import ant
from brax.experimental.composer.observers import LambdaObserver as lo
from brax.experimental.composer.observers import SimObserver as so

ENV_DESCS = {
    'ant_run':
        dict(
            components=dict(
                ant1=dict(
                    component='ant',
                    pos=(0, 0, 0),
                    reward_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            offset=5,
                            target_goal=(4, 0))),
                    score_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            target_goal=(4, 0))),
                ),)),
    'ant_chase':
        dict(
            components=dict(
                ant1=dict(component='ant', pos=(0, 0, 0)),
                ant2=dict(
                    component='ant',
                    pos=(0, 2, 0),
                    reward_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            offset=5,
                            scale=1,
                            target_goal=(4, 0)),),
                    score_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            target_goal=(4, 0)),),
                ),
            ),
            edges=dict(
                ant1__ant2=dict(
                    extra_observers=[
                        dict(observer_type='root_vec', indices=(0, 1)),
                    ],
                    reward_fns=dict(
                        dist=dict(
                            reward_type='root_dist', min_dist=1, offset=5)),
                    score_fns=dict(
                        dist=dict(reward_type='root_dist', min_dist=1)),
                ),)),
    'ant_push':
        dict(
            components=dict(
                ant1=dict(
                    component='ant',
                    pos=(0, 0, 0),
                ),
                cap1=dict(
                    component='singleton',
                    component_params=dict(size=0.5),
                    pos=(1, 0, 0),
                    observers=('root_z_joints',),
                    reward_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            offset=5,
                            scale=1,
                            target_goal=(4, 0))),
                    score_fns=dict(
                        goal=dict(
                            reward_type='root_goal',
                            sdcomp='vel',
                            indices=(0, 1),
                            target_goal=(4, 0))),
                ),
            ),
            edges=dict(
                ant1__cap1=dict(
                    extra_observers=[
                        dict(observer_type='root_vec', indices=(0, 1)),
                    ],
                    reward_fns=dict(
                        dist=dict(reward_type='root_dist', offset=5)),
                    score_fns=dict(dist=dict(reward_type='root_dist')),
                ),)),
    'uni_ant':
        dict(components=dict(ant1=dict(component='ant', pos=(0, 0, 0)),),),
    'bi_ant':
        dict(
            components=dict(
                ant1=dict(component='ant', pos=(0, 1, 0)),
                ant2=dict(component='ant', pos=(0, -1, 0)),
            ),
            extra_observers=[
                lo(name='delta_pos',
                   fn='-',
                   observers=[
                       so('body', 'pos', ant.ROOT, 'ant1'),
                       so('body', 'pos', ant.ROOT, 'ant2')
                   ]),
                lo(name='delta_vel',
                   fn='-',
                   observers=[
                       so('body', 'vel', ant.ROOT, 'ant1'),
                       so('body', 'vel', ant.ROOT, 'ant2')
                   ]),
            ],
            edges=dict(ant1__ant2=dict(collide_type=None),)),
    'tri_ant':
        dict(
            components=dict(
                ant1=dict(component='ant', pos=(0, 1, 0)),
                ant2=dict(component='ant', pos=(0, -1, 0)),
                ant3=dict(component='ant', pos=(1, 0, 0)),
            ),
            edges=dict(),
        ),
}
