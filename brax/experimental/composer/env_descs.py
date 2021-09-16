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
    'ant_cheetah':
        dict(
            components=dict(
                ant1=dict(component='ant', pos=(0, 1, 0)),
                cheetah2=dict(component='halfcheetah', pos=(0, -1, 0)),
            ),
            edges=dict(ant1__cheetah2=dict(collide_type=None),)),
}
