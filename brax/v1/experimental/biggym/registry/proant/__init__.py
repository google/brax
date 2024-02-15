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

"""Example: a Component + env rewards."""
AUTHORS = ('Shixiang Shane Gu',)
CONTACTS = ('shanegu@google.com',)
AFFILIATIONS = ('google.com',)
DESCRIPTIONS = ('ant with different number of legs running',)

ENVS = dict(
    run=dict(
        module='ant:Run',
        tracks=('rl',),
    ),)

COMPONENTS = dict(
    ant=dict(
        module='ant',
        tracks=('race', 'race_ma', 'sumo', 'onigokko_predator',
                'onigokko_prey'),
    ),)
