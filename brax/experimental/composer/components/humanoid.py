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

"""Humanoid."""
# pylint:disable=protected-access
import functools
from brax.envs import humanoid
from brax.experimental.composer import component_editor
from brax.experimental.composer.components import common


COLLIDES = ('torso', 'left_shin', 'right_shin')

ROOT = 'torso'

DEFAULT_OBSERVERS = ('root_z_joints',)

TERM_FN = functools.partial(
    common.height_term_fn, max_height=2.1, min_height=0.7)


def get_specs():
  return dict(
      message_str=component_editor.filter_message_str(
          humanoid._SYSTEM_CONFIG_SPRING, 'floor'),
      collides=COLLIDES,
      root=ROOT,
      term_fn=TERM_FN,
      observers=DEFAULT_OBSERVERS)
