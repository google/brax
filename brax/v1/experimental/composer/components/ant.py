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

"""Ant."""
# pylint:disable=protected-access
import brax.v1 as brax
from brax.v1.envs import ant
from brax.v1.experimental.composer import component_editor
from brax.v1.experimental.composer.components import common

COLLIDES = ('$ Torso', '$ Body 4', '$ Body 7', '$ Body 10', '$ Body 13')

ROOT = '$ Torso'

DEFAULT_OBSERVERS = ('root_z_joints', 'cfrc')


def term_fn(done, sys, qp: brax.QP, info: brax.Info, component,
            **unused_kwargs):
  """Termination."""
  done = common.height_term_fn(
      done,
      sys,
      qp,
      info,
      component,
      max_height=1.0,
      min_height=0.2,
      **unused_kwargs)
  done = common.upright_term_fn(done, sys, qp, info, component, **unused_kwargs)
  return done


def get_specs():
  return dict(
      message_str=component_editor.filter_message_str(ant._SYSTEM_CONFIG_SPRING,
                                                      'Ground'),
      collides=COLLIDES,
      root=ROOT,
      term_fn=term_fn,
      observers=DEFAULT_OBSERVERS)
