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

"""Component loader and editor."""
# pylint:disable=protected-access
# pylint:disable=g-complex-comprehension
import copy
# import importlib
import itertools
import json
from typing import Any, Tuple, Dict, List

import brax
from brax.experimental.composer.components import ant
from brax.experimental.composer.components import ground
from brax.experimental.composer.components import halfcheetah
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

DEFAULT_GLOBAL_OPTIONS_STR = """
friction: 0.6
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.05
substeps: 10
"""

FIX_XZ_OPTIONS_STR = """
frozen {
  position {
    y: 1.0
  }
  rotation {
    x: 1.0
    z: 1.0
  }
}
"""

NAME_FIELDS = {
    'bodies': ('name',),
    'joints': (
        'name',
        'parent',
        'child',
    ),
    'actuators': (
        'name',
        'joint',
    ),
    'collide_include': (
        'first',
        'second',
    ),
}

COMPONENT_MAPPING = {}


def load_component(component: str,
                   observers: List[str] = None,
                   **kwargs) -> Dict[str, Any]:
  """Load component config information."""
  if component == 'ant':
    lib = ant
  elif component == 'ground':
    lib = ground
  elif component == 'halfcheetah':
    lib = halfcheetah
  else:
    raise NotImplementedError(component)
  # load_path = COMPONENT_MAPPING.get(
  #     component, f'brax.experimental.composer.components.{component}')
  # lib = importlib.import_module(load_path)
  return dict(
      message_str=lib.SYSTEM_CONFIG,
      collides=lib.COLLIDES,
      root=lib.ROOT,
      term_fn=lib.TERM_FN,
      observers=observers or lib.DEFAULT_OBSERVERS,
      **kwargs)


def message_str2message(message_str: str) -> brax.Config:
  return text_format.Parse(message_str, brax.Config())


def message_str2json(message_str: str) -> Dict[str, Any]:
  message = text_format.Parse(message_str, brax.Config())
  json_str = MessageToJson(message)
  return json.loads(json_str)


def json2message_str(config_dict: Dict[str, Any]) -> str:
  json_str = json.dumps(config_dict)
  message = Parse(json_str, brax.Config())
  return text_format.MessageToString(message)


def json_global_options(fix_xz=False, **kwargs):
  message_str = DEFAULT_GLOBAL_OPTIONS_STR
  if fix_xz:
    message_str += FIX_XZ_OPTIONS_STR
  options = message_str2json(message_str)
  return dict(copy.deepcopy(options), **kwargs)


def json_collides(first_collides: Tuple[str],
                  second_collides: Tuple[str]) -> Dict[str, Any]:
  collides = []
  for first, second in itertools.product(first_collides, second_collides):
    collides += [dict(first=first, second=second)]
  return dict(collide_include=collides)


def add_suffix(name: str, suffix: str):
  """Add suffix to string."""
  return f'{name}_{suffix}'


def json_add_suffix(
    config_dict: Any,
    suffix: str = '',
    parents: Tuple[str] = (),
    excludes: Tuple[str] = ('Ground',),
    force_add: bool = False,
) -> Dict[str, Any]:
  """Add suffix to all name references in config."""
  if isinstance(config_dict, dict):
    return {
        key: json_add_suffix(
            value,
            suffix=suffix,
            parents=parents + (key,),
            excludes=excludes,
            force_add=force_add) for key, value in config_dict.items()
    }
  elif isinstance(config_dict, (list, tuple)):
    return type(config_dict)([
        json_add_suffix(
            value,
            suffix=suffix,
            parents=parents,
            excludes=excludes,
            force_add=force_add) for value in config_dict
    ])
  else:
    if force_add or (len(parents) >= 2 and
                     parents[-1] in NAME_FIELDS.get(parents[-2], ())):
      assert isinstance(config_dict, str), config_dict
      if config_dict not in excludes:
        config_dict = add_suffix(config_dict, suffix)
    return config_dict
