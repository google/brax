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

"""Component loader and editor."""
# pylint:disable=protected-access
# pylint:disable=g-complex-comprehension
# pylint:disable=g-doc-args
# pylint:disable=g-doc-return-or-yield
import copy
import itertools
import json
from typing import Any, Tuple, Dict

import brax
# pylint:disable=unused-import
from brax.experimental.composer.components import register_component
from brax.experimental.composer.components import register_default_components
# pylint:enable=unused-import
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

DEFAULT_GLOBAL_OPTIONS_STR = """
friction: 1.0
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.05
substeps: 10
dynamics_mode: "legacy_spring"
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
    'qps': ('name',),
    'angles': ('name',),
}


def message_str2message(message_str: str) -> brax.Config:
  return text_format.Parse(message_str, brax.Config())


def message_str2json(message_str: str) -> Dict[str, Any]:
  message = text_format.Parse(message_str, brax.Config())
  json_str = MessageToJson(message)
  return json.loads(json_str)


DEFAULT_GLOBAL_OPTIONS_JSON = message_str2json(DEFAULT_GLOBAL_OPTIONS_STR)


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


SPLITTER = '___'
COMP_SPLITTER = '__'


def concat_comps(*comp_names):
  """Concat comp_name."""
  if not comp_names:
    return None
  return COMP_SPLITTER.join(sorted(comp_names))


def match_name(name: str, *comp_names):
  """Check comp_name matches name, e.g. ('a1__a2___dist','a1','a2')-->True."""
  comp_name = concat_comps(*comp_names)
  assert comp_name
  return name.startswith(f'{comp_name}{SPLITTER}')


def concat_name(name: str, *comp_names):
  """Add comp_name to name, e.g. ('dist', 'a2', 'a1')-->'a1__a2___dist'."""
  comp_name = concat_comps(*comp_names)
  if comp_name:
    return f'{comp_name}{SPLITTER}{name}'
  return name


def split_name(name: str):
  """Split the name, e.g. ('a1__a2___dist')-->('dist', ('a1', 'a2'))."""
  if SPLITTER not in name:
    return name, []
  comp_name, name = name.split(SPLITTER, 1)
  comp_names = comp_name.split(COMP_SPLITTER)
  return name, comp_names


def json_concat_name(
    config_dict: Any,
    comp_name: str = '',
    parents: Tuple[str] = (),
    excludes: Tuple[str] = ('Ground',),
    force_add: bool = False,
) -> Dict[str, Any]:
  """Add comp_name to all name references in config."""
  if isinstance(config_dict, dict):
    return {
        key: json_concat_name(
            value,
            comp_name=comp_name,
            parents=parents + (key,),
            excludes=excludes,
            force_add=force_add) for key, value in config_dict.items()
    }
  elif isinstance(config_dict, (list, tuple)):
    return type(config_dict)([
        json_concat_name(
            value,
            comp_name=comp_name,
            parents=parents,
            excludes=excludes,
            force_add=force_add) for value in config_dict
    ])
  else:
    if force_add or (len(parents) >= 2 and
                     parents[-1] in NAME_FIELDS.get(parents[-2], ())):
      assert isinstance(config_dict, str), config_dict
      if config_dict not in excludes:
        config_dict = concat_name(config_dict, comp_name)
    return config_dict


def filter_json(config_dict: Dict[str, Any], ground_name: str):
  """Filter config dict and remove ground & global configs."""
  # filter global configs
  config_dict = {
      k: v
      for k, v in config_dict.items()
      if k not in DEFAULT_GLOBAL_OPTIONS_JSON
  }
  # filter ground body and contacts
  collide_include = config_dict.get('collideInclude', None)
  assert collide_include, f'missing collideInclude?: {sorted(config_dict)}'
  collide_include = [
      c for c in collide_include
      if not any(v == ground_name for v in c.values())
  ]
  config_dict['collide_include'] = collide_include
  bodies = config_dict.get('bodies', None)
  assert bodies, f'missing bodies?: {sorted(config_dict)}'
  bodies = [c for c in bodies if c['name'] != ground_name]
  config_dict['bodies'] = bodies
  return config_dict


def filter_message_str(config_str: str, *args, **kwargs):
  """Filter config str and remove ground & global configs."""
  config_dict = message_str2json(config_str)
  config_dict = filter_json(config_dict, *args, **kwargs)
  return json2message_str(config_dict)
