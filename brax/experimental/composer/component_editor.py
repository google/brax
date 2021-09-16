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
import importlib
import itertools
import json
from typing import Any, Tuple, Dict

import brax
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

DEFAULT_REGISTER_COMPONENTS = ('ant', 'ground', 'halfcheetah')

DEFAULT_GLOBAL_OPTIONS_STR = """
friction: 1.0
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


def register_component(component: str,
                       component_params: Dict[str, Any] = None,
                       component_specs: Any = None,
                       override: bool = False):
  """Register component config information."""
  global COMPONENT_MAPPING
  if not override and component in COMPONENT_MAPPING:
    specs = COMPONENT_MAPPING[component]
    return copy.deepcopy(specs)
  if component_specs is None:
    if '.' not in component:
      load_path = f'brax.experimental.composer.components.{component}'
    else:
      load_path = component
    component_lib = importlib.import_module(load_path)
    component_specs = component_lib.get_specs(**(component_params or {}))
  COMPONENT_MAPPING[component] = component_specs
  return copy.deepcopy(component_specs)


def register_default_components():
  """Register all default components."""
  for component in DEFAULT_REGISTER_COMPONENTS:
    register_component(component)


def load_component(component: str,
                   component_specs: Dict[str, Any] = None,
                   **override_specs) -> Dict[str, Any]:
  """Load component config information."""
  default_specs = register_component(
      component=component, component_specs=component_specs)
  default_specs.update(override_specs)
  return default_specs


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
  if suffix:
    return f'{name}_{suffix}'
  else:
    return name


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
