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

"""Components.

In Braxlines Composer, each environment is built reusing
components defined in this folder. This faciliates reusability,
procedural generation, and randomizability of Brax environments.
"""
import copy
import importlib
from typing import Any, Dict

DEFAULT_REGISTER_COMPONENTS = ('ant', 'ground', 'halfcheetah', 'singleton',
                               'pro_ant', 'octopus', 'humanoid')

COMPONENT_MAPPING = {}


def list_components():
  return sorted(COMPONENT_MAPPING)


def exists(component: str):
  return component in COMPONENT_MAPPING


def register_component(component: str,
                       component_specs: Any = None,
                       load_path: str = None,
                       override: bool = False):
  """Register component library."""
  global COMPONENT_MAPPING
  if not override and component in COMPONENT_MAPPING:
    return COMPONENT_MAPPING[component]
  if component_specs is None:
    if not load_path:
      if '.' not in component:
        load_path = f'brax.v1.experimental.composer.components.{component}'
      else:
        load_path = component
    component_lib = importlib.import_module(load_path)
  COMPONENT_MAPPING[component] = component_lib
  return component_lib


def register_default_components():
  """Register all default components."""
  for component in DEFAULT_REGISTER_COMPONENTS:
    register_component(component)


def load_component(component: str,
                   component_specs: Dict[str, Any] = None,
                   component_params: Dict[str, Any] = None,
                   **override_specs) -> Dict[str, Any]:
  """Load component config information."""
  if isinstance(component, str):
    # if string, load a library under composer/components
    component_lib = register_component(
        component=component, component_specs=component_specs)
    specs_fn = component_lib.get_specs
  else:
    # else, assume it's a custom get_specs()
    specs_fn = component
  default_specs = specs_fn(**(component_params or {}))
  default_specs = copy.deepcopy(default_specs)
  default_specs.update(override_specs)
  return default_specs
