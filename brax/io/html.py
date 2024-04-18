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

# pylint:disable=g-multiple-import
"""Exports a system config and trajectory as an html view."""

import base64
from typing import List, Optional, Union
import zlib

import brax
from brax.base import State, System
from brax.io import json
from etils import epath
import jinja2


def save(path: str, sys: System, states: List[State]):
  """Saves trajectory as an HTML text file."""
  path = epath.Path(path)
  if not path.parent.exists():
    path.parent.mkdir(parents=True)
  path.write_text(render(sys, states))


def render_from_json(
    sys: str, height: Union[int, str], colab: bool, base_url: Optional[str]
) -> str:
  """Returns an HTML string that visualizes the brax system json string."""
  html_path = epath.resource_path('brax') / 'visualizer/index.html'
  template = jinja2.Template(html_path.read_text())

  js_url = base_url
  if base_url is None:
    base_url = 'https://cdn.jsdelivr.net/gh/google/brax'
    js_url = f'{base_url}@v{brax.__version__}/brax/visualizer/js/viewer.js'

  sys = base64.b64encode(zlib.compress(bytes(sys, 'utf-8'))).decode('ascii')
  html = template.render(
      system_json_b64=sys, height=height, js_url=js_url, colab=colab
  )
  return html


def render(
    sys: System,
    states: List[State],
    height: Union[int, str] = 480,
    colab: bool = True,
    base_url: Optional[str] = None,
) -> str:
  """Returns an HTML string for the brax system and trajectory.

  Args:
    sys: brax System object
    states: list of system states to render
    height: the height of the render window
    colab: whether to use css styles for colab
    base_url: the base url for serving the visualizer files. By default, a CDN
      url is used

  Returns:
    string containing HTML for the brax visualizer
  """
  return render_from_json(json.dumps(sys, states), height, colab, base_url)
