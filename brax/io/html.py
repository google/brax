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

"""Exports a system config and trajectory as an html view."""

import json
import os
from typing import List

import brax
from brax.io.file import File
from brax.io.file import MakeDirs
from brax.io.json import JaxEncoder

from google.protobuf import json_format


def save_html(path: str,
              sys: brax.System,
              qps: List[brax.QP],
              make_dir: bool = False):
  """Saves trajectory as a HTML file."""
  if make_dir and path:
    MakeDirs(os.path.dirname(path))
  with File(path, 'w') as fout:
    fout.write(render(sys, qps))


def render(sys: brax.System, qps: List[brax.QP], height: int = 480) -> str:
  """Returns an HTML page that visualizes the system and qps trajectory."""
  if any((len(qp.pos.shape), len(qp.rot.shape)) != (2, 2) for qp in qps):
    raise RuntimeError('unexpected shape in qp.')
  d = {
      'config': json_format.MessageToDict(sys.config, True),
      'pos': [qp.pos for qp in qps],
      'rot': [qp.rot for qp in qps],
  }
  system = json.dumps(d, cls=JaxEncoder)
  html = _HTML.replace('<!-- system json goes here -->', system)
  html = html.replace('<!-- viewer height goes here -->', f'{height}px')
  return html


_HTML = """
<html>
  <head>
    <title>brax visualizer</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }
      #brax-viewer {
        margin: 0;
        padding: 0;
        height: <!-- viewer height goes here -->;
      }
    </style>
  </head>
  <body>
    <script type="application/javascript">
    var system = <!-- system json goes here -->;
    </script>
    <div id="brax-viewer"></div>
    <script type="module">
      import {Viewer} from 'https://cdn.jsdelivr.net/gh/google/brax@v0.0.13/js/viewer.js';
      const domElement = document.getElementById('brax-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""
