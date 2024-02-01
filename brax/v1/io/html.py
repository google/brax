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

"""Exports a system config and trajectory as an html view."""

import os
from typing import List, Optional

import brax.v1 as brax
from brax.v1.io.file import File
from brax.v1.io.file import MakeDirs
from brax.v1.io.json import dumps


def save_html(path: str,
              sys: brax.System,
              qps: List[brax.QP],
              make_dir: bool = False):
  """Saves trajectory as a HTML file."""
  if make_dir and path:
    MakeDirs(os.path.dirname(path))
  with File(path, 'w') as fout:
    fout.write(render(sys, qps))


def render(sys: brax.System,
           qps: List[brax.QP],
           height: int = 480,
           info: Optional[brax.Info] = None) -> str:
  """Returns an HTML page that visualizes the system and qps trajectory."""
  if any((len(qp.pos.shape), len(qp.rot.shape)) != (2, 2) for qp in qps):
    raise RuntimeError('unexpected shape in qp.')
  system = dumps(sys, qps, info)
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
      import {Viewer} from 'https://cdn.jsdelivr.net/gh/google/brax@v0.1.0/js/viewer.js';
      const domElement = document.getElementById('brax-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""
