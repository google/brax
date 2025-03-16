# Copyright 2025 The Brax Authors.
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

"""Viewer utilities."""

import mujoco

MAX_VIEWPORTS = 12
MAX_LINE_POINTS = 200
y_range = [0, 0.01]
gridsize = [3, 4]
figures = {}


def get_menu_text():
  text_1 = 'SHIFT + M\nSHIFT + O\nSPACE\nRIGHT/LEFT\nUP/DOWN\nSHIFT + H\nTAB'
  text_2 = (
      'Toggle metrics\nToggle pixel obs\nPause/play\nNext/prev env\nNext/prev'
      ' eval\nToggle help\nToggle left UI'
  )
  return text_1, text_2


def get_viewports(num_viewports: int, viewer_rect: mujoco.MjrRect):
  """Calculate viewports for the given number of plots."""
  denom = 6 if num_viewports >= 6 else max(num_viewports, 4)
  viewport_width = viewer_rect.width // denom
  viewport_height = viewer_rect.height // denom
  max_viewports_per_column = 6
  viewports = []
  for i in range(min(num_viewports, MAX_VIEWPORTS)):
    column = i // max_viewports_per_column
    row = i % max_viewports_per_column
    left = viewer_rect.left + viewer_rect.width - (column + 1) * viewport_width
    bottom = (
        viewer_rect.bottom + viewer_rect.height - (row + 1) * viewport_height
    )
    viewports.append(
        mujoco.MjrRect(
            bottom=bottom,
            height=viewport_height,
            left=left,
            width=viewport_width,
        )
    )
  return viewports


def reset_figures(metrics_keys_list):
  """Initialize/reset figures for each metric."""
  global figures
  # Ensure that 'reward' appears first.
  metrics_keys = metrics_keys_list.copy()
  if 'reward' in metrics_keys:
    metrics_keys.remove('reward')
    metrics_keys.insert(0, 'reward')
  for key in metrics_keys:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.flg_extend = 1
    fig.gridsize = gridsize
    fig.range[1] = y_range[1]
    fig.figurergba[-1] = 0.5
    fig.title = key
    for i in range(MAX_LINE_POINTS):
      fig.linedata[0][2 * i] = -float(i)
    figures[key] = fig


def add_data_to_fig(metric_key, data):
  """Add a new data point to the figure for a given metric."""
  fig = figures[metric_key]
  pnt = min(MAX_LINE_POINTS, fig.linepnt[0] + 1)
  for i in range(pnt - 1, 0, -1):
    fig.linedata[0][2 * i + 1] = fig.linedata[0][2 * i - 1]
  fig.linepnt[0] = pnt
  fig.linedata[0][1] = data
