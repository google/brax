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

"""Experiment configuration loader and runner."""
# pylint:disable=broad-except
# pylint:disable=g-complex-comprehension
import datetime
import functools
import importlib
import math
import os
import pprint
import re
from typing import Dict, Any, List, Tuple, Callable
from brax.v1.envs.env import State
from brax.v1.experimental.braxlines.common import config_utils
from brax.v1.experimental.braxlines.common import logger_utils
from brax.v1.experimental.braxlines.experiments import defaults
from brax.v1.io import file
import numpy as np

DEFAULT_LIB_PATH_TEMPLATE = 'brax.experimental.braxlines.experiments'


def load_experiment(experiment_name: str):
  load_path = f'{DEFAULT_LIB_PATH_TEMPLATE}.{experiment_name}'
  experiment_lib = importlib.import_module(load_path)
  return experiment_lib.AGENT_MODULE, experiment_lib.CONFIG


def run_experiment(experiment_name: str = None,
                   output_path: str = '/tmp/sweep',
                   start_count: int = 0,
                   end_count: int = int(1e6),
                   ignore_errors: bool = False,
                   agent_module: str = None,
                   config: Dict[str, Any] = None):
  """Run experiments defined by `config` serially."""
  if not agent_module and not config:
    agent_module, config = load_experiment(experiment_name)
  if isinstance(agent_module, str):
    agent_module = importlib.import_module(agent_module)

  prefix_keys = config_utils.list_keys_to_expand(config)
  for c, p in zip(config, prefix_keys):
    c.update(dict(prefix_keys=p))
  config_count = config_utils.count_configuration(config)
  start_count = max(start_count, 0)
  end_count = min(end_count, sum(config_count))
  print(f'Loaded experiment_name={experiment_name}')
  print(f'Loaded {sum(config_count)}({config_count}) experiment configurations')
  print(f'Set start_count={start_count}, end_count={end_count}')
  print(f'Set prefix_keys={prefix_keys}')
  print(f'Set output_dir={output_path}')

  # @title Launch experiments
  for i in range(start_count, end_count):
    c, _ = config_utils.index_configuration(config, index=i, count=config_count)
    task_name = config_utils.get_compressed_name_from_keys(
        c, agent_module.TASK_KEYS)
    experiment_name = config_utils.get_compressed_name_from_keys(
        c, c.pop('prefix_keys'))
    output_dir = f'{output_path}/{task_name}/{experiment_name}'
    print(f'[{i+1}/{sum(config_count)}] Starting experiment...')
    print(f'\t config: {pprint.pformat(c, indent=2)}')
    print(f'\t output_dir={output_dir}')
    return_dict = {}
    if ignore_errors:
      try:
        agent_module.train(c, output_dir=output_dir, return_dict=return_dict)
      except Exception as e:
        print(
            f'[{i+1}/{sum(config_count)}] FAILED experiment {e.__class__.__name__}: {e.message}'
        )
    else:
      agent_module.train(c, output_dir=output_dir, return_dict=return_dict)
    print(f'\t time_to_jit={return_dict.get("time_to_train", None)}')
    print(f'\t time_to_train={return_dict.get("time_to_jit", None)}')


def load_data(csv_files: List[str], max_print: int = 5):
  """Load data from a list of csv_file paths."""
  data = {}
  for i, csv_file in enumerate(csv_files):
    key = os.path.basename(os.path.dirname(csv_file))
    verbose = i < max_print or i == len(csv_files) - 1
    if verbose:
      print(f'[{i+1}/{len(csv_files)}] key={key}')
    data[key] = {
        k: np.array(v)
        for k, v in logger_utils.parse_csv(csv_file, verbose=verbose).items()
    }
    data[key]['_csv_file'] = csv_file
  return data


COMMON_TAG_VALUES = {
    'an_diayn': 'DIAYN',
    'an_cdiayn': 'cDIAYN',
    'an_gcrl': 'GCRL',
    'an_diayn_full': 'DIAYN_FULL',
    'rt_gail': 'GAIL',
    'rt_airl': 'AIRL',
    'rt_gail2': 'GAIL2',
    'rt_fairl': 'FARIL',
    'rt_mle': 'MLE',
}
COMMON_TAG_VALUES.update({f'de.cpl_{i}': f'num_legs={i}' for i in range(20)})
COMMON_TAG_VALUES.update({f'de.cf_{i}': f'scale={i}' for i in (0.2, 0.5, 1)})


def split_tag(tag_str: str,
              match_tags: Tuple[str] = (),
              replace_common_tag_values: bool = False,
              tag_splitter: str = '__',
              tag_value_splitter: str = '_'):
  """Split the tag_str into two strs based on matching tags."""
  if replace_common_tag_values:
    ctv = COMMON_TAG_VALUES
  else:
    ctv = {}  # replace none
  tag_values = tag_str.split(tag_splitter)
  tag_values = [vv.split(tag_value_splitter, 1) for vv in tag_values]
  match_str = tag_splitter.join([
      ctv.get(tag_value_splitter.join(vv), tag_value_splitter.join(vv))
      for vv in tag_values
      if vv[0] in match_tags
  ])
  rest_str = tag_splitter.join([
      ctv.get(tag_value_splitter.join(vv), tag_value_splitter.join(vv))
      for vv in tag_values
      if vv[0] not in match_tags
  ])
  return match_str, rest_str


def compute_statistics(data,
                       merge_tags: Tuple[str] = ('s',),
                       max_print: int = 5,
                       tag_splitter_kwargs: Dict[str, Any] = None):
  """Merge data to derive mean/std's."""
  statistics = {}
  filepaths = {}
  n = len(data)
  for i, (k, v) in enumerate(data.items()):
    merge_key, data_key = split_tag(k, merge_tags, **(tag_splitter_kwargs or
                                                      {}))
    statistics[data_key] = statistics.get(data_key, {})
    c = len(statistics[data_key])
    filepaths[data_key] = filepaths.get(data_key, []) + [v.pop('_csv_file')]
    statistics[data_key][merge_key] = v
    if i < max_print or i == len(data) - 1 or c == 0:
      print(f'[{i+1}/{n}] {k} -> {data_key}[{c}], {merge_key}')

  for i, k in enumerate(statistics):
    v = statistics[k]
    print(f'[{i+1}/{len(statistics)}] len({k})={len(v)}')
    stat_keys = list(v.values())[0].keys()
    new_v = {}
    for stat_key in stat_keys:
      d = [v[stat_key] for v in v.values()]
      mean = np.mean(d, axis=0)
      std = np.std(d, axis=0)
      new_v[stat_key] = dict(mean=mean, std=std)
    statistics[k] = new_v
  return statistics, filepaths


def color_spec(n):
  t = np.linspace(-510, 510, n)
  return np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255) / 255.


def plot_states(states: List[State], **kwargs):
  """Plot trajectory."""
  x = np.array([s.info['steps'] for s in states])
  ys = {}
  ys.update({
      f'rewards/{k}': np.array([s.info['rewards'][k] for s in states
                               ]) for k in states[-1].info.get('rewards', {})
  })
  ys.update({
      f'scores/{k}': np.array([s.info['scores'][k] for s in states
                              ]) for k in states[-1].info.get('scores', {})
  })
  if 'score' in states[-1].info:
    ys.update({'score': np.array([s.info['score'] for s in states])})
  ys.update({'reward': np.array([s.reward for s in states])})
  plotdata = {k: dict(x=x, y=y) for k, y in ys.items()}
  plot_curves(plotdata=plotdata, xlabel='time steps', **kwargs)


def plot_curves(plotdata: Dict[str, Any],
                plotpatterns: List[str] = None,
                xlabel: str = '# environment steps',
                xlim: List[float] = None,
                max_ncols: int = 5,
                output_name: str = 'training_curves',
                output_path: str = None):
  """Plot data."""
  # pylint:disable=g-import-not-at-top
  import matplotlib.pyplot as plt
  # pylint:enable=g-import-not-at-top
  if plotpatterns:
    plotkeys = [key for key in plotdata if any(x in key for x in plotpatterns)]
  else:
    plotkeys = list(plotdata)
  num_figs = len(plotkeys)
  ncols = min(num_figs, max_ncols)
  nrows = int(math.ceil(num_figs / ncols))
  fig, axs = plt.subplots(
      ncols=ncols, nrows=nrows, figsize=(3.5 * ncols, 3 * nrows))
  for i, key in enumerate(plotkeys):
    col, row = i % ncols, int(i / ncols)
    ax = axs
    if nrows > 1:
      ax = ax[row]
    if ncols > 1:
      ax = ax[col]
    ax.plot(plotdata[key]['x'], plotdata[key]['y'])
    ax.set(xlabel=xlabel, ylabel=key)
    if xlim:
      ax.set_xlim(xlim)
  fig.tight_layout()
  if output_path:
    with file.File(f'{output_path}/{output_name}.png', 'wb') as f:
      plt.savefig(f)


def get_progress_fn(plotpatterns: List[str],
                    times: List[float],
                    return_dict: Dict[str, float] = None,
                    progress_dict: Dict[str, float] = None,
                    xlabel: str = '# environment steps',
                    xlim: List[float] = None,
                    max_ncols: int = 5,
                    update_metrics_fn: Callable[..., Any] = None,
                    pre_plot_fn: Callable[..., Any] = None,
                    post_plot_fn: Callable[..., Any] = None,
                    tab: logger_utils.Tabulator = None):
  """Get progress function for training."""
  plotdata = {}
  return_dict = return_dict or {}
  progress_dict = progress_dict or {}

  plot = functools.partial(
      plot_curves,
      plotdata=plotdata,
      plotpatterns=plotpatterns,
      xlabel=xlabel,
      xlim=xlim,
      max_ncols=max_ncols)

  def progress(num_steps, metrics, params):
    if update_metrics_fn:
      update_metrics_fn(num_steps, metrics, params)
    times.append(datetime.datetime.now())
    for key, v in metrics.items():
      assert not np.isnan(v), f'{key} {num_steps} NaN'
      plotdata[key] = plotdata.get(key, dict(x=[], y=[]))
      plotdata[key]['x'] += [num_steps]
      plotdata[key]['y'] += [v]
    if num_steps > 0:
      tab.add(num_steps=num_steps, **metrics)
      tab.dump()
      return_dict.update(dict(num_steps=num_steps, **metrics))
      progress_dict.update(dict(num_steps=num_steps, **metrics))
    if pre_plot_fn:
      pre_plot_fn()
    plot()
    if post_plot_fn:
      post_plot_fn()

  return progress, plot, plotdata, progress_dict


def plot_statistics(statistics: Dict[str, Any],
                    key_include_re: str = '',
                    key_exclude_re: str = '',
                    xmax: int = 1e10,
                    xlabel: str = '',
                    ylabel_re: str = '',
                    ylabels: Tuple[str] = (),
                    ncols: int = 5,
                    legend_tags: Tuple[str] = None,
                    replace_common_tag_values: bool = True,
                    tag_splitter_kwargs: Dict[str, Any] = None,
                    output_path: str = '',
                    output_name: str = 'statistics'):
  """Plot statistics."""
  # pylint:disable=g-import-not-at-top
  import matplotlib.pyplot as plt
  # pylint:enable=g-import-not-at-top
  include_re = f'.*{key_include_re}.*'
  exclude_re = f'.*{key_exclude_re}.*' if key_exclude_re else None
  ylabel_re = f'.*{ylabel_re}.*'

  plot_data = {
      k: v for k, v in statistics.items() if re.match(include_re, k) and
      (not exclude_re or not re.match(exclude_re, k))
  }
  if legend_tags is not None:
    plot_data = {
        split_tag(
            k,
            legend_tags,
            replace_common_tag_values=replace_common_tag_values,
            **(tag_splitter_kwargs or {}))[0]: v for k, v in plot_data.items()
    }
  if not ylabels:
    ylabels = sorted(list(plot_data.values())[0].keys())
    ylabels = [y for y in ylabels if re.match(ylabel_re, y)]
  nrows = int(math.ceil(len(ylabels) / ncols))

  fig, axs = plt.subplots(
      ncols=ncols, nrows=nrows, figsize=(4.5 * ncols, 4 * nrows))
  colors = color_spec(len(plot_data))
  summaries = {}
  for i, y in enumerate(ylabels):
    axxmin = None
    axxmax = None
    axcolor = None
    axalpha = None
    ax = axs
    row, col = int(i / ncols), i % ncols
    if nrows > 1:
      ax = ax[row]
    if ncols > 1:
      ax = ax[col]
    summary = {}
    for (k, v), c in zip(sorted(plot_data.items()), colors):
      if y in v:
        indices = v[xlabel]['mean'] <= xmax
        xv = v[xlabel]['mean'][indices]
        if axxmin is None:
          axxmin = xv.min()
        if axxmax is None:
          axxmax = xv.max()
        yv = v[y]['mean'][indices]
        yvstd = v[y]['std'][indices]
        summary[k] = dict(x=xv[-1], ymean=yv[-1], ystd=yvstd[-1])
        ax.plot(xv, yv, label=k, c=c, alpha=0.8)
        ax.fill_between(xv, yv - yvstd, yv + yvstd, color=c, alpha=0.2)
        if axcolor:
          ax.patch.set_facecolor(axcolor)
        if axalpha is not None:
          ax.patch.set_alpha(axalpha)
    ax.set(xlim=(axxmin, axxmax))
    ax.set(xlabel=xlabel, ylabel=y)
    ax.legend()
    summaries[y] = summary
  fig.tight_layout()
  if output_path:
    file.MakeDirs(output_path)
    with file.File(f'{output_path}/{output_name}.png', 'wb') as f:
      plt.savefig(f)
  return summaries


def print_summmary_to_latex_table(summaries,
                                  summary_keys,
                                  multipliers,
                                  prefix: str = ''):
  """Print summary to LaTeX-friendly format."""
  tags = sorted(list(list(summaries.values())[0].keys()))
  s = ''
  for tag in tags:
    # pylint:disable=anomalous-backslash-in-string
    tag2 = tag.replace('_', '\_')
    # pylint:enable=anomalous-backslash-in-string
    s += f'{prefix}{tag2}'
    for skey, mult in zip(summary_keys, multipliers):
      s += ' & $ '
      v = summaries[skey][tag]['ymean'] * mult
      if abs(max([v['ymean'] * mult for v in summaries[skey].values()]) -
             v) < 1e-6:
        is_max = True
      else:
        is_max = False
      if is_max:
        s += '\\mathbf{'
      s += f'{v:.3f} \\pm {summaries[skey][tag]["ystd"]:.3f}'
      if is_max:
        s += '}'
      s += ' $'
    s += '\\\\\n'
  return s
