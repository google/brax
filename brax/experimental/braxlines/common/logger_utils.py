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

"""Logging utilities."""






import collections
import copy
import csv
import logging
import os
import pprint
import sys
import time
from brax.io import file as io_file
import numpy as np

_tabulators = {}
_timers = {}
_record_time = False


def save_config(output_path, config, verbose=False):
  io_file.MakeDirs(os.path.dirname(output_path))
  config_str = pprint.pformat(config, indent=2)
  with io_file.File(output_path, 'w') as f:
    f.write(config_str)
  if verbose:
    print(f'Saved {output_path}')


def load_config(path, verbose=False):
  with io_file.File(path, 'r') as f:
    config_str = f.read()
  config = eval(config_str)
  if verbose:
    print(f'Loaded {path}')
  return config


class Graph(object):
  """Visualize data in dynamic graphs."""

  def __init__(
      self,
      max_length=100,
  ):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    self.max_length = max_length
    self.app = QtGui.QApplication([])
    self.win = pg.GraphicsWindow()
    self.ps = {}
    self.curves = {}
    self.dats = {}

  def add_plot(self, key):
    if key not in self.ps:
      self.ps[key] = self.win.addPlot(colspan=2)
      self.ps[key].setLabel(axis='top', text=key)
      self.win.nextRow()
      self.curves[key] = self.ps[key].plot()
      self.dats[key] = collections.deque()

  def update(self, **values):
    for key, value in values.items():
      self.add_plot(key)
      if len(self.dats[key]) > self.max_length:
        self.dats[key].popleft()
      self.dats[key].append(value)
      self.curves[key].setData(self.dats[key])
    self.app.processEvents()


class Tabulator(object):
  """Tabulate data and incrementally format into a csv."""

  def __init__(self, output_path=None, append=True, cast_type=None):
    self._output_path = output_path
    self._cast_type = cast_type
    self._curr_values = collections.OrderedDict()
    self._history_values = collections.OrderedDict()
    self._curr_counts = {}
    if append and output_path and io_file.Exists(self._output_path):
      self.finalize_from_file()
    else:
      self._finalized = False

  def finalize_from_file(self):
    data = parse_csv(self._output_path)
    self._history_values = data
    for key, value in data.items():
      self._curr_values[key] = value[-1]
    self._finalized = True

  def get_statistics(self, indices=None, stat='mean', tag='', key_filter=None):
    """Get statistics (average, max, min) values in the table."""
    ret = {}
    for key, values in self._history_values.items():
      if key_filter and not key_filter(key):
        continue
      target_values = np.array(values)
      if indices is None:
        pass
      elif isinstance(indices, (int, tuple, list, np.ndarray)):
        if isinstance(indices, int):
          indices = [indices]
        target_values = target_values[indices]
      elif isinstance(indices, str) and ':' in indices:
        first_index, second_index = [
            int(s) if s else None for s in indices.split(':', 1)
        ]
        target_values = target_values[first_index:second_index]
      else:
        raise NotImplementedError(indices, type(indices))
      if tag:
        key += tag
      if stat == 'mean':
        ret[key] = np.mean(target_values, axis=0)
      elif stat == 'max':
        ret[key] = np.max(target_values, axis=0)
      elif stat == 'min':
        ret[key] = np.min(target_values, axis=0)
      else:
        raise NotImplementedError(stat)
    return ret

  def get_last(self):
    return self.get_statistics(indices=-1)

  def get_curr(self):
    return self._curr_values

  def add(self, accumulate=False, **entries):
    """Add an entry of data."""
    for key, value in sorted(entries.items()):
      if key not in self._history_values:
        assert not self._finalized, ('Cannot add a new key {} once tabulator is'
                                     ' finalized.').format(key)
        self._history_values[key] = []
      value = copy.deepcopy(value)
      if accumulate:
        value += self._curr_values.get(key, 0.0)
        self._curr_counts[key] = self._curr_counts.get(key, 0) + 1
      value = self.cast(value)
      self._curr_values[key] = value

  def cast(self, value):
    if self._cast_type:
      try:
        value = self._cast_type(value)
      except TypeError as e:
        raise TypeError('{}: Failed to cast {} as {}'.format(
            e, value, self._cast_type))
    return value

  def finalize(self):
    output_dir = os.path.dirname(self._output_path)
    if output_dir:
      io_file.MakeDirs(output_dir)
    with io_file.File(self._output_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(self._history_values.keys())
    self._finalized = True

  def dump(self, output_path=None, average=True):
    """Dump to a csv file."""
    output_path = output_path or self._output_path
    if not self._curr_values:
      return  # empty
    if output_path:
      if not self._finalized:
        self.finalize()  # finalize
      with io_file.File(output_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(self._curr_values.values())
    for key, value in self._history_values.items():
      v = copy.deepcopy(self._curr_values[key])
      if average:
        v /= self._curr_counts.get(key, 1.0)
        value = self.cast(value)
      value.append(v)
    self._curr_counts = {}
    self._curr_values = {k: 0.0 for k in self._curr_values}


def parse_csv(filename: str, verbose: bool = False):
  """Parse a csv file."""
  with io_file.File(filename, 'r') as f:
    csv_data = np.genfromtxt(f, delimiter=',', names=True, deletechars='')
  data = collections.OrderedDict()
  try:
    for i, key in enumerate(csv_data.dtype.names):
      data[key] = [d[i] for d in csv_data]
  except TypeError:
    # 0-D array errors out
    for key in csv_data.dtype.names:
      data[key] = np.array([csv_data[key]])
  if verbose:
    print(f'Loaded len={len(list(data.values())[0])}, '
          f'keys={sorted(list(data.keys()))} from {filename}')
  return data


def parse_csv_parallel(filenames, n_threads=1):
  import multiprocessing
  with multiprocessing.pool.ThreadPool(n_threads) as pool:
    jobs = {
        filename: pool.apply_async(parse_csv, [filename], error_callback=print)
        for filename in filenames
    }
    data = {key: value.get() for key, value in jobs.items()}
  return data


def timeit():
  global _record_time
  _record_time = True


def tic(name):
  global _timers
  _timers[name] = time.time()


def toc(name, indent=0):
  global _timers, _record_time
  assert name in _timers
  dt = time.time() - _timers[name]
  del _timers[name]
  if _record_time:
    print('{}[{}] runtime: {}s'.format(''.join(['\t'] * indent), name, dt))
  return dt


def get_level(name):
  """Get level."""
  level = 'info'  # default level
  os_level = os.getenv('LEVEL')
  if os_level is not None:
    if ',' in os_level:
      os_levels = os_level.split(',')
      if name in os_levels[1:]:
        level = os_levels[0]
    else:
      level = os_level
  return level


class LoggerWrapper(object):
  """LoggerWrapper."""

  def __init__(self, logger, name):
    self.logger = logger
    self.name = name

  def format(self, content='', name=None, **kwargs):
    """Format content to str."""
    if name is None:
      name = self.name
    else:
      name = self.name + ':' + name
    s = '[{}]'.format(name)
    if content:
      s += ' ' + pprint.pformat(content)
    if kwargs:
      s += ' ' + pprint.pformat(kwargs)
    return s

  def add_name(self, name):
    self.name = ':'.join((self.name, name))

  def pop_name(self):
    self.name = ':'.join(self.name.split(':')[:-1])

  def debug(self, content='', name=None, **kwargs):
    level = get_level(self.name)
    if level in ('debug',):
      self.logger.debug(self.format(content=content, name=name, **kwargs))

  def info(self, content='', name=None, **kwargs):
    self.logger.info(self.format(content=content, name=name, **kwargs))


def get_logger(level=None, name=__name__):
  """Get logger.

  If `level` is not specified, it consults os.getenv('LEVEL').
      e.g. LEVEL=debug: print all debug messages.
           LEVEL=debug,name1,name2: print all debug messages,
              only for loggers with `name1` or `name2`,
              and use default level (`info`) for others.

  Args:
      level: a string, e.g. 'info', 'debug', 'error'.
      name: a string, identifier for logger.

  Returns:
      A logging.logger object.
  """
  name = name.split('.')[-1]  # base name
  if level is None:
    level = get_level(name)
  logger = logging.getLogger(name)
  out_hdlr = logging.StreamHandler(sys.stdout)
  out_hdlr.setFormatter(
      logging.Formatter('[{}] %(asctime)s %(message)s'.format(name)))
  out_hdlr.setLevel(getattr(logging, level.upper()))
  logger.addHandler(out_hdlr)
  logger.setLevel(getattr(logging, level.upper()))
  return LoggerWrapper(logger, name=name)


def get_tabulator(name=__name__, **kwargs):
  """Get a tabulator."""
  global _tabulators
  if name not in _tabulators:
    _tabulators[name] = Tabulator(**kwargs)
  return _tabulators[name]


if __name__ == '__main__':
  tab = get_tabulator(append=False)
  tab.dump()  # do nothing
  tab.add(a=3, b=2, c=4)
  tab.add(b=4, d=6)
  tab.dump()
  tab.add(a=1, d=4)
  tab.dump()
  tab2 = get_tabulator(append=True)
  tab2.add(a=4, b=1, c=2, d=3)
  tab2.dump()
