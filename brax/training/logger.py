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

"""Logger for training metrics."""

import collections
import logging
from jax import numpy as jnp
import numpy as np


class EpisodeMetricsLogger:
  """Logs training metrics for each episode."""

  def __init__(
      self, buffer_size=100, steps_between_logging=1e5, progress_fn=None
  ):
    self._metrics_buffer = collections.defaultdict(
        lambda: collections.deque(maxlen=buffer_size)
    )
    self._buffer_size = buffer_size
    self._steps_between_logging = steps_between_logging
    self._num_steps = 0
    self._last_log_steps = 0
    self._log_count = 0
    self._progress_fn = progress_fn

  def update_episode_metrics(self, metrics, dones):
    self._num_steps += np.prod(dones.shape)
    if jnp.sum(dones) > 0:
      for name, metric in metrics.items():
        done_metrics = metric[dones.astype(bool)].flatten().tolist()
        self._metrics_buffer[name].extend(done_metrics)
    if self._num_steps - self._last_log_steps >= self._steps_between_logging:
      self.log_metrics()
      self._last_log_steps = self._num_steps

  def log_metrics(self, pad=35):
    """Log metrics to console."""
    self._log_count += 1
    log_string = (
        f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
    )
    mean_metrics = {}
    for metric_name in self._metrics_buffer:
      mean_metrics[metric_name] = np.mean(self._metrics_buffer[metric_name])
      log_string += (
          f"{f'Episode {metric_name}:':>{pad}}"
          f" {mean_metrics[metric_name]:.4f}\n"
      )
    logging.info(log_string)
    if self._progress_fn is not None:
      self._progress_fn(
          int(self._num_steps),
          {f"episode/{name}": value for name, value in mean_metrics.items()},
      )
