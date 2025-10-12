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

"""Logger for training metrics."""

import collections
import logging

import numpy as np
from jax import numpy as jnp


class MetricsLogger:
    """Logs training metrics at each training step."""

    def __init__(self, buffer_size, steps_between_logging, progress_fn):
        self._metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._buffer_size = buffer_size
        self._steps_between_logging = steps_between_logging
        self._num_steps = 0
        self._last_log_steps = 0
        self._log_count = 0
        self._progress_fn = progress_fn
        self._episodic_metrics_updated = set()

    def update_env_metrics(self, metrics, dones, env_steps):
        self._num_steps = int((np.uint64(env_steps.hi) << 32) + np.uint64(env_steps.lo))

        # Log metrics of whole episodes that finished at this step
        if jnp.sum(dones) > 0:
            for name, metric in metrics.items():
                done_metrics = np.mean(metric[dones.astype(bool)].flatten()).item()
                metric_key = f"episodic/{name}"
                self._metrics_buffer[metric_key].append(done_metrics)
                self._episodic_metrics_updated.add(metric_key)

        # Also log episodic metrics as step averages
        for name, metric in metrics.items():
            arr = np.asarray(metric)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            # Compute step average across all environments, divided by episode length
            if name == 'length':
                # Don't divide length by itself
                step_avg = arr.reshape(-1).mean().item()
            else:
                # Get episode lengths for normalization
                if 'length' in metrics:
                    lengths = np.asarray(metrics['length'])
                    # Avoid division by zero
                    lengths = np.maximum(lengths, 1.0)
                    # Compute step average by dividing metric by episode length
                    normalized_metric = arr / lengths
                    step_avg = normalized_metric.reshape(-1).mean().item()
                else:
                    # Fallback if length not available
                    step_avg = arr.reshape(-1).mean().item()

            self._metrics_buffer[f"training/{name}"].append(step_avg)

        self.maybe_log_metrics()

    def update_train_metrics(self, metrics, env_steps):
        """Update training metrics buffer and log if needed.

        Args:
          metrics: Dictionary of training metrics (losses, lambda values, etc.)
          env_steps: Current environment step count
        """
        self._num_steps = int((np.uint64(env_steps.hi) << 32) + np.uint64(env_steps.lo))

        # Add metrics to buffer
        for name, metric in metrics.items():
            arr = np.asarray(metric)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            value = arr.reshape(-1).mean().item()
            self._metrics_buffer[f"training/{name}"].append(value)
        self.maybe_log_metrics()

    def maybe_log_metrics(self, pad=35):
        """Log metrics to console."""
        # Log if enough steps have passed
        if self._num_steps - self._last_log_steps < self._steps_between_logging:
            return
        self._last_log_steps = self._num_steps
        self._log_count += 1
        log_string = (
            f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
        )
        mean_metrics = {}
        for metric_name in self._metrics_buffer:
            if len(self._metrics_buffer[metric_name]) > 0:
                if not metric_name.startswith("episodic/") or metric_name in self._episodic_metrics_updated:
                    avg = np.mean(self._metrics_buffer[metric_name])
                    mean_metrics[metric_name] = avg
                    log_string += (f"{f'Training {metric_name}:':>{pad}} {avg:.4f}\n")

        # Clear the updated episodic metrics set after logging
        self._episodic_metrics_updated.clear()

        logging.info(log_string)
        self._progress_fn(int(self._num_steps), mean_metrics)
