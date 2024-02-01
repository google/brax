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



"""General purpose metrics writer interface."""

from absl import logging

try:
  from tensorboardX import SummaryWriter  # type: ignore
finally:
  pass



class Writer:
  """General purpose metrics writer."""

  def __init__(self, logdir=''):
    self._writer = SummaryWriter(logdir=logdir)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._writer.close()

  def write_hparams(self, hparams):
    """Writes global hparams."""
    logging.info('Hyperparameters: %s', hparams)
    self._writer.add_hparams(hparams, {})

  def write_scalars(self, step, scalars):
    """Writers scalar metrics."""
    values = [
        f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}'
        for k, v in sorted(scalars.items())
    ]
    logging.info('[%d] %s', step, ', '.join(values))
    for k, v in scalars.items():
      self._writer.add_scalar(k, v, step)
