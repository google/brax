# Copyright 2021 The Brax Authors.
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

# pylint:disable=g-import-not-at-top
"""Tools for exporting brax policies to other frameworks."""

from typing import Any, Callable
import warnings


def to_tf_model(path: str, inference_fn: Callable[..., Any], *trace_args):
  """Exports a brax inference function to a tensorflow saved model."""
  try:
    from jax.experimental import jax2tf
    import tensorflow as tf
  except ImportError:
    warnings.warn("""to_tf_model requires tensorflow.  Please run
    `pip install tensorflow` for this function to work.""")
    raise
  model = tf.Module()
  model.f = tf.function(jax2tf.convert(inference_fn), autograph=False)
  # for input tracing so that the model has the correct shapes
  model.f(*trace_args)
  tf.saved_model.save(model, path)
