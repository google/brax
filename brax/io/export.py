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

# pylint:disable=g-import-not-at-top
"""Tools for exporting brax policies to other frameworks."""

from collections import abc
from typing import Any, Callable
import warnings


def _fix_frozen(d):
  """Changes any mappings (e.g. frozendict) back to dict."""
  if isinstance(d, list):
    return [_fix_frozen(v) for v in d]
  elif isinstance(d, tuple):
    return tuple(_fix_frozen(v) for v in d)
  elif not isinstance(d, abc.Mapping):
    return d
  d = dict(d)
  for k, v in d.items():
    d[k] = _fix_frozen(v)
  return d


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
  # saved_model doesn't recognize flax FrozenDict - convert them back to dict
  trace_args = _fix_frozen(trace_args)
  # for input tracing so that the model has the correct shapes
  model.f(*trace_args)
  tf.saved_model.save(model, path)
