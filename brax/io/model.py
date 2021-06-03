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

"""Loading/saving of inference functions."""

from typing import Any, Callable
from flax import serialization

from jax.experimental import jax2tf
import tensorflow as tf
from tensorflow.io import gfile

# TODO: would it better to just load/save native jax?


def load(path: str) -> Callable[..., Any]:
  return tf.saved_model.load(path).f


def save(path: str, inference_fn: Callable[..., Any], *trace_args):
  model = tf.Module()
  model.f = tf.function(jax2tf.convert(inference_fn), autograph=False)
  # for input tracing so that the model has the correct shapes
  model.f(*trace_args)
  tf.saved_model.save(model, path)


def save_params(path: str, params: Any):
  """Saves parameters in Flax format."""
  with gfile.GFile(path, 'wb') as fout:
    fout.write(serialization.to_bytes(params))
