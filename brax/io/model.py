# Copyright 2026 The Brax Authors.
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

import pickle
from typing import Any
import warnings

from etils import epath
import jax
import msgpack
import numpy as np


class SecurityWarning(UserWarning):
  """Warning category for insecure model loading."""

  pass


def _encode_pytree(obj: Any) -> Any:
  """Recursively converts a Pytree into msgpack-compatible types."""
  if isinstance(obj, (jax.Array, np.ndarray)):
    # Standard metadata-preserving array format
    return {
        '__type__': 'array',
        'data': obj.tobytes(),
        'shape': obj.shape,
        'dtype': str(obj.dtype),
    }
  # Handle flax.struct.dataclass and NamedTuples (like RunningStatisticsState)
  if hasattr(obj, '__dict__') and hasattr(obj, '_asdict'):
    return {
        '__type__': obj.__class__.__name__,
        'data': {k: _encode_pytree(v) for k, v in obj._asdict().items()},
    }
  # Handle nested containers
  if isinstance(obj, dict):
    return {k: _encode_pytree(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return {
        '__type__': obj.__class__.__name__,
        'data': [_encode_pytree(x) for x in obj],
    }
  return obj


def _decode_pytree(obj: Any) -> Any:
  """Reconstructs Pytree types from serialized dictionaries."""
  if isinstance(obj, dict):
    type_name = obj.get('__type__')
    # Reconstruct Arrays
    if type_name == 'array':
      return jax.numpy.frombuffer(obj['data'], dtype=obj['dtype']).reshape(
          obj['shape']
      )

    # Reconstruct specialized Brax/Flax types
    data = obj.get('data')
    if type_name == 'RunningStatisticsState':
      from brax.training.acme import running_statistics

      return running_statistics.RunningStatisticsState(**_decode_pytree(data))
    if type_name == 'UInt64':
      from brax.training import types

      return types.UInt64(**_decode_pytree(data))

    # Reconstruct containers
    if type_name == 'tuple':
      return tuple(_decode_pytree(x) for x in data)
    if type_name == 'list':
      return [_decode_pytree(x) for x in data]

    # Generic nested dicts
    return {k: _decode_pytree(v) for k, v in (data if data else obj).items()}

  return obj


def save_params(path: str, params: Any):
  """Saves parameters safely using msgpack."""
  encoded = _encode_pytree(params)
  with epath.Path(path).open('wb') as fout:
    fout.write(msgpack.packb(encoded))


def load_params(path: str, allow_pickle: bool = False) -> Any:
  """Loads parameters safely, with a security-gated legacy path."""
  with epath.Path(path).open('rb') as fin:
    buf = fin.read()

  if buf.startswith(b'\x80'):  # Pickle Protocol 2+ Header
    if not allow_pickle:
      raise RuntimeError(
          'SECURITY ERROR: Insecure pickle file detected. For security reasons,'
          ' loading is blocked. Use allow_pickle=True if you trust the source.'
      )

    warnings.warn(
        'SECURITY WARNING: Loading legacy pickle files is insecure and '
        'deprecated. Please migrate your models to the new secure format.',
        category=SecurityWarning,
    )
    return pickle.loads(buf)

  return _decode_pytree(msgpack.unpackb(buf))
