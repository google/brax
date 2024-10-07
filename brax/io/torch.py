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

"""Functions to convert Jax Arrays into PyTorch Tensors and vice-versa."""
from collections import abc
import functools
from typing import Any, Dict, Union
import warnings

import jax
from jax import dlpack as jax_dlpack

try:
  # pylint:disable=g-import-not-at-top
  import torch
  from torch.utils import dlpack as torch_dlpack
except ImportError:
  warnings.warn(
      "brax.io.torch requires PyTorch. Please run `pip install torch` to use "
      "functions from this module.")
  raise

Device = Union[str, torch.device]


@functools.singledispatch
def torch_to_jax(value: Any) -> Any:
  """Converts PyTorch tensors to JAX arrays.

  Args:
    value: torch tensor

  Returns:
    a JAX array
  """
  del value


@torch_to_jax.register(torch.Tensor)
def _tensor_to_jax(value: torch.Tensor) -> jax.Array:
  """Converts a PyTorch Tensor into a jax.Array."""
  tensor = torch_dlpack.to_dlpack(value)
  tensor = jax_dlpack.from_dlpack(tensor)
  return tensor


@torch_to_jax.register(abc.Mapping)
def _torch_dict_to_jax(
    value: Dict[str, Union[torch.Tensor, Any]]
) -> Dict[str, Union[jax.Array, Any]]:
  """Converts a dict of PyTorch tensors into a dict of jax.Arrays."""
  return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})  # type: ignore


@functools.singledispatch
def jax_to_torch(value: Any, device: Union[Device, None] = None) -> Any:
  """Convert JAX values to PyTorch Tensors.

  Args:
    value: jax array or pytree
    device: device to copy value to (or None to leave on same device)

  Returns:
    Torch tensor on device

  By default, the returned tensors are on the same device as the Jax inputs,
  but if `device` is passed, the tensors will be moved to that device.
  """
  del value, device


@jax_to_torch.register(jax.Array)
def _jaxarray_to_tensor(
    value: jax.Array, device: Union[Device, None] = None
) -> torch.Tensor:
  """Converts a jax.Array into PyTorch Tensor."""
  dpack = jax_dlpack.to_dlpack(value.astype("float32"))
  tensor = torch_dlpack.from_dlpack(dpack)
  if device:
    return tensor.to(device=device)
  return tensor


@jax_to_torch.register(abc.Mapping)
def _jax_dict_to_torch(
    value: Dict[str, Union[jax.Array, Any]],
    device: Union[Device, None] = None) -> Dict[str, Union[torch.Tensor, Any]]:
  """Converts a dict of jax.Arrays into a dict of PyTorch tensors."""
  return type(value)(
      **{k: jax_to_torch(v, device=device) for k, v in value.items()})  # type: ignore
