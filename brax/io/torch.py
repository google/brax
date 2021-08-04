""" Generic functions to convert Jax DeviceArrays into PyTorch Tensors and vice-versa.
"""
import warnings
from functools import singledispatch
from typing import Any, Union, Dict
from collections import abc

from jax._src import dlpack as jax_dlpack
from jaxlib.xla_extension import DeviceArray

try:
    import torch
except ImportError:
    warnings.warn(
        "to_torch requires PyTorch. Please run `pip install torch` for this function "
        "to work."
    )
    raise

from torch import Tensor
from torch.utils import dlpack as torch_dlpack
Device = Union[str, torch.device]


@singledispatch
def torch_to_jax(value: Any) -> Any:
    """Converts values to JAX tensors."""
    # Don't do anything by default, and when a handler is registered for this type of
    # value, it gets used to convert it to a Jax DeviceArray.
    # NOTE: The alternative would be to raise an error when an unsupported value is
    # encountered:
    # raise NotImplementedError(f"Don't know how to convert {v} to a Jax tensor")
    return value


@torch_to_jax.register(Tensor)
def _tensor_to_jax(value: Tensor) -> DeviceArray:
    """Converts a PyTorch Tensor into a Jax DeviceArray."""
    tensor = torch_dlpack.to_dlpack(value)
    tensor = jax_dlpack.from_dlpack(tensor)
    return tensor


@torch_to_jax.register(abc.Mapping)
def _torch_dict_to_jax(
    value: Dict[str, Union[Tensor, Any]]
) -> Dict[str, Union[DeviceArray, Any]]:
    """Converts a dict of PyTorch tensors into a dict of Jax DeviceArrays."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})


@singledispatch
def jax_to_torch(value: Any, device: Device = None) -> Any:
    """Convert JAX values to PyTorch Tensors.

    By default, the returned tensors are on the same device as the Jax inputs, but if
    `device` is passed, the tensors will be moved to that device.
    """
    # Don't do anything by default, and when a handler is registered for this type of
    # value, it gets used to convert it to a torch tensor.
    # NOTE: The alternative would be to raise an error when an unsupported value is
    # encountered:
    # raise NotImplementedError(f"Don't know how to convert {v} to a Torch tensor")
    return value


@jax_to_torch.register(DeviceArray)
def _devicearray_to_tensor(value: DeviceArray, device: Device = None) -> Tensor:
    """Converts a Jax DeviceArray into PyTorch Tensor."""
    dpack = jax_dlpack.to_dlpack(value)
    tensor = torch_dlpack.from_dlpack(dpack)
    if device:
        return tensor.to(device=device)
    return tensor


@jax_to_torch.register(abc.Mapping)
def _jax_dict_to_torch(
    value: Dict[str, Union[DeviceArray, Any]], device: Device = None
) -> Dict[str, Union[Tensor, Any]]:
    """Converts a dict of Jax DeviceArrays into a dict of PyTorch tensors."""
    return type(value)(**{k: jax_to_torch(v, device=device) for k, v in value.items()})
