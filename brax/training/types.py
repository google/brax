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

"""Brax training types."""

from typing import Any, Mapping, NamedTuple, Tuple, TypeVar, Union

from brax.training.acme.types import NestedArray
import flax
import jax
import jax.numpy as jnp
import numpy as np

# Protocol was introduced into typing in Python >=3.8
# via https://www.python.org/dev/peps/pep-0544/
# Before that, its status was DRAFT and available via typing_extensions
try:
  from typing import Protocol  # pylint:disable=g-import-not-at-top
except ImportError:
  from typing_extensions import Protocol  # pylint:disable=g-import-not-at-top

Params = Any
PRNGKey = jnp.ndarray
Metrics = Mapping[str, jnp.ndarray]
Observation = Union[jnp.ndarray, Mapping[str, jnp.ndarray]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]
Action = jnp.ndarray
Extra = Mapping[str, Any]
PolicyParams = Any
PreprocessorParams = Any
PolicyParams = Tuple[PreprocessorParams, Params]
NetworkType = TypeVar('NetworkType')


class Transition(NamedTuple):
  """Container for a transition."""

  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class Policy(Protocol):

  def __call__(
      self,
      observation: Observation,
      key: PRNGKey,
  ) -> Tuple[Action, Extra]:
    pass


class PreprocessObservationFn(Protocol):

  def __call__(
      self,
      observation: Observation,
      preprocessor_params: PreprocessorParams,
  ) -> jnp.ndarray:
    pass


def identity_observation_preprocessor(
    observation: Observation, preprocessor_params: PreprocessorParams
):
  del preprocessor_params
  return observation


class NetworkFactory(Protocol[NetworkType]):

  def __call__(
      self,
      observation_size: ObservationSize,
      action_size: int,
      preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
  ) -> NetworkType:
    pass


@flax.struct.dataclass
class UInt64:
  """Custom 64-bit integer implementation using two 32-bit parts.

  This class implements 64-bit unsigned integer arithmetic using two 32-bit
  parts
  (hi and lo) to work around JAX's limitation of not supporting 64-bit
  unsigned integers directly when jax_enable_x64 is False.
  """

  hi: Union[int, np.ndarray, jax.Array]
  lo: Union[int, np.ndarray, jax.Array]

  def to_numpy(self):
    """Convert UInt64 to numpy uint64."""
    hi_np = np.array(self.hi, dtype=np.uint64)
    lo_np = np.array(self.lo, dtype=np.uint64)
    return (hi_np << np.uint64(32)) | lo_np

  def __post_init__(self):
    """Cast post init."""
    object.__setattr__(self, "hi", jnp.uint32(self.hi))
    object.__setattr__(self, "lo", jnp.uint32(self.lo))

  def __add__(self, other):
    other = _sanitize_uint64_input(other)
    return _add_uint64(self, other)

  def __repr__(self):
    return f"UInt64(hi={self.hi}, lo={self.lo})"

  def __int__(self):
    """Convert UInt64 to Python int."""
    return int(self.to_numpy())


def _sanitize_uint64_input(other: Union[int, np.ndarray, jax.Array, UInt64]):
  """Sanitizes input for UInt64 arithmetic.

  Args:
    other: Input value, either int, np.ndarray or UInt64.

  Returns:
    UInt64 representation of the input.

  Raises:
    NotImplementedError: If the input type is not supported.
  """
  if isinstance(other, (int, np.ndarray, jax.Array)):
    other_lo = other & jnp.array(0xFFFFFFFF, dtype=jnp.uint32)
    other_hi = other >> 32
    other = UInt64(
        hi=jnp.array(other_hi, dtype=jnp.uint32),
        lo=jnp.array(other_lo, dtype=jnp.uint32),
    )
  elif isinstance(other, UInt64):
    pass
  else:
    raise NotImplementedError(f"Cannot perform op on {type(other)} and UInt64.")
  return other


def _add_uint64(a: UInt64, b: UInt64) -> UInt64:
  """Adds two UInt64 numbers together.

  Args:
    a: First UInt64 number
    b: Second UInt64 number

  Returns:
    New UInt64 number representing the sum
  """
  result_lo = a.lo + b.lo
  carry_bit_pattern = (a.lo & b.lo) | ((a.lo | b.lo) & (~result_lo))
  carry = carry_bit_pattern >> 31
  result_hi = a.hi + b.hi + carry
  return UInt64(hi=result_hi, lo=result_lo)
