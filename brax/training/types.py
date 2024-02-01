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

"""Brax training types."""

from typing import Any, Mapping, NamedTuple, Tuple, TypeVar

from brax.training.acme.types import NestedArray
import jax.numpy as jnp

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
Observation = jnp.ndarray
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


def identity_observation_preprocessor(observation: Observation,
                                      preprocessor_params: PreprocessorParams):
  del preprocessor_params
  return observation


class NetworkFactory(Protocol[NetworkType]):

  def __call__(
      self,
      observation_size: int,
      action_size: int,
      preprocess_observations_fn:
      PreprocessObservationFn = identity_observation_preprocessor
  ) -> NetworkType:
    pass
