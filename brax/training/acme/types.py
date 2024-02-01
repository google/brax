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

"""Common types used throughout Acme.

This file was taken from acme and modified to simplify dependencies:

https://github.com/deepmind/acme/blob/master/acme/types.py
"""
from typing import Any, Iterable, Mapping, Union

from brax.training.acme import specs
import jax.numpy as jnp

# Define types for nested arrays and tensors.
NestedArray = jnp.ndarray
NestedTensor = Any

# pytype: disable=not-supported-yet
NestedSpec = Union[
    specs.Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],
]
# pytype: enable=not-supported-yet

Nest = Union[NestedArray, NestedTensor, NestedSpec]
