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

"""Input normalization utils."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def bcast_local_devices(value, local_devices_to_use=1):
  """Broadcasts an object to all local devices."""
  devices = jax.local_devices()[:local_devices_to_use]
  return jax.device_put_replicated(value, devices)


def synchronize_hosts():
  if jax.process_count() == 1:
    return
  # Make sure all processes stay up until the end of main.
  x = jnp.ones([jax.local_device_count()])
  x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
  assert x[0] == jax.device_count()


def _fingerprint(x: Any) -> float:
  sums = jax.tree_map(jnp.sum, x)
  return jax.tree_util.tree_reduce(lambda x, y: x + y, sums)


def is_synchronized(x: Any, axis_name: str) -> jnp.ndarray:
  fp = _fingerprint(x)
  return jax.lax.pmin(
      fp, axis_name=axis_name) == jax.lax.pmax(
          fp, axis_name=axis_name)


def pmap_with_synchro(f: Callable[..., Tuple[Any, ...]],
                      axis_name: str,
                      check_input: bool = True,
                      check_output: bool = True):
  """Pmap f and perform sanity checks on the input and output.

  Args:
    f: The function to be pmapped.
    axis_name: The name of the pmap axis.
    check_input: If True, assert that the first input of f is the same on all
      devices.
    check_output: If True, expect that f's output is a Tuple and assert that the
      first output of f is the same on all devices.

  Returns:
    Pmapped f. The output should not be called in a jitted function or
      assertions will fail.
  """

  def g(*args, **kwargs):
    synchro_input = True
    if check_input:
      synchro_input = is_synchronized(args[0], axis_name)
    outputs = f(*args, **kwargs)
    synchro_output = True
    if check_output:
      synchro_output = is_synchronized(outputs[0], axis_name)
    return outputs, synchro_input, synchro_output

  g = jax.pmap(g, axis_name=axis_name)

  def h(*args, **kwargs):
    output, synchro_input, synchro_output = g(*args, **kwargs)
    assert synchro_input[0]
    assert synchro_output[0]
    return output

  return h
