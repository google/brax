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

"""Flax-style Dense module with Spectral Normalization.

Reference:
  Dense: https://github.com/google/flax/blob/main/flax/linen/linear.py
  Spectral Normalization:
    - https://arxiv.org/abs/1802.05957
    - https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/spectral_norm.py
"""
from typing import Any, Callable, Tuple

from brax.training.types import PRNGKey
from flax import linen
from flax.linen.initializers import lecun_normal, normal, zeros
from jax import lax
import jax.numpy as jnp


Array = Any
Shape = Tuple[int]
Dtype = Any


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.
  This specialized function exists for numerical stability reasons.
  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.
  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class SNDense(linen.Module):
  """A linear transformation applied over the last dimension of the input
  with spectral normalization (https://arxiv.org/abs/1802.05957).

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    eps: The constant used for numerical stability.
    n_steps: How many steps of power iteration to perform to approximate the
      singular value of the input.
  """
  features: int
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  eps: float = 1e-4
  n_steps: int = 1

  @linen.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = jnp.asarray(kernel, self.dtype)

    """for spectral normalization"""
    kernel_shape = kernel.shape
    # Handle scalars.
    if kernel.ndim <= 1:
      raise ValueError("Spectral normalization is not well defined for "
                       "scalar inputs.")
    # Handle higher-order tensors.
    elif kernel.ndim > 2:
      kernel = jnp.reshape(kernel, [-1, kernel.shape[-1]])
    key = self.make_rng('sing_vec')
    u0_state = self.variable('sing_vec', 'u0', normal(stddev=1.), key, (1, kernel.shape[-1]))
    u0 = u0_state.value

    # Power iteration for the weight's singular value.
    for _ in range(self.n_steps):
      v0 = _l2_normalize(jnp.matmul(u0, kernel.transpose([1, 0])), eps=self.eps)
      u0 = _l2_normalize(jnp.matmul(v0, kernel), eps=self.eps)

    u0 = lax.stop_gradient(u0)
    v0 = lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, kernel), jnp.transpose(u0))[0, 0]

    kernel /= sigma
    kernel = kernel.reshape(kernel_shape)

    u0_state.value = u0

    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y
