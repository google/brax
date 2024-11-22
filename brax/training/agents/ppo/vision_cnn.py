"""Networks.
"""

from functools import partial
from typing import Sequence, Tuple, Any, Callable

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import flax
from flax import linen
import jax
import jax.numpy as jp

ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]

class NatureCNN(linen.Module):
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  layer_norm: bool = False
  dtype: Any = jp.float32

  @linen.compact
  def __call__(self, data: jp.ndarray):
    conv = partial(linen.Conv, use_bias=False, dtype=self.dtype)
    hidden = data

    hidden = conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1')(hidden)
    hidden = self.activation(hidden)

    hidden = conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2')(hidden)
    hidden = self.activation(hidden)

    hidden = conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3')(hidden)
    hidden = self.activation(hidden)

    hidden = jp.mean(hidden, axis=(-2, -3))

    for i, layer_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
        layer_size, kernel_init=self.kernel_init, name=f'dense_{i}')(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden