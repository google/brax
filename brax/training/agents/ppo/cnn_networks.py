"""
Network implementations
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


class CNN(linen.Module):
  """CNN module.
  Warning: this expects the images to be 3D; convention NHWC
  num_filters: the number of filters per layer
  kernel_sizes: also per layer
  """
  num_filters: Sequence[int]
  kernel_sizes: Sequence[Tuple]
  strides: Sequence[Tuple]
  activation: ActivationFn = linen.relu
  use_bias: bool = True

  @linen.compact
  def __call__(self, data: jp.ndarray):
    hidden = data
    for i, (num_filter, kernel_size, stride) in enumerate(
      zip(self.num_filters, self.kernel_sizes, self.strides)):
      
      hidden = linen.Conv(
          num_filter,
          kernel_size=kernel_size,
          strides=stride,
          use_bias=self.use_bias)(
              hidden)
      
      hidden = self.activation(hidden)
    return hidden


class VisionMLP(linen.Module):
  # Apply a CNN backbone then an MLP.
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: dict):
    natureCNN = partial(CNN,
                        num_filters=[32, 64, 64],
                        kernel_sizes=[(8, 8), (4, 4), (3, 3)],
                        strides=[(4, 4), (2, 2), (1, 1)],
                        activation=linen.relu,
                        use_bias=False)
    cnn_outs = [natureCNN()(data[key]) for key in data.keys() if key.startswith('pixels/')]
    cnn_outs = [jp.mean(cnn_out, axis=(-2, -3)) for cnn_out in cnn_outs]
    if 'state' in data:
      cnn_outs.append(data['state'])

    hidden = jp.concatenate(cnn_outs, axis=-1)
    return networks.MLP(layer_sizes=self.layer_sizes,
                        activation=self.activation,
                        kernel_init=self.kernel_init,
                        activate_final=self.activate_final,
                        layer_norm=self.layer_norm)(hidden)
