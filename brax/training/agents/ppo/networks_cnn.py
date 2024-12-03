"""
Network implementations
"""

from functools import partial
from typing import Any, Callable, Sequence

from flax import linen
import jax
import jax.numpy as jp

from brax.training import networks

ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]


class VisionMLP(linen.Module):
  """ 
  Applies a CNN backbone then an MLP.
  
  The CNN architecture originates from the paper:
  "Human-level control through deep reinforcement learning",
  Nature 518, no. 7540 (2015): 529-533
  """
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  layer_norm: bool = False
  normalise_channels: bool = False
  state_obs_key: str = ""

  @linen.compact
  def __call__(self, data: dict):
    pixels_hidden = {k: v for k, v in data.items() if k.startswith("pixels/")}
    if self.normalise_channels:
      # Calculates shared statistics over an entire 2D image.
      image_layernorm = partial(
        linen.LayerNorm, use_bias=False, use_scale=False, reduction_axes=(-1, -2)
      )
      def ln_per_chan(v: jax.Array):
        normalised = [image_layernorm()(v[..., chan]) for chan in range(v.shape[-1])]
        return jp.stack(normalised, axis=-1)

      pixels_hidden = jax.tree.map(ln_per_chan, pixels_hidden)

    natureCNN = partial(
      networks.CNN,
      num_filters=[32, 64, 64],
      kernel_sizes=[(8, 8), (4, 4), (3, 3)],
      strides=[(4, 4), (2, 2), (1, 1)],
      activation=linen.relu,
      use_bias=False,
    )
    cnn_outs = [natureCNN()(pixels_hidden[key]) for key in pixels_hidden]
    cnn_outs = [jp.mean(cnn_out, axis=(-2, -3)) for cnn_out in cnn_outs]
    if self.state_obs_key:
      cnn_outs.append(
        data[self.state_obs_key]
      )  # TODO: Try with dedicated state network

    hidden = jp.concatenate(cnn_outs, axis=-1)
    return networks.MLP(
      layer_sizes=self.layer_sizes,
      activation=self.activation,
      kernel_init=self.kernel_init,
      activate_final=self.activate_final,
      layer_norm=self.layer_norm,
    )(hidden)
