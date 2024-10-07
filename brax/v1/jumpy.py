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

# pylint:disable=redefined-builtin
"""Numpy backend for JAX that is called for non-jit/non-jax arrays."""

import builtins
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings

import jax
from jax import custom_jvp
from jax import numpy as jnp
import numpy as onp

ndarray = Union[onp.ndarray, jnp.ndarray]  # pylint:disable=invalid-name
tree_map = jax.tree_util.tree_map  # works great with jax or numpy as-is
pi = onp.pi
inf = onp.inf
float32 = onp.float32
int32 = onp.int32

warnings.warn(
    'brax.v1 is deprecated and will be removed in a future release.',
    DeprecationWarning)


def _in_jit() -> bool:
  """Returns true if currently inside a jax.jit call or jit is disabled."""
  if jax.config.jax_disable_jit:
    return True

  if jax.__version_info__ <= (0, 4, 33):
    return 'DynamicJaxprTrace' in str(
        jax.core.thread_local_state.trace_state.trace_stack
    )

  return jax.core.unsafe_am_i_under_a_jit_DO_NOT_USE()


def _which_np(*args):
  checker = lambda a: (
      isinstance(a, (jnp.ndarray, jax.interpreters.batching.BatchTracer)) and
      not isinstance(a, onp.ndarray))
  if builtins.any(jax.tree_util.tree_leaves(tree_map(checker, args))):
    return jnp
  return onp


F = TypeVar('F', bound=Callable)


def vmap(fun: F, include: Optional[Sequence[bool]] = None) -> F:
  """Creates a function which maps ``fun`` over argument axes."""
  if _in_jit():
    in_axes = 0
    if include:
      in_axes = [0 if inc else None for inc in include]
    return jax.vmap(fun, in_axes=in_axes)

  def _batched(*args, include=include):
    if include is not None and len(include) != len(args):
      raise RuntimeError('Len of `args` list must match length of `include`.')

    # by default, vectorize over every arg
    if include is None:
      include = [True for _ in args]

    # determine number of parallel evaluations to unroll into serial evals
    batch_size = None
    for a, inc in zip(args, include):
      if inc:
        flat_args, _ = jax.tree_util.tree_flatten(a)
        batch_size = flat_args[0].shape[0]
        break

    # rebuild b_args for each serial evaluation
    rets = []
    for b_idx in range(batch_size):
      b_args = []
      for a, inc in zip(args, include):
        if inc:
          b_args.append(take(a, b_idx))  # pytype: disable=wrong-arg-types  # jax-ndarray
        else:
          b_args.append(a)
      rets.append(fun(*b_args))

    np = _which_np(*rets)
    return jax.tree_util.tree_map(lambda *x: np.stack(x), *rets)

  return _batched


Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  """Scan a function over leading array axes while carrying along state."""
  if _in_jit():
    return jax.lax.scan(f, init, xs, length, reverse, unroll)
  else:
    xs_flat, xs_tree = jax.tree_util.tree_flatten(xs)
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [x[i] for x in xs_flat]
      carry, y = f(carry, jax.tree_util.tree_unflatten(xs_tree, xs_slice))
      ys.append(y)
    mry = maybe_reversed(ys)
    np = _which_np(*mry)
    stacked_y = jax.tree_util.tree_map(lambda *y: np.stack(y), *mry)
    return carry, stacked_y


def while_loop(cond_fun: Callable[[X], Any], body_fun: Callable[[X], X],
               init_val: X) -> X:
  """Call body_fun while cond_fun is true, starting with init_val."""
  if _in_jit():
    return jax.lax.while_loop(cond_fun, body_fun, init_val)
  else:
    val = init_val
    while cond_fun(val):
      val = body_fun(val)
    return val


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, X], X],
              init_val: X) -> X:
  """Call body_fun over range from lower to upper, starting with init_val."""
  if _in_jit():
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)
  else:
    val = init_val
    for i in range(lower, upper):
      val = body_fun(i, val)
    return val


def take(
    tree: Any, i: Union[int, ndarray, Sequence[int]], axis: int = 0
) -> Any:
  """Returns tree sliced by i."""
  np = _which_np(i)
  if isinstance(i, list) or isinstance(i, tuple):
    i = np.array(i, dtype=int)
  return jax.tree_util.tree_map(lambda x: np.take(x, i, axis=axis, mode='clip'),
                                tree)


def norm(x: ndarray,
         axis: Optional[Union[Tuple[int, ...], int]] = None) -> ndarray:
  """Returns the array norm."""
  return _which_np(x, axis).linalg.norm(x, axis=axis)


def index_update(x: ndarray, idx: ndarray, y: ndarray) -> ndarray:
  """Pure equivalent of x[idx] = y."""
  if _which_np(x, idx, y) is jnp:
    return jnp.array(x).at[idx].set(jnp.array(y))
  x = onp.copy(x)
  x[idx] = y
  return x


def safe_norm(x: ndarray,
              axis: Optional[Union[Tuple[int, ...], int]] = None) -> ndarray:
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.

  Avoids a poorly defined gradient for jnp.linal.norm(0) see
  https://github.com/google/jax/issues/3058 for details
  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm

  Returns:
    Norm of the array x.
  """
  np = _which_np(x)
  if np is jnp:
    is_zero = jnp.allclose(x, 0.)
    # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis)
    n = jnp.where(is_zero, 0., n)
  else:
    n = onp.linalg.norm(x, axis=axis)
  return n


def expand_dims(x: ndarray,
                axis: Union[Tuple[int, ...], int] = 0) -> ndarray:
  """Increases batch dimensionality along axis."""

  return _which_np(x).expand_dims(x, axis=axis)


def any(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether any array element along a given axis evaluates to True."""
  return _which_np(a).any(a, axis=axis)


def all(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether all array elements along a given axis evaluate to True."""
  return _which_np(a).all(a, axis=axis)


def mean(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Compute the arithmetic mean along the specified axis."""
  return _which_np(a).mean(a, axis=axis)


def var(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Compute the variance along the specified axis."""
  return _which_np(a).var(a, axis=axis)


def arange(start: int, stop: int) -> ndarray:
  """Return evenly spaced values within a given interval."""
  return _which_np().arange(start, stop)


def dot(x: ndarray, y: ndarray) -> ndarray:
  """Returns dot product of two arrays."""
  return _which_np(x, y).dot(x, y)


def outer(a: ndarray, b: ndarray) -> ndarray:
  """Compute the outer product of two vectors."""
  return _which_np(a, b).outer(a, b)


def matmul(x1: ndarray, x2: ndarray) -> ndarray:
  """Matrix product of two arrays."""
  return _which_np(x1, x2).matmul(x1, x2)


def inv(a: ndarray) -> ndarray:
  """Compute the (multiplicative) inverse of a matrix."""
  return _which_np(a).linalg.inv(a)


def roll(x: ndarray, shift, axis=None) -> ndarray:
  """Rolls array elements along a given axis."""
  return _which_np(x).roll(x, shift, axis=axis)


def square(x: ndarray) -> ndarray:
  """Return the element-wise square of the input."""
  return _which_np(x).square(x)


def tile(x: ndarray, reps: Union[Tuple[int, ...], int]) -> ndarray:
  """Construct an array by repeating A the number of times given by reps."""
  return _which_np(x).tile(x, reps)


def repeat(a: ndarray, repeats: Union[int, ndarray], *args,
           **kwargs) -> ndarray:
  """Repeat elements of an array."""
  return _which_np(a, repeats).repeat(a, repeats=repeats, *args, **kwargs)


def floor(x: ndarray) -> ndarray:
  """Returns the floor of the input, element-wise.."""
  return _which_np(x).floor(x)


def cross(x: ndarray, y: ndarray) -> ndarray:
  """Returns cross product of two arrays."""
  return _which_np(x, y).cross(x, y)


def sin(angle: ndarray) -> ndarray:
  """Returns trigonometric sine, element-wise."""
  return _which_np(angle).sin(angle)


def cos(angle: ndarray) -> ndarray:
  """Returns trigonometric cosine, element-wise."""
  return _which_np(angle).cos(angle)


def arctan2(x1: ndarray, x2: ndarray) -> ndarray:
  """Returns element-wise arc tangent of x1/x2 choosing the quadrant correctly."""
  return _which_np(x1, x2).arctan2(x1, x2)


def arctanh(x: ndarray) -> ndarray:
  """Returns element-wise arctanh of x."""
  return _which_np(x).arctanh(x)


def tanh(x: ndarray) -> ndarray:
  """Returns element-wise tanh of x."""
  return _which_np(x).tanh(x)


def arccos(x: ndarray) -> ndarray:
  """Trigonometric inverse cosine, element-wise."""
  return _which_np(x).arccos(x)


@custom_jvp
def safe_arccos(x: ndarray) -> ndarray:
  """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
  return _which_np(x).arccos(x)


@safe_arccos.defjvp
def _safe_arccos_jvp(primal, tangent):
  x, = primal
  x_dot, = tangent
  primal_out = safe_arccos(x)
  tangent_out = -x_dot / sqrt(1. - clip(x, -1 + 1e-7, 1 - 1e-7)**2.)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return primal_out, tangent_out


def arcsin(x: ndarray) -> ndarray:
  """Trigonometric inverse sine, element-wise."""
  return _which_np(x).arcsin(x)


@custom_jvp
def safe_arcsin(x: ndarray) -> ndarray:
  """Trigonometric inverse sine, element-wise with safety clipping in grad."""
  return _which_np(x).arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
  x, = primal
  x_dot, = tangent
  primal_out = safe_arccos(x)
  tangent_out = x_dot / sqrt(1. - clip(x, -1 + 1e-7, 1 - 1e-7)**2.)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return primal_out, tangent_out


def logical_not(x: ndarray) -> ndarray:
  """Returns the truth value of NOT x element-wise."""
  return _which_np(x).logical_not(x)


def logical_and(x1: ndarray, x2: ndarray) -> ndarray:
  """Returns the truth value of x1 AND x2 element-wise."""
  return _which_np(x1, x2).logical_and(x1, x2)


def logical_or(x1: ndarray, x2: ndarray) -> ndarray:
  """Returns the truth value of x1 OR x2 element-wise."""
  return _which_np(x1, x2).logical_or(x1, x2)


def multiply(x1: ndarray, x2: ndarray) -> ndarray:
  """Multiply arguments element-wise."""
  return _which_np(x1, x2).multiply(x1, x2)


def minimum(x1: ndarray, x2: ndarray) -> ndarray:
  """Element-wise minimum of array elements."""
  return _which_np(x1, x2).minimum(x1, x2)


def maximum(x1: ndarray, x2: ndarray) -> ndarray:
  """Element-wise maximum of array elements."""
  return _which_np(x1, x2).maximum(x1, x2)


def amin(x: ndarray, *args, **kwargs) -> ndarray:
  """Returns the minimum along a given axis."""
  return _which_np(x).amin(x, *args, **kwargs)


def amax(x: ndarray, *args, **kwargs) -> ndarray:
  """Returns the maximum along a given axis."""
  return _which_np(x).amax(x, *args, **kwargs)


def argmin(x: ndarray, *args, **kwargs) -> ndarray:
  """Returns the argmin along a given axis."""
  return _which_np(x).argmin(x, *args, **kwargs)


def argmax(x: ndarray, *args, **kwargs) -> ndarray:
  """Returns the argmax along a given axis."""
  return _which_np(x).argmax(x, *args, **kwargs)


def exp(x: ndarray) -> ndarray:
  """Returns the exponential of all elements in the input array."""
  return _which_np(x).exp(x)


def sign(x: ndarray) -> ndarray:
  """Returns an element-wise indication of the sign of a number."""
  return _which_np(x).sign(x)


def sum(a: ndarray, axis: Optional[int] = None):
  """Returns sum of array elements over a given axis."""
  return _which_np(a).sum(a, axis=axis)


def random_prngkey(seed: int) -> ndarray:
  """Returns a PRNG key given a seed."""
  if _which_np() is jnp:
    return jax.random.PRNGKey(seed)
  else:
    rng = onp.random.default_rng(seed)
    return rng.integers(low=0, high=2**32, dtype='uint32', size=2)


def random_uniform(rng: ndarray,
                   shape: Tuple[int, ...] = (),
                   low: Optional[float] = 0.0,
                   high: Optional[float] = 1.0) -> ndarray:
  """Sample uniform random values in [low, high) with given shape/dtype."""
  if _which_np(rng) is jnp:
    return jax.random.uniform(rng, shape=shape, minval=low, maxval=high)
  else:
    return onp.random.default_rng(rng).uniform(size=shape, low=low, high=high)


def random_split(rng: ndarray, num: int = 2) -> ndarray:
  """Splits a PRNG key into num new keys by adding a leading axis."""
  if _which_np(rng) is jnp:
    return jax.random.split(rng, num=num)
  else:
    rng = onp.random.default_rng(rng)
    return rng.integers(low=0, high=2**32, dtype='uint32', size=(num, 2))


def randint(rng: ndarray,
            shape: Tuple[int, ...] = (),
            low: Optional[int] = 0,
            high: Optional[int] = 1) -> ndarray:
  """Sample integers in [low, high) with given shape."""
  if _which_np(rng) is jnp:
    return jax.random.randint(rng, shape=shape, minval=low, maxval=high)
  else:
    return onp.random.default_rng(rng).integers(low=low, high=high, size=shape)


def choice(rng: ndarray,
           a: Union[int, Any],
           shape: Tuple[int, ...] = (),
           replace: bool = True,
           p: Optional[Any] = None,
           axis: int = 0) -> ndarray:
  """Generate sample(s) from given array."""
  if _which_np(rng) is jnp:
    return jax.random.choice(
        rng, a, shape=shape, replace=replace, p=p, axis=axis)
  else:
    return onp.random.default_rng(rng).choice(
        a, size=shape, replace=replace, p=p, axis=axis)


def segment_sum(data: ndarray,
                segment_ids: ndarray,
                num_segments: Optional[int] = None) -> ndarray:
  """Computes the sum within segments of an array."""
  if _which_np(data, segment_ids) is jnp:
    s = jax.ops.segment_sum(data, segment_ids, num_segments)
  else:
    if num_segments is None:
      num_segments = onp.amax(segment_ids) + 1
    s = onp.zeros((num_segments,) + data.shape[1:])
    onp.add.at(s, segment_ids, data)
  return s


def top_k(operand: ndarray, k: int) -> ndarray:
  """Returns the ordered top k values and their indices along the last axis of operand."""
  if _which_np(operand) is jnp:
    return jax.lax.top_k(operand, k)  # pytype: disable=bad-return-type  # jax-ndarray
  else:
    top_ind = onp.argpartition(operand, -k)[-k:]
    sorted_ind = top_ind[onp.argsort(-operand[top_ind])]
    return operand[sorted_ind], sorted_ind  # pytype: disable=bad-return-type  # jax-ndarray


def stack(x: List[ndarray], axis=0) -> ndarray:
  """Join a sequence of arrays along a new axis."""
  return _which_np(*x).stack(x, axis=axis)


def concatenate(x: Sequence[ndarray], axis=0) -> ndarray:
  """Join a sequence of arrays along an existing axis."""
  return _which_np(*x).concatenate(x, axis=axis)


def sqrt(x: ndarray) -> ndarray:
  """Returns the non-negative square-root of an array, element-wise."""
  return _which_np(x).sqrt(x)


def where(condition: jax.typing.ArrayLike, x: jax.typing.ArrayLike,
          y: jax.typing.ArrayLike) -> ndarray:
  """Return elements chosen from `x` or `y` depending on `condition`."""
  return _which_np(condition, x, y).where(condition, x, y)


def cond(pred, true_fun: Callable[..., bool], false_fun: Callable[..., bool],
         *operands: Any):
  """Conditionally apply true_fun or false_fun to operands."""
  if _in_jit():
    return jax.lax.cond(pred, true_fun, false_fun, *operands)
  else:
    if pred:
      return true_fun(operands)
    else:
      return false_fun(operands)


def diag(v: ndarray, k: int = 0) -> ndarray:
  """Extract a diagonal or construct a diagonal array."""
  return _which_np(v).diag(v, k)


def clip(a: ndarray, a_min: ndarray, a_max: ndarray) -> ndarray:
  """Clip (limit) the values in an array."""
  return _which_np(a, a_min, a_max).clip(a, a_min, a_max)


def eye(n: int) -> ndarray:
  """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
  return _which_np().eye(n)


def zeros(shape, dtype=float) -> ndarray:
  """Return a new array of given shape and type, filled with zeros."""
  return _which_np().zeros(shape, dtype=dtype)


def zeros_like(a: ndarray) -> ndarray:
  """Return an array of zeros with the same shape and type as a given array."""
  return _which_np(a).zeros_like(a)


def ones(shape, dtype=float) -> ndarray:
  """Return a new array of given shape and type, filled with ones."""
  return _which_np().ones(shape, dtype=dtype)


def ones_like(a: ndarray) -> ndarray:
  """Return an array of ones with the same shape and type as a given array."""
  return _which_np(a).ones_like(a)


def reshape(a: ndarray, newshape: Union[Tuple[int, ...], int]) -> ndarray:
  """Gives a new shape to an array without changing its data."""
  return _which_np(a).reshape(a, newshape)


def atleast_1d(*arys) -> ndarray:
  """Ensure arrays are all at least 1d (dimensions added to beginning)."""
  return _which_np(*arys).atleast_1d(*arys)


def atleast_2d(*arys) -> ndarray:
  """Ensure arrays are all at least 2d (dimensions added to beginning)."""
  return _which_np(*arys).atleast_2d(*arys)


def atleast_3d(*arys) -> ndarray:
  """Ensure arrays are all at least 3d (dimensions added to beginning)."""
  return _which_np(*arys).atleast_3d(*arys)


def array(object: Any, dtype=None) -> ndarray:
  """Creates an array given a list."""
  try:
    np = _which_np(*object)
  except TypeError:
    np = _which_np(object)  # object is not iterable (e.g. primitive type)
  return np.array(object, dtype)


def abs(a: ndarray) -> ndarray:
  """Calculate the absolute value element-wise."""
  return _which_np(a).abs(a)


def meshgrid(*xi,
             copy: bool = True,
             sparse: bool = False,
             indexing: str = 'xy') -> ndarray:
  """Create N-D coordinate matrices from 1D coordinate vectors."""
  if _which_np(xi[0]) is jnp:
    return jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)  # pytype: disable=bad-return-type  # jnp-type
  return onp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
