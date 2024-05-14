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

# pylint:disable=g-multiple-import
"""Functions for scanning pytrees given an order defined by a system."""
from typing import Callable, Sequence, TypeVar

from brax.base import QD_WIDTHS, Q_WIDTHS, System
import jax
from jax import numpy as jp

L = TypeVar('L')
Q = TypeVar('Q')
D = TypeVar('D')
Y = TypeVar('Y')


def _take(obj: Y, idxs: Sequence[int]) -> Y:
  """Takes idxs on any pytree given to it.

  XLA executes x[jp.array([1, 2, 3])] slower than x[1:4], so we detect when
  take indices are contiguous, and convert them to slices.

  Args:
    obj: an input pytree
    idxs: indices to take

  Returns:
    obj pytree with leaves taken by idxs
  """

  def take(x):
    if idxs == list(range(idxs[0], idxs[-1] + 1)):
      x = x[idxs[0] : idxs[-1] + 1]
    else:
      x = x.take(jp.array(idxs), axis=0, mode='wrap')
    return x

  return jax.tree.map(take, obj)


def tree(
    sys: System,
    f: Callable[..., Y],
    in_types: str,
    *args,
    reverse: bool = False,
) -> Y:
  r"""Scan a function over each level of a system while carrying along state.

  f is called once for each level of a system tree.  running `tree` on tall tree
  layouts (like long kinematic chains/ropes) may produce long unrolls that are
  slow to jit.

  Args:
    sys: a brax system defining the scan order
    f: a function to be scanned with the following type signature:\
        def f(y, *args) -> y
       where
         ``y`` is the carry value and return value, in link parent order
         ``*args`` are input arguments with types matching ``in_types``
    in_types: string specifying the type of each input arg:
        'l' is an input to be split according to link ranges
        'q' is an input to be split according to q ranges
        'd' is an input to be split according to qd ranges
    *args: the input arguments corresponding to ``in_types``
    reverse: if True, scans up the system tree from leaves to root, otherwise
      root to leaves

  Returns:
    The stacked outputs of ``f``, matching the system link order.
  """
  if len(args) != len(in_types):
    return ValueError('len(args) must match len(in_types)')

  depth_fn = lambda i, p=sys.link_parents: p[i] + 1 and 1 + depth_fn(p[i])
  q_idx, qd_idx, depth_idxs = 0, 0, []
  for i, t in enumerate(sys.link_types):
    depth = depth_fn(i)
    while depth >= len(depth_idxs):
      depth_idxs.append({'l': [], 'q': [], 'd': []})
    depth_idxs[depth]['l'].append(i)
    depth_idxs[depth]['q'].extend(range(q_idx, q_idx + Q_WIDTHS[t]))
    depth_idxs[depth]['d'].extend(range(qd_idx, qd_idx + QD_WIDTHS[t]))
    q_idx, qd_idx = q_idx + Q_WIDTHS[t], qd_idx + QD_WIDTHS[t]
  y, ys = None, []

  if reverse:
    for depth in range(len(depth_idxs) - 1, -1, -1):
      in_args = [_take(a, depth_idxs[depth][t]) for a, t in zip(args, in_types)]

      if y is not None:
        link_idxs = depth_idxs[depth]['l']
        parent_idxs = [sys.link_parents[i] for i in depth_idxs[depth + 1]['l']]
        parent_map = [link_idxs.index(p) for p in parent_idxs]

        def index_sum(x, b=(len(link_idxs),), p=jp.array(parent_map)):
          return jp.zeros(b + x.shape[1:]).at[p].add(x)

        y = jax.tree.map(index_sum, y)

      y = f(y, *in_args)
      ys.insert(0, y)
  else:
    for depth in range(len(depth_idxs)):
      in_args = [_take(a, depth_idxs[depth][t]) for a, t in zip(args, in_types)]

      if y is not None:
        parent_idxs = [sys.link_parents[i] for i in depth_idxs[depth]['l']]
        parent_map = [depth_idxs[depth - 1]['l'].index(p) for p in parent_idxs]
        y = _take(y, parent_map)

      y = f(y, *in_args)
      ys.append(y)

  y = jax.tree.map(lambda *x: jp.concatenate(x), *ys)

  # we concatenated results out of order, so put back in order if needed
  order = sum([d['l'] for d in depth_idxs], [])
  if order != list(range(len(order))):
    y = _take(y, [order.index(i) for i in range(len(order))])

  return y


def link_types(
    sys: System, f: Callable[..., Y], in_types: str, out_types: str, *args
) -> Y:
  r"""Scan a function over System link type ranges.

  Args:
    sys: system defining the kinematic tree and other properties
    f: a function to be scanned with the following type signature:\
          def f(typ, *args) -> y
        where
          ``typ`` is the actuator, link type string
          ``*args`` are input arguments with types matching ``in_types``
          ``y`` is an output arguments with types matching ``out_type``
    in_types: string specifying the type of each input arg:
        'l' is an input to be split according to link ranges
        'q' is an input to be split according to q ranges
        'd' is an input to be split according to qd ranges
    out_types: string specifying the types of the outputs
    *args: the input arguments corresponding to ``in_types``

  Returns:
    The stacked outputs of ``f`` matching the system link order.
  """
  q_idx, qd_idx, typ_order_idxs = 0, 0, []
  typ_order = sorted(set(sys.link_types), key=sys.link_types.find)
  for i, t in enumerate(sys.link_types):
    order = typ_order.index(t)
    while order >= len(typ_order_idxs):
      typ_order_idxs.append({'l': [], 'q': [], 'd': []})
    typ_order_idxs[order]['l'].append(i)
    typ_order_idxs[order]['q'].extend(range(q_idx, q_idx + Q_WIDTHS[t]))
    typ_order_idxs[order]['d'].extend(range(qd_idx, qd_idx + QD_WIDTHS[t]))
    q_idx, qd_idx = q_idx + Q_WIDTHS[t], qd_idx + QD_WIDTHS[t]

  ys = []

  for typ, typ_idxs in zip(typ_order, typ_order_idxs):
    in_args = [_take(a, typ_idxs[t]) for a, t in zip(args, in_types)]
    ys.append(f(typ, *in_args))

  y = jax.tree.map(lambda *x: jp.concatenate(x), *ys)

  # we concatenated results out of order, so put back in order if needed

  ys = [y] if len(out_types) == 1 else y
  out_ys = []
  for idxs, ot in enumerate(out_types):
    order = sum([t[ot] for t in typ_order_idxs], [])
    if order != list(range(len(order))):
      out_ys.append(
          _take(ys[idxs], [order.index(i) for i in range(len(order))])
      )
    else:
      out_ys.append(ys[idxs])
  y = out_ys[0] if len(out_types) == 1 else out_ys

  return y
