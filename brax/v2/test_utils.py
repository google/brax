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

# pylint:disable=g-multiple-import
"""Some shared test functions."""
import copy
import time
from typing import Iterable, Tuple

from brax.v2.base import System
from brax.v2.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco
import numpy as np


def load_fixture_mujoco(path: str) -> mujoco.MjModel:
  path = epath.resource_path('brax') / f'v2/test_data/{path}'
  xml = mjcf.fuse_bodies(path.read_text())
  model = mujoco.MjModel.from_xml_string(xml)

  return model


def sample_mujoco_states(
    path: str, count: int = 500, modulo: int = 20, force_pgs: bool = False
) -> Iterable[Tuple[mujoco.MjData, mujoco.MjData]]:
  """Samples count / modulo states from mujoco for comparison."""
  path = epath.resource_path('brax') / f'v2/test_data/{path}'
  xml = mjcf.fuse_bodies(path.read_text())
  model = mujoco.MjModel.from_xml_string(xml)
  if force_pgs:
    model.opt.solver = 0
  data = mujoco.MjData(model)
  # give a little kick to avoid symmetry
  data.qvel = np.random.uniform(low=-0.01, high=0.01, size=(model.nv,))
  for i in range(count):
    before = copy.deepcopy(data)
    mujoco.mj_step(model, data)
    if i % modulo == 0:
      # hijack subtree_angmom, subtree_linvel (unused) to store xang, xvel
      for i in range(model.nbody):
        vel = np.zeros((6,))
        mujoco.mj_objectVelocity(model, data, 2, i, vel, 1)
        data.subtree_angmom[i] = vel[:3]
        data.subtree_linvel[i] = vel[3:]
      yield before, data


def load_fixture(path: str) -> System:
  path = epath.resource_path('brax') / f'v2/test_data/{path}'
  sys = mjcf.load(path)

  return sys


def benchmark(
    name: str, init_fn, step_fn, batch_size: int = 256, length: int = 1000
) -> float:
  """Reports jit time and op time for a function."""

  @jax.jit
  def run_batch(seed: jp.ndarray):
    rngs = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    init_state = jax.vmap(init_fn)(rngs)

    @jax.vmap
    def run(state):
      def step(state, _):
        state = step_fn(state)
        return state, ()

      return jax.lax.scan(step, state, (), length=length)

    return run(init_state)

  times = []
  for i in range(5):
    t = time.time()
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), run_batch(i))
    times.append(time.time() - t)
  op_time = jp.mean(jp.array(times[1:]))  # ignore JIT time

  batch_sps = batch_size * length / op_time
  print(
      f'{name} jit time: {times[0] - op_time:.3f}s, '
      f'op time: {op_time:.3f}s, '
      f'{batch_sps:,.0f} batch steps/sec on '
      f'{jax.devices()[0].device_kind}'
  )

  return batch_sps
