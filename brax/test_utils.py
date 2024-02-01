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
"""Some shared test functions."""
import copy
import time
from typing import Iterable, Tuple

from brax.base import System
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco
import numpy as np


def load_fixture_mujoco(path: str) -> mujoco.MjModel:
  full_path = epath.resource_path('brax') / f'test_data/{path}'
  if not full_path.exists():
    full_path = epath.resource_path('brax') / f'envs/assets/{path}'
  xml = mjcf.fuse_bodies(full_path.read_text())
  model = mujoco.MjModel.from_xml_string(xml)

  return model


def _normalize_q(model: mujoco.MjModel, q: np.ndarray):
  """Normalizes the quaternion part of q."""
  q = np.array(q)
  q_idx = 0
  for typ in model.jnt_type:
    q_dim = 7 if typ == 0 else 1
    if typ == 0:
      q[q_idx + 3:q_idx + 7] = (
          q[q_idx + 3:q_idx + 7] / np.linalg.norm(q[q_idx + 3:q_idx + 7]))
    q_idx += q_dim
  return q


def sample_mujoco_states(
    path: str, count: int = 500, modulo: int = 20, force_pgs: bool = False,
    random_init: bool = False, random_q_scale: float = 1.0,
    random_qd_scale: float = 0.1, vel_to_local: bool = True, seed: int = 42
) -> Iterable[Tuple[mujoco.MjData, mujoco.MjData]]:
  """Samples count / modulo states from mujoco for comparison."""
  np.random.seed(seed)
  model = load_fixture_mujoco(path)
  model.opt.iterations = 50  # return to default for high-precision comparison
  if force_pgs:
    model.opt.solver = 0
  data = mujoco.MjData(model)
  # give a little kick to avoid symmetry
  data.qvel = np.random.uniform(low=-0.01, high=0.01, size=(model.nv,))
  if random_init:
    data.qpos = np.random.uniform(model.nq) * random_q_scale
    data.qpos = _normalize_q(model, data.qpos)
    data.qvel = np.random.uniform(size=(model.nv,)) * random_qd_scale
  for i in range(count):
    before = copy.deepcopy(data)
    mujoco.mj_step(model, data)
    if i % modulo == 0:
      # hijack subtree_angmom, subtree_linvel (unused) to store xang, xvel
      for i in range(model.nbody):
        vel = np.zeros((6,))
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY.value, i, vel, vel_to_local)
        data.subtree_angmom[i] = vel[:3]
        data.subtree_linvel[i] = vel[3:]
      yield before, data


def load_fixture(path: str) -> System:
  full_path = epath.resource_path('brax') / f'test_data/{path}'
  if not full_path.exists():
    full_path = epath.resource_path('brax') / f'envs/assets/{path}'
  sys = mjcf.load(full_path)

  return sys


def benchmark(
    name: str, init_fn, step_fn, batch_size: int = 256, length: int = 1000
) -> float:
  """Reports jit time and op time for a function."""

  @jax.jit
  def run_batch(seed: jax.Array):
    rngs = jax.random.split(jax.random.PRNGKey(seed), batch_size)  # pytype: disable=wrong-arg-types  # jax-ndarray
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

  return batch_sps  # pytype: disable=bad-return-type  # jnp-type
