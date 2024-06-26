import itertools
import os
import time
from pathlib import Path

from brax.envs.wrappers.training import VmapWrapper

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"
import copy
import functools
from typing import Dict, Union, List, Tuple, Callable

import jax
import jax.numpy as jp

from brax import math, base, System, envs
from brax.envs.base import Wrapper, Env, State
from flax import struct


### SYSTEM DEFINITION INTERACTION LOGIC ###

def _traverse_sys(sys: System, attr: List[str], val: Union[jp.ndarray, None]):
    if val is None:
        READ = True
    else:
        READ = False

    def _thunk(sys, attr):
        """"Replaces attributes in sys with val."""
        if not attr:
            return sys
        if len(attr) == 2 and attr[0] == 'geoms':
            geoms = copy.deepcopy(sys.geoms)
            if not hasattr(val, '__iter__'):
                for i, g in enumerate(geoms):
                    if not hasattr(g, attr[1]):
                        continue
                    geoms[i] = g.replace(**{attr[1]: val})
            else:
                sizes = [g.transform.pos.shape[0] for g in geoms]
                g_idx = 0
                for i, g in enumerate(geoms):
                    if not hasattr(g, attr[1]):
                        continue
                    size = sizes[i]
                    geoms[i] = g.replace(**{attr[1]: val[g_idx:g_idx + size].T})
                    g_idx += size

            if READ:
                return geoms
            else:
                return sys.replace(geoms=geoms)

        if len(attr) == 1:

            if READ:
                return getattr(sys, attr[0])
            else:
                return sys.replace(**{attr[0]: val})

        subsys = getattr(sys, attr[0])
        if READ:
            return _thunk(subsys, attr[1:])
        else:
            return sys.replace(**{attr[0]: _thunk(subsys, attr[1:])})

    return _thunk(sys, attr)


def _write_sys(sys, attr, val):
    """"Replaces attributes in sys with val."""
    if not attr:
        return sys
    if len(attr) == 2 and attr[0] == 'geoms':
        geoms = copy.deepcopy(sys.geoms)
        if not hasattr(val, '__iter__'):
            for i, g in enumerate(geoms):
                if not hasattr(g, attr[1]):
                    continue
                geoms[i] = g.replace(**{attr[1]: val})
        else:
            sizes = [g.transform.pos.shape[0] for g in geoms]
            g_idx = 0
            for i, g in enumerate(geoms):
                if not hasattr(g, attr[1]):
                    continue
                size = sizes[i]
                geoms[i] = g.replace(**{attr[1]: val[g_idx:g_idx + size].T})
                g_idx += size
        return sys.replace(geoms=geoms)
    if len(attr) == 1:
        return sys.replace(**{attr[0]: val})
    return sys.replace(**{attr[0]:
                              _write_sys(getattr(sys, attr[0]), attr[1:], val)})


# TODO traverse_sys and write_sys can probably be collapsed into one function

### RANDOMIZATION CONFIG LOGIC ###

@struct.dataclass
class SysKeyRange:
    """Defines ranges of values for specified System keys

    <base>, <min>, and <max> should be of shape (num elements in sys.<...key>,)

    We compute new values to set sys to with this formula:
        sys.<...key>[i] = <base>[i] + U(<min>[i], <max>[i]) \ForAll i \in [len(sys.<...key>)]

    # If <base>[i] is <None>, we use the default value in sys, and the formula becomes:
    #    sys.<...key>[i] = sys.<...key>[i] + U(<min>[i], <max>[i]) \ForAll i \in [len(sys.<...key>)]
    """
    key: str
    base: jp.ndarray
    min: jp.ndarray
    max: jp.ndarray


KEY_SEP = "."


def make_skrs(sys: Union[Env, System], randomization_config_path: Union[str, Path]):
    if isinstance(sys, Env):
        sys = sys.unwrapped.sys

    def yaml_to_basic_skrs():
        """
        Special yaml values for the <base>:
        - "r" r reads the corresponding key from the system.
        - "n" n disables randomization for this specific <key,index> pair.

        The <base> field is optional; when missing, it will automatically be set to "r".
        """
        import yaml
        with open(f"{randomization_config_path}", "r") as stream:
            loaded_yaml = yaml.safe_load(stream)

        def make_skr(key, min, max, base=None):
            if base is None:
                fixed_base = ["r"] * len(min)
            else:
                fixed_base = [float(x) if not isinstance(x, str) else x for x in base]

            assert len(min) == len(max) == len(fixed_base)
            return SysKeyRange(key, fixed_base, min, max)

        def is_skr_descr(subdict):
            minimal_skr_keys_set = set(["min", "max"])
            skr_keys_set = minimal_skr_keys_set.union(set(["base"]))

            subkeys = set(list(subdict.keys()))
            return subkeys == minimal_skr_keys_set or subkeys == skr_keys_set

        def collect(dic, collected_skr: Tuple):
            if len(dic) == 0:
                return collected_skr

            next_dic = {}
            for key, subdict in dic.items():
                if is_skr_descr(subdict):
                    collected_skr = collected_skr + (make_skr(key=key, **subdict),)
                    continue

                for subkey, val in subdict.items():
                    next_key = f"{key}{KEY_SEP}{subkey}"
                    assert isinstance(val, dict)
                    next_dic[next_key] = val

            return collect(next_dic, collected_skr)

        return collect(loaded_yaml, tuple())

    def read_sys_onto_skr(skr):
        key = skr.key
        new_base = skr.base.copy()
        new_max = skr.max.copy()
        new_min = skr.min.copy()

        read_vals = _traverse_sys(sys, key.split(KEY_SEP), None)

        for i, x in enumerate(skr.base):
            if isinstance(x, str):
                assert x in ["r", "n"]
                new_base[i] = read_vals[i]
                if x == "n":
                    new_max[i] = 0.0
                    new_min[i] = 0.0

        new_base = jp.array(new_base)
        new_min = jp.array(new_min)
        new_max = jp.array(new_max)

        return skr.replace(base=new_base, max=new_max, min=new_min)

    ret = list(map(read_sys_onto_skr, yaml_to_basic_skrs()))
    # they arrive in reverse order, so we're gonna swap em

    ret = list(reversed(ret))
    return ret




### WRAPPER LOGIC ###

class _VSysWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        def _find_batch_size(e):
            if not hasattr(e, "env"):
                return None
            e = e.env
            if isinstance(e, VmapWrapper):
                return e.batch_size
            return _find_batch_size(e)

        self.batch_size = _find_batch_size(env)
        self.single_env = False

        if self.batch_size is None:
            self.batch_size = 1
            self.single_env = True


class IdentityVSysWrapper(_VSysWrapper):
    """Uses the same sys for every leading vmap axes."""

    def __init__(self, env: Env):
        super().__init__(env)

        self.sys = env.unwrapped.sys
        if not self.single_env:
            def identity(_):
                return self.sys
            self.sys = jax.vmap(identity)(jax.numpy.arange(self.batch_size))

    def reset(self, rng: jp.ndarray) -> State:
        return self.env.reset(self.sys, rng)


def split_key(key, num):
    reset_rng, *set_rng = jax.random.split(key, num + 1)
    set_rng = jp.reshape(jp.stack(set_rng), (num, 2))
    return reset_rng, set_rng


class DomainRandVSysWrapper(_VSysWrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, skrs: List[SysKeyRange], inital_rng: jax.random.PRNGKey, do_on_reset: int, do_every_N_step: Union[Callable[[jax.random.PRNGKeyArray], int], int], do_at_creation: int):
        super().__init__(env)

        self.baseline_sys = self.env.unwrapped.sys
        self.baseline_vsys = jax.vmap(functools.partial(lambda x: self.baseline_sys))(jp.arange(self.batch_size))

        keys = [skr.key.split(KEY_SEP) for skr in skrs]

        def receive_sys(sys: System, mask: jp.ndarray, vals: List[jp.ndarray]):
            def identity(sys: System, v: List[jp.ndarray]) -> System:
                return sys
            def resamply(sys: System, vals: List[jp.ndarray]) -> System:
                """Sets params in the System."""
                for key, val in zip(keys, vals):
                    sys = _write_sys(sys, key, val)
                return sys

            return jax.lax.cond(
                    mask > 0,
                    resamply,
                    identity,
                    sys, vals
                )

        self._sysset = jax.vmap(receive_sys)

        def get_current_skrs_vals(sys: System):
            """Sets params in the System."""
            ret = []
            for key in keys:
                read = _traverse_sys(sys, key, None)
                ret.append(read)
            return ret
        self._sysread = jax.vmap(get_current_skrs_vals)

        self.skrs = skrs


        def _sample_for_one_key(rng):
            vals = []
            for skr in skrs:
                rng, key = jax.random.split(rng)

                rand = jax.random.uniform(key=key, minval=skr.min, maxval=skr.max, shape=skr.max.shape)
                val = skr.base + rand
                vals.append(val)
            return vals
        self._sample_batch_skrs = jax.vmap(_sample_for_one_key)

        self.do_on_reset = do_on_reset >= 1

        if isinstance(do_every_N_step, int):
            def _do_every_N_step(rng: jax.random.PRNGKey):
                return do_every_N_step * jp.ones(self.batch_size)
            self.do_every_N_step = _do_every_N_step
        else:
            def vmapped_do_every_N_step(rng: jax.random.PRNGKey):
                _, rngs = split_key(rng, self.batch_size)
                return jax.vmap(do_every_N_step)(rngs)
            self.do_every_N_step = vmapped_do_every_N_step

        self.do_at_creation = do_at_creation >= 1
        if self.do_at_creation:
            self.baseline_vsys = self.randomize_vsys(inital_rng, self.baseline_vsys, mask=jp.ones(self.batch_size,))

    def randomize_vsys(self, key: jp.ndarray, current_sys: System, mask: jp.ndarray):
        """
        Args:
            rng: a rng key vector of size self.batch_size
            mask: only resample for positive nonzero mask indices

        Returns:

        """
        key, batch_rng = split_key(key, self.batch_size)
        vals = self._sample_batch_skrs(batch_rng)
        current_sys = self._sysset(current_sys, mask, vals)
        self.current_skrs_vals = self._sysread(current_sys)
        return current_sys

    def reset(self, key: jp.ndarray) -> State:
        key, set_rng = jax.random.split(key)

        mask = jp.ones(self.batch_size,) * self.do_on_reset
        sys = self.randomize_vsys(key=set_rng, current_sys=self.baseline_vsys, mask=mask)

        key, reset_rng = jax.random.split(key)
        state = self.env.reset(sys=sys, rng=reset_rng)

        key, stepcount_rng = jax.random.split(key)
        vsys_stepcount = self.do_every_N_step(stepcount_rng)

        state = state.replace(sys=sys, vsys_stepcount=vsys_stepcount, vsys_rng=split_key(key, self.batch_size)[-1])
        return state

    def step(self, state, action):
        key = state.vsys_rng[0]
        key, stepcount_rng = jax.random.split(key)

        mask_resample = state.vsys_stepcount == 1
        decr_vsys_stepcount = jp.clip(state.vsys_stepcount - 1, 0, jp.inf)
        maybe_new_vsys_stepcount = self.do_every_N_step(stepcount_rng)
        new_vsys_stepcount = decr_vsys_stepcount * jp.logical_not(mask_resample) + maybe_new_vsys_stepcount * mask_resample

        mask_resample = mask_resample + (state.done * self.do_on_reset)
        key, rng_randomize = jax.random.split(key)
        resampled_sys = self.randomize_vsys(key=rng_randomize, current_sys=state.sys, mask=mask_resample)

        state = state.replace(sys=resampled_sys, vsys_rng=split_key(key, self.batch_size)[-1], vsys_stepcount=new_vsys_stepcount)
        state = self.env.step(state, action)

        #state_info = state.info
        #state_info["skrs_vals"] = self.current_skrs_vals
        #state_info["skrs_resampled"] = mask_resample

        #state = state.replace(info=state_info)
        return state


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng, 2)
    ret = jax.random.normal(key)

    # num_train_envs * num_frames * num_params ^ discr_level = num_desired_envs

    env = envs.create(
        "halfcheetah",
        backend="spring",
        episode_length=1000,
        auto_reset=True,
        batch_size=32,
        no_vsys=False
    )

    x = make_skrs(env, "./inertia.yaml")

    env = DomainRandVSysWrapper(env, x, jax.random.PRNGKey(10), do_on_reset=True, do_at_creation=False, do_every_N_step=5)
    #env = IdentityVSysWrapper(env)
    #env = DomainCartesianVSysWrapper(env, x, DISCRETIZATION_LEVEL)
    key = jax.random.PRNGKey(0)

    USE_JIT = True
    if USE_JIT:
        reset_func = jax.jit(env.reset)
        step_func = jax.jit(env.step)
    else:
        reset_func = env.reset
        step_func = env.step
    state = reset_func(key)

    def randact():
        act = jax.random.uniform(jax.random.PRNGKey(0), (env.batch_size, env.action_size,))
        return act

    state = step_func(state, randact())
    start = time.time()
    for i in range(4):
        print(env.current_skrs_vals)
        print(i)
        state = step_func(state, randact())
    before = env.current_skrs_vals
    for i in range(4):
        print(i)
        state = step_func(state, randact())
    print(state.info["skrs_vals"])
    end = time.time()
    after = env.current_skrs_vals
    print(end - start)
    i = 9
