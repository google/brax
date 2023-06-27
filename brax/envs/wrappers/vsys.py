import os
import time
from pathlib import Path

from jax._src.tree_util import Partial
from jax.random import KeyArray

from brax.envs.wrappers.training import VmapWrapper

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"
import copy
import functools
from typing import Dict, Union, List, Tuple

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


def set_sys(sys, params: Dict[str, jp.ndarray]):
    """Sets params in the System."""
    for k in params.keys():
        sys = _write_sys(sys, k.split('.'), params[k])
    return sys


def set_sys_capsules(sys, lengths, radii):
    """Sets the system with new capsule lengths/radii."""
    sys2 = set_sys(sys, {'geoms.length': lengths})
    sys2 = set_sys(sys2, {'geoms.radius': radii})

    # we assume inertia.transform.pos is (0,0,0), as is often the case for
    # capsules

    # get the new joint transform
    cur_len = sys.geoms[1].length[:, None]
    joint_dir = jax.vmap(math.normalize)(sys.link.joint.pos)[0]
    joint_dist = sys.link.joint.pos - 0.5 * cur_len * joint_dir
    joint_transform = 0.5 * lengths[:, None] * joint_dir + joint_dist
    sys2 = set_sys(sys2, {'link.joint.pos': joint_transform})

    # get the new link transform
    parent_idx = jp.array([sys.link_parents])
    sys2 = set_sys(
        sys2,
        {
            'link.transform.pos': -(
                    sys2.link.joint.pos
                    + joint_dist
                    + 0.5 * lengths[parent_idx].T * joint_dir
            )
        },
    )
    return sys2


def util_vmap_set(sys, keys, vals):
    dico = dict(zip(keys, vals))

    return set_sys(sys, dico)


def randomize(sys, rng):
    return set_sys(sys,
                   {'link.inertia.mass': sys.link.inertia.mass + jax.random.uniform(rng, shape=(sys.num_links(),))})


@jax.jit
def randomize_sys_capsules(
        rng: jp.ndarray,
        sys: base.System,
        min_length: float = 0.0,
        max_length: float = 0.0,
        min_radius: float = 0.0,
        max_radius: float = 0.0,
):
    """Randomizes joint offsets, assume capsule geoms appear in geoms[1]."""
    rng, key1, key2 = jax.random.split(rng, 3)
    length_u = jax.random.uniform(
        key1, shape=(sys.num_links(),), minval=min_length, maxval=max_length
    )
    radius_u = jax.random.uniform(
        key2, shape=(sys.num_links(),), minval=min_radius, maxval=max_radius
    )
    length = length_u + sys.geoms[1].length  # pytype: disable=attribute-error
    radius = radius_u + sys.geoms[1].radius  # pytype: disable=attribute-error
    return set_sys_capsules(sys, length, radius)


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

    return list(map(read_sys_onto_skr, yaml_to_basic_skrs()))


def make_sysset(skrs: List[SysKeyRange], batch_size: Union[int, None]):
    keys = [skr.key.split(KEY_SEP) for skr in skrs]

    def identity(sys: System, vals: List[jp.ndarray]) -> System:
        return sys

    def resamply(sys: System, vals: List[jp.ndarray]) -> System:
        """Sets params in the System."""
        for key, val in zip(keys, vals):
            sys = _write_sys(sys, key, val)
        return sys

    def apply_mask(sys: System, mask: jp.ndarray, vals: List[jp.ndarray]):
        sys = jax.lax.cond(
            mask,
            resamply,   # true
            identity,   # false
            sys, vals   # operands
        )
        return sys

    def receive_sys(sys: System, mask: jp.ndarray, vals: List[jp.ndarray]):
        mask = mask > 0

        if batch_size is None or batch_size == 1:
            return apply_mask(sys, mask[0], vals)
        else:
            return jax.vmap(apply_mask)(sys, mask, vals)

    return receive_sys


### WRAPPER LOGIC ###

class VSysWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        def _find_batch_size(e):
            if not hasattr(e, "env"):
                return 1
            e = e.env
            if isinstance(e, VmapWrapper):
                return e.batch_size
            return _find_batch_size(e)

        self.batch_size = _find_batch_size(env)

class IdentityVSysWrapper(VSysWrapper):
    """Uses the same sys for every leading vmap axes."""

    def __init__(self, env: Env):
        super().__init__(env)

        def _reset(sys, rng):
            return self.env.reset(sys=sys, rng=rng)

        self.sys = env.unwrapped.sys
        if self.batch_size != 1:
            def identity(_):
                return self.sys
            self.sys = jax.vmap(identity)(jax.numpy.arange(self.batch_size))

        _reset = functools.partial(_reset, sys=self.sys)
        self._reset = jax.jit(_reset)

    def reset(self, rng: jp.ndarray) -> State:
        return self._reset(rng=rng)


class DomainRandVSysWrapper(VSysWrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, seed: Union[jax.random.PRNGKey, int], skrs: List[SysKeyRange], randomize_every_nsteps: Union[int, None]):
        super().__init__(env)

        self.baseline_sys = self.env.unwrapped.sys
        def _make_vsys():
            def _identity(sys, _):
                return sys
            return jax.vmap(functools.partial(_identity, self.baseline_sys))(jp.arange(self.batch_size))
        self.baseline_vsys = _make_vsys() if (self.batch_size is not None and self.batch_size != 1) else self.baseline_sys
        self.current_sys = _make_vsys()

        if isinstance(seed, int):
            seed = jax.random.PRNGKey(seed)
        self.rng = seed

        self.skrs = skrs

        def _sample_for_one_key(skrs, rng):
            vals = []
            for skr in skrs:
                rng, key = jax.random.split(rng)

                rand = jax.random.uniform(key=key, minval=skr.min, maxval=skr.max, shape=skr.max.shape)
                val = skr.base + rand
                vals.append(val)

            return rng, vals
        _sample_for_one_key = jax.jit(Partial(_sample_for_one_key, skrs))

        if self.batch_size == 1 or self.batch_size is None:
            @jax.jit
            def __sample_for_one_key(rng):
                ret_rng, ret_vals = _sample_for_one_key(rng)
                return ret_rng, ret_vals
            self._sample_batch_skrs = __sample_for_one_key
        else:
            self._sample_batch_skrs = jax.vmap(_sample_for_one_key)

        self._sysset = make_sysset(skrs, self.batch_size)

        self.randomize_every_nsteps = randomize_every_nsteps

    @property
    def observation_size(self) -> int:
        return self.obs_size

    def generic_set(self, rng: jp.ndarray, current_sys: System, mask: jp.ndarray):
        """
        Args:
            rng: a rng key vector of size self.batch_size
            mask: only resample for positive nonzero mask indices

        Returns:

        """
        state_rng, vals = self._sample_batch_skrs(rng)
        current_sys = self._sysset(current_sys, mask, vals)
        return state_rng, current_sys

    def reset(self, rng: jp.ndarray) -> State:
        reset_rng, *set_rng = jax.random.split(rng, self.batch_size+1)
        set_rng = jp.reshape(jp.stack(set_rng), (self.batch_size, 2))
        if self.batch_size is None or self.batch_size == 1:
            set_rng = set_rng[0]

        state_rng, sys = self.generic_set(rng=set_rng, current_sys=self.baseline_vsys, mask=jp.ones(self.batch_size,))

        state = self.env.reset(sys=sys, rng=reset_rng)
        state = state.replace(vsys_stepcount=jp.zeros(self.batch_size), vsys_rng=state_rng)
        return state

    def step(self, state, action):
        step_count = state.vsys_stepcount + 1

        state = self.env.step(state, action)

        # TODO when resets here, gotta update SOME of the systems......

        step_count = step_count * (1-state.done)

        step_count_tripped = -jp.mod(step_count, self.randomize_every_nsteps) + 1
        # step_count_tripped: 1 if need to resample, <0 value otherwise
        resample_mask = state.done + step_count_tripped
        state_rng, resampled_sys = self.generic_set(rng=state.vsys_rng, current_sys=state.sys, mask=resample_mask)
        state = state.replace(sys=resampled_sys, vsys_rng=state_rng, vsys_stepcount=step_count)

        return state


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng, 2)
    ret = jax.random.normal(key)

    SINGLE_ENV = False

    env = envs.create(
        "halfcheetah",
        backend="spring",
        episode_length=1000,
        auto_reset=True,
        batch_size=None if SINGLE_ENV else 4000,
        no_vsys=False
    )

    x = make_skrs(env, "./inertia.yaml")

    env = DomainRandVSysWrapper(env, 0, x, 100)
    #env = IdentityVSysWrapper(env)
    key = jax.random.PRNGKey(0)

    reset_func = jax.jit(env.reset)
    step_func = jax.jit(env.step)
    state = reset_func(key)

    def randact():
        act = jax.random.uniform(jax.random.PRNGKey(0), (env.batch_size, env.action_size,))
        if SINGLE_ENV:
            return act[0]
        return act

    start = time.time()
    for i in range(100):
        print(i)
        state = step_func(state, randact())
    end = time.time()
    print(end - start)
    i = 9
