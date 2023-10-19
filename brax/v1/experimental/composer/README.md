# Braxlines Composer

<img src="https://github.com/google/brax/raw/main/docs/img/composer/ant_push.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/composer/ant_chase.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/composer/pro_ant2.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/composer/pro_ant1.gif" width="150" height="107"/>

Braxlines Composer allows modular composition of environments.
The composed environment is compatible with training algorithms in
[Brax](https://github.com/google/brax) and
[Braxlines](https://github.com/google/brax/tree/main/brax/experimental/braxlines). See:
* [composer.py](https://github.com/google/brax/tree/main/brax/experimental/composer/composer.py) for descriptions of the API
* [observers.py](https://github.com/google/brax/tree/main/brax/experimental/composer/observers.py) for observation definition utilities
* [reward_functions.py](https://github.com/google/brax/tree/main/brax/experimental/composer/reward_functions.py) for reward definition utilities
* [envs/](https://github.com/google/brax/tree/main/brax/experimental/composer/envs) for examples of environment composition configs
* [components/](https://github.com/google/brax/tree/main/brax/experimental/composer/components) for examples of environment component configs
* [agent_utils.py](https://github.com/google/brax/tree/main/brax/experimental/composer/agent_utils.py) for multi-agent RL support

## Usage

For composing an environment registered in [envs/](https://github.com/google/brax/tree/main/brax/experimental/composer/envs),
during creation:
```python
from brax.v1.experimental.composer import composer
env = composer.create(env_name='pro_ant_run', num_legs=2)
```

To inspect what environments are pre-registered, and what are configurable environment parameters e.g. `num_legs` for an environment e.g. `pro_ant_run`, use the following. `support_kwargs==True` means that the environment may have unlisted keyword parameters:
```python
env_list = composer.list_env()
env_params, support_kwargs = composer.inspect_env(env_name='pro_ant_run')
```

For composing an environment from a description `env_desc`:
```python
# env_desc captures the full information about the environment
#   including rewards, observations, resets
env_desc = dict(
    components=dict(  # component information
        agent1=dict(
            component='pro_ant',  # components/pro_ant.py
            component_params=dict(num_legs=6)),  # 6 legs
        cap1=dict(
            component='singleton',  # components/singleton.py
            component_params=dict(size=0.5),  # adjust object size
            pos=(1, 0, 0),  # where to place a capsule object
            reward_fns=dict(
                goal=dict(  # reward1: a target velocity for the object
                    reward_type='root_goal', sdcomp='vel',
                    target_goal=(4, 0, 0))))),
    edges=dict(  # edge information
        agent1__cap1=dict(  # edge names use sorted component names
            extra_observers=[  # add agent-object position diff as an extra obs
                dict(observer_type='root_vec')],
            reward_fns=dict(  # reward2: make the agent close to the object
                dist=dict(reward_type='root_dist')),),))
env = composer.create(env_desc=env_desc)

# you may also register envs and create later with `env_name`
env_name = 'ant_push_6legs'
composer.register_env(env_name=env_name, env_desc=env_desc)
env = composer.create(env_name=env_name)
```

Lastly, while less recommended, `desc_edits` can be used for dynamic environment editing by directly modifying `env_desc` dictionary object. For examples, see [envs/ant_descs.py](https://github.com/google/brax/tree/main/brax/experimental/composer/envs/ant_descs.py).

*These are illustrative examples. For full examples, see [envs/ant_descs.py](https://github.com/google/brax/tree/main/brax/experimental/composer/envs/ant_descs.py) for standard Brax envs and [envs/ma_descs.py](https://github.com/google/brax/tree/main/brax/experimental/composer/envs/ma_descs.py) for multi-agent RL Brax envs.*

## Colab Notebooks

Explore Composer easily and quickly through:
* [Composer Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/composer/composer.ipynb) dynamically composes an environment and trains it using PPO within a few minutes.
* [Experiment Sweep](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/experiment_sweep.ipynb) provides a basic example for running a hyperparameter sweep. Set `experiment`=`composer_sweep`.
* [Experiment Viewer](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/experiment_viewer.ipynb) provides a basic example for visualizing results from a hyperparameter sweep.

Tips:
* `env_desc` and `config_json` are full descriptions of the environment and the system, and are accessible through `env.env_desc` and `env.metadata.config_json`.
* for debugging NaNs, use:
```python
from jax import config
config.update("jax_debug_nans", True)
```

## Learn More

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/sketches.png" width="540" height="220"/>

For a deep dive into Braxlines, please see
our paper, [Braxlines: Fast and Interactive Toolkit for RL-driven Behavior Generation Beyond Reward Maximization](https://arxiv.org/abs/2110.04686).

*Braxlines is under rapid development. While API is stabilizing,
feel free to send documentation and feature questions and requests through git or email.*

## Citing Composer

If you would like to reference Braxlines in a publication, please use:

```
@article{gu2021braxlines,
  title={Braxlines: Fast and Interactive Toolkit for RL-driven Behavior Engineering beyond Reward Maximization},
  author={Gu, Shixiang Shane and Diaz, Manfred and Freeman, Daniel C and Furuta, Hiroki and Ghasemipour, Seyed Kamyar Seyed and Raichuk, Anton and David, Byron and Frey, Erik and Coumans, Erwin and Bachem, Olivier},
  journal={arXiv preprint arXiv:2110.04686},
  year={2021}
}
@software{brax2021github,
  author = {C. Daniel Freeman and Erik Frey and Anton Raichuk and Sertan Girgin and Igor Mordatch and Olivier Bachem},
  title = {Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation},
  url = {http://github.com/google/brax},
  version = {0.0.5},
  year = {2021},
}
```
