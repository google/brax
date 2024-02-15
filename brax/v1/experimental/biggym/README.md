# BIG-Gym

BIG-Gym is a *crowd-sourcing* challenge for RL *environments* and *behaviors*, inspired by [BIG-Bench](https://github.com/google/BIG-bench). *Our goal is to create the "ImageNet" for continuous control, with diversity in agent morphologies, environment scenes, objects, and tasks.* We solicit submissions for two tracks: **Open-Ended Creativity Track** and **Goal-Oriented Competition Track**. Details (timelines, submission instructions) are [here](https://sites.google.com/view/rlbiggym).

```python
from brax.v1.experimental import biggym

# register all in registry/__init__.py
biggym.register_all(verbose=True)
#   `biggym.ENVS_BY_TRACKS` shows which envs are registered under each track
#   `biggym.COMPONENTS_BY_TRACK` shows ... components ...

# OR, register a specific folder under registry/
env_names, component_names, task_env_names, metadata = biggym.register(registry_name)

# create an environment under Open-Ended Creativity Track
# (optional) inspect and get default configurable parameters of an environment
env0_params, _ = biggym.inspect_env(env_names[0])
env0 = biggym.create(env_names[0], **env0_params)

# create an environment under Goal-Oriented Competition Track
env1_params, _ = biggym.inspect_env(task_env_names[0])
env1 = biggym.create(task_env_names[0], **env1_params)

# register all combinations of matches among components in Goal-Oriented Track
match_env_names = biggym.register_matches()

# create an environment between agents under Goal-Oriented Competition Track
env2_params, _ = biggym.inspect_env(match_env_names[0])
env2 = biggym.create(match_env_names[0], **env2_params)
```

## Colab Notebooks

Explore BIG-Gym easily and quickly through:
* [BIG-Gym Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/biggym/biggym_rl.ipynb)

Tips:
* for debugging NaNs, use:
```python
from jax import config
config.update("jax_debug_nans", True)
```

## Citing

If you use BIG-Gym in a publication, please cite referenced libraries:

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

## Organizers
* [Shixiang Shane Gu](https://sites.google.com/view/gugurus/home) (Google Brain), [Hiroki Furuta](https://frt03.github.io/) (University of Tokyo), [Manfred Diaz](https://manfreddiaz.github.io/) (University of Montreal)
* [Brax](https://github.com/google/brax)/[Braxlines](https://arxiv.org/abs/2110.04686) teams
* [NeurIPS 2021 EcoRL workshop](https://sites.google.com/view/ecorl2021/home) organizers
