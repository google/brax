# Braxlines Composer

Braxlines Composer allows modular composition of environments.
The composed environment is compatible with training algorithms in
[Brax](https://github.com/google/brax) and
[Braxlines](https://github.com/google/brax/tree/main/brax/experimental/braxlines).
See [composer.py](https://github.com/google/brax/tree/main/brax/experimental/composer/composer.py)
for descriptions of the API, and
[env_descs.py](https://github.com/google/brax/tree/main/brax/experimental/composer/env_descs.py)
for examples of environment composition configs.

## Colab Notebooks

Explore Composer easily and quickly through:
* [Composer Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/composer.ipynb) dynamically composes an environment and trains it using PPO within a few minutes.

<img src="https://github.com/google/brax/raw/main/docs/img/composer/ant_push.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/composer/ant_chase.gif" width="150" height="107"/>

Tips:
* for debugging, use:
```python
from jax.config import config
config.update("jax_debug_nans", True)
```

## Learn More

For a deep dive into Composer, please see
our paper, [Braxlines: Fast and Interactive Toolkit for RL-driven Behavior Generation Beyond Reward Maximization](https://openreview.net/forum?id=-W0LCm8wE2S).

## Citing Composer

If you would like to reference Braxlines in a publication, please use:

```
@article{gu2021braxlines,
  title={Braxlines: Fast and Interactive Toolkit for RL-driven Behavior Generation Beyond Reward Maximization},
  author={Gu, Shixiang Shane and Diaz, Manfred and Freeman, C Daniel and Furuta, Hiroki and Ghasemipour, Seyed Kamyar Seyed and Raichuk, Anton and David, Byron and Frey, Erik and Coumans, Erwin and Bachem, Olivier},
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
