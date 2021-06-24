# `BRAX`

Brax is a differentiable physics engine that simulates environments made up
of rigid bodies, joints, and actuators. It's also a suite of learning algorithms
to train agents to operate in these environments (PPO, SAC, evolutionary
strategy, and direct trajectory optimization are implemented).

Brax is written in [JAX](https://github.com/google/jax) and is designed for
use on acceleration hardware. It is both efficient for single-core training, and
scalable to massively parallel simulation, without the need for pesky
datacenters.

<img src="https://github.com/google/brax/raw/main/docs/img/ant.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/fetch.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/grasp.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/halfcheetah.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/humanoid.gif" width="150" height="107"/>

*Some policies trained via Brax. Brax simulates these environments at millions of physics steps per second on TPU.*

## Colab Notebooks

Explore Brax easily and quickly through a series of colab notebooks:

* [Brax Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb) introduces the Brax API, and shows how to simulate basic physics primitives.
* [Brax Training](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb) introduces Brax environments and training algorithms, and lets you train your own policies directly within the colab.

## Using Brax locally

To install Brax from source, clone this repo, `cd` to it, and then:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

To train a model:

```
learn
```

Training on NVidia GPU is supported, but you must first install [CUDA, CuDNN,
and JAX with GPU support](https://github.com/google/jax#installation).

## Citing Brax

If you would like to reference Brax in a publication, please use:

```
@software{brax2021github,
  author = {C. Daniel Freeman and Erik Frey and Anton Raichuk and Sertan Girgin and Igor Mordatch and Olivier Bachem},
  title = {Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation},
  url = {http://github.com/google/brax},
  version = {0.1.0},
  year = {2021},
}
```
