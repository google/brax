<img src="https://github.com/google/brax/raw/main/docs/img/brax_logo.gif" width="336" height="80" alt="BRAX"/>

Brax is a fast and fully differentiable physics engine used for research and
development of robotics, human perception, materials science, reinforcement
learning, and other simulation-heavy applications.

Brax is written in [JAX](https://github.com/google/jax) and is designed for use
on acceleration hardware. It is both efficient for single-device simulation, and
scalable to massively parallel simulation on multiple devices, without the need
for pesky datacenters.

<img src="https://github.com/google/brax/raw/main/docs/img/humanoid_v2.gif" width="160" height="160"/><img src="https://github.com/google/brax/raw/main/docs/img/a1.gif" width="160" height="160"/><img src="https://github.com/google/brax/raw/main/docs/img/ant_v2.gif" width="160" height="160"/><img src="https://github.com/google/brax/raw/main/docs/img/ur5e.gif" width="160" height="160"/>

Brax simulates environments at millions of physics steps per second on TPU, and includes a suite of learning algorithms that train agents in seconds
to minutes:

*   Baseline learning algorithms such as
    [PPO](https://github.com/google/brax/blob/main/brax/training/agents/ppo),
    [SAC](https://github.com/google/brax/blob/main/brax/training/agents/sac),
    [ARS](https://github.com/google/brax/blob/main/brax/training/agents/ars), and
    [evolutionary strategies](https://github.com/google/brax/blob/main/brax/training/agents/es).
*   Learning algorithms that leverage the differentiability of the simulator, such as [analytic policy gradients](https://github.com/google/brax/blob/main/brax/training/agents/apg).

## One API, Three Pipelines

Brax offers three distinct physics pipelines that are easy to swap:

* [Generalized](https://github.com/google/brax/blob/main/brax/v2/generalized/)
calculates motion in [generalized coordinates](https://en.wikipedia.org/wiki/Generalized_coordinates) using the same accurate robot
dynamics algorithms as [MuJoCo](https://mujoco.org/) and [TDS](https://github.com/erwincoumans/tiny-differentiable-simulator).

  NOTE: We plan to import [MuJoCo XLA - MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) as another physics backend, eventually replacing `generalized`. Check out our recent [announcement](https://github.com/google/brax/discussions/409).

* [Positional](https://github.com/google/brax/blob/main/brax/v2/positional/)
uses [Position Based Dynamics](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf),
a fast but stable method of resolving joint and collision constraints.
* [Spring](https://github.com/google/brax/blob/main/brax/v2/spring/) provides
fast and cheap simulation for rapid experimentation, using simple impulse-based
methods often found in video games.

These pipelines share the same API and can run side-by-side within the same
simulation.  This makes Brax well suited for experiments in transfer learning
and closing the gap between simulation and the real world.

## Quickstart: Colab in the Cloud

Explore Brax easily and quickly through a series of colab notebooks:

* [Brax Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb) introduces the Brax API, and shows how to simulate basic physics primitives.
* [Brax Training](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb) introduces Brax's training algorithms, and lets you train your own policies directly within the colab. It also demonstrates loading and saving policies.
* [Brax Training with MuJoCo XLA - MJX](https://colab.sandbox.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb) demonstrates training in Brax using the `MJX` physics simulator.
* [Brax Training with PyTorch on GPU](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training_torch.ipynb) demonstrates how Brax can be used in other ML frameworks for fast training, in this case PyTorch.

## Using Brax Locally

To install Brax from pypi, install it with:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install brax
```

You may also install from [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://github.com/mamba-org/mamba):

```
conda install -c conda-forge brax  # s/conda/mamba for mamba
```

Alternatively, to install Brax from source, clone this repo, `cd` to it, and then:

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

Training on NVidia GPU is supported, but you must first install
[CUDA, CuDNN, and JAX with GPU support](https://github.com/google/jax#installation).

## Learn More

For a deep dive into Brax's design and performance characteristics, please see
our paper, [Brax -- A Differentiable Physics Engine for Large Scale Rigid Body Simulation
](https://arxiv.org/abs/2106.13281), which appeared in the [Datasets and Benchmarks Track](https://neurips.cc/Conferences/2021/CallForDatasetsBenchmarks) at [NeurIPS 2021](https://nips.cc/Conferences/2021).

## Citing Brax

If you would like to reference Brax in a publication, please use:

```
@software{brax2021github,
  author = {C. Daniel Freeman and Erik Frey and Anton Raichuk and Sertan Girgin and Igor Mordatch and Olivier Bachem},
  title = {Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation},
  url = {http://github.com/google/brax},
  version = {0.9.3},
  year = {2021},
}
```

## Acknowledgements

Brax has come a long way since its original publication.  We offer gratitude and
effusive praise to the following people:

* Manu Orsini and Nikola Momchev who provided a major refactor of Brax's
training algorithms to make them more accessible and reusable.
* Erwin Coumans who has graciously offered advice and mentorship, and many
useful references from [Tiny Differentiable Simulator](https://github.com/erwincoumans/tiny-differentiable-simulator).
* Baruch Tabanpour, a colleague who helped launch brax v2 and overhauled the contact library.
* [Shixiang Shane Gu](https://sites.google.com/corp/view/gugurus) and [Hiroki Furuta](https://frt03.github.io/), who contributed BIG-Gym and Braxlines, and a scene composer to Brax.
* Our awesome [open source collaborators and contributors](https://github.com/google/brax/graphs/contributors).  Thank you!
