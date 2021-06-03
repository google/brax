# `BRAX`

Brax is a differentiable physics engine that simulates environments made up
of rigid bodies, joints, and actuators. It's also a suite of learning algorithms
to train agents to operate in these environments (PPO, SAC, evolutionary
strategy, and direct trajectory optimization are implemented).

Brax is written in [JAX](https://github.com/google/jax) and is designed for
use on acceleration hardware. It is both efficient for single-core training, and
scalable to massively parallel simulation, without the need for pesky
datacenters.

![ant locomotion](./docs/img/ant_loco.gif)

*Locomotion policy (run to target) trained in Brax in ~20 seconds. Brax
simulates this environment >3M frames per second on a single v100 GPU.*

## Getting Started

To install brax from source, clone this repo, `cd` to it, and then:

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
