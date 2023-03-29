# Braxlines

Braxlines is a series of minimalistic implementations for RL problem
formulations beyond simple reward maximization.
It is built on [JAX](https://github.com/google/jax)-based physics simulator
[Brax](https://github.com/google/brax) designed for
use on acceleration hardware. It is both efficient for single-core training, and
scalable to massively parallel simulation, without the need for pesky
datacenters.

*Policies can be trained via Braxlines **under a few minutes**. Brax simulates these environments at millions of physics steps per second on TPU.*

## Colab Notebooks

Explore Braxlines easily and quickly through a series of colab notebooks.

### Mutual Information Maximization (MI-Max) RL

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_diayn.png" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_diayn_skill1.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_diayn_skill2.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_diayn_skill4.gif" width="150" height="107"/>

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_diayn.png" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_diayn_skill1.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_diayn_skill2.gif" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_diayn_skill3.gif" width="150" height="107"/>

* [VGCRL Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/mimax.ipynb) implements [Variational GCRL](https://arxiv.org/abs/2106.01404) algorithms, which include [goal-conditioned RL](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.3077) and [DIAYN](https://arxiv.org/abs/1802.06070) as special cases. These algorithms can all be considered as maximizing mutual information (MI) between latent intents and state marginal distribution.

### Divergence Minimization (D-Min) RL

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_smm.png" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_smm.gif" width="150" height="107"/>
<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_smm.png" width="150" height="107"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_smm.gif" width="150" height="107"/>

* [IRL_SMM Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/dmin.ipynb) implements a family of adversarial inverse RL algorithms, which includes [GAIL](https://arxiv.org/abs/1606.03476), [AIRL](https://arxiv.org/abs/1710.11248), and [FAIRL](https://arxiv.org/abs/1911.02256) as special cases. These algorithms minimize the divergence between the policy's state marginal distribution and a given target distribution. As discussed in [f-MAX](https://arxiv.org/abs/1911.02256), these algorithms could also be used for [state-marginal matching](https://arxiv.org/abs/1906.05274) RL besides imitation learning.

### Hyperparameter Sweep on Colab

Since each experiment runs under a few minutes, a reasonable amount of hyperparameter sweep can be directly run in series using the free colab TPU.

* [Experiment Sweep](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/experiment_sweep.ipynb) provides a basic example for running a hyperparameter sweep.
* [Experiment Viewer](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/experiment_viewer.ipynb) provides a basic example for visualizing results from a hyperparameter sweep.

## Learn More

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/sketches.png" width="540" height="220"/>

For a deep dive into Braxlines, please see
our paper, [Braxlines: Fast and Interactive Toolkit for RL-driven Behavior Generation Beyond Reward Maximization](https://arxiv.org/abs/2110.04686).

*Braxlines is under rapid development. While API is stabilizing,
feel free to send documentation and feature questions and requests through git or email.*

## Citing Braxlines

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
