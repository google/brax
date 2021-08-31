# Braxlines

Braxlines is a series of minimalistic implementation for popular RL problem
formulations beyond simple reward maximization.
It is built on [JAX](https://github.com/google/jax)-based physics simulator
[Brax](https://github.com/google/brax) designed for
use on acceleration hardware. It is both efficient for single-core training, and
scalable to massively parallel simulation, without the need for pesky
datacenters.

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_smm.gif" width="150" height="107"/>

*Some policies trained via Braxlines under a few minutes. Brax simulates these environments at millions of physics steps per second on TPU.*

## Colab Notebooks

Explore Braxlines easily and quickly through a series of colab notebooks.

### Mutual Information Maximization (MI-Max) RL

Goal-reaching and empowerment are two popular approaches for skill discovery
and learning in RL.
[VGCRL Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/mimax.ipynb)
implements
[Variational GCRL](https://arxiv.org/abs/2106.01404) algorithms, which include [goal-conditioned RL](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.3077) and [DIAYN](https://arxiv.org/abs/1802.06070) as special cases.

### Divergence Minimization (D-Min) RL

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_smm.gif" width="150" height="107"/>

Matching a target marginal state distribution is the foundation for [adversarial inverse RL](https://arxiv.org/abs/1911.02256) algorithms
in imitation learning, as well as recently popularized [state marginal matching](https://arxiv.org/abs/1906.05274) methods.
[IRL_SMM Basics](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/dmin.ipynb)
runs a family of adversarial inverse RL algorithms, which includes [GAIL](https://arxiv.org/abs/1606.03476), [AIRL](https://arxiv.org/abs/1710.11248), and [FAIRL](https://arxiv.org/abs/1911.02256) as special cases. These algorithms minimize D(p(s,a), p*(s,a)) or D(p(s), p*(s)), the divergence D between the policy's state(-action) marginal distribution p(s,a) or p(s), and a given target distribution p*(s,a) or p*(s). As discussed in [f-MAX](https://arxiv.org/abs/1911.02256), these algorithms could also be used for [state-marginal matching](https://arxiv.org/abs/1906.05274) RL besides imitation learning.
