<img src="https://github.com/google/brax/raw/main/docs/img/brax_logo.gif" width="336" height="80" alt="BRAX"/>

[English](README.md) | [简体中文](README.zh.md)

> **警告**
> 自 0.13.0 版本起，仅 `brax/training` 处于积极维护状态。对于环境部分，建议用户使用 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 替代 `brax/envs`，其中的所有环境均能与 `brax/training` 良好配合训练。如果您希望将 Brax 用于物理模拟，请使用 [github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) 中提供的 MJX（`pip install mujoco_mjx`）或 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)，而非将 Brax 作为 MuJoCo 物理模拟的封装器。我们未来可能会将 `brax` 纯粹重新定位为强化学习（RL）库。

Brax 是一个快速且完全可微的物理引擎，用于机器人技术、人类感知、材料科学、强化学习以及其他重度依赖仿真的应用领域的研究与开发。

Brax 基于 [JAX](https://github.com/google/jax) 编写，专为加速硬件设计。它既能实现高效的单设备仿真，又可扩展至跨多设备的海量并行仿真，无需依赖繁琐的数据中心架构。

<img src="https://github.com/google/brax/raw/main/docs/img/humanoid_v2.gif" width="160" height="160"/><img src="https://github.com/google/brax/raw/main/docs/img/a1.gif" width="160" height="160"/><img src="https://github.com/google/brax/raw/main/docs/img/ant_v2.gif" width="160" height="160"/><img src="https://github.com/google/brax/raw/main/docs/img/ur5e.gif" width="160" height="160"/>

Brax 在 TPU 上能够以每秒数百万个物理步长的速度仿真环境，并包含一套学习算法套件，可在数秒到数分钟内完成智能体训练：

*   基线学习算法，例如
    [PPO](https://github.com/google/brax/blob/main/brax/training/agents/ppo)、
    [SAC](https://github.com/google/brax/blob/main/brax/training/agents/sac)、
    [ARS](https://github.com/google/brax/blob/main/brax/training/agents/ars) 以及
    [演化策略（Evolutionary Strategies）](https://github.com/google/brax/blob/main/brax/training/agents/es)。
*   充分利用仿真器可微特性的学习算法，例如[解析策略梯度（Analytic Policy Gradients）](https://github.com/google/brax/blob/main/brax/training/agents/apg)。

## 统一 API，四大管线

Brax 提供了四种截然不同且易于互相切换的物理管线：

* [MuJoCo XLA - MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) - MuJoCo 物理引擎的 JAX 重新实现版本。
* [Generalized](https://github.com/google/brax/blob/main/brax/generalized/)
在[广义坐标（generalized coordinates）](https://en.wikipedia.org/wiki/Generalized_coordinates)下计算运动，采用类似于 [MuJoCo](https://mujoco.org/) 和 [TDS](https://github.com/erwincoumans/tiny-differentiable-simulator) 的动力学算法。
* [Positional](https://github.com/google/brax/blob/main/brax/positional/)
采用[基于位置的动力学（Position Based Dynamics）](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf)，这是一种求解关节与碰撞约束的快速且稳定的方法。
* [Spring](https://github.com/google/brax/blob/main/brax/spring/) 提供快速且开销极低的仿真以供快速实验，采用电子游戏中常见的简易基于冲量的方法。

这些管线共享相同的 API，并可以在同一个仿真中并排运行。这使得 Brax 非常适合用于迁移学习实验以及弥合仿真与现实世界之间的差距（Sim-to-Real）。

## 快速开始：云端 Colab

通过一系列 Colab 笔记本轻松快速地探索 Brax：

* [Brax 基础（Brax Basics）](https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb) 介绍 Brax API，并展示如何仿真基础物理图元。
* [Brax 训练（Brax Training）](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb) 介绍 Brax 的训练算法，并支持直接在 Colab 中训练自定义策略，同时演示策略的加载与保存。
* [使用 MuJoCo XLA - MJX 进行 Brax 训练](https://colab.sandbox.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb) 演示在 Brax 中使用 `MJX` 物理仿真器进行训练。
* [在 GPU 上结合 PyTorch 进行 Brax 训练](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training_torch.ipynb) 演示如何将 Brax 与其他机器学习框架（此处为 PyTorch）结合以实现快速训练。

## 本地使用 Brax

如需从 PyPI 安装 Brax，请运行：

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install brax
```

您也可以通过 [Conda](https://docs.conda.io/en/latest/) 或 [Mamba](https://github.com/mamba-org/mamba) 安装：

```
conda install -c conda-forge brax  # 若使用 mamba 则替换为 mamba
```

或者，如需从源码安装 Brax，请克隆本仓库并进入目录，然后运行：

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

训练模型：

```
learn
```

支持在 NVIDIA GPU 上进行训练，但必须首先安装
[CUDA、cuDNN 以及支持 GPU 的 JAX](https://github.com/google/jax#installation)。

## 了解更多

如需深入了解 Brax 的设计与性能特性，请参阅发表于 [NeurIPS 2021](https://nips.cc/Conferences/2021) [数据集与基准赛道（Datasets and Benchmarks Track）](https://neurips.cc/Conferences/2021/CallForDatasetsBenchmarks) 的论文：[Brax -- A Differentiable Physics Engine for Large Scale Rigid Body Simulation](https://arxiv.org/abs/2106.13281)。

## 引用 Brax

如果您在学术出版物中引用了 Brax，请使用以下 BibTeX 格式：

```
@software{brax2021github,
  author = {C. Daniel Freeman and Erik Frey and Anton Raichuk and Sertan Girgin and Igor Mordatch and Olivier Bachem},
  title = {Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation},
  url = {http://github.com/google/brax},
  version = {0.14.2},
  year = {2021},
}
```

## 致谢

自最初发布以来，Brax 经历了长足的发展。我们由衷感谢并赞扬以下各位同仁：

* Manu Orsini 与 Nikola Momchev，他们对 Brax 的训练算法进行了重大重构，使其更易用、复用性更高。
* Erwin Coumans，他热心提供了指导与建议，以及来自 [Tiny Differentiable Simulator](https://github.com/erwincoumans/tiny-differentiable-simulator) 的诸多宝贵参考。
* Baruch Tabanpour，协助发布 Brax v2 并全面重构了接触库（contact library）的同事。
* [Shixiang Shane Gu](https://sites.google.com/corp/view/gugurus) 与 [Hiroki Furuta](https://frt03.github.io/)，他们为 Brax 贡献了 BIG-Gym、Braxlines 以及场景组合器。
* 我们优秀的[开源协作者与贡献者](https://github.com/google/brax/graphs/contributors)。衷心感谢大家！

---

> **免责声明**：本文档由社区志愿者进行本地化翻译，旨在促进跨语言的技术交流与学习。如翻译内容与[英文原版](README.md)存在出入，请以英文原版为准。
