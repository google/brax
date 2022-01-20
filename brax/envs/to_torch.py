# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper around a Brax GymWrapper, that converts outputs to PyTorch tensors.

This conversion happens directly on-device, without moving values to the CPU.
"""
from typing import Optional, Union

from brax.envs import wrappers
# NOTE: The following line will emit a warning and raise ImportError if `torch`
# isn't available.
from brax.io import torch
import gym


class JaxToTorchWrapper(gym.Wrapper):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self,
               env: Union[wrappers.GymWrapper, wrappers.VectorGymWrapper],
               device: Optional[torch.Device] = None):
    """Creates a Wrapper around a `GymWrapper` or `VectorGymWrapper` that outputs PyTorch tensors."""
    super().__init__(env)
    self.device: Optional[torch.Device] = device

  def observation(self, observation):
    return torch.jax_to_torch(observation, device=self.device)

  def action(self, action):
    return torch.torch_to_jax(action)

  def reward(self, reward):
    return torch.jax_to_torch(reward, device=self.device)

  def done(self, done):
    return torch.jax_to_torch(done, device=self.device)

  def info(self, info):
    return torch.jax_to_torch(info, device=self.device)

  def reset(self):
    obs = super().reset()
    return self.observation(obs)

  def step(self, action):
    action = self.action(action)
    obs, reward, done, info = super().step(action)
    obs = self.observation(obs)
    reward = self.reward(reward)
    done = self.done(done)
    info = self.info(info)
    return obs, reward, done, info
