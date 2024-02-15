# Copyright 2024 The Brax Authors.
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
from typing import Optional

# NOTE: The following line will emit a warning and raise ImportError if `torch`
# isn't available.
from brax.io import torch
import gym


class TorchWrapper(gym.Wrapper):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self, env: gym.Env, device: Optional[torch.Device] = None):
    """Creates a gym Env to one that outputs PyTorch tensors."""
    super().__init__(env)
    self.device = device

  def reset(self):
    obs = super().reset()
    return torch.jax_to_torch(obs, device=self.device)

  def step(self, action):
    action = torch.torch_to_jax(action)
    obs, reward, done, info = super().step(action)
    obs = torch.jax_to_torch(obs, device=self.device)
    reward = torch.jax_to_torch(reward, device=self.device)
    done = torch.jax_to_torch(done, device=self.device)
    info = torch.jax_to_torch(info, device=self.device)
    return obs, reward, done, info
