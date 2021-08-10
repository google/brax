""" Wrapper around a Brax GymWrapper, that converts its outputs to PyTorch tensors.

This conversion happens directly on-device, witout the need to move values to the CPU.
"""
from typing import Optional, Union
import gym

from brax.envs.wrappers import GymWrapper, VectorGymWrapper

# NOTE: The following line will raise a warning and raise ImportError if `torch` isn't
# available.
from brax.io.torch import torch_to_jax, jax_to_torch, Device


class JaxToTorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(
        self, env: Union[GymWrapper, VectorGymWrapper], device: Optional[Device] = None
    ):
        """Creates a Wrapper around a `GymWrapper` or `VectorGymWrapper` so it outputs
        PyTorch tensors.

        Parameters
        ----------
        env : Union[GymWrapper, VectorGymWrapper]
            A `GymWrapper` or `VectorGymWrapper` to wrap.
        device : Union[torch.device, str], optional
            device on which to move the Tensors. Defaults to `None`, in which case the
            tensors will be on the same devices as their Jax equivalents.
        """
        super().__init__(env)
        self.device: Optional[Device] = device

    def observation(self, observation):
        return jax_to_torch(observation, device=self.device)

    def action(self, action):
        return torch_to_jax(action)

    def reward(self, reward):
        return jax_to_torch(reward, device=self.device)

    def done(self, done):
        return jax_to_torch(done, device=self.device)

    def info(self, info):
        return jax_to_torch(info, device=self.device)

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
