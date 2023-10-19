# Copyright 2023 The Brax Authors.
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

"""setup.py for Brax.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="brax",
    version="0.9.3",
    description=("A differentiable physics engine written in JAX."),
    author="Brax Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/google/brax",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=["bin/learn"],
    install_requires=[
        "absl-py",
        "dataclasses; python_version < '3.7'",
        # TODO: remove dm_env after dropping legacy v1 code
        "dm_env",
        "etils",
        "flask",
        "flask_cors",
        "flax",
        # TODO: remove grpcio and gym after dropping legacy v1 code
        "grpcio",
        "gym",
        "jax>=0.4.6",
        "jaxlib>=0.4.6",
        "jaxopt",
        "jinja2",
        "ml_collections",
        "mujoco",
        "numpy",
        "optax",
        "Pillow",
        "pytinyrenderer",
        "scipy",
        "tensorboardX",
        "trimesh",
        "typing-extensions",
    ],
    extras_require={
        "develop": ["pytest", "transforms3d"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="JAX reinforcement learning rigidbody physics",
)
