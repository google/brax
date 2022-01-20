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

"""General-purpose file interface to support different backing file APIs."""


import glob
import os
from typing import List


def Glob(pattern: str) -> List[str]:
  return glob.glob(pattern)


def Exists(pathname: str) -> bool:
  """Check if a path exists."""
  exist = None
  if exist is None:
    exist = os.path.exists(pathname)
  return exist


def MakeDirs(dirname: str):
  """Make directory if it doesn't exist."""
  exist_dir = False
  if not exist_dir:
    os.makedirs(dirname, exist_ok=True)


class File:
  """General purpose file resource."""

  def __init__(self, fileName: str, mode='r'):
    self.f = None
    if not self.f:
      self.f = open(fileName, mode)

  def __enter__(self):
    return self.f

  def __exit__(self, exc_type, exc_value, traceback):
    self.f.close()
