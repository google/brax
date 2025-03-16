# Copyright 2025 The Brax Authors.
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

"""Model loader."""

import pickle

import mujoco

from brax.training.rscope.config import META_PATH


def load_model_and_data():
  """Load meta information and create the Mujoco model and data."""
  with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
  mj_model = mujoco.MjModel.from_xml_path(
      meta['xml_path'], assets=meta['model_assets']
  )
  mj_data = mujoco.MjData(mj_model)
  return mj_model, mj_data, meta
