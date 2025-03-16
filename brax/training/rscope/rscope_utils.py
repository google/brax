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

"""Brax training rscope utils."""

import datetime
import os
import shutil
import pickle
import jax
import numpy as np
from pathlib import PosixPath
from typing import Union, Optional, Dict, Any


BASE_PATH = "/tmp/rscope/active_run"
TEMP_PATH = "/tmp/rscope/temp"


def rscope_init(xml_path: Union[PosixPath, str], 
                model_assets: Optional[Dict[str, Any]] = None):
    # clear the active run directory.
    if os.path.exists(BASE_PATH):
        shutil.rmtree(BASE_PATH)
    os.makedirs(BASE_PATH)

    if not isinstance(xml_path, str):
        xml_path = xml_path.as_posix()

    rscope_meta = {
        "xml_path": xml_path,
        "model_assets": model_assets
    }
    # Make the base path and temp path if they don't exist.
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)

    with open(os.path.join(BASE_PATH, "rscope_meta.pkl"), "wb") as f:
        pickle.dump(rscope_meta, f)


def dump_eval(eval: dict):
    # write to <datetime>.mj_unroll.
    now = datetime.datetime.now()
    now_str = now.strftime("%Y_%m_%d-%H_%M_%S")
    # ensure it's numpy.
    eval = jax.tree.map(lambda x: np.array(x), eval)
    # 2 stages to ensure atomicity.
    temp_path = os.path.join(TEMP_PATH, f"partial_transition.tmp")
    final_path = os.path.join(BASE_PATH, f"{now_str}.mj_unroll")
    with open(temp_path, "wb") as f:
        pickle.dump(eval, f)    
    os.rename(temp_path, 
              final_path)
