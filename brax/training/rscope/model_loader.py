import pickle
import mujoco
from config import META_PATH

def load_model_and_data():
    """Load meta information and create the Mujoco model and data."""
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    mj_model = mujoco.MjModel.from_xml_path(meta['xml_path'], assets=meta['model_assets'])
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data, meta
