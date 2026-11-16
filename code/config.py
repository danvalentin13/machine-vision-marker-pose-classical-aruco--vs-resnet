"""config.py -- the ONE place where the project's folders are defined.

Every script imports its paths from here (`from config import DATA_DIR, OUTPUT_DIR, TRAINING_DIR`), so when you
move to another machine you change at most the lines marked EDIT HERE -- or nothing at all.

Resolution order for each folder:
  1. environment variable (MV_DATA_DIR / MV_OUTPUT_DIR / MV_TRAINING_DIR), handy for a one-off run;
  2. the EDIT HERE value below, if you set one;
  3. the default: a folder next to `code/` (data, output, training), so it works wherever the project folder is copied.
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ----------------------------------------------------------------------------------------------
# EDIT HERE (leave as None to use the defaults  <project>/data, <project>/output, <project>/training)
# Example:  DATA_DIR_OVERRIDE = r"D:\mv_data"
DATA_DIR_OVERRIDE = None      # inputs: markers.json, calibration, test recordings
OUTPUT_DIR_OVERRIDE = None    # generated: figures, GIF/MP4, result tables
TRAINING_DIR_OVERRIDE = None  # everything for training the network: synthetic, real (training frames), models
# ----------------------------------------------------------------------------------------------

DATA_DIR = os.path.abspath(os.environ.get("MV_DATA_DIR") or DATA_DIR_OVERRIDE
                           or os.path.join(PROJECT_ROOT, "data"))
OUTPUT_DIR = os.path.abspath(os.environ.get("MV_OUTPUT_DIR") or OUTPUT_DIR_OVERRIDE
                             or os.path.join(PROJECT_ROOT, "output"))
TRAINING_DIR = os.path.abspath(os.environ.get("MV_TRAINING_DIR") or TRAINING_DIR_OVERRIDE
                               or os.path.join(PROJECT_ROOT, "training"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)


def data_path(*parts: str) -> str:
    """Path inside the data folder, e.g. data_path("calibration", "camera.npz")."""
    return os.path.join(DATA_DIR, *parts)


def training_path(*parts: str) -> str:
    """Path inside the training folder, e.g. training_path("synthetic", "synth_v1")."""
    return os.path.join(TRAINING_DIR, *parts)


def output_path(*parts: str) -> str:
    """Path inside the output folder; creates the parent folders."""
    p = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR    :", DATA_DIR)
    print("OUTPUT_DIR  :", OUTPUT_DIR)
    print("TRAINING_DIR:", TRAINING_DIR)
