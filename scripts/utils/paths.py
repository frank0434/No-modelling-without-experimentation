"""
utils/paths.py
--------------
Single source for the project root and common directories, computed from this
file's location (cwd-independent). Import instead of re-deriving a project-root
finder in every script:

    from utils.paths import ROOT, DATA, OUTPUT
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/utils/ -> scripts/ -> ch3_model/
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
