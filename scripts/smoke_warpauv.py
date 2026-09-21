#!/usr/bin/env python3
"""Start the required fresh 64-environment, one-iteration WarpAUV smoke run."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    train_script = Path(__file__).with_name("train.py")
    defaults = [
        "--task",
        "FinsSim-WarpAUV-Direct-v0",
        "--num_envs",
        "64",
        "--max_iterations",
        "1",
        "--headless",
    ]
    sys.argv = [str(train_script), *defaults, *sys.argv[1:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()

