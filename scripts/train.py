#!/usr/bin/env python3
"""Run the official unified Isaac Lab RSL-RL trainer with FinsSim tasks."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    configured_path = os.environ.get("ISAACLAB_PATH")
    if not configured_path:
        raise EnvironmentError("Set ISAACLAB_PATH to the Isaac Lab installation directory.")
    isaaclab_path = Path(configured_path).expanduser().resolve()
    runner = isaaclab_path / "scripts" / "reinforcement_learning" / "train.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Isaac Lab RSL-RL train script not found: {runner}")
    forwarded_args = sys.argv[1:]
    if "--external_callback" not in forwarded_args:
        forwarded_args.extend(["--external_callback", "finssim_isaaclab_tasks.register.register_environments"])
    os.chdir(isaaclab_path)
    # The unified dispatcher imports its sibling ``common.py`` as a top-level
    # module. ``runpy`` does not add the target script directory.
    if str(runner.parent) not in sys.path:
        sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), "--rl_library", "rsl_rl", *forwarded_args]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
