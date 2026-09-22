"""Run an upstream SKRL script after registering FinsSim Isaac tasks."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def run(action: str) -> None:
    if action not in {"train", "play"}:
        raise ValueError(f"Unsupported SKRL action: {action}")
    configured_path = os.environ.get("ISAACLAB_PATH")
    if not configured_path:
        raise EnvironmentError("Set ISAACLAB_PATH to the Isaac Lab installation directory.")
    isaaclab_path = Path(configured_path).expanduser().resolve()
    runner = isaaclab_path / "scripts" / "reinforcement_learning" / f"{action}.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Isaac Lab SKRL {action} script not found: {runner}")

    # The SKRL task runner does not expose an --external_callback argument.
    # Importing this package executes the Gym
    # registrations before upstream resolve_task_config() is called.
    from finssim_isaaclab_tasks.register import register_environments

    register_environments()
    os.chdir(isaaclab_path)
    if str(runner.parent) not in sys.path:
        sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), "--rl_library", "skrl", *sys.argv[1:]]
    runpy.run_path(str(runner), run_name="__main__")
