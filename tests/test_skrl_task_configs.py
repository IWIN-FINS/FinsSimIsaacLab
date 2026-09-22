from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import yaml

from finssim_isaaclab_tasks.register import register_environments


def test_finsrov_tasks_register_skrl_ppo_configs() -> None:
    """Both FinsROV policy contracts expose distinct SKRL PPO entry points."""
    register_environments()
    expected = {
        "FinsSim-FinsROV-HoldForPosition-v0": (16, 8),
        "FinsSim-FinsROV-HoldForPosition-Wrench-v0": (16, 6),
    }
    for task_id, (observation_dim, action_dim) in expected.items():
        spec = gym.spec(task_id)
        entry_point = spec.kwargs["skrl_cfg_entry_point"]
        module_name, filename = entry_point.split(":", maxsplit=1)
        module = __import__(module_name, fromlist=["__file__"])
        config_path = Path(module.__file__).parent / filename
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert config["agent"]["class"] == "PPO"
        assert config["agent"]["rollouts"] == 24
        assert config["agent"]["rewards_shaper_scale"] == 1.0
        assert config["agent"]["observation_preprocessor"] is None
        assert config["models"]["policy"]["network"][0]["layers"] == [64, 64]
        assert observation_dim == 16
        assert action_dim in {6, 8}
