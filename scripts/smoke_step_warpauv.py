#!/usr/bin/env python3
"""Launch one WarpAUV environment, reset it, and perform a deterministic step."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch

    from finssim_isaaclab_tasks.register import register_environments
    from finssim_isaaclab_tasks.tasks.direct.warpauv.env_cfg import WarpAUVEnvCfg

    register_environments()
    cfg = WarpAUVEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    env = gym.make("FinsSim-WarpAUV-Direct-v0", cfg=cfg)
    try:
        observations, _ = env.reset(seed=42)
        actions = torch.zeros((1, 6), dtype=torch.float32, device=env.unwrapped.device)
        next_observations, reward, terminated, truncated, _ = env.step(actions)
        assert observations["policy"].shape == (1, 17)
        assert next_observations["policy"].shape == (1, 17)
        assert reward.shape == (1,)
        assert torch.isfinite(reward).all()
        print(
            "WarpAUV one-env reset/step: ok "
            f"reward={reward.item():.6f} terminated={terminated.item()} truncated={truncated.item()}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
