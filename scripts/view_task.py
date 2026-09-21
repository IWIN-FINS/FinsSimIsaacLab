#!/usr/bin/env python3
"""Open a registered FinsSim task in the Isaac Sim GUI without training."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Registered Gymnasium task id.")
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from finssim_isaaclab_tasks.register import register_environments

    register_environments()
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    try:
        env.reset(seed=42)
        action_dim = env.unwrapped.action_space.shape[-1]
        actions = torch.zeros(
            (args_cli.num_envs, action_dim),
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        print(f"Viewing task {args_cli.task} with {args_cli.num_envs} environment(s). Press Ctrl-C to exit.")
        while simulation_app.is_running():
            _, _, terminated, truncated, _ = env.step(actions)
            if torch.any(terminated | truncated):
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
