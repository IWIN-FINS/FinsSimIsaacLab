#!/usr/bin/env python3
"""Reset and step any registered FinsSim Isaac task."""

from __future__ import annotations

import argparse
import traceback
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=4)
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
        observations, _ = env.reset(seed=42)
        for _ in range(args_cli.steps):
            actions = torch.zeros((args_cli.num_envs, env.unwrapped.action_space.shape[-1]), device=env.unwrapped.device)
            observations, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all()
        policy = observations["policy"] if isinstance(observations, dict) else observations
        print({"task": args_cli.task, "envs": args_cli.num_envs, "steps": args_cli.steps, "observation_shape": tuple(policy.shape), "reward_mean": float(reward.mean()), "terminated": int(terminated.sum()), "truncated": int(truncated.sum())})
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:  # Kit may replace sys.excepthook; report explicitly.
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
