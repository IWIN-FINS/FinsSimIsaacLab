"""MDP terms for underwater pose hold."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import reset_root_state_uniform, time_out  # noqa: F401
from isaaclab.managers import SceneEntityCfg


def _position_error_to_default(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return position error from the asset's per-environment hold target.

    Manager reset events place an asset at ``default_root_pose + env_origin``.
    Using only ``env_origin`` silently makes any non-zero default depth part of
    the tracking error and can terminate the environment immediately.
    """
    robot = env.scene[asset_cfg.name]
    target_position_w = robot.data.default_root_pose.torch[:, :3] + env.scene.env_origins
    return robot.data.root_pos_w.torch - target_position_w


def pose_hold_observation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    position_error_w = _position_error_to_default(env, asset_cfg)
    return torch.cat((position_error_w, robot.data.root_quat_w.torch, robot.data.root_lin_vel_b.torch, robot.data.root_ang_vel_b.torch), dim=-1)


def position_hold_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    error = _position_error_to_default(env, asset_cfg)
    return torch.exp(-torch.sum(error.square(), dim=-1))


def orientation_hold_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    quat = env.scene[asset_cfg.name].data.root_quat_w.torch
    return quat[:, 3].square()


def out_of_bounds(env, limit_m: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    error = _position_error_to_default(env, asset_cfg)
    return torch.any(error.abs() > limit_m, dim=-1)
