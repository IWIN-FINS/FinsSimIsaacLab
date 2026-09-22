from __future__ import annotations

import math

import gymnasium as gym
import torch

from finssim_isaaclab_tasks.register import register_environments
from finssim_isaaclab_tasks.tasks.direct.finsrov_trajectory_tracking.contract import (
    build_trajectory_observation,
    trajectory_yaw_error,
)
from finssim_isaaclab_tasks.tasks.direct.finsrov_trajectory_tracking.trajectory import (
    CIRCLE,
    STRAIGHT_LINE,
    circle_feasible_angular_speed,
    evaluate_trajectory,
)


def test_finsrov_trajectory_task_registers_30d_8d_contract() -> None:
    register_environments()
    spec = gym.spec("FinsSim-FinsROV-TrajectoryTracking-v0")
    assert spec.kwargs["env_cfg_entry_point"].endswith("env_cfg:FinsROVTrajectoryTrackingEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "rsl_rl_ppo_cfg:FinsROVTrajectoryTrackingPPORunnerCfg"
    )


def test_line_and_circle_references_use_isaac_task_axes() -> None:
    origin = torch.zeros(2, 3)
    scale = torch.tensor([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    profile = torch.tensor([STRAIGHT_LINE, CIRCLE])
    position, velocity = evaluate_trajectory(
        profile,
        origin,
        scale,
        phase=torch.zeros(2),
        angular_speed=torch.tensor([0.0, 1.0]),
        line_direction=torch.ones(2),
        time_s=torch.zeros(2),
        duration_s=4.0,
    )

    assert torch.allclose(position, torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    # Unity [x forward, y up, z left] maps to Isaac [x forward, y left, z up].
    assert torch.allclose(velocity[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(circle_feasible_angular_speed(scale, 0.2, 0.16), torch.full((2,), 0.1))


def test_trajectory_observation_is_30d_and_yaw_term_is_horizontal() -> None:
    root_pos = torch.zeros(2, 3)
    root_quat = torch.zeros(2, 4)
    root_quat[:, 3] = 1.0
    reference_velocity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    preview = torch.zeros(2, 4, 3)
    preview[0, :, 0] = torch.arange(4) * 0.1
    preview[1, :, 1] = torch.arange(4) * 0.1

    observation = build_trajectory_observation(
        root_pos,
        root_quat,
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        preview,
        reference_velocity,
        progress=torch.zeros(2),
    )
    assert observation.shape == (2, 30)
    assert torch.allclose(observation[:, 21:24], torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1))

    yaw_error, valid = trajectory_yaw_error(root_quat, reference_velocity)
    assert torch.allclose(yaw_error, torch.tensor([0.0, math.pi / 2]), atol=1.0e-5)
    assert torch.all(valid == 1)

