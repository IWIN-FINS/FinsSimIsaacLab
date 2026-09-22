from __future__ import annotations

import torch

from finssim_isaaclab_tasks.tasks.direct.finsrov_t3.contract import (
    build_moving_target_observation,
)


def test_finsrov_t3_observation_is_13d() -> None:
    batch = 4
    observation = build_moving_target_observation(
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch),
        position_scale=3.0,
        target_velocity_scale=0.2,
        linear_velocity_scale=1.0,
        angular_velocity_scale=1.0,
        velocity_clip=2.0,
    )
    assert observation.shape == (batch, 13)
    assert torch.isfinite(observation).all()
