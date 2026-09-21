"""Unity ``ControlForMovingTargetReward`` observation contract for T3."""

from __future__ import annotations

import torch

from ..finsrov_hold_for_position.contract import isaac_to_unity_angular, isaac_to_unity_polar

OBSERVATION_SIZE = 13


def _clip_normalized(value: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    safe_scale = max(abs(float(scale)), 1.0e-6)
    return torch.clamp(value / safe_scale, -abs(float(clip)), abs(float(clip)))


def build_moving_target_observation(
    body_target_offset_i: torch.Tensor,
    target_velocity_b_i: torch.Tensor,
    vehicle_linear_velocity_b_i: torch.Tensor,
    vehicle_angular_velocity_b_i: torch.Tensor,
    target_distance: torch.Tensor,
    *,
    position_scale: float,
    target_velocity_scale: float,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
    velocity_clip: float,
) -> torch.Tensor:
    """Build the Unity T3 13D observation in its declared field order."""

    target_offset_u = isaac_to_unity_polar(body_target_offset_i)
    target_velocity_u = isaac_to_unity_polar(target_velocity_b_i)
    vehicle_linear_velocity_u = isaac_to_unity_polar(vehicle_linear_velocity_b_i)
    vehicle_angular_velocity_u = isaac_to_unity_angular(vehicle_angular_velocity_b_i)
    distance = torch.clamp(target_distance.reshape(-1, 1) / max(abs(position_scale), 1.0e-6), 0.0, 1.0)
    observation = torch.cat(
        (
            _clip_normalized(target_offset_u, position_scale, velocity_clip),
            _clip_normalized(target_velocity_u, target_velocity_scale, velocity_clip),
            _clip_normalized(vehicle_linear_velocity_u, linear_velocity_scale, velocity_clip),
            _clip_normalized(vehicle_angular_velocity_u, angular_velocity_scale, velocity_clip),
            distance,
        ),
        dim=-1,
    )
    if observation.shape[-1] != OBSERVATION_SIZE:
        raise RuntimeError(f"T3 observation has {observation.shape[-1]} values, expected {OBSERVATION_SIZE}")
    return observation
