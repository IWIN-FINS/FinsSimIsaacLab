"""Batched straight-line and circular references for the FinsROV T2 task.

The reference equations follow Unity's ``TrajectoryTrackingMath`` for the two
profiles retained by the Isaac task.  Local reference vectors are first
expressed in the Unity/controller convention ``[forward, up, left]`` and then
mapped to Isaac's task convention ``[forward, left, up]``.
"""

from __future__ import annotations

from typing import Final

import torch


STRAIGHT_LINE: Final[int] = 0
CIRCLE: Final[int] = 1


def evaluate_trajectory(
    profile: torch.Tensor,
    origin_w_i: torch.Tensor,
    scale_u: torch.Tensor,
    phase: torch.Tensor,
    angular_speed: torch.Tensor,
    line_direction: torch.Tensor,
    time_s: torch.Tensor,
    duration_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a batch of Unity-compatible references in Isaac world axes.

    Args:
        profile: Integer profile IDs, ``0`` for line and ``1`` for circle.
        origin_w_i: Reference origins in Isaac world/task axes ``[x, y, z]``.
        scale_u: Geometric scales in Unity/controller axes ``[x, y, z]``.
        phase: Initial phase for each environment.
        angular_speed: Signed circular angular speed in rad/s.
        line_direction: Signed line direction, either ``-1`` or ``+1``.
        time_s: Elapsed episode time for each environment.
        duration_s: Episode duration in seconds.

    Returns:
        Position and velocity in Isaac world/task axes.
    """

    safe_duration = max(float(duration_s), 1.0e-6)
    time_s = torch.as_tensor(time_s, device=origin_w_i.device, dtype=origin_w_i.dtype).reshape(-1)
    phase_t = phase + angular_speed * torch.clamp(time_s, min=0.0)

    line_progress = torch.clamp(time_s / safe_duration, 0.0, 1.0)
    start_x = -line_direction * scale_u[:, 0]
    end_x = line_direction * scale_u[:, 0]
    line_x = torch.lerp(start_x, end_x, line_progress)
    line_velocity_x = torch.where(
        time_s < safe_duration,
        (end_x - start_x) / safe_duration,
        torch.zeros_like(time_s),
    )

    circle_x = scale_u[:, 0] * torch.cos(phase_t)
    circle_z = scale_u[:, 2] * torch.sin(phase_t)
    circle_velocity_x = -scale_u[:, 0] * torch.sin(phase_t) * angular_speed
    circle_velocity_z = scale_u[:, 2] * torch.cos(phase_t) * angular_speed

    is_circle = profile == CIRCLE
    local_position_u = torch.stack(
        (
            torch.where(is_circle, circle_x, line_x),
            torch.zeros_like(line_x),
            torch.where(is_circle, circle_z, torch.zeros_like(line_x)),
        ),
        dim=-1,
    )
    local_velocity_u = torch.stack(
        (
            torch.where(is_circle, circle_velocity_x, line_velocity_x),
            torch.zeros_like(line_velocity_x),
            torch.where(is_circle, circle_velocity_z, torch.zeros_like(line_velocity_x)),
        ),
        dim=-1,
    )

    # Unity/controller [x forward, y up, z left] -> Isaac [x forward, y left, z up].
    local_position_i = local_position_u[..., (0, 2, 1)]
    local_velocity_i = local_velocity_u[..., (0, 2, 1)]
    return origin_w_i + local_position_i, local_velocity_i


def circle_feasible_angular_speed(
    scale_u: torch.Tensor,
    surge_limit_mps: torch.Tensor | float,
    sway_limit_mps: torch.Tensor | float,
    *,
    phase_samples: int = 96,
) -> torch.Tensor:
    """Compute a per-environment circular speed under axis velocity limits."""

    device = scale_u.device
    dtype = scale_u.dtype
    phases = torch.arange(phase_samples, device=device, dtype=dtype) * (2.0 * torch.pi / phase_samples)
    sin_phase = torch.sin(phases).unsqueeze(0)
    cos_phase = torch.cos(phases).unsqueeze(0)
    unit_surge = scale_u[:, 0:1] * sin_phase.abs()
    unit_sway = scale_u[:, 2:3] * cos_phase.abs()
    surge_limit = torch.as_tensor(surge_limit_mps, device=device, dtype=dtype).reshape(-1, 1)
    sway_limit = torch.as_tensor(sway_limit_mps, device=device, dtype=dtype).reshape(-1, 1)
    feasible = torch.full((scale_u.shape[0],), float("inf"), device=device, dtype=dtype)
    feasible = torch.minimum(feasible, torch.where(unit_surge > 1.0e-6, surge_limit / unit_surge, feasible[:, None]).amin(dim=1))
    feasible = torch.minimum(feasible, torch.where(unit_sway > 1.0e-6, sway_limit / unit_sway, feasible[:, None]).amin(dim=1))
    return torch.nan_to_num(feasible, nan=0.0, posinf=0.0).clamp_min(1.0e-4)
