"""Unity T2 30D observation and reward contract in Isaac tensors."""

from __future__ import annotations

import torch

from ..finsrov_hold_for_position.contract import isaac_to_unity_angular, isaac_to_unity_polar


OBSERVATION_SIZE = 30
PREVIEW_COUNT = 4


def _clip_normalized(value: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    safe_scale = max(abs(float(scale)), 1.0e-6)
    return torch.clamp(value / safe_scale, -abs(float(clip)), abs(float(clip)))


def build_trajectory_observation(
    root_pos_w_i: torch.Tensor,
    root_quat_w_i: torch.Tensor,
    root_lin_vel_b_i: torch.Tensor,
    root_ang_vel_b_i: torch.Tensor,
    reference_positions_w_i: torch.Tensor,
    reference_velocity_w_i: torch.Tensor,
    *,
    position_scale: float = 3.0,
    linear_velocity_scale: float = 1.0,
    angular_velocity_scale: float = 1.0,
    observation_clip: float = 2.0,
    progress: torch.Tensor,
) -> torch.Tensor:
    """Build Unity T2's 30 scalar observations.

    ``reference_positions_w_i`` has shape ``[N, 4, 3]`` and contains the
    current point followed by three 0.1 s preview points.
    """

    from isaaclab.utils.math import quat_apply, quat_conjugate

    if reference_positions_w_i.ndim != 3 or reference_positions_w_i.shape[1:] != (PREVIEW_COUNT, 3):
        raise ValueError("reference_positions_w_i must have shape [num_envs, 4, 3]")

    inverse_root = quat_conjugate(root_quat_w_i)
    # Isaac Lab's TorchScript ``quat_apply`` expects the quaternion and vector
    # batches to have the same leading shape; it does not broadcast [N, 4]
    # over [N, preview_count, 3].  Flatten the preview axis explicitly and
    # restore it after the body-frame rotation.
    num_envs = root_pos_w_i.shape[0]
    preview_count = reference_positions_w_i.shape[1]
    preview_quat = inverse_root[:, None, :].expand(-1, preview_count, -1).reshape(-1, 4)
    preview_offset_w_i = (reference_positions_w_i - root_pos_w_i[:, None, :]).reshape(-1, 3)
    target_offset_b_i = quat_apply(preview_quat, preview_offset_w_i).reshape(num_envs, preview_count, 3)
    target_offset_b_u = isaac_to_unity_polar(target_offset_b_i)
    target_offset_obs = _clip_normalized(target_offset_b_u, position_scale, observation_clip).reshape(root_pos_w_i.shape[0], -1)

    reference_velocity_b_i = quat_apply(inverse_root, reference_velocity_w_i)
    reference_velocity_b_u = isaac_to_unity_polar(reference_velocity_b_i)
    reference_velocity_obs = _clip_normalized(reference_velocity_b_u, linear_velocity_scale, observation_clip)

    linear_velocity_b_u = isaac_to_unity_polar(root_lin_vel_b_i)
    linear_velocity_obs = _clip_normalized(linear_velocity_b_u, linear_velocity_scale, observation_clip)

    angular_velocity_b_u = isaac_to_unity_angular(root_ang_vel_b_i)
    angular_velocity_obs = _clip_normalized(angular_velocity_b_u, angular_velocity_scale, observation_clip)

    world_up_i = torch.zeros_like(root_pos_w_i)
    world_up_i[:, 2] = 1.0
    body_up_i = quat_apply(inverse_root, world_up_i)
    body_up_u = isaac_to_unity_polar(body_up_i)

    tangent_speed_horizontal = torch.linalg.vector_norm(reference_velocity_b_u[:, (0, 2)], dim=-1)
    tangent_yaw = torch.atan2(reference_velocity_b_u[:, 2], reference_velocity_b_u[:, 0])
    tangent_yaw = torch.where(tangent_speed_horizontal > 1.0e-5, tangent_yaw, torch.zeros_like(tangent_yaw))
    tangent_yaw_obs = torch.stack((torch.sin(tangent_yaw), torch.cos(tangent_yaw)), dim=-1)

    progress = torch.clamp(progress.reshape(-1), 0.0, 1.0)
    progress_obs = torch.stack(
        (
            torch.sin(2.0 * torch.pi * progress),
            torch.cos(2.0 * torch.pi * progress),
            torch.sin(4.0 * torch.pi * progress),
            torch.cos(4.0 * torch.pi * progress),
        ),
        dim=-1,
    )

    observation = torch.cat(
        (
            target_offset_obs,
            reference_velocity_obs,
            linear_velocity_obs,
            angular_velocity_obs,
            body_up_u,
            tangent_yaw_obs,
            progress_obs,
        ),
        dim=-1,
    )
    if observation.shape[-1] != OBSERVATION_SIZE:
        raise RuntimeError(f"trajectory observation has {observation.shape[-1]} values, expected {OBSERVATION_SIZE}")
    return observation


def trajectory_yaw_error(
    root_quat_w_i: torch.Tensor,
    reference_velocity_w_i: torch.Tensor,
    *,
    minimum_reference_speed_mps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute horizontal yaw error and a valid-reference mask.

    The reference heading is the horizontal tangent of the trajectory.  The
    output angle is in radians and the mask disables the term when the
    reference has no meaningful horizontal direction.
    """

    from isaaclab.utils.math import quat_apply

    forward_w_i = quat_apply(
        root_quat_w_i,
        torch.tensor((1.0, 0.0, 0.0), device=root_quat_w_i.device, dtype=root_quat_w_i.dtype).expand(root_quat_w_i.shape[0], -1),
    )
    forward_xy = forward_w_i[:, :2]
    reference_xy = reference_velocity_w_i[:, :2]
    reference_speed = torch.linalg.vector_norm(reference_xy, dim=-1)
    valid = reference_speed > minimum_reference_speed_mps
    forward_heading = torch.atan2(forward_xy[:, 1], forward_xy[:, 0])
    reference_heading = torch.atan2(reference_xy[:, 1], reference_xy[:, 0])
    delta = reference_heading - forward_heading
    error = torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))
    return error, valid.to(dtype=root_quat_w_i.dtype)
