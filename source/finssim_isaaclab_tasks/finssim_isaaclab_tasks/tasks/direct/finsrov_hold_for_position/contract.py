"""The exact Unity ``HoldForPosition`` policy interface for FinsROV.

The simulator uses Isaac's body axes ``[forward, left, up]`` while the Unity
agent uses ``[forward, up, left]``.  This module is deliberately free of Isaac
runtime objects so its coordinate transform and policy ABI can be unit-tested.
Quaternions throughout this task are Isaac Lab 3 ``xyzw`` quaternions.
"""

from __future__ import annotations

from typing import Final

import torch


# Unity ``FinsROVAgentRuntime.DefaultThrusterOrder``.  These labels also match
# the ROS/hardware canonical names shown after the equals sign.
THRUSTER_NAMES: Final[tuple[str, ...]] = (
    "Vertical1=V_LF",
    "Vertical2=V_LB",
    "Vertical3=V_RB",
    "Vertical4=V_RF",
    "Horizontal1=H_LF",
    "Horizontal2=H_LB",
    "Horizontal3=H_RB",
    "Horizontal4=H_RF",
)

OBSERVATION_FIELDS: Final[tuple[str, ...]] = (
    "target_offset_body_x_norm",
    "target_offset_body_y_norm",
    "target_offset_body_z_norm",
    "relative_target_rot6d_00",
    "relative_target_rot6d_10",
    "relative_target_rot6d_20",
    "relative_target_rot6d_01",
    "relative_target_rot6d_11",
    "relative_target_rot6d_21",
    "linear_velocity_body_x_norm",
    "linear_velocity_body_y_norm",
    "linear_velocity_body_z_norm",
    "angular_velocity_body_x_norm",
    "angular_velocity_body_y_norm",
    "angular_velocity_body_z_norm",
    "target_distance_norm",
)

# v_isaac = B @ v_unity.  B is symmetric, involutory, and has det(B) = -1.
# A polar vector therefore maps Isaac -> Unity by component order [x, z, y].
# Angular velocity is an axial vector and receives the additional minus sign.
UNITY_TO_ISAAC_BASIS: Final[tuple[tuple[float, float, float], ...]] = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)

# Source: FinsSim/tools/finsrov_wrench_geometry.yaml, converted from Unity
# [x forward, y up, z left] to Isaac [x forward, y left, z up].  Positions are
# relative to Unity's Rigidbody center of mass, not the visual-root origin.
THRUSTER_POSITIONS_FROM_COM_B: Final[tuple[tuple[float, float, float], ...]] = (
    (0.08312216, 0.13304673, 0.049069572),
    (-0.13687684, 0.13304667, 0.049069580),
    (-0.13687672, -0.14295308, 0.049069563),
    (0.08312222, -0.14295302, 0.049069580),
    (0.19834375, 0.12746146, 0.037169563),
    (-0.23286586, 0.14669480, 0.037169548),
    (-0.23286586, -0.15660119, 0.037169563),
    (0.19834371, -0.13736765, 0.037169577),
)

THRUSTER_DIRECTIONS_B: Final[tuple[tuple[float, float, float], ...]] = (
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (-1.0, -1.0, 0.0),
)


def isaac_to_unity_polar(vector_i: torch.Tensor) -> torch.Tensor:
    """Map Isaac polar vectors into Unity FinsROV body coordinates."""

    return torch.stack((vector_i[..., 0], vector_i[..., 2], vector_i[..., 1]), dim=-1)


def isaac_to_unity_angular(vector_i: torch.Tensor) -> torch.Tensor:
    """Map Isaac angular velocity into Unity FinsROV body coordinates."""

    return -isaac_to_unity_polar(vector_i)


def _quat_to_matrix_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-not ``xyzw`` quaternions to rotation matrices."""

    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def relative_target_rotation_6d_unity(root_quat_w_i: torch.Tensor, goal_quat_w_i: torch.Tensor) -> torch.Tensor:
    """Return Unity's two-column 6D encoding of ``inv(root) * target``.

    Unity's ``AddRotation6DObservation`` writes matrix column zero followed by
    column one: ``[m00,m10,m20,m01,m11,m21]``.  The reflection between the two
    body-frame conventions must be applied on both sides of the relative
    rotation, not merely by permuting the six final values.
    """

    root_rotation_i = _quat_to_matrix_xyzw(root_quat_w_i)
    goal_rotation_i = _quat_to_matrix_xyzw(goal_quat_w_i)
    relative_rotation_i = root_rotation_i.transpose(-1, -2) @ goal_rotation_i
    basis = torch.as_tensor(UNITY_TO_ISAAC_BASIS, dtype=relative_rotation_i.dtype, device=relative_rotation_i.device)
    relative_rotation_unity = basis.transpose(-1, -2) @ relative_rotation_i @ basis
    return torch.cat((relative_rotation_unity[..., :, 0], relative_rotation_unity[..., :, 1]), dim=-1)


def yaw_error_magnitude_xyzw(goal_quat_w_i: torch.Tensor, root_quat_w_i: torch.Tensor) -> torch.Tensor:
    """Return the unsigned horizontal-heading error in Isaac's z-up world frame.

    The task controls only yaw.  Projecting the body-forward axis onto the
    Isaac world ``xy`` plane deliberately makes this error independent of
    roll/pitch.  The input convention is Isaac Lab 3 ``xyzw``.
    """

    goal_forward_w = _quat_to_matrix_xyzw(goal_quat_w_i)[..., :, 0]
    root_forward_w = _quat_to_matrix_xyzw(root_quat_w_i)[..., :, 0]
    goal_yaw = torch.atan2(goal_forward_w[..., 1], goal_forward_w[..., 0])
    root_yaw = torch.atan2(root_forward_w[..., 1], root_forward_w[..., 0])
    yaw_delta = goal_yaw - root_yaw
    return torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))


def _normalize_and_clip(value: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    safe_scale = max(abs(scale), 1.0e-6)
    safe_clip = max(abs(clip), 1.0e-6)
    return torch.clamp(value / safe_scale, min=-safe_clip, max=safe_clip)


def build_hold_for_position_observation(
    body_goal_offset_i: torch.Tensor,
    root_quat_w_i: torch.Tensor,
    goal_quat_w_i: torch.Tensor,
    linear_velocity_b_i: torch.Tensor,
    angular_velocity_b_i: torch.Tensor,
    *,
    position_scale: float,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
    velocity_clip: float,
) -> torch.Tensor:
    """Build Unity ``HoldForPosition.CollectObservations`` byte-for-byte by field.

    The returned policy tensor has exactly 16 scalar elements.  The target,
    linear velocity, and distance are polar vectors/scalars; angular velocity
    follows the axial-vector reflection rule described above.
    """

    local_target_offset = isaac_to_unity_polar(body_goal_offset_i)
    relative_rotation_6d = relative_target_rotation_6d_unity(root_quat_w_i, goal_quat_w_i)
    linear_velocity = isaac_to_unity_polar(linear_velocity_b_i)
    angular_velocity = isaac_to_unity_angular(angular_velocity_b_i)
    safe_position_scale = max(abs(position_scale), 1.0e-6)
    distance = torch.clamp(
        torch.linalg.vector_norm(local_target_offset, dim=-1, keepdim=True) / safe_position_scale,
        min=0.0,
        max=1.0,
    )
    return torch.cat(
        (
            local_target_offset / safe_position_scale,
            relative_rotation_6d,
            _normalize_and_clip(linear_velocity, linear_velocity_scale, velocity_clip),
            _normalize_and_clip(angular_velocity, angular_velocity_scale, velocity_clip),
            distance,
        ),
        dim=-1,
    )


def normalized_action_to_calibrated_force_request(
    actions: torch.Tensor,
    *,
    max_forward_force_n: torch.Tensor | tuple[float, ...],
    max_reverse_force_n: torch.Tensor | tuple[float, ...],
) -> torch.Tensor:
    """Map normalized per-thruster actions to calibrated force requests.

    Positive and negative thrust authority is asymmetric. For each canonical
    FinsROV thruster ``i`` this applies ``F_i = a_i * Fmax_i`` after clipping
    ``a_i`` to ``[-1, 1]``, selecting the calibrated forward or reverse limit
    by the action sign. No additional global action-force scale is used.
    """

    clipped_actions = torch.clamp(actions, -1.0, 1.0)
    forward_limit = torch.as_tensor(max_forward_force_n, device=actions.device, dtype=actions.dtype).reshape(-1)
    reverse_limit = torch.as_tensor(max_reverse_force_n, device=actions.device, dtype=actions.dtype).reshape(-1)
    if forward_limit.numel() != actions.shape[-1] or reverse_limit.numel() != actions.shape[-1]:
        raise ValueError(
            "calibrated force-limit count must equal the action dimension: "
            f"actions={actions.shape[-1]}, forward={forward_limit.numel()}, reverse={reverse_limit.numel()}"
        )
    if not bool(torch.isfinite(forward_limit).all() and torch.isfinite(reverse_limit).all()):
        raise ValueError("calibrated force limits must be finite")
    if bool(torch.any(forward_limit <= 0.0) or torch.any(reverse_limit <= 0.0)):
        raise ValueError("calibrated force limits must be strictly positive magnitudes")
    return torch.where(
        clipped_actions >= 0.0,
        clipped_actions * forward_limit,
        clipped_actions * reverse_limit,
    )
