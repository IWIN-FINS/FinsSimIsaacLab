from __future__ import annotations

from typing import Sequence

import numpy as np

from .frame import FINS_TO_ISAAC_BASIS, convert_quaternion_xyzw_to_wxyz, transform_body_angular_vector


def build_warpauv_observation(
    *,
    target_position_fins: Sequence[float],
    target_quaternion_fins_xyzw: Sequence[float],
    position_fins: Sequence[float],
    orientation_fins_xyzw: Sequence[float],
    linear_velocity_body_fins: Sequence[float],
    angular_velocity_body_fins: Sequence[float],
) -> np.ndarray:
    """Build the exact 17-value observation expected by WarpAUV.

    The Isaac task uses ``goal_quaternion_wxyz``, then the body-frame offset
    from the goal position to the vehicle, current quaternion, body linear
    velocity and body angular velocity. The ROS state is FinsROV's x-forward,
    y-up, z-left basis and is converted to Isaac's x-forward, y-left, z-up.
    """

    target_position = np.asarray(target_position_fins, dtype=np.float64).reshape(3)
    position = np.asarray(position_fins, dtype=np.float64).reshape(3)
    orientation = np.asarray(orientation_fins_xyzw, dtype=np.float64).reshape(4)
    target_orientation = np.asarray(target_quaternion_fins_xyzw, dtype=np.float64).reshape(4)
    offset_world_isaac = FINS_TO_ISAAC_BASIS @ (target_position - position)
    orientation_wxyz = convert_quaternion_xyzw_to_wxyz(target_orientation, FINS_TO_ISAAC_BASIS)
    current_wxyz = convert_quaternion_xyzw_to_wxyz(orientation, FINS_TO_ISAAC_BASIS)
    linear_body = FINS_TO_ISAAC_BASIS @ np.asarray(linear_velocity_body_fins, dtype=np.float64).reshape(3)
    angular_body = transform_body_angular_vector(angular_velocity_body_fins)

    # Isaac quat_apply(quat_conjugate(root_quat), target-root) is equivalent
    # to rotating the world offset by the inverse current body orientation.
    current_rotation = _quat_wxyz_to_matrix(current_wxyz)
    offset_body = current_rotation.T @ offset_world_isaac
    observation = np.concatenate(
        [
            orientation_wxyz,
            offset_body,
            current_wxyz,
            linear_body,
            angular_body,
        ]
    )
    return observation.astype(np.float32, copy=False)


def _quat_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
