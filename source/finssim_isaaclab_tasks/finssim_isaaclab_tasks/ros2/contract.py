"""Pure-Python contract shared by the Isaac-side ROS 2 runner and its tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Fins controller: x forward, y up, z left.
# Isaac:            x forward, y left, z up.
# This is an improper basis change, so axial vectors need an extra minus sign.
FINS_TO_ISAAC_BASIS = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64
)


@dataclass(frozen=True)
class Ros2Topics:
    """Stable, isolated DDS interface of the Isaac FinsROV simulation."""

    clock: str = "/clock"
    thrusters: str = "/sim/finsrov/thrusters_out"
    reset: str = "/sim/finsrov/reset"
    pose: str = "/sim/finsrov/controller/pose"
    imu: str = "/sim/finsrov/controller/imu"
    depth: str = "/sim/finsrov/controller/depth"
    dvl: str = "/sim/finsrov/controller/dvl"
    odom: str = "/sim/finsrov/ground_truth/odom"
    bridge_status: str = "/sim/finsrov/bridge/status"


@dataclass(frozen=True)
class TruthState:
    """One Fins-controller-frame truth sample, with quaternion in ``xyzw``."""

    position_world: np.ndarray
    orientation_world_body_xyzw: np.ndarray
    linear_velocity_body: np.ndarray
    angular_velocity_body: np.ndarray
    linear_acceleration_body: np.ndarray


class ActionCommandBuffer:
    """Validate, clamp and expire the eight direct force commands safely."""

    def __init__(self, command_timeout_sec: float = 0.5) -> None:
        if command_timeout_sec < 0.0:
            raise ValueError("command_timeout_sec must be non-negative")
        self.command_timeout_sec = float(command_timeout_sec)
        self._force_n = np.zeros(8, dtype=np.float32)
        self._received_sim_time: float | None = None
        self._reset_requested = False

    def receive_force_n(self, values: Sequence[float], sim_time_sec: float) -> bool:
        force_n = np.asarray(values, dtype=np.float32).reshape(-1)
        if force_n.size != 8 or not np.all(np.isfinite(force_n)):
            return False
        self._force_n = np.clip(force_n, -7.0, 7.0)
        self._received_sim_time = float(sim_time_sec)
        return True

    def command_force_n(self, sim_time_sec: float) -> np.ndarray:
        if self._received_sim_time is None:
            return np.zeros(8, dtype=np.float32)
        if self.command_timeout_sec > 0.0 and sim_time_sec - self._received_sim_time > self.command_timeout_sec:
            return np.zeros(8, dtype=np.float32)
        return self._force_n.copy()

    def request_reset(self) -> None:
        self._reset_requested = True
        self._force_n.fill(0.0)
        self._received_sim_time = None

    def take_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested


def _quat_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm((x, y, z, w))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = np.asarray((x, y, z, w), dtype=np.float64) / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = ((m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale, 0.25 * scale)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = (0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[2, 1] - m[1, 2]) / scale)
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale, (m[0, 2] - m[2, 0]) / scale)
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale, (m[1, 0] - m[0, 1]) / scale)
    result = np.asarray(q, dtype=np.float64)
    return (result / max(np.linalg.norm(result), 1.0e-12)).astype(np.float32)


def isaac_truth_to_fins(
    position_world_isaac: Sequence[float],
    orientation_world_body_isaac_xyzw: Sequence[float],
    linear_velocity_body_isaac: Sequence[float],
    angular_velocity_body_isaac: Sequence[float],
    linear_acceleration_body_isaac: Sequence[float],
) -> TruthState:
    """Map Isaac truth into the Unity-compatible Fins controller convention."""

    basis = FINS_TO_ISAAC_BASIS
    orientation = basis @ _quat_xyzw_to_matrix(orientation_world_body_isaac_xyzw) @ basis.T
    return TruthState(
        position_world=(basis @ np.asarray(position_world_isaac, dtype=np.float64).reshape(3)).astype(np.float32),
        orientation_world_body_xyzw=_matrix_to_quat_xyzw(orientation),
        linear_velocity_body=(basis @ np.asarray(linear_velocity_body_isaac, dtype=np.float64).reshape(3)).astype(np.float32),
        angular_velocity_body=(-basis @ np.asarray(angular_velocity_body_isaac, dtype=np.float64).reshape(3)).astype(np.float32),
        linear_acceleration_body=(basis @ np.asarray(linear_acceleration_body_isaac, dtype=np.float64).reshape(3)).astype(np.float32),
    )
