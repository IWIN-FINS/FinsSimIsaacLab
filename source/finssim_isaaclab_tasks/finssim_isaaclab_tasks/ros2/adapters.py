"""Fins-specific semantics used by the native OmniGraph ROS 2 graph.

This module deliberately has no Isaac Kit or rclpy imports.  It is the single
home for Fins coordinate conversion and safe thruster-command interpretation,
and is shared by the runner, tests, and the documented OmniGraph node schema.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contract import ActionCommandBuffer, TruthState, isaac_truth_to_fins


class FinsRos2StateAdapter:
    """Convert Isaac task truth to the controller-facing Fins convention."""

    @staticmethod
    def from_isaac(
        position_world_isaac: Sequence[float],
        orientation_world_body_isaac_xyzw: Sequence[float],
        linear_velocity_body_isaac: Sequence[float],
        angular_velocity_body_isaac: Sequence[float],
        linear_acceleration_body_isaac: Sequence[float],
    ) -> TruthState:
        return isaac_truth_to_fins(
            position_world_isaac,
            orientation_world_body_isaac_xyzw,
            linear_velocity_body_isaac,
            angular_velocity_body_isaac,
            linear_acceleration_body_isaac,
        )


class FinsRos2CommandAdapter:
    """Validate native ROS2 Subscriber output before it reaches actuators."""

    def __init__(self, command_timeout_sec: float) -> None:
        self._buffer = ActionCommandBuffer(command_timeout_sec)

    @property
    def command_timeout_sec(self) -> float:
        return self._buffer.command_timeout_sec

    def receive_thrusters(self, values: Sequence[float] | None, sim_time_sec: float) -> bool:
        if values is None:
            return False
        return self._buffer.receive_force_n(values, sim_time_sec)

    def receive_reset(self, values: Sequence[float] | None) -> bool:
        if values is None:
            return False
        request = np.asarray(values, dtype=np.float32).reshape(-1)
        if request.size != 1 or not np.isfinite(request[0]) or request[0] <= 0.0:
            return False
        self._buffer.request_reset()
        return True

    def command_force_n(self, sim_time_sec: float) -> np.ndarray:
        return self._buffer.command_force_n(sim_time_sec)

    def take_reset_request(self) -> bool:
        return self._buffer.take_reset_request()
