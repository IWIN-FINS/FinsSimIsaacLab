from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


class IsaacWarpAUVThrustModel:
    """Reproduce WarpAUV's 6-action -> six-DOF wrench calculation.

    Isaac local axes are x-forward, y-left, z-up. Action order is
    [drive_left, drive_right, rear_left, rear_right, front_left, front_right].
    The first-order dynamics are disabled by default because the checked-in
    Isaac task currently zeroes its computed alpha before applying it.
    """

    ACTION_NAMES = ("drive_left", "drive_right", "rear_left", "rear_right", "front_left", "front_right")
    POSITIONS = np.asarray(
        [
            [-0.4127, 0.1506, -0.0889],
            [-0.4127, -0.1506, -0.0889],
            [-0.3030, 0.1461, -0.1587],
            [-0.3030, -0.1461, -0.1587],
            [0.0585, 0.1461, -0.0540],
            [0.0585, -0.1461, -0.0540],
        ],
        dtype=np.float64,
    )
    DIRECTIONS = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.7071067811865476, 0.7071067811865476],
            [0.0, -0.7071067811865476, 0.7071067811865476],
            [0.0, 0.7071067811865476, -0.7071067811865476],
            [0.0, -0.7071067811865476, -0.7071067811865476],
        ],
        dtype=np.float64,
    )

    def __init__(self, *, rotor_constant: float = 0.001, deadzone: float = 0.08) -> None:
        self.rotor_constant = float(rotor_constant)
        self.deadzone = abs(float(deadzone))

    @staticmethod
    def action_to_angular_speed(action: np.ndarray) -> np.ndarray:
        positive = -139.0 * action * action + 500.0 * action + 8.28
        negative = 161.0 * action * action + 517.86 * action - 5.72
        return np.where(action >= 0.0, positive, negative)

    def action_to_force(self, action: Sequence[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.size != 6:
            raise ValueError(f"Isaac WarpAUV action must have 6 values, got {values.size}")
        values = np.clip(values, -1.0, 1.0)
        active = np.where(np.abs(values) >= self.deadzone, values, 0.0)
        omega = self.action_to_angular_speed(active)
        force = self.rotor_constant * np.abs(omega) * omega
        return np.where(np.abs(values) >= self.deadzone, force, 0.0)

    def action_to_wrench(self, action: Sequence[float]) -> np.ndarray:
        forces = self.action_to_force(action)
        force_vectors = forces[:, None] * self.DIRECTIONS
        moments = np.cross(self.POSITIONS, force_vectors)
        return np.concatenate([force_vectors.sum(axis=0), moments.sum(axis=0)]).astype(np.float32)

    def wrench_envelope(self) -> np.ndarray:
        """Maximum absolute Isaac wrench over the six action box corners."""

        samples = np.asarray(list(product((-1.0, 1.0), repeat=6)), dtype=np.float64)
        wrenches = np.asarray([self.action_to_wrench(sample) for sample in samples], dtype=np.float64)
        return np.max(np.abs(wrenches), axis=0).astype(np.float32)
