from __future__ import annotations

from typing import Sequence

import numpy as np

from .isaac_thruster_model import IsaacWarpAUVThrustModel


class IsaacLabToFinsRovAdapter:
    """Convert Isaac's virtual wrench into the existing FinsROV wrench API.

    The auto scale maps the Isaac six-action envelope to the configured FinsROV
    allocator ranges. This is a normalization/capability mapping, not a claim
    that the two vehicles have identical hydrodynamics.
    """

    def __init__(
        self,
        *,
        thrust_model: IsaacWarpAUVThrustModel | None = None,
        auto_scale_from_envelope: bool = True,
        fins_reference_wrench: Sequence[float] = (10.0, 10.0, 10.0, 0.1, 0.1, 0.1),
        virtual_to_fins_scale: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        force_axis_permutation: Sequence[int] = (0, 2, 1),
        force_axis_signs: Sequence[float] = (1.0, 1.0, 1.0),
        torque_axis_permutation: Sequence[int] = (0, 2, 1),
        torque_axis_signs: Sequence[float] = (-1.0, -1.0, -1.0),
    ) -> None:
        self.thrust_model = thrust_model or IsaacWarpAUVThrustModel()
        self.auto_scale_from_envelope = bool(auto_scale_from_envelope)
        self.fins_reference_wrench = self._vec6(fins_reference_wrench, "fins_reference_wrench")
        self.virtual_to_fins_scale = self._vec6(virtual_to_fins_scale, "virtual_to_fins_scale")
        self.force_axis_permutation = self._perm3(force_axis_permutation, "force_axis_permutation")
        self.force_axis_signs = self._vec3(force_axis_signs, "force_axis_signs")
        self.torque_axis_permutation = self._perm3(torque_axis_permutation, "torque_axis_permutation")
        self.torque_axis_signs = self._vec3(torque_axis_signs, "torque_axis_signs")
        envelope_isaac = self.thrust_model.wrench_envelope()
        # The envelope is a magnitude. Preserve the configured torque signs
        # for actual wrench conversion, but never let them enter the scaling
        # denominator.
        self.isaac_wrench_envelope = np.abs(self._reorder_wrench(envelope_isaac))
        if self.auto_scale_from_envelope:
            self.wrench_scale = self.fins_reference_wrench / np.maximum(self.isaac_wrench_envelope, 1e-6)
        else:
            self.wrench_scale = np.ones(6, dtype=np.float32)

    @staticmethod
    def _vec3(values: Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size != 3 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain 3 finite values")
        return array

    @staticmethod
    def _vec6(values: Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size != 6 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain 6 finite values")
        return array

    @staticmethod
    def _perm3(values: Sequence[int], name: str) -> tuple[int, int, int]:
        permutation = tuple(int(value) for value in values)
        if sorted(permutation) != [0, 1, 2]:
            raise ValueError(f"{name} must be a permutation of [0, 1, 2]")
        return permutation

    def _reorder_wrench(self, wrench_isaac: Sequence[float]) -> np.ndarray:
        wrench = np.asarray(wrench_isaac, dtype=np.float32).reshape(6)
        force = wrench[:3][list(self.force_axis_permutation)] * self.force_axis_signs
        # Force is a polar vector: F_fins = B F_isaac. Torque is an axial
        # vector: tau_fins = det(B) B tau_isaac = -B tau_isaac.
        torque = wrench[3:][list(self.torque_axis_permutation)] * self.torque_axis_signs
        return np.concatenate([force, torque]).astype(np.float32, copy=False)

    def action_to_virtual_wrench_isaac(self, action: Sequence[float]) -> np.ndarray:
        return self.thrust_model.action_to_wrench(action)

    def action_to_fins_wrench(self, action: Sequence[float]) -> np.ndarray:
        virtual = self._reorder_wrench(self.action_to_virtual_wrench_isaac(action))
        return (virtual * self.wrench_scale * self.virtual_to_fins_scale).astype(np.float32, copy=False)

    def diagnostics(self) -> dict[str, list[float] | bool]:
        return {
            "auto_scale_from_envelope": self.auto_scale_from_envelope,
            "isaac_wrench_envelope_fins_order": self.isaac_wrench_envelope.tolist(),
            "fins_reference_wrench": self.fins_reference_wrench.tolist(),
            "wrench_scale": self.wrench_scale.tolist(),
            "virtual_to_fins_scale": self.virtual_to_fins_scale.tolist(),
        }
