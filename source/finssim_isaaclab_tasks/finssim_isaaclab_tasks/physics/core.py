"""Engine-independent, batched FinsSim hydrodynamics.

The formulas and modes mirror ``Marus.Hydrodynamics``.  Inputs and outputs are
always body-frame tensors; Isaac Lab tasks only perform the final PhysX write.
Quaternions use Isaac Lab 3's ``xyzw`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

HydrodynamicsMode = Literal[
    "off", "hydrostatic_only", "simplified_fossen", "fossen", "mesh", "fossen_mesh_residual"
]


@dataclass(frozen=True)
class HydrodynamicsProfileCfg:
    """Isaac-side equivalent of Unity's ``HydrodynamicsProfile``.

    Coefficients are expressed in the vehicle body frame in ``[u,v,w,p,q,r]``
    order.  The profile deliberately contains values, rather than Isaac objects,
    so it is usable from unit tests and Warp/Torch batch kernels.
    """

    mode: HydrodynamicsMode = "fossen"
    water_density: float = 1027.0
    gravity_magnitude: float = 9.81
    displaced_volume: float = 0.0
    center_of_buoyancy_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_damping: tuple[float, float, float, float, float, float] = (0.0,) * 6
    quadratic_damping: tuple[float, float, float, float, float, float] = (0.0,) * 6
    forward_speed_damping: tuple[float, float, float, float, float, float] = (0.0,) * 6
    added_mass_diagonal: tuple[float, float, float, float, float, float] = (0.0,) * 6
    added_mass_full: tuple[tuple[float, float, float, float, float, float], ...] | None = None
    enable_added_mass_force: bool = True
    enable_added_mass_coriolis: bool = True
    acceleration_filter_alpha: float = 0.3
    max_explicit_added_mass_acceleration: tuple[float, float, float, float, float, float] = (0.0,) * 6
    surface_residual_scale: float = 0.25
    form_drag_coefficient: float = 1.0
    skin_drag_coefficient: float = 0.02
    form_drag_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    form_drag_torque_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    skin_drag_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    skin_drag_torque_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_buoyancy_volume_scale: float = 1.0
    min_triangle_area: float = 1.0e-6
    max_force_magnitude: float = 10000.0
    max_torque_magnitude: float = 10000.0


@dataclass
class HydrodynamicsState:
    """Batched rigid-body and fluid state, with leading dimension ``num_envs``."""

    quat_w_xyzw: torch.Tensor
    linear_velocity_b: torch.Tensor
    angular_velocity_b: torch.Tensor
    water_linear_velocity_b: torch.Tensor
    water_angular_velocity_b: torch.Tensor
    submerged_fraction: torch.Tensor | None = None

    @property
    def relative_velocity(self) -> torch.Tensor:
        return torch.cat(
            (self.linear_velocity_b - self.water_linear_velocity_b, self.angular_velocity_b - self.water_angular_velocity_b),
            dim=-1,
        )


@dataclass
class HydrodynamicsDiagnostics:
    wrench_b: torch.Tensor
    hydrostatic_b: torch.Tensor
    damping_b: torch.Tensor
    added_mass_acceleration_b: torch.Tensor
    added_mass_coriolis_b: torch.Tensor
    mesh_b: torch.Tensor
    relative_velocity_b: torch.Tensor
    relative_acceleration_b: torch.Tensor
    submerged_volume: torch.Tensor
    wetted_area: torch.Tensor

    @classmethod
    def zeros(cls, count: int, device: torch.device, dtype: torch.dtype) -> "HydrodynamicsDiagnostics":
        vector = torch.zeros(count, 6, device=device, dtype=dtype)
        scalar = torch.zeros(count, device=device, dtype=dtype)
        return cls(vector, vector.clone(), vector.clone(), vector.clone(), vector.clone(), vector.clone(), vector.clone(), vector.clone(), scalar, scalar.clone())


def _quat_rotate_inverse_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into body coordinates for normalized ``xyzw`` quaternions."""

    xyz = quat[..., :3]
    w = quat[..., 3:4]
    return vector - 2.0 * torch.cross(xyz, torch.cross(xyz, vector, dim=-1) + w * vector, dim=-1)


class UnderwaterHydrodynamics:
    """Stateful vectorized implementation of every parametric MARUS backend."""

    def __init__(self, profile: HydrodynamicsProfileCfg, num_envs: int, device: torch.device | str):
        self.profile = profile
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._previous_relative_velocity = torch.zeros(num_envs, 6, device=self.device)
        self._filtered_relative_acceleration = torch.zeros(num_envs, 6, device=self.device)
        self._has_previous_velocity = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._previous_relative_velocity[env_ids] = 0.0
        self._filtered_relative_acceleration[env_ids] = 0.0
        self._has_previous_velocity[env_ids] = False

    def compute(self, state: HydrodynamicsState, dt: float, mesh_wrench_b: torch.Tensor | None = None,
                submerged_volume: torch.Tensor | None = None, wetted_area: torch.Tensor | None = None) -> HydrodynamicsDiagnostics:
        count = state.linear_velocity_b.shape[0]
        dtype = state.linear_velocity_b.dtype
        result = HydrodynamicsDiagnostics.zeros(count, state.linear_velocity_b.device, dtype)
        relative_velocity = state.relative_velocity
        result.relative_velocity_b = relative_velocity
        mode = self.profile.mode
        if mode == "off":
            return result

        fraction = state.submerged_fraction
        if fraction is None:
            fraction = torch.ones(count, device=self.device, dtype=dtype)
        volume = submerged_volume if submerged_volume is not None else fraction * self.profile.displaced_volume
        result.submerged_volume = volume
        result.hydrostatic_b = self._hydrostatic_wrench(state.quat_w_xyzw, volume)

        if mode == "hydrostatic_only":
            result.wrench_b = result.hydrostatic_b
            return self._clamp(result)

        if mode in ("simplified_fossen", "fossen", "fossen_mesh_residual"):
            result.damping_b = self._diagonal_damping(relative_velocity)

        if mode in ("fossen", "fossen_mesh_residual"):
            acceleration = self._estimate_relative_acceleration(relative_velocity, dt)
            result.relative_acceleration_b = acceleration
            added_mass = self._added_mass(dtype)
            if self.profile.enable_added_mass_force:
                result.added_mass_acceleration_b = -torch.einsum("ij,nj->ni", added_mass, acceleration)
            if self.profile.enable_added_mass_coriolis:
                result.added_mass_coriolis_b = -self._added_mass_coriolis(added_mass, relative_velocity)

        if mode in ("mesh", "fossen_mesh_residual") and mesh_wrench_b is None:
            raise ValueError(f"Hydrodynamics mode '{mode}' requires mesh_wrench_b")
        if mode in ("mesh", "fossen_mesh_residual"):
            result.mesh_b = mesh_wrench_b
            if mode == "fossen_mesh_residual":
                result.mesh_b = result.mesh_b * self.profile.surface_residual_scale
        if wetted_area is not None:
            result.wetted_area = wetted_area

        result.wrench_b = (
            result.hydrostatic_b + result.damping_b + result.added_mass_acceleration_b + result.added_mass_coriolis_b + result.mesh_b
        )
        return self._clamp(result)

    def _hydrostatic_wrench(self, quat_w_xyzw: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        buoyancy_w = torch.zeros(quat_w_xyzw.shape[0], 3, device=self.device, dtype=quat_w_xyzw.dtype)
        buoyancy_w[:, 2] = volume * self.profile.water_density * self.profile.gravity_magnitude
        force_b = _quat_rotate_inverse_xyzw(quat_w_xyzw, buoyancy_w)
        cob = torch.tensor(self.profile.center_of_buoyancy_b, device=self.device, dtype=force_b.dtype).expand_as(force_b)
        return torch.cat((force_b, torch.cross(cob, force_b, dim=-1)), dim=-1)

    def _diagonal_damping(self, relative_velocity: torch.Tensor) -> torch.Tensor:
        linear = torch.tensor(self.profile.linear_damping, device=self.device, dtype=relative_velocity.dtype).abs()
        quadratic = torch.tensor(self.profile.quadratic_damping, device=self.device, dtype=relative_velocity.dtype).abs()
        forward = torch.tensor(self.profile.forward_speed_damping, device=self.device, dtype=relative_velocity.dtype).abs()
        coefficient = linear + quadratic * relative_velocity.abs() + forward * relative_velocity[:, :1].abs()
        return -coefficient * relative_velocity

    def _added_mass(self, dtype: torch.dtype) -> torch.Tensor:
        if self.profile.added_mass_full is not None:
            return torch.tensor(self.profile.added_mass_full, device=self.device, dtype=dtype).abs()
        return torch.diag(torch.tensor(self.profile.added_mass_diagonal, device=self.device, dtype=dtype).abs())

    def _estimate_relative_acceleration(self, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        if dt <= 1e-6:
            return torch.zeros_like(velocity)
        raw = (velocity - self._previous_relative_velocity) / dt
        alpha = max(0.0, min(1.0, self.profile.acceleration_filter_alpha))
        filtered = self._filtered_relative_acceleration * (1.0 - alpha) + raw * alpha
        filtered = torch.where(self._has_previous_velocity[:, None], filtered, torch.zeros_like(filtered))
        limits = torch.tensor(self.profile.max_explicit_added_mass_acceleration, device=self.device, dtype=velocity.dtype)
        if torch.any(limits > 0):
            filtered = torch.maximum(torch.minimum(filtered, limits), -limits)
        self._previous_relative_velocity.copy_(velocity)
        self._filtered_relative_acceleration.copy_(filtered)
        self._has_previous_velocity.fill_(True)
        return filtered

    @staticmethod
    def _added_mass_coriolis(added_mass: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        a = torch.einsum("ij,nj->ni", added_mass, velocity)
        a1, a2 = a[:, :3], a[:, 3:]
        v1, v2 = velocity[:, :3], velocity[:, 3:]
        return torch.cat((-torch.cross(a1, v2, dim=-1), -torch.cross(a1, v1, dim=-1) - torch.cross(a2, v2, dim=-1)), dim=-1)

    def _clamp(self, diagnostics: HydrodynamicsDiagnostics) -> HydrodynamicsDiagnostics:
        force_limit, torque_limit = self.profile.max_force_magnitude, self.profile.max_torque_magnitude
        if force_limit > 0:
            diagnostics.wrench_b[:, :3] = diagnostics.wrench_b[:, :3] * torch.clamp(
                force_limit / diagnostics.wrench_b[:, :3].norm(dim=-1, keepdim=True).clamp_min(force_limit), max=1.0
            )
        if torque_limit > 0:
            diagnostics.wrench_b[:, 3:] = diagnostics.wrench_b[:, 3:] * torch.clamp(
                torque_limit / diagnostics.wrench_b[:, 3:].norm(dim=-1, keepdim=True).clamp_min(torque_limit), max=1.0
            )
        diagnostics.wrench_b.nan_to_num_(0.0, 0.0, 0.0)
        return diagnostics
