"""Learning-to-Swim equivalent-box water model and Unity FinsROV actuators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate

from .contract import THRUSTER_DIRECTIONS_B, THRUSTER_POSITIONS_FROM_COM_B, normalized_action_to_calibrated_force_request


@dataclass(frozen=True)
class EquivalentBoxHydrodynamicsCfg:
    """Parameters for the rectangular equivalent-inertia-body model.

    The formula is intentionally the one in Learning to Swim's
    ``rigid_body_hydrodynamics.py``.  Values below are FinsROV values expressed
    in Isaac's ``[forward, left, up]`` body convention.
    """

    water_density: float
    water_dynamic_viscosity: float
    gravity_magnitude: float
    mass: float
    inertia_b: tuple[float, float, float]
    displaced_volume: float
    com_to_cob_b: tuple[float, float, float]
    com_to_cob_randomization_radius: float
    volume_range: tuple[float, float]
    enable_domain_randomization: bool = True


@dataclass
class EquivalentBoxHydrodynamicsDiagnostics:
    """Batched body-frame wrench decomposition, useful for regression tests."""

    wrench_b: torch.Tensor
    buoyancy_b: torch.Tensor
    quadratic_drag_b: torch.Tensor
    viscous_drag_b: torch.Tensor


class EquivalentBoxHydrodynamics:
    """Batched Learning-to-Swim hydrostatics plus equivalent-box drag.

    Gravity is still supplied by PhysX.  This class supplies the opposing
    hydrostatic buoyancy and the two drag terms, exactly as the reference task.
    """

    def __init__(self, cfg: EquivalentBoxHydrodynamicsCfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.masses = torch.full((num_envs, 1), cfg.mass, device=self.device)
        self.inertias = torch.tensor(cfg.inertia_b, device=self.device).reshape(1, 3).repeat(num_envs, 1)
        self.base_com_to_cob = torch.tensor(cfg.com_to_cob_b, device=self.device).reshape(1, 3)
        self.com_to_cob = self.base_com_to_cob.repeat(num_envs, 1)
        self.volumes = torch.full((num_envs, 1), cfg.displaced_volume, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.com_to_cob[env_ids] = self.base_com_to_cob
        self.volumes[env_ids] = self.cfg.displaced_volume
        if not self.cfg.enable_domain_randomization:
            return
        self.com_to_cob[env_ids] += self._sample_sphere(len(env_ids), self.cfg.com_to_cob_randomization_radius)
        self.volumes[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.cfg.volume_range)

    def compute(
        self,
        root_quat_w_xyzw: torch.Tensor,
        linear_velocity_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
    ) -> EquivalentBoxHydrodynamicsDiagnostics:
        """Compute Learning-to-Swim buoyancy, quadratic, and viscous wrenches."""

        buoyancy_w = torch.zeros_like(linear_velocity_b)
        buoyancy_w[:, 2] = self.cfg.water_density * self.cfg.gravity_magnitude * self.volumes.squeeze(-1)
        buoyancy_force_b = quat_apply(quat_conjugate(root_quat_w_xyzw), buoyancy_w)
        buoyancy_torque_b = torch.cross(self.com_to_cob, buoyancy_force_b, dim=-1)

        half_dimensions = self._inferred_half_dimensions()
        rj = torch.roll(half_dimensions, shifts=1, dims=1)
        rk = torch.roll(half_dimensions, shifts=-1, dims=1)
        quadratic_force_b = -2.0 * self.cfg.water_density * rj * rk * linear_velocity_b.abs() * linear_velocity_b
        quadratic_torque_b = (
            -0.5
            * self.cfg.water_density
            * half_dimensions
            * (rj.pow(4) + rk.pow(4))
            * angular_velocity_b.abs()
            * angular_velocity_b
        )

        equivalent_radius = half_dimensions.mean(dim=1, keepdim=True)
        viscous_force_b = -6.0 * self.cfg.water_dynamic_viscosity * torch.pi * equivalent_radius * linear_velocity_b
        viscous_torque_b = -8.0 * self.cfg.water_dynamic_viscosity * torch.pi * equivalent_radius.pow(3) * angular_velocity_b

        buoyancy_b = torch.cat((buoyancy_force_b, buoyancy_torque_b), dim=-1)
        quadratic_drag_b = torch.cat((quadratic_force_b, quadratic_torque_b), dim=-1)
        viscous_drag_b = torch.cat((viscous_force_b, viscous_torque_b), dim=-1)
        return EquivalentBoxHydrodynamicsDiagnostics(
            wrench_b=buoyancy_b + quadratic_drag_b + viscous_drag_b,
            buoyancy_b=buoyancy_b,
            quadratic_drag_b=quadratic_drag_b,
            viscous_drag_b=viscous_drag_b,
        )

    def _inferred_half_dimensions(self) -> torch.Tensor:
        return torch.sqrt(
            (3.0 / (2.0 * self.masses.repeat(1, 3)))
            * (torch.roll(self.inertias, shifts=1, dims=1) + torch.roll(self.inertias, shifts=-1, dims=1) - self.inertias)
        )

    def _sample_sphere(self, count: int, radius: float) -> torch.Tensor:
        directions = torch.randn(count, 3, device=self.device)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return directions * radius * torch.rand(count, 1, device=self.device).pow(1.0 / 3.0)


@dataclass(frozen=True)
class CalibratedNormalizedThrusterCfg:
    """FinsROV direct-thruster settings with calibrated force authority."""

    max_forward_force_n: tuple[float, ...] = (1.0,) * len(THRUSTER_POSITIONS_FROM_COM_B)
    max_reverse_force_n: tuple[float, ...] = (1.0,) * len(THRUSTER_POSITIONS_FROM_COM_B)
    command_delay_s: float = 0.02
    force_time_constant_s: float = 0.01
    max_force_slew_rate_n_s: float = 1000.0


class CalibratedNormalizedThrusters:
    """Eight FinsROV thrusters driven by normalized calibrated actions."""

    def __init__(self, cfg: CalibratedNormalizedThrusterCfg, num_envs: int, device: torch.device | str, physics_dt: float):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.physics_dt = physics_dt
        self.positions_b = torch.tensor(THRUSTER_POSITIONS_FROM_COM_B, device=self.device)
        directions_b = torch.tensor(THRUSTER_DIRECTIONS_B, device=self.device)
        self.directions_b = directions_b / torch.linalg.vector_norm(directions_b, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.max_forward_force_n = self._validate_force_limits(cfg.max_forward_force_n, "max_forward_force_n")
        self.max_reverse_force_n = self._validate_force_limits(cfg.max_reverse_force_n, "max_reverse_force_n")
        self.applied_force = torch.zeros(num_envs, len(THRUSTER_POSITIONS_FROM_COM_B), device=self.device)

        delay_steps = cfg.command_delay_s / max(physics_dt, 1.0e-6)
        self._delay_floor = int(delay_steps)
        self._delay_fraction = delay_steps - self._delay_floor
        self.command_history = torch.zeros(num_envs, self._delay_floor + 2, len(THRUSTER_POSITIONS_FROM_COM_B), device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.applied_force[env_ids] = 0.0
        self.command_history[env_ids] = 0.0

    def _validate_force_limits(self, values: tuple[float, ...], name: str) -> torch.Tensor:
        limits = torch.as_tensor(values, device=self.device, dtype=torch.float32).reshape(-1)
        expected_count = len(THRUSTER_POSITIONS_FROM_COM_B)
        if limits.numel() != expected_count:
            raise ValueError(f"{name} must contain {expected_count} values, got {limits.numel()}")
        if not bool(torch.isfinite(limits).all()) or bool(torch.any(limits <= 0.0)):
            raise ValueError(f"{name} must contain finite positive force magnitudes")
        return limits

    def compute(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return summed force, torque about CoM, and per-thruster applied force."""

        requested_force = normalized_action_to_calibrated_force_request(
            actions,
            max_forward_force_n=self.max_forward_force_n,
            max_reverse_force_n=self.max_reverse_force_n,
        )
        self.command_history = torch.roll(self.command_history, shifts=1, dims=1)
        self.command_history[:, 0] = requested_force
        delayed_force = (
            (1.0 - self._delay_fraction) * self.command_history[:, self._delay_floor]
            + self._delay_fraction * self.command_history[:, self._delay_floor + 1]
        )

        if self.cfg.force_time_constant_s > 0.0:
            alpha = 1.0 - torch.exp(torch.tensor(-self.physics_dt / self.cfg.force_time_constant_s, device=self.device))
            delayed_force = self.applied_force + alpha * (delayed_force - self.applied_force)
        if self.cfg.max_force_slew_rate_n_s > 0.0:
            max_delta = self.cfg.max_force_slew_rate_n_s * self.physics_dt
            delayed_force = torch.clamp(delayed_force, self.applied_force - max_delta, self.applied_force + max_delta)
        self.applied_force.copy_(delayed_force)

        forces_b = delayed_force.unsqueeze(-1) * self.directions_b.unsqueeze(0)
        torques_b = torch.cross(self.positions_b.unsqueeze(0), forces_b, dim=-1)
        return forces_b.sum(dim=1), torques_b.sum(dim=1), delayed_force
