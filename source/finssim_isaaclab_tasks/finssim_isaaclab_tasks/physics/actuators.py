"""Batched thruster curves and actuator dynamics migrated from MARUS."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ThrusterModelCfg:
    positions_b: tuple[tuple[float, float, float], ...]
    directions_b: tuple[tuple[float, float, float], ...]
    max_forward_force_n: float = 0.0
    max_reverse_force_n: float = 0.0
    deadzone: float = 0.0
    rotor_constant: float | None = None
    force_time_constant_s: float = 0.0
    max_force_slew_rate_n_s: float = 0.0


class ThrusterSystem:
    """Converts normalized actions to a body wrench with per-env actuator state."""

    def __init__(self, cfg: ThrusterModelCfg, num_envs: int, device: torch.device | str):
        self.cfg, self.num_envs, self.device = cfg, num_envs, torch.device(device)
        self.positions_b = torch.tensor(cfg.positions_b, device=self.device, dtype=torch.float32)
        directions = torch.tensor(cfg.directions_b, device=self.device, dtype=torch.float32)
        self.directions_b = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.applied_force = torch.zeros(num_envs, len(cfg.positions_b), device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.applied_force[torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids] = 0.0

    def compute(self, actions: torch.Tensor, dt: float, angular_speed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        command = torch.where(actions.abs() >= self.cfg.deadzone, actions, torch.zeros_like(actions))
        if angular_speed is not None and self.cfg.rotor_constant is not None:
            requested = self.cfg.rotor_constant * angular_speed.abs() * angular_speed
        else:
            requested = torch.where(command >= 0, command * self.cfg.max_forward_force_n, command * self.cfg.max_reverse_force_n)
        if self.cfg.force_time_constant_s > 0:
            alpha = min(1.0, dt / self.cfg.force_time_constant_s)
            requested = self.applied_force + alpha * (requested - self.applied_force)
        if self.cfg.max_force_slew_rate_n_s > 0:
            delta = self.cfg.max_force_slew_rate_n_s * dt
            requested = torch.clamp(requested, self.applied_force - delta, self.applied_force + delta)
        self.applied_force.copy_(requested)
        forces = requested.unsqueeze(-1) * self.directions_b.unsqueeze(0)
        torques = torch.cross(self.positions_b.unsqueeze(0), forces, dim=-1)
        return forces.sum(dim=1), torques.sum(dim=1), requested
