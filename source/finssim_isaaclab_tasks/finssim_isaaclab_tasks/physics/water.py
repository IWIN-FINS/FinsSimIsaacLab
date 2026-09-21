"""Vectorized water kinematics providers replacing Unity/DWP2 interop."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class WaterKinematicsCfg:
    height_m: float = 0.0
    current_velocity_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gauss_markov_tau_s: float = 0.0
    gauss_markov_sigma_m_s: float = 0.0
    wave_amplitude_m: float = 0.0
    wave_frequency_hz: float = 0.0
    wave_direction_w: tuple[float, float, float] = (1.0, 0.0, 0.0)


class WaterKinematics:
    def __init__(self, cfg: WaterKinematicsCfg, num_envs: int, device: torch.device | str):
        self.cfg, self.device = cfg, torch.device(device)
        self.stochastic_current_w = torch.zeros(num_envs, 3, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.stochastic_current_w.shape[0], device=self.device) if env_ids is None else env_ids
        self.stochastic_current_w[ids] = 0.0

    def sample(self, positions_w: torch.Tensor, dt: float, time_s: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.tensor(self.cfg.current_velocity_w, device=self.device, dtype=positions_w.dtype)
        if self.cfg.gauss_markov_tau_s > 0 and self.cfg.gauss_markov_sigma_m_s > 0:
            alpha = torch.exp(torch.tensor(-dt / self.cfg.gauss_markov_tau_s, device=self.device))
            noise = torch.randn_like(self.stochastic_current_w) * self.cfg.gauss_markov_sigma_m_s
            self.stochastic_current_w.mul_(alpha).add_(noise * torch.sqrt(1.0 - alpha * alpha))
        current = base + self.stochastic_current_w.to(positions_w.dtype)
        height = torch.full((positions_w.shape[0],), self.cfg.height_m, device=self.device, dtype=positions_w.dtype)
        if self.cfg.wave_amplitude_m and self.cfg.wave_frequency_hz:
            direction = torch.tensor(self.cfg.wave_direction_w, device=self.device, dtype=positions_w.dtype)
            direction = direction / direction.norm().clamp_min(1e-6)
            phase = positions_w @ direction + 2.0 * torch.pi * self.cfg.wave_frequency_hz * time_s
            height = height + self.cfg.wave_amplitude_m * torch.sin(phase)
        return height, current.expand_as(positions_w)
