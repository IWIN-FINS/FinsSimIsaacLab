"""FinsROV HoldForPosition with normalized 6D wrench policy actions."""

from __future__ import annotations

import torch

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.env import FinsROVHoldForPositionEnv

from .allocator import FinsROVPhysicalWrenchAllocator, FinsROVPhysicalWrenchAllocatorCfg
from .env_cfg import FinsROVHoldForPositionWrenchEnvCfg


class FinsROVHoldForPositionWrenchEnv(FinsROVHoldForPositionEnv):
    """Keep the 16D task unchanged while allocating a 6D wrench to eight thrusters."""

    cfg: FinsROVHoldForPositionWrenchEnvCfg

    def __init__(self, cfg: FinsROVHoldForPositionWrenchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.wrench_allocator = FinsROVPhysicalWrenchAllocator(
            FinsROVPhysicalWrenchAllocatorCfg(
                wrench_limits=cfg.wrench_limits,
                max_forward_force_n=cfg.max_forward_force_n,
                max_reverse_force_n=cfg.max_reverse_force_n,
                enable_roll_pitch_moments=cfg.enable_roll_pitch_moments,
            ),
            self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        if not self.wrench_allocator.roll_pitch_moments_enabled:
            self.actions = self.actions.clone()
            self.actions[:, 3:5] = 0.0

    def set_roll_pitch_moments_enabled(self, enabled: bool) -> None:
        """Toggle policy roll/pitch wrench requests without changing 6D I/O."""

        self.wrench_allocator.set_roll_pitch_moments_enabled(enabled)

    def _compute_wrench(self, wrench_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        thruster_actions = self.wrench_allocator.allocate(wrench_actions)
        return super()._compute_wrench(thruster_actions)
