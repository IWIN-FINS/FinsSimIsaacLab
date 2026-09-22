"""Eight-thruster FinsROV Direct pose-hold environment."""

from __future__ import annotations

import torch

from finssim_isaaclab_tasks.physics.actuators import ThrusterSystem
from finssim_isaaclab_tasks.physics.core import HydrodynamicsState, UnderwaterHydrodynamics
from finssim_isaaclab_tasks.tasks.direct.warpauv.env import WarpAUVEnv
from finssim_isaaclab_tasks.vehicles.finsrov import FINSROV_HYDRODYNAMICS, FINSROV_THRUSTERS

from .env_cfg import FinsROVEnvCfg


class FinsROVEnv(WarpAUVEnv):
    cfg: FinsROVEnvCfg

    def __init__(self, cfg: FinsROVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 8, device=self.device)
        self.thrusters = ThrusterSystem(FINSROV_THRUSTERS, self.num_envs, self.device)
        self.hydrodynamics = UnderwaterHydrodynamics(FINSROV_HYDRODYNAMICS, self.num_envs, self.device)

    def _compute_wrench(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.cfg.sim.dt * self.cfg.decimation
        thruster_force, thruster_torque, _ = self.thrusters.compute(actions, dt)
        state = HydrodynamicsState(
            quat_w_xyzw=self.robot.data.root_quat_w.torch,
            linear_velocity_b=self.robot.data.root_lin_vel_b.torch,
            angular_velocity_b=self.robot.data.root_ang_vel_b.torch,
            water_linear_velocity_b=torch.zeros_like(self.robot.data.root_lin_vel_b.torch),
            water_angular_velocity_b=torch.zeros_like(self.robot.data.root_ang_vel_b.torch),
        )
        hydro = self.hydrodynamics.compute(state, dt)
        return thruster_force + hydro.wrench_b[:, :3], thruster_torque + hydro.wrench_b[:, 3:]

