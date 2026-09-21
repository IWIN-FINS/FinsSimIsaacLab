"""Manager action term that owns the complete underwater force pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.physics.actuators import ThrusterSystem
from finssim_isaaclab_tasks.physics.core import HydrodynamicsState, UnderwaterHydrodynamics
from finssim_isaaclab_tasks.vehicles.finsrov import FINSROV_HYDRODYNAMICS, FINSROV_THRUSTERS
from finssim_isaaclab_tasks.vehicles.warpauv import WARPAUV_HYDRODYNAMICS, WARPAUV_THRUSTERS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


class UnderwaterThrusterAction(ActionTerm):
    cfg: "UnderwaterThrusterActionCfg"

    def __init__(self, cfg: "UnderwaterThrusterActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: RigidObject = env.scene[cfg.asset_name]
        profile, thruster_cfg = (
            (FINSROV_HYDRODYNAMICS, FINSROV_THRUSTERS) if cfg.vehicle == "finsrov" else (WARPAUV_HYDRODYNAMICS, WARPAUV_THRUSTERS)
        )
        if cfg.hydrodynamics_mode is not None:
            profile = replace(profile, mode=cfg.hydrodynamics_mode)
        self.thrusters = ThrusterSystem(thruster_cfg, self.num_envs, self.device)
        self.hydrodynamics = UnderwaterHydrodynamics(profile, self.num_envs, self.device)
        self._raw_actions = torch.zeros(self.num_envs, len(thruster_cfg.positions_b), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(torch.clamp(actions, -1.0, 1.0))

    def apply_actions(self) -> None:
        dt = self._env.physics_dt
        angular_speed = None
        if self.cfg.vehicle == "warpauv":
            positive = -139.0 * self._raw_actions.square() + 500.0 * self._raw_actions + 8.28
            negative = 161.0 * self._raw_actions.square() + 517.86 * self._raw_actions - 5.72
            angular_speed = torch.where(self._raw_actions >= 0.0, positive, negative)
        thrust, moment, _ = self.thrusters.compute(self._raw_actions, dt, angular_speed)
        state = HydrodynamicsState(
            self.robot.data.root_quat_w.torch,
            self.robot.data.root_lin_vel_b.torch,
            self.robot.data.root_ang_vel_b.torch,
            torch.zeros_like(self.robot.data.root_lin_vel_b.torch),
            torch.zeros_like(self.robot.data.root_ang_vel_b.torch),
        )
        hydro = self.hydrodynamics.compute(state, dt)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=(thrust + hydro.wrench_b[:, :3]).unsqueeze(1),
            torques=(moment + hydro.wrench_b[:, 3:]).unsqueeze(1),
        )

    def reset(self, env_ids=None):
        self._raw_actions[env_ids] = 0.0
        self.thrusters.reset(env_ids)
        self.hydrodynamics.reset(env_ids)


@configclass
class UnderwaterThrusterActionCfg(ActionTermCfg):
    class_type: type = UnderwaterThrusterAction
    asset_name: str = "robot"
    vehicle: Literal["finsrov", "warpauv"] = "finsrov"
    hydrodynamics_mode: str | None = None

