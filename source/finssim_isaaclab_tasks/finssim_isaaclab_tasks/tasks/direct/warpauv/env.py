"""WarpAUV six-thruster position/orientation hold task for Isaac Lab 3.

The task keeps the reference vehicle's force curve, thruster positions and
buoyancy/drag model.  Unlike the legacy Lab 2 task, all task quaternions are
explicitly Lab 3 ``xyzw`` tensors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from .env_cfg import WarpAUVEnvCfg
from finssim_isaaclab_tasks.physics.actuators import ThrusterSystem
from finssim_isaaclab_tasks.physics.core import HydrodynamicsState, UnderwaterHydrodynamics
from finssim_isaaclab_tasks.physics.water import WaterKinematics, WaterKinematicsCfg
from finssim_isaaclab_tasks.vehicles.warpauv import WARPAUV_HYDRODYNAMICS, WARPAUV_THRUSTERS


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 3] = 1.0
        self.goal_pos_w = self.scene.env_origins.clone()
        self.default_env_origins = self.scene.env_origins.clone()
        self.thrusters = ThrusterSystem(WARPAUV_THRUSTERS, self.num_envs, self.device)
        self.hydrodynamics = UnderwaterHydrodynamics(WARPAUV_HYDRODYNAMICS, self.num_envs, self.device)
        self.water = WaterKinematics(WaterKinematicsCfg(), self.num_envs, self.device)
        # PhysX exposes ``_ALL_INDICES`` as a Warp array, whereas DirectRLEnv
        # bookkeeping uses Torch advanced indexing.
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        thrust, moment = self._compute_wrench(self.actions)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=thrust.unsqueeze(1), torques=moment.unsqueeze(1)
        )

    def _get_observations(self) -> dict:
        root_quat_w = self.robot.data.root_quat_w.torch
        root_pos_w = self.robot.data.root_pos_w.torch
        body_goal_offset = quat_apply(quat_conjugate(root_quat_w), self.goal_pos_w - root_pos_w)
        observation = torch.cat(
            (
                self.goal_quat_w,
                body_goal_offset,
                root_quat_w,
                self.robot.data.root_lin_vel_b.torch,
                self.robot.data.root_ang_vel_b.torch,
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        body_goal_offset = quat_apply(
            quat_conjugate(self.robot.data.root_quat_w.torch), self.goal_pos_w - self.robot.data.root_pos_w.torch
        )
        position_reward = self.cfg.rew_scale_pos * torch.exp(-torch.sum(body_goal_offset.square(), dim=-1))
        orientation_dot = torch.sum(self.goal_quat_w * self.robot.data.root_quat_w.torch, dim=-1).square()
        orientation_reward = self.cfg.rew_scale_ang * torch.exp(-(1.0 - orientation_dot))
        angular_reward = self.cfg.rew_scale_ang_vel * torch.exp(
            -torch.sum(self.robot.data.root_ang_vel_b.torch.square(), dim=-1)
        )
        action_reward = self.cfg.rew_scale_actions * torch.exp(-torch.sum(self.actions.square(), dim=-1))
        return position_reward + orientation_reward + angular_reward + action_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        local_position = self.robot.data.root_pos_w.torch - self.scene.env_origins
        out_of_bounds = torch.any(torch.abs(local_position[:, :2]) > self.cfg.max_auv_extent, dim=-1)
        out_of_bounds |= torch.abs(local_position[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_extent
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.starting_depth
        root_pose[:, :3] += self._sample_sphere(count, self.cfg.goal_spawn_radius)
        root_velocity.zero_()
        self.default_env_origins[env_ids] = self.scene.env_origins[env_ids]
        self.default_env_origins[env_ids, 2] += self.cfg.starting_depth
        self.goal_pos_w[env_ids] = self.default_env_origins[env_ids]
        self.goal_quat_w[env_ids] = self._random_quaternions(count)
        self.thrusters.reset(env_ids)
        self.hydrodynamics.reset(env_ids)
        self.water.reset(env_ids)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def _compute_wrench(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive = -139.0 * actions.square() + 500.0 * actions + 8.28
        negative = 161.0 * actions.square() + 517.86 * actions - 5.72
        angular_speed = torch.where(actions >= 0.0, positive, negative)
        angular_speed = torch.where(torch.abs(actions) >= self.cfg.deadzone, angular_speed, torch.zeros_like(angular_speed))
        thruster_force, thruster_torque, _ = self.thrusters.compute(
            actions, self.cfg.sim.dt * self.cfg.decimation, angular_speed=angular_speed
        )
        state = HydrodynamicsState(
            quat_w_xyzw=self.robot.data.root_quat_w.torch,
            linear_velocity_b=self.robot.data.root_lin_vel_b.torch,
            angular_velocity_b=self.robot.data.root_ang_vel_b.torch,
            water_linear_velocity_b=torch.zeros_like(self.robot.data.root_lin_vel_b.torch),
            water_angular_velocity_b=torch.zeros_like(self.robot.data.root_ang_vel_b.torch),
        )
        hydro = self.hydrodynamics.compute(state, self.cfg.sim.dt * self.cfg.decimation)
        return thruster_force + hydro.wrench_b[:, :3], thruster_torque + hydro.wrench_b[:, 3:]

    def _sample_sphere(self, count: int, radius: float) -> torch.Tensor:
        directions = torch.randn(count, 3, device=self.device)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-6)
        return directions * radius * torch.rand(count, 1, device=self.device).pow(1.0 / 3.0)

    def _random_quaternions(self, count: int) -> torch.Tensor:
        quaternions = torch.randn(count, 4, device=self.device)
        return quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True).clamp_min(1e-6)
