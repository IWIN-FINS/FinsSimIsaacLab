"""Scheme-A IsaacLab FinsROV trajectory-tracking environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz

from finssim_isaaclab_tasks.assets.finsrov import FINSROV_CFG
from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.dynamics import (
    CalibratedNormalizedThrusterCfg,
    CalibratedNormalizedThrusters,
    EquivalentBoxHydrodynamics,
    EquivalentBoxHydrodynamicsCfg,
)

from .contract import build_trajectory_observation, trajectory_yaw_error
from .env_cfg import FinsROVTrajectoryTrackingEnvCfg
from .trajectory import CIRCLE, STRAIGHT_LINE, circle_feasible_angular_speed, evaluate_trajectory


class FinsROVTrajectoryTrackingEnv(DirectRLEnv):
    """30D Unity-T2 observation with eight calibrated FinsROV thrusters."""

    cfg: FinsROVTrajectoryTrackingEnvCfg

    def __init__(self, cfg: FinsROVTrajectoryTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 8, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

        self.trajectory_profile = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trajectory_origin_w_i = self.scene.env_origins.clone()
        self.trajectory_scale_u = torch.zeros(self.num_envs, 3, device=self.device)
        self.trajectory_phase = torch.zeros(self.num_envs, device=self.device)
        self.trajectory_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self.trajectory_line_direction = torch.ones(self.num_envs, device=self.device)

        self.thrusters = CalibratedNormalizedThrusters(
            CalibratedNormalizedThrusterCfg(
                max_forward_force_n=cfg.max_forward_force_n,
                max_reverse_force_n=cfg.max_reverse_force_n,
                command_delay_s=cfg.command_delay_s,
                force_time_constant_s=cfg.force_time_constant_s,
                max_force_slew_rate_n_s=cfg.max_force_slew_rate_n_s,
            ),
            self.num_envs,
            self.device,
            cfg.sim.dt,
        )
        self.hydrodynamics = EquivalentBoxHydrodynamics(
            EquivalentBoxHydrodynamicsCfg(
                water_density=cfg.water_density,
                water_dynamic_viscosity=cfg.water_dynamic_viscosity,
                gravity_magnitude=abs(cfg.sim.gravity[2]),
                mass=cfg.mass,
                inertia_b=cfg.inertia_b,
                displaced_volume=cfg.displaced_volume,
                com_to_cob_b=cfg.com_to_cob_b,
                com_to_cob_randomization_radius=cfg.com_to_cob_randomization_radius,
                volume_range=cfg.volume_range,
                enable_domain_randomization=cfg.enable_domain_randomization,
            ),
            self.num_envs,
            self.device,
        )
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, self.cfg.pool_bottom_z),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        thrust_b, torque_b = self._compute_wrench(self.actions)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=thrust_b.unsqueeze(1), torques=torque_b.unsqueeze(1)
        )

    def _get_observations(self) -> dict:
        root_pos_w_i = self.robot.data.root_pos_w.torch
        root_quat_w_i = self.robot.data.root_quat_w.torch
        root_lin_vel_b_i = self.robot.data.root_lin_vel_b.torch
        root_ang_vel_b_i = self.robot.data.root_ang_vel_b.torch
        current_time = self._current_time()
        reference_positions, reference_velocity = self._evaluate_reference_grid(current_time)
        observation = build_trajectory_observation(
            root_pos_w_i,
            root_quat_w_i,
            root_lin_vel_b_i,
            root_ang_vel_b_i,
            reference_positions,
            reference_velocity,
            position_scale=self.cfg.position_observation_scale,
            linear_velocity_scale=self.cfg.linear_velocity_observation_scale,
            angular_velocity_scale=self.cfg.angular_velocity_observation_scale,
            observation_clip=self.cfg.observation_clip,
            progress=current_time / self.cfg.episode_length_s,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w_i = self.robot.data.root_pos_w.torch
        root_quat_w_i = self.robot.data.root_quat_w.torch
        root_ang_vel_b_i = self.robot.data.root_ang_vel_b.torch
        current_time = self._current_time()
        reference_position, reference_velocity = self._evaluate_reference(current_time)

        tracking_error = torch.linalg.vector_norm(reference_position - root_pos_w_i, dim=-1)
        pose_reward = self.cfg.rew_scale_pose * torch.exp(-self.cfg.pose_distance_scale * tracking_error)

        body_up_w_i = quat_apply(
            root_quat_w_i,
            torch.tensor((0.0, 0.0, 1.0), device=self.device, dtype=root_quat_w_i.dtype).expand(self.num_envs, -1),
        )
        tilt_error = torch.abs(1.0 - body_up_w_i[:, 2])
        up_reward = self.cfg.rew_scale_up / (1.0 + tilt_error.square())

        # Isaac body z is the yaw axis. The Unity/controller axial-vector
        # conversion makes the corresponding Unity yaw rate -Isaac omega_z.
        yaw_rate = -root_ang_vel_b_i[:, 2]
        spin_reward = self.cfg.rew_scale_spin / (1.0 + yaw_rate.pow(4))

        yaw_error, yaw_valid = trajectory_yaw_error(root_quat_w_i, reference_velocity)
        yaw_alignment_reward = self.cfg.rew_scale_yaw * yaw_valid * torch.exp(-yaw_error)

        effort_reward = self.cfg.rew_scale_effort * torch.exp(-torch.mean(torch.abs(self.actions), dim=-1))
        action_delta = self.actions - self.previous_actions
        smooth_reward = self.cfg.rew_scale_smoothness * torch.exp(-torch.mean(action_delta.square(), dim=-1))

        reward = pose_reward + pose_reward * (up_reward + spin_reward) + yaw_alignment_reward + effort_reward + smooth_reward
        self.previous_actions.copy_(self.actions)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w_i = self.robot.data.root_pos_w.torch
        local_position = root_pos_w_i - self.scene.env_origins
        current_time = self._current_time()
        reference_position, _ = self._evaluate_reference(current_time)
        tracking_error = torch.linalg.vector_norm(reference_position - root_pos_w_i, dim=-1)

        invalid = ~torch.isfinite(root_pos_w_i).all(dim=-1)
        invalid |= ~torch.isfinite(self.robot.data.root_lin_vel_b.torch).all(dim=-1)
        invalid |= ~torch.isfinite(self.robot.data.root_ang_vel_b.torch).all(dim=-1)
        out_of_bounds = invalid | (tracking_error > self.cfg.max_path_error)
        out_of_bounds |= torch.linalg.vector_norm(local_position[:, :2], dim=-1) > self.cfg.max_distance_from_origin
        safe_x = self.cfg.pool_length_x * 0.5 - self.cfg.vehicle_half_extent_x - self.cfg.pool_wall_clearance
        safe_y = self.cfg.pool_width_y * 0.5 - self.cfg.vehicle_half_extent_y - self.cfg.pool_wall_clearance
        safe_surface = self.cfg.water_surface_z - self.cfg.vehicle_half_extent_z - self.cfg.water_surface_clearance
        safe_bottom = self.cfg.pool_bottom_z + self.cfg.vehicle_half_extent_z + self.cfg.pool_bottom_clearance
        out_of_bounds |= torch.abs(local_position[:, 0]) > safe_x
        out_of_bounds |= torch.abs(local_position[:, 1]) > safe_y
        out_of_bounds |= local_position[:, 2] > safe_surface
        out_of_bounds |= local_position[:, 2] < safe_bottom
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        count = len(env_ids)
        self._sample_trajectory_parameters(env_ids)
        zero_time = torch.zeros(count, device=self.device)
        start_position, _ = self._evaluate_reference_for_envs(env_ids, zero_time)

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] = start_position
        jitter = torch.empty(count, 3, device=self.device).uniform_(-self.cfg.initial_position_jitter, self.cfg.initial_position_jitter)
        root_pose[:, :3] += jitter
        safe_surface = self.cfg.water_surface_z - self.cfg.vehicle_half_extent_z - self.cfg.water_surface_clearance
        safe_bottom = self.cfg.pool_bottom_z + self.cfg.vehicle_half_extent_z + self.cfg.pool_bottom_clearance
        root_pose[:, 2] = root_pose[:, 2].clamp(
            min=self.scene.env_origins[env_ids, 2] + safe_bottom,
            max=self.scene.env_origins[env_ids, 2] + safe_surface,
        )

        roll_pitch_limit = torch.deg2rad(torch.tensor(self.cfg.initial_roll_pitch_range_deg, device=self.device))
        roll = torch.empty(count, device=self.device).uniform_(-roll_pitch_limit, roll_pitch_limit)
        pitch = torch.empty(count, device=self.device).uniform_(-roll_pitch_limit, roll_pitch_limit)
        yaw = torch.empty(count, device=self.device).uniform_(-torch.pi, torch.pi)
        root_pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        root_velocity.zero_()
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.thrusters.reset(env_ids)
        self.hydrodynamics.reset(env_ids)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def _compute_wrench(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        thruster_force_b, thruster_torque_b, _ = self.thrusters.compute(actions)
        hydro = self.hydrodynamics.compute(
            self.robot.data.root_quat_w.torch,
            self.robot.data.root_lin_vel_b.torch,
            self.robot.data.root_ang_vel_b.torch,
        )
        return thruster_force_b + hydro.wrench_b[:, :3], thruster_torque_b + hydro.wrench_b[:, 3:]

    def _current_time(self) -> torch.Tensor:
        return self.episode_length_buf.to(dtype=torch.float32) * (self.cfg.sim.dt * self.cfg.decimation)

    def _evaluate_reference(self, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return evaluate_trajectory(
            self.trajectory_profile,
            self.trajectory_origin_w_i,
            self.trajectory_scale_u,
            self.trajectory_phase,
            self.trajectory_angular_speed,
            self.trajectory_line_direction,
            time_s,
            self.cfg.episode_length_s,
        )

    def _evaluate_reference_for_envs(self, env_ids: torch.Tensor, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return evaluate_trajectory(
            self.trajectory_profile[env_ids],
            self.trajectory_origin_w_i[env_ids],
            self.trajectory_scale_u[env_ids],
            self.trajectory_phase[env_ids],
            self.trajectory_angular_speed[env_ids],
            self.trajectory_line_direction[env_ids],
            time_s,
            self.cfg.episode_length_s,
        )

    def _evaluate_reference_grid(self, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(self.cfg.preview_count, device=self.device, dtype=time_s.dtype) * self.cfg.preview_step_s
        count = self.num_envs
        preview_time = time_s[:, None] + offsets[None, :]
        position, _ = evaluate_trajectory(
            self.trajectory_profile[:, None].expand(count, self.cfg.preview_count).reshape(-1),
            self.trajectory_origin_w_i[:, None, :].expand(count, self.cfg.preview_count, 3).reshape(-1, 3),
            self.trajectory_scale_u[:, None, :].expand(count, self.cfg.preview_count, 3).reshape(-1, 3),
            self.trajectory_phase[:, None].expand(count, self.cfg.preview_count).reshape(-1),
            self.trajectory_angular_speed[:, None].expand(count, self.cfg.preview_count).reshape(-1),
            self.trajectory_line_direction[:, None].expand(count, self.cfg.preview_count).reshape(-1),
            preview_time.reshape(-1),
            self.cfg.episode_length_s,
        )
        _, current_velocity = self._evaluate_reference(time_s)
        return position.reshape(count, self.cfg.preview_count, 3), current_velocity

    def _sample_trajectory_parameters(self, env_ids: torch.Tensor) -> None:
        stage = int(self.cfg.trajectory_curriculum_stage)
        if stage not in (0, 1, 2):
            raise ValueError("trajectory_curriculum_stage must be 0, 1, or 2")
        if stage == 0:
            x_range, z_range = self.cfg.stage0_scale_x_range, self.cfg.stage0_scale_z_range
            surge_limit, sway_limit = self.cfg.stage0_surge_limit_mps, self.cfg.stage0_sway_limit_mps
            profile = torch.full((len(env_ids),), STRAIGHT_LINE, dtype=torch.long, device=self.device)
        elif stage == 1:
            x_range, z_range = self.cfg.stage1_scale_x_range, self.cfg.stage1_scale_z_range
            surge_limit, sway_limit = self.cfg.stage1_surge_limit_mps, self.cfg.stage1_sway_limit_mps
            profile = torch.randint(0, 2, (len(env_ids),), device=self.device)
        else:
            x_range, z_range = self.cfg.stage2_scale_x_range, self.cfg.stage2_scale_z_range
            surge_limit, sway_limit = self.cfg.stage2_surge_limit_mps, self.cfg.stage2_sway_limit_mps
            profile = torch.randint(0, 2, (len(env_ids),), device=self.device)

        count = len(env_ids)
        scale = torch.zeros(count, 3, device=self.device)
        scale[:, 0] = torch.empty(count, device=self.device).uniform_(*x_range)
        scale[:, 2] = torch.empty(count, device=self.device).uniform_(*z_range)
        origin = self.scene.env_origins[env_ids].clone()
        origin[:, 2] += torch.empty(count, device=self.device).uniform_(*self.cfg.trajectory_origin_depth_range)
        phase = torch.empty(count, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        line_direction = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            torch.full((count,), -1.0, device=self.device),
            torch.ones(count, device=self.device),
        )
        angular_speed = torch.zeros(count, device=self.device)
        circle_mask = profile == CIRCLE
        if bool(circle_mask.any()):
            feasible = circle_feasible_angular_speed(scale, surge_limit, sway_limit)
            requested = 2.0 * torch.pi * self.cfg.requested_trajectory_cycles / self.cfg.episode_length_s
            direction = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                torch.full((count,), -1.0, device=self.device),
                torch.ones(count, device=self.device),
            )
            angular_speed = direction * torch.minimum(feasible, torch.full_like(feasible, requested))
            angular_speed = torch.where(circle_mask, angular_speed, torch.zeros_like(angular_speed))

        self.trajectory_profile[env_ids] = profile
        self.trajectory_origin_w_i[env_ids] = origin
        self.trajectory_scale_u[env_ids] = scale
        self.trajectory_phase[env_ids] = phase
        self.trajectory_angular_speed[env_ids] = angular_speed
        self.trajectory_line_direction[env_ids] = line_direction
