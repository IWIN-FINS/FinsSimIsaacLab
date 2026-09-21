"""16D Unity-compatible, eight-thruster FinsROV HoldForPosition environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from .contract import build_hold_for_position_observation, yaw_error_magnitude_xyzw
from .dynamics import (
    CalibratedNormalizedThrusterCfg,
    CalibratedNormalizedThrusters,
    EquivalentBoxHydrodynamics,
    EquivalentBoxHydrodynamicsCfg,
)
from .env_cfg import FinsROVHoldForPositionEnvCfg


class FinsROVHoldForPositionEnv(DirectRLEnv):
    """FinsROV task with Unity ABI and Learning-to-Swim dynamics/training horizon."""

    cfg: FinsROVHoldForPositionEnvCfg

    def __init__(self, cfg: FinsROVHoldForPositionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 8, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 3] = 1.0
        self.goal_pos_w = self.scene.env_origins.clone()
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
        # The pool floor is one metre below the water surface (z=0), not at
        # the Isaac stage origin.  This keeps the collision plane at the
        # physical pool bottom.
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
        root_quat_w = self.robot.data.root_quat_w.torch
        root_pos_w = self.robot.data.root_pos_w.torch
        body_goal_offset_i = quat_apply(quat_conjugate(root_quat_w), self.goal_pos_w - root_pos_w)
        observation = build_hold_for_position_observation(
            body_goal_offset_i,
            root_quat_w,
            self.goal_quat_w,
            self.robot.data.root_lin_vel_b.torch,
            self.robot.data.root_ang_vel_b.torch,
            position_scale=self.cfg.position_observation_scale,
            linear_velocity_scale=self.cfg.linear_velocity_observation_scale,
            angular_velocity_scale=self.cfg.angular_velocity_observation_scale,
            velocity_clip=self.cfg.normalized_velocity_observation_clip,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        body_goal_offset = quat_apply(
            quat_conjugate(self.robot.data.root_quat_w.torch), self.goal_pos_w - self.robot.data.root_pos_w.torch
        )
        position_reward = self.cfg.rew_scale_pos * torch.exp(-torch.linalg.vector_norm(body_goal_offset, dim=-1).square())
        orientation_reward = self.cfg.rew_scale_ang * torch.exp(
            -yaw_error_magnitude_xyzw(self.goal_quat_w, self.robot.data.root_quat_w.torch)
        )
        angular_velocity_reward = self.cfg.rew_scale_ang_vel * torch.exp(
            -torch.linalg.vector_norm(self.robot.data.root_ang_vel_b.torch, dim=-1).square()
        )
        action_reward = self.cfg.rew_scale_actions * torch.exp(-torch.linalg.vector_norm(self.actions, dim=-1).square())
        return position_reward + orientation_reward + angular_velocity_reward + action_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        local_position = self.robot.data.root_pos_w.torch - self.scene.env_origins
        out_of_bounds = torch.any(torch.abs(local_position[:, :2]) > self.cfg.max_auv_extent, dim=-1)
        out_of_bounds |= torch.abs(local_position[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_extent
        pool_boundary_violation = (local_position[:, 2] > self.cfg.water_surface_z - self.cfg.water_surface_clearance) | (
            local_position[:, 2] < self.cfg.pool_bottom_z + self.cfg.pool_bottom_clearance
        )
        out_of_bounds |= pool_boundary_violation
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
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        # Keep the Learning-to-Swim horizontal reset radius, but sample depth
        # independently inside the physical 1 m water column.  A 3D sphere
        # around z=-0.5 would otherwise place some resets above the surface.
        root_pose[:, :2] += self._sample_horizontal_disk(count, self.cfg.goal_spawn_radius)
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + self._sample_water_depth(count)
        root_velocity.zero_()

        self.goal_pos_w[env_ids] = self.scene.env_origins[env_ids]
        self.goal_pos_w[env_ids, 2] += self._sample_goal_depth(count)
        self.goal_quat_w[env_ids] = self._random_yaw_quaternions(count)

        guidance_mask = torch.rand(count, device=self.device) < self.cfg.init_guidance_rate
        guided_env_ids = env_ids[guidance_mask]
        root_pose[guidance_mask, :3] = self.goal_pos_w[guided_env_ids]
        root_pose[guidance_mask, 3:7] = self.goal_quat_w[guided_env_ids]

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

    def _sample_horizontal_disk(self, count: int, radius: float) -> torch.Tensor:
        """Sample a uniform solid disk in the local horizontal x-y plane."""

        angle = 2.0 * torch.pi * torch.rand(count, device=self.device)
        disk_radius = radius * torch.sqrt(torch.rand(count, device=self.device))
        return torch.stack((disk_radius * torch.cos(angle), disk_radius * torch.sin(angle)), dim=-1)

    def _sample_water_depth(self, count: int) -> torch.Tensor:
        """Sample reset CoM depth with explicit surface/bottom clearances."""

        lower = self.cfg.pool_bottom_z + self.cfg.pool_bottom_clearance
        upper = self.cfg.water_surface_z - self.cfg.water_surface_clearance
        if lower >= upper:
            raise ValueError("pool depth and water clearances leave no valid reset interval")
        return lower + (upper - lower) * torch.rand(count, device=self.device)

    def _sample_goal_depth(self, count: int) -> torch.Tensor:
        """Sample a target depth from the configured safe water interval."""

        lower, upper = self.cfg.goal_depth_range
        safe_lower = self.cfg.pool_bottom_z + self.cfg.pool_bottom_clearance
        safe_upper = self.cfg.water_surface_z - self.cfg.water_surface_clearance
        if lower < safe_lower or upper > safe_upper or lower >= upper:
            raise ValueError(
                "goal_depth_range must be an increasing interval inside the safe pool water column"
            )
        return lower + (upper - lower) * torch.rand(count, device=self.device)

    def _random_yaw_quaternions(self, count: int) -> torch.Tensor:
        """Sample Unity/Isaac-equivalent target headings with zero roll/pitch."""

        yaw = (2.0 * torch.rand(count, device=self.device) - 1.0) * torch.pi
        half_yaw = 0.5 * yaw
        quaternions = torch.zeros(count, 4, device=self.device)
        quaternions[:, 2] = torch.sin(half_yaw)
        quaternions[:, 3] = torch.cos(half_yaw)
        return quaternions
