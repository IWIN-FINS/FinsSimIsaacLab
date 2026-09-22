"""Independent FinsROV T3 moving-target environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

from ..finsrov_hold_for_position.env import FinsROVHoldForPositionEnv
from .contract import build_moving_target_observation
from .env_cfg import FinsROVT3MovingTargetEnvCfg


class FinsROVT3MovingTargetEnv(FinsROVHoldForPositionEnv):
    """FinsROV T3: 13D moving-target observation and eight thruster actions."""

    cfg: FinsROVT3MovingTargetEnvCfg

    def __init__(self, cfg: FinsROVT3MovingTargetEnvCfg, render_mode: str | None = None, **kwargs):
        # The parent constructor performs an initial reset.  Route that reset
        # through the parent before T3-only state tensors exist, then perform
        # the real T3 reset after all moving-target state is initialized.
        self._t3_bootstrap = True
        super().__init__(cfg, render_mode, **kwargs)
        self._t3_bootstrap = False

        self.target_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_acceleration_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_desired_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_min_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_max_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_motion_time_s = torch.zeros(self.num_envs, device=self.device)
        self.target_next_retarget_time_s = torch.zeros(self.num_envs, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, 8, device=self.device)
        self.previous_target_distance = torch.zeros(self.num_envs, device=self.device)
        self.stable_success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if getattr(self, "_t3_bootstrap", False):
            return super()._reset_idx(env_ids)

        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        count = len(env_ids)
        self.target_velocity_w[env_ids] = self._sample_initial_target_velocity(count)
        self.target_acceleration_w[env_ids].zero_()
        self.target_desired_velocity_w[env_ids] = self.target_velocity_w[env_ids]
        self.target_motion_time_s[env_ids].zero_()
        self.target_next_retarget_time_s[env_ids] = self._sample_retarget_interval(count)
        self._set_target_bounds(env_ids)
        self.previous_actions[env_ids].zero_()
        self.stable_success_steps[env_ids].zero_()
        self.success[env_ids] = False
        self.previous_target_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_pos_w[env_ids] - self.robot.data.root_pos_w.torch[env_ids], dim=-1
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)

    def _apply_action(self) -> None:
        self._advance_target_motion(self.cfg.sim.dt)
        super()._apply_action()

    def _get_observations(self) -> dict:
        root_pos_w = self.robot.data.root_pos_w.torch
        root_quat_w = self.robot.data.root_quat_w.torch
        inverse_root = quat_conjugate(root_quat_w)
        target_offset_b = quat_apply(inverse_root, self.goal_pos_w - root_pos_w)
        target_velocity_b = quat_apply(inverse_root, self.target_velocity_w)
        distance = torch.linalg.vector_norm(target_offset_b, dim=-1)
        observation = build_moving_target_observation(
            target_offset_b,
            target_velocity_b,
            self.robot.data.root_lin_vel_b.torch,
            self.robot.data.root_ang_vel_b.torch,
            distance,
            position_scale=self.cfg.target_spawn_radius,
            target_velocity_scale=self.cfg.target_velocity_observation_scale,
            linear_velocity_scale=self.cfg.linear_velocity_observation_scale,
            angular_velocity_scale=self.cfg.angular_velocity_observation_scale,
            velocity_clip=self.cfg.normalized_velocity_observation_clip,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w.torch
        root_quat_w = self.robot.data.root_quat_w.torch
        to_target_w = self.goal_pos_w - root_pos_w
        distance = torch.linalg.vector_norm(to_target_w, dim=-1)
        direction_w = to_target_w / distance[:, None].clamp_min(1.0e-6)
        vehicle_velocity_w = quat_apply(root_quat_w, self.robot.data.root_lin_vel_b.torch)
        relative_velocity_w = vehicle_velocity_w - self.target_velocity_w
        relative_speed = torch.linalg.vector_norm(relative_velocity_w, dim=-1)
        direction_alignment = torch.sum(relative_velocity_w * direction_w, dim=-1)
        direction_alignment = torch.where(relative_speed > 1.0e-6, direction_alignment / relative_speed, torch.zeros_like(relative_speed))
        approach_velocity = torch.clamp(torch.sum(relative_velocity_w * direction_w, dim=-1), -1.0, 1.0)
        near_ratio = 1.0 - torch.clamp(distance / max(self.cfg.near_target_distance, 1.0e-6), 0.0, 1.0)
        velocity_match = near_ratio * (
            1.0 - torch.clamp(relative_speed / max(self.cfg.relative_velocity_match_speed, 1.0e-6), 0.0, 1.0)
        )
        angular_speed = torch.linalg.vector_norm(self.robot.data.root_ang_vel_b.torch, dim=-1)
        distance_progress = self.previous_target_distance - distance
        action_energy = torch.mean(torch.abs(self.actions), dim=-1)
        action_change = torch.mean(torch.abs(self.actions - self.previous_actions), dim=-1)
        stable = (distance < self.cfg.success_distance) & (
            relative_speed < self.cfg.stable_success_relative_speed
        ) & (angular_speed < self.cfg.stable_success_angular_velocity)
        self.stable_success_steps = torch.where(
            stable, self.stable_success_steps + 1, torch.zeros_like(self.stable_success_steps)
        )
        self.success = self.stable_success_steps >= self.cfg.stable_success_steps_required

        reward = (
            self.cfg.rew_scale_step
            + self.cfg.rew_scale_distance_progress * distance_progress
            + self.cfg.rew_scale_direction_alignment * direction_alignment
            + self.cfg.rew_scale_approach_velocity * approach_velocity
            + self.cfg.rew_scale_near_target * near_ratio
            + self.cfg.rew_scale_relative_velocity * velocity_match
            + self.cfg.rew_scale_stable_hold * near_ratio.square() * stable.to(dtype=distance.dtype)
            - self.cfg.rew_scale_ang_vel_penalty * angular_speed
            - self.cfg.rew_scale_near_target_speed_penalty * near_ratio * relative_speed
            - self.cfg.rew_scale_action_energy_penalty * action_energy
            - self.cfg.rew_scale_action_change_penalty * action_change
            + self.cfg.success_reward * self.success.to(dtype=distance.dtype)
        )
        self.previous_target_distance.copy_(distance)
        self.previous_actions.copy_(self.actions)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        return terminated | self.success, time_out

    def _set_target_bounds(self, env_ids: torch.Tensor) -> None:
        extent = torch.as_tensor(
            self.cfg.target_motion_extents, device=self.device, dtype=self.goal_pos_w.dtype
        ).expand(len(env_ids), -1)
        safe_lower = self.scene.env_origins[env_ids] + torch.tensor(
            (0.0, 0.0, self.cfg.pool_bottom_z + self.cfg.pool_bottom_clearance),
            device=self.device,
            dtype=self.goal_pos_w.dtype,
        )
        safe_upper = self.scene.env_origins[env_ids] + torch.tensor(
            (0.0, 0.0, self.cfg.water_surface_z - self.cfg.water_surface_clearance),
            device=self.device,
            dtype=self.goal_pos_w.dtype,
        )
        self.target_min_w[env_ids] = self.goal_pos_w[env_ids] - extent
        self.target_max_w[env_ids] = self.goal_pos_w[env_ids] + extent
        self.target_min_w[env_ids, 2] = torch.maximum(self.target_min_w[env_ids, 2], safe_lower[:, 2])
        self.target_max_w[env_ids, 2] = torch.minimum(self.target_max_w[env_ids, 2], safe_upper[:, 2])

    def _advance_target_motion(self, dt_s: float) -> None:
        self.target_motion_time_s += dt_s
        retarget = self.target_motion_time_s >= self.target_next_retarget_time_s
        if retarget.any():
            env_ids = torch.nonzero(retarget, as_tuple=False).squeeze(-1)
            self.target_desired_velocity_w[env_ids] = self._sample_initial_target_velocity(len(env_ids))
            self.target_next_retarget_time_s[env_ids] = self.target_motion_time_s[env_ids] + self._sample_retarget_interval(len(env_ids))

        old_velocity = self.target_velocity_w.clone()
        delta_velocity = self.target_desired_velocity_w - old_velocity
        max_delta = self.cfg.target_motion_max_acceleration * dt_s
        delta_norm = torch.linalg.vector_norm(delta_velocity, dim=-1, keepdim=True)
        delta_velocity = delta_velocity * torch.clamp(max_delta / delta_norm.clamp_min(1.0e-8), max=1.0)
        next_velocity = old_velocity + delta_velocity
        next_position = self.goal_pos_w + next_velocity * dt_s

        below = next_position < self.target_min_w
        above = next_position > self.target_max_w
        next_position = torch.maximum(torch.minimum(next_position, self.target_max_w), self.target_min_w)
        next_velocity = torch.where(below | above, torch.zeros_like(next_velocity), next_velocity)
        self.goal_pos_w.copy_(next_position)
        self.target_velocity_w.copy_(next_velocity)
        self.target_acceleration_w.copy_((next_velocity - old_velocity) / max(dt_s, 1.0e-6))

    def _sample_initial_target_velocity(self, count: int) -> torch.Tensor:
        direction = torch.randn(count, 3, device=self.device)
        direction[:, 2] *= self.cfg.target_motion_vertical_direction_scale
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        speed = torch.empty(count, device=self.device).uniform_(
            self.cfg.target_motion_min_speed, self.cfg.target_motion_max_speed
        )
        return direction * speed[:, None]

    def _sample_retarget_interval(self, count: int) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(*self.cfg.target_motion_retarget_interval)
