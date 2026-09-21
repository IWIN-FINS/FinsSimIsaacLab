"""Configuration for the independent FinsROV T3 moving-target task."""

from isaaclab.utils.configclass import configclass

from ..finsrov_hold_for_position.env_cfg import FinsROVHoldForPositionEnvCfg


@configclass
class FinsROVT3MovingTargetEnvCfg(FinsROVHoldForPositionEnvCfg):
    """Eight-thruster T3 configuration with a smooth moving target."""

    observation_space = 13
    action_space = 8

    # Unity ControlForMovingTargetReward parameters, expressed in Isaac axes
    # [x forward, y left, z up].  The target is bounded inside the one-metre
    # pool and has an explicitly maintained velocity and acceleration.
    target_spawn_radius = 3.0
    target_velocity_observation_scale = 0.2
    target_motion_extents = (2.5, 2.5, 0.35)
    target_motion_min_speed = 0.0
    target_motion_max_speed = 0.10
    target_motion_max_acceleration = 0.08
    target_motion_vertical_direction_scale = 0.4
    target_motion_retarget_interval = (1.5, 4.0)

    # Unity moving-target reward coefficients.
    rew_scale_step = -0.001
    rew_scale_distance_progress = 1.2
    rew_scale_direction_alignment = 0.03
    rew_scale_approach_velocity = 0.05
    rew_scale_near_target = 0.08
    rew_scale_relative_velocity = 0.06
    relative_velocity_match_speed = 0.25
    rew_scale_ang_vel_penalty = 0.01
    rew_scale_near_target_speed_penalty = 0.04
    rew_scale_action_energy_penalty = 0.0015
    rew_scale_action_change_penalty = 0.001
    rew_scale_stable_hold = 0.04
    success_reward = 2.0
    out_of_bounds_penalty = -1.0
    success_distance = 0.25
    near_target_distance = 0.8
    stable_success_relative_speed = 0.12
    stable_success_angular_velocity = 0.3
    stable_success_steps_required = 10
