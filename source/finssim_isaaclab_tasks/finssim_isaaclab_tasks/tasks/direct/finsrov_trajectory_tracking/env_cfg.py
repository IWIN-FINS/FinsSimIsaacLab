"""Configuration for the FinsROV T2 trajectory-tracking task."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.env_cfg import FinsROVHoldForPositionEnvCfg


@configclass
class FinsROVTrajectoryTrackingEnvCfg(FinsROVHoldForPositionEnvCfg):
    """Scheme-A FinsROV task: Unity T2 contract on Isaac FinsROV physics."""

    decimation = 2
    episode_length_s = 30.0
    action_space = 8
    observation_space = 30
    state_space = 0

    # Unity T2 previewDecisionStride=5 at a 50 Hz fixed step equals 0.1 s.
    preview_step_s = 0.1
    preview_count = 4
    position_observation_scale = 3.0
    linear_velocity_observation_scale = 1.0
    angular_velocity_observation_scale = 1.0
    observation_clip = 2.0

    # Stage 0 is deliberately line-only. Stages 1 and 2 progressively add
    # circles and expand up to the largest physically safe pool-constrained
    # range; the Unity open-water ranges are intentionally not reused.
    trajectory_curriculum_stage = 0
    requested_trajectory_cycles = 1.0
    # The real tank is approximately 2 m long (Isaac x) and 1 m wide
    # (Isaac y), with its origin at the centre of the water surface.  These
    # ranges are for the trajectory centre/reference, not the vehicle hull;
    # the task's rectangular boundary check also accounts for the hull size.
    stage0_scale_x_range = (0.20, 0.35)
    stage0_scale_z_range = (0.08, 0.12)
    stage1_scale_x_range = (0.25, 0.45)
    stage1_scale_z_range = (0.08, 0.15)
    stage2_scale_x_range = (0.35, 0.60)
    stage2_scale_z_range = (0.10, 0.20)
    stage0_surge_limit_mps = 0.15
    stage0_sway_limit_mps = 0.12
    stage1_surge_limit_mps = 0.20
    stage1_sway_limit_mps = 0.16
    stage2_surge_limit_mps = 0.30
    stage2_sway_limit_mps = 0.20

    # The reference is horizontal for the retained line/circle profiles. Its
    # depth is randomized while leaving room for the initial +/-5 cm jitter.
    trajectory_origin_depth_range = (-0.78, -0.22)
    initial_position_jitter = 0.03
    initial_roll_pitch_range_deg = 20.0

    # Pool bounds and conservative FinsROV half-extents obtained from the
    # task-frame USD visual envelope (about 0.56 x 0.42 x 0.25 m).  The
    # boundary checks use the root/CoM envelope as a conservative proxy for
    # the full hull and leave an additional 5 cm wall clearance.
    pool_length_x = 2.0
    pool_width_y = 1.0
    pool_wall_clearance = 0.05
    vehicle_half_extent_x = 0.30
    vehicle_half_extent_y = 0.22
    vehicle_half_extent_z = 0.13

    # T2 safety and termination settings, adapted to the 2 m x 1 m physical
    # pool.  The rectangular pool check below is authoritative; the radial
    # limit remains a secondary numerical runaway guard.
    max_path_error = 0.8
    max_distance_from_origin = 1.0
    out_of_bounds_penalty = -1.0

    # Unity T2 reward coefficients plus an explicit tangent-yaw alignment
    # term. The yaw-rate term remains part of the original T2 reward.
    rew_scale_pose = 0.5
    pose_distance_scale = 1.6
    rew_scale_up = 0.5
    rew_scale_spin = 0.5
    rew_scale_yaw = 0.25
    rew_scale_effort = 0.1
    rew_scale_smoothness = 0.0
