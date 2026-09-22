"""Configuration for the 16D Unity-compatible FinsROV HoldForPosition task."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.assets.finsrov import FINSROV_CFG


@configclass
class FinsROVHoldForPositionEnvCfg(DirectRLEnvCfg):
    """Learning-to-Swim rollout/PPO setup with FinsROV interface and physics."""

    decimation = 2
    # Give the FinsROV enough time to traverse the Learning-to-Swim reset
    # distribution and settle near the target before timeout.
    # At dt=1/120 and decimation=2 this is 600 policy steps per episode.
    episode_length_s = 10.0
    action_space = 8
    observation_space = 16
    state_space = 0

    # Learning to Swim uses 120 Hz physics, two physics updates per policy step.
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg = FINSROV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # Physical pool geometry.  Isaac's water surface is z=0 and the real pool
    # is 1 m deep.  Keep both the goal and reset CoM strictly inside the water
    # column rather than reusing Learning-to-Swim's deeper open-water depth.
    water_surface_z = 0.0
    pool_bottom_z = -1.0
    water_surface_clearance = 0.05
    pool_bottom_clearance = 0.05
    # Target depth is randomized independently for each reset, but remains
    # inside the same safe water column as the initial vehicle CoM.
    goal_depth_range = (-0.95, -0.05)
    starting_depth = -0.5
    # Goal orientation is yaw-only; roll and pitch are not task objectives.
    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    max_auv_extent = 7.0

    # Exact Unity HoldForPosition observation normalization.
    position_observation_scale = 3.0
    linear_velocity_observation_scale = 1.0
    angular_velocity_observation_scale = 1.0
    normalized_velocity_observation_clip = 2.0

    # Learning-to-Swim equivalent rectangular-body hydrodynamic formula, with
    # calibrated FinsROV mass/inertia/CoM-to-CoB in Isaac [x forward,y left,z up].
    water_density = 1027.0
    water_dynamic_viscosity = 0.001306
    mass = 12.11
    inertia_b = (0.13154832, 0.23185866, 0.24651341)
    displaced_volume = 0.0118
    com_to_cob_b = (0.0, 0.0, 0.11)
    enable_domain_randomization = True
    com_to_cob_randomization_radius = 0.05
    # Learning to Swim uses approximately +/-13.2% volume DR.  Keep the same
    # relative disturbance for FinsROV instead of copying its +/-0.003 m^3
    # absolute range, which would exceed FinsROV's vertical thrust authority.
    volume_range = (0.0103, 0.0133)

    # Current V4 Pro1 calibrated force authority in canonical order:
    # [V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]. Values are derived
    # from the active signed-quadratic c1/RPM limits in
    # finsrov_hardware_bridge_v4_pro1.yaml and match the Fossen prefab.
    # A normalized policy action maps directly to the corresponding signed
    # force limit: F_i = a_i * Fmax_i.
    max_forward_force_n = (8.474877, 8.797188, 8.797188, 8.474877, 8.797188, 8.474877, 8.797188, 8.474877)
    max_reverse_force_n = (7.974983, 8.272793, 8.272793, 7.974983, 8.272793, 7.974983, 8.272793, 7.974983)
    command_delay_s = 0.02
    force_time_constant_s = 0.01
    max_force_slew_rate_n_s = 1000.0

    # Position + yaw hold reward. Keep the reward coefficients aligned with
    # Learning to Swim: angular-velocity stabilization is disabled.
    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_ang_vel = 0.0
    rew_scale_actions = 0.2
