"""Configuration for the Isaac Lab 3 WarpAUV direct task."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.assets.warpauv import WARPAUV_CFG


@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 3.0
    action_space = 6
    observation_space = 17
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

    starting_depth = 8.0
    goal_spawn_radius = 2.0
    max_auv_extent = 7.0
    water_density = 997.0
    water_viscosity = 0.001306
    volume = 0.022747843530591776
    mass = 22.701
    rotor_constant = 0.001
    deadzone = 0.08
    com_to_cob_offset = (0.0, 0.0, 0.01)
    inertia = (0.37, 0.97, 1.19)

    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_ang_vel = 0.0
    rew_scale_actions = 0.2

