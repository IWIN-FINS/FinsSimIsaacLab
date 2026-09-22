"""FinsROV pose-hold configuration using the calibrated Fossen baseline."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.assets.finsrov import FINSROV_CFG
from finssim_isaaclab_tasks.tasks.direct.warpauv.env_cfg import WarpAUVEnvCfg


@configclass
class FinsROVEnvCfg(WarpAUVEnvCfg):
    action_space = 8
    robot_cfg = FINSROV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    starting_depth = -2.0
    goal_spawn_radius = 0.5
    max_auv_extent = 4.0

