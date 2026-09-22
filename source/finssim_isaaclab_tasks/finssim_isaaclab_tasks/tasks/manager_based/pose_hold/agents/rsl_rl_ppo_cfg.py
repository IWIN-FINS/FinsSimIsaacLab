from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.warpauv.agents.rsl_rl_ppo_cfg import WarpAUVPPORunnerCfg


@configclass
class FinsROVPoseHoldPPORunnerCfg(WarpAUVPPORunnerCfg):
    experiment_name = "finssim_finsrov_pose_hold"


@configclass
class WarpAUVPoseHoldPPORunnerCfg(WarpAUVPPORunnerCfg):
    experiment_name = "finssim_warpauv_pose_hold"

