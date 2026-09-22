"""RSL-RL PPO configuration for the FinsROV T2 trajectory task."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.warpauv.agents.rsl_rl_ppo_cfg import WarpAUVPPORunnerCfg


@configclass
class FinsROVTrajectoryTrackingPPORunnerCfg(WarpAUVPPORunnerCfg):
    """Learning-to-Swim PPO settings for the 30D/8D trajectory contract."""

    experiment_name = "finssim_finsrov_trajectory_tracking_30d_8d"
