"""PPO settings copied from Learning to Swim's RSL-RL setup."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.warpauv.agents.rsl_rl_ppo_cfg import WarpAUVPPORunnerCfg


@configclass
class FinsROVHoldForPositionPPORunnerCfg(WarpAUVPPORunnerCfg):
    """24-step rollouts, 400 iterations, 64-64 ELU PPO, for 16D/8D yaw hold."""

    experiment_name = "finssim_finsrov_hold_for_position_yaw_16d_8d"
