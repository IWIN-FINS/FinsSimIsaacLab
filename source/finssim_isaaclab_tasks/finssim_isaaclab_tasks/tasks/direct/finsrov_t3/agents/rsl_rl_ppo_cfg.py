"""RSL-RL PPO configuration for the FinsROV T3 moving-target task."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.agents.rsl_rl_ppo_cfg import (
    FinsROVHoldForPositionPPORunnerCfg,
)


@configclass
class FinsROVT3MovingTargetPPORunnerCfg(FinsROVHoldForPositionPPORunnerCfg):
    """Learning-to-Swim PPO settings for the 13D/8D T3 contract."""

    experiment_name = "finssim_finsrov_t3_moving_target_13d_8d"
