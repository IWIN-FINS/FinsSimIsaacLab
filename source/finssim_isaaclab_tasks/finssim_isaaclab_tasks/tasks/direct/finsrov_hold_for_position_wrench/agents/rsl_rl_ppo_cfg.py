"""Learning-to-Swim PPO configuration for the six-action wrench policy."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.agents.rsl_rl_ppo_cfg import (
    FinsROVHoldForPositionPPORunnerCfg,
)


@configclass
class FinsROVHoldForPositionWrenchPPORunnerCfg(FinsROVHoldForPositionPPORunnerCfg):
    experiment_name = "finssim_finsrov_hold_for_position_wrench_16d"
