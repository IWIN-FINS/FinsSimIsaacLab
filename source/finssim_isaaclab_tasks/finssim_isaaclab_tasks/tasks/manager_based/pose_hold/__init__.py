"""Manager-based pose-hold registrations for both underwater vehicles."""

import gymnasium as gym

from . import agents

for vehicle, cfg_name, runner_name in (
    ("FinsROV", "FinsROVPoseHoldEnvCfg", "FinsROVPoseHoldPPORunnerCfg"),
    ("WarpAUV", "WarpAUVPoseHoldEnvCfg", "WarpAUVPoseHoldPPORunnerCfg"),
):
    gym.register(
        id=f"FinsSim-{vehicle}-PoseHold-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_name}",
        },
    )
