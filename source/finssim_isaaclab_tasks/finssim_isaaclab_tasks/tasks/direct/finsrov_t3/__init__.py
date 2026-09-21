"""FinsROV T3 moving-target task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="FinsSim-FinsROV-T3-MovingTarget-v0",
    entry_point=f"{__name__}.env:FinsROVT3MovingTargetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FinsROVT3MovingTargetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinsROVT3MovingTargetPPORunnerCfg",
    },
)
