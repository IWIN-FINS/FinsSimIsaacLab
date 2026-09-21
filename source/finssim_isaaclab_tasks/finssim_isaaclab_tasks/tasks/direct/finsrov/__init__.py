"""FinsROV DirectRLEnv registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="FinsSim-FinsROV-Direct-v0",
    entry_point=f"{__name__}.env:FinsROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FinsROVEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinsROVPPORunnerCfg",
    },
)

