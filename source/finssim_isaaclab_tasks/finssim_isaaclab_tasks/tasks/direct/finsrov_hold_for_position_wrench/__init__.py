"""Six-wrench-action FinsROV HoldForPosition task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="FinsSim-FinsROV-HoldForPosition-Wrench-v0",
    entry_point=f"{__name__}.env:FinsROVHoldForPositionWrenchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FinsROVHoldForPositionWrenchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinsROVHoldForPositionWrenchPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
