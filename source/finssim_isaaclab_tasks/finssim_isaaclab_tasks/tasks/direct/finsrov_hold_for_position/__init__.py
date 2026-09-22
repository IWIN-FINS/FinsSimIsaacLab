"""Unity-compatible 16D FinsROV HoldForPosition task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="FinsSim-FinsROV-HoldForPosition-v0",
    entry_point=f"{__name__}.env:FinsROVHoldForPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FinsROVHoldForPositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinsROVHoldForPositionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
