"""Configuration for the six-wrench-action FinsROV HoldForPosition task."""

from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.env_cfg import FinsROVHoldForPositionEnvCfg

from .allocator import POLICY_WRENCH_LIMITS


@configclass
class FinsROVHoldForPositionWrenchEnvCfg(FinsROVHoldForPositionEnvCfg):
    """Same 16D Learning-to-Swim task; policy action is six physical wrenches."""

    action_space = 6
    wrench_limits = POLICY_WRENCH_LIMITS
    # This wrench task follows the Unity simulation profile: every channel is
    # symmetric +/-7 N.  Do not inherit the direct task's hardware calibration.
    max_forward_force_n = (7.0,) * 8
    max_reverse_force_n = (7.0,) * 8
    # Keep a six-value policy ABI while controlling position and yaw only.
    enable_roll_pitch_moments: bool = False
