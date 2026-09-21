"""WarpAUV vehicle data separated from task code."""

from __future__ import annotations

from finssim_isaaclab_tasks.physics.actuators import ThrusterModelCfg
from finssim_isaaclab_tasks.physics.core import HydrodynamicsProfileCfg


WARPAUV_HYDRODYNAMICS = HydrodynamicsProfileCfg(
    mode="simplified_fossen",
    water_density=997.0,
    displaced_volume=0.022747843530591776,
    center_of_buoyancy_b=(0.0, 0.0, 0.01),
    # Equivalent-box coefficients used by the former task-local model
    # (mass=22.701 kg, inertia=(.37,.97,1.19), beta=.001306 Pa s).
    linear_damping=(0.0084663058, 0.0048606398, 0.0024508293, 0.0013351559, 0.0002526565, 0.0000323884),
    quadratic_damping=(39.19611829, 68.27214837, 135.40164656, 0.27740403, 1.38660248, 0.76970424),
)

WARPAUV_THRUSTERS = ThrusterModelCfg(
    positions_b=(
        (-0.4127, 0.1506, -0.0889), (-0.4127, -0.1506, -0.0889),
        (-0.3030, 0.1461, -0.1587), (-0.3030, -0.1461, -0.1587),
        (0.0585, 0.1461, -0.0540), (0.0585, -0.1461, -0.0540),
    ),
    directions_b=(
        (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 0.7071067811865476, 0.7071067811865476), (0.0, -0.7071067811865476, 0.7071067811865476),
        (0.0, 0.7071067811865476, -0.7071067811865476), (0.0, -0.7071067811865476, -0.7071067811865476),
    ),
    deadzone=0.08,
    rotor_constant=0.001,
)
