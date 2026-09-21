"""FinsROV Fossen baseline imported from the calibrated Unity profile.

Source: ``Assets/Models/FinsROV/Calibrated/FinsROV_Fossen.prefab`` and
``Hydrodynamics/FinsROV_HydrodynamicsProfile.asset``.  Unity stays unchanged;
this is an intentionally independent Isaac Lab config snapshot.
"""

from __future__ import annotations

from finssim_isaaclab_tasks.physics.actuators import ThrusterModelCfg
from finssim_isaaclab_tasks.physics.core import HydrodynamicsProfileCfg


FINSROV_HYDRODYNAMICS = HydrodynamicsProfileCfg(
    mode="fossen",
    water_density=1027.0,
    displaced_volume=0.0118,
    # Unity profile CoB=(0,+0.03,0), CoM=(0,-0.08,0).  Isaac's z axis is up.
    center_of_buoyancy_b=(0.0, 0.0, 0.11),
    linear_damping=(39.073593, 52.30703, 80.09677, 1.0, 1.0, 0.3003185),
    quadratic_damping=(0.0, 0.0, 100.0, 2.0, 2.0, 0.47763214),
    added_mass_diagonal=(15.0346985, 28.505947, 23.354355, 0.1, 0.1, 0.5903969),
    max_explicit_added_mass_acceleration=(2.0, 2.0, 2.0, 4.0, 4.0, 4.0),
    surface_residual_scale=0.25,
)

# Canonical order from FinsSim: [V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF].
# The USD import manifest is responsible for mapping this body frame onto its prim axes.
FINSROV_THRUSTERS = ThrusterModelCfg(
    positions_b=(
        (-0.19, 0.10, 0.18), (-0.19, 0.10, -0.18), (-0.19, -0.10, -0.18), (-0.19, -0.10, 0.18),
        (0.15, -0.10, 0.18), (0.15, -0.10, -0.18), (0.15, 0.10, -0.18), (0.15, 0.10, 0.18),
    ),
    directions_b=((0.0, 1.0, 0.0),) * 4 + ((1.0, 0.0, 0.0),) * 4,
    max_forward_force_n=7.4022,
    max_reverse_force_n=7.4022,
    force_time_constant_s=0.12,
    max_force_slew_rate_n_s=500.0,
)
