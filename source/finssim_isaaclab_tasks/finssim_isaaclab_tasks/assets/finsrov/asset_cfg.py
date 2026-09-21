"""Isaac Lab spawn config for the converted FinsROV Fossen baseline."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


# ``finsrov.usd`` is the raw Y-up, centimetre FBX conversion.  Spawn the
# wrapper instead so the RigidObject body frame matches the task's explicit
# [forward, left, up] thrust and observation convention.  The wrapper also
# authors the calibrated Unity CoM in that frame, so PhysX, hydrodynamics, and
# the per-thruster r-cross-F torque use one reference point.
USD_PATH = Path(__file__).resolve().parent / "data" / "finsrov_task_frame.usda"

FINSROV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=12.11),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
)
