"""GPU batched physics bench for FinsROV and WarpAUV backend regression."""

from __future__ import annotations

import argparse
import torch

from finssim_isaaclab_tasks.physics.core import HydrodynamicsProfileCfg, HydrodynamicsState, UnderwaterHydrodynamics
from finssim_isaaclab_tasks.physics.actuators import ThrusterSystem
from finssim_isaaclab_tasks.vehicles.finsrov import FINSROV_HYDRODYNAMICS, FINSROV_THRUSTERS
from finssim_isaaclab_tasks.vehicles.warpauv import WARPAUV_HYDRODYNAMICS, WARPAUV_THRUSTERS


parser = argparse.ArgumentParser()
parser.add_argument("--vehicle", choices=("finsrov", "warpauv"), required=True)
parser.add_argument("--backend", default=None)
parser.add_argument("--num-envs", type=int, default=64)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
profile, thruster_cfg = (FINSROV_HYDRODYNAMICS, FINSROV_THRUSTERS) if args.vehicle == "finsrov" else (WARPAUV_HYDRODYNAMICS, WARPAUV_THRUSTERS)
if args.backend:
    profile = HydrodynamicsProfileCfg(**{**profile.__dict__, "mode": args.backend})
hydro = UnderwaterHydrodynamics(profile, args.num_envs, device)
thrusters = ThrusterSystem(thruster_cfg, args.num_envs, device)
state = HydrodynamicsState(
    quat_w_xyzw=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(args.num_envs, 1),
    linear_velocity_b=torch.tensor([0.4, -0.2, 0.1], device=device).repeat(args.num_envs, 1),
    angular_velocity_b=torch.tensor([0.1, 0.0, -0.1], device=device).repeat(args.num_envs, 1),
    water_linear_velocity_b=torch.zeros(args.num_envs, 3, device=device),
    water_angular_velocity_b=torch.zeros(args.num_envs, 3, device=device),
)
diagnostics = hydro.compute(state, 1.0 / 120.0)
force, torque, applied = thrusters.compute(torch.full((args.num_envs, len(thruster_cfg.positions_b)), 0.2, device=device), 1.0 / 120.0)
print({"vehicle": args.vehicle, "backend": profile.mode, "device": str(device), "envs": args.num_envs, "hydro_wrench_b": diagnostics.wrench_b[0].tolist(), "thruster_force_b": force[0].tolist(), "thruster_torque_b": torque[0].tolist(), "applied_force_n": applied[0].tolist()})
