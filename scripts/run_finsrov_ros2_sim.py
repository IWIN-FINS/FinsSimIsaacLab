"""Run one Isaac FinsROV environment as a ROS 2 DDS simulator endpoint."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--command-timeout-sec", type=float, default=0.5)
parser.add_argument("--state-publish-hz", type=float, default=60.0)
parser.add_argument("--seed", type=int, default=None)
# Open the committed USD before Isaac Lab builds its DirectRLEnv scene.  Kit
# then materializes the Action Graph as part of normal stage loading; the
# runner never constructs graph nodes programmatically.
parser.add_argument(
    "--open_usd",
    default=str(
        Path(__file__).resolve().parents[1]
        / "source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/ros2/graph/finsrov_ros2.usda"
    ),
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop after this many simulator control steps; 0 runs until Kit exits.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# OGN node types must exist before Kit opens ``finsrov_ros2.usda``.  Appending
# rather than replacing keeps user-provided Kit settings available.
args_cli.kit_args = "--enable isaacsim.ros2.bridge --enable omni.graph.action " + args_cli.kit_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _numpy(tensor):
    value = tensor.torch if hasattr(tensor, "torch") else tensor
    return value.detach().cpu().numpy()


def main() -> None:
    import torch
    from finssim_isaaclab_tasks.ros2.action_graph import FinsROSRos2ActionGraph, enable_native_ros2_bridge
    from finssim_isaaclab_tasks.ros2.adapters import FinsRos2CommandAdapter, FinsRos2StateAdapter
    from finssim_isaaclab_tasks.ros2.env import FinsROVRos2Env
    from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.env_cfg import (
        FinsROVHoldForPositionEnvCfg,
    )

    if args_cli.state_publish_hz <= 0.0:
        raise ValueError("--state-publish-hz must be positive")
    enable_native_ros2_bridge()
    cfg = FinsROVHoldForPositionEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 4.0
    cfg.sim.device = args_cli.device
    cfg.enable_domain_randomization = False
    cfg.init_guidance_rate = 1.0
    # The ROS controller profile sends direct physical force in N. Keep the
    # simulator's actuator authority exactly symmetric to its published ABI.
    cfg.max_forward_force_n = (7.0,) * 8
    cfg.max_reverse_force_n = (7.0,) * 8
    env = FinsROVRos2Env(cfg)
    command_adapter = FinsRos2CommandAdapter(args_cli.command_timeout_sec)
    sim_time_sec = 0.0
    physics_dt = float(cfg.sim.dt) * int(cfg.decimation)
    publish_period = 1.0 / args_cli.state_publish_hz
    next_publish_time = 0.0
    previous_linear_velocity = None
    steps = 0
    try:
        # Isaac Lab's reset(None) routes through Replicator's global-seed path.
        # In headless Isaac Sim 6 this can create an invalid SDG graph, whereas a
        # plain reset correctly leaves the environment unseeded.
        if args_cli.seed is None:
            env.reset()
        else:
            env.reset(seed=args_cli.seed)
        # The direct environment reset completes Isaac Lab's scene/stage
        # construction.  Attach the committed graph only afterwards: loading a
        # USD sublayer before this point lets the framework replace its stage
        # and leaves stale OmniGraph node handles behind.
        graph = FinsROSRos2ActionGraph()
        # ``is_running()`` becomes false for a headless AppLauncher after the
        # first reset in Isaac Sim 6, despite the simulation context remaining
        # valid.  ``is_exiting()`` is the lifecycle signal we need here.
        while not simulation_app.is_exiting():
            thrusters, reset = graph.command_messages()
            command_adapter.receive_thrusters(thrusters, sim_time_sec)
            command_adapter.receive_reset(reset)
            if command_adapter.take_reset_request():
                env.reset_from_ros()
                previous_linear_velocity = None

            force_n = command_adapter.command_force_n(sim_time_sec)
            actions = torch.as_tensor(force_n / 7.0, dtype=torch.float32, device=env.device).unsqueeze(0)
            env.step(actions)
            sim_time_sec += physics_dt
            steps += 1

            if sim_time_sec + 1.0e-9 >= next_publish_time:
                position = _numpy(env.robot.data.root_pos_w)[0]
                quaternion_wxyz = _numpy(env.robot.data.root_quat_w)[0]
                linear_velocity = _numpy(env.robot.data.root_lin_vel_b)[0]
                angular_velocity = _numpy(env.robot.data.root_ang_vel_b)[0]
                linear_acceleration = (
                    (linear_velocity - previous_linear_velocity) / physics_dt
                    if previous_linear_velocity is not None
                    else linear_velocity * 0.0
                )
                previous_linear_velocity = linear_velocity.copy()
                state = FinsRos2StateAdapter.from_isaac(
                    position,
                    (quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]),
                    linear_velocity,
                    angular_velocity,
                    linear_acceleration,
                )
                graph.publish_truth(sim_time_sec, state, command_adapter.command_timeout_sec)
                next_publish_time += publish_period
            if args_cli.max_steps > 0 and steps >= args_cli.max_steps:
                break
        print({"ros2_sim_steps": steps, "sim_time_sec": round(sim_time_sec, 6)})
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001 - Kit can replace sys.excepthook.
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
