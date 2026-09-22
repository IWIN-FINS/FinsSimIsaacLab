"""Load and bind the committed FinsROV native ROS 2 Action Graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import Ros2Topics, TruthState


GRAPH_PATH = "/World/FinsROVROS2"
GRAPH_ASSET = Path(__file__).with_name("graph") / "finsrov_ros2.usda"


def enable_native_ros2_bridge() -> None:
    """Enable Isaac Sim's official ROS 2 bridge before loading the USD graph."""

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    # This package is installed as a Python extension in Isaac Lab, so Kit does
    # not necessarily process its extension.toml dependency list at app launch.
    # Enable the native bridge explicitly and wait for its OGN node interfaces
    # to register before the static USD graph is composed.
    for extension_name in ("omni.graph.action", "isaacsim.ros2.bridge"):
        manager.set_extension_enabled_immediate(extension_name, True)
    app = omni.kit.app.get_app()
    for _ in range(4):
        app.update()
    missing = [
        extension_name
        for extension_name in ("omni.graph.action", "isaacsim.ros2.bridge")
        if not manager.is_extension_enabled(extension_name)
    ]
    if missing:
        raise RuntimeError(f"Isaac Sim extensions did not enable: {missing}")
    import omni.graph.core as og

    unavailable = [
        node_type
        for node_type in ("omni.graph.action.OnPlaybackTick", "isaacsim.ros2.bridge.ROS2PublishClock")
        if og.get_node_type(node_type) is None
    ]
    if unavailable:
        raise RuntimeError(f"Isaac Sim OmniGraph node types did not register: {unavailable}")


def load_authored_graph() -> None:
    """Assert that Kit opened the committed graph before scene construction."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(GRAPH_PATH).IsValid():
        raise RuntimeError(
            "FinsROV ROS2 graph is absent. Start Kit with --open_usd "
            f"{GRAPH_ASSET} and enable isaacsim.ros2.bridge before loading it."
        )


class FinsROSRos2ActionGraph:
    """Typed binding over native Publisher/Subscriber attributes in the USD graph."""

    def __init__(self, *, topics: Ros2Topics | None = None) -> None:
        import omni.graph.core as og
        import omni.kit.app
        import omni.usd

        self._og = og
        self._topics = topics or Ros2Topics()
        # Allow the graph compiler to realize authored node specs after the
        # environment has completed its first physics-context initialization.
        omni.kit.app.get_app().update()
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(GRAPH_PATH).IsValid():
            raise RuntimeError(f"FinsROV ROS2 Action Graph was not loaded from {GRAPH_ASSET}")
        if og.get_graph_by_path(GRAPH_PATH) is None:
            prim = stage.GetPrimAtPath(GRAPH_PATH)
            node_prim = stage.GetPrimAtPath(f"{GRAPH_PATH}/Clock")
            raise RuntimeError(
                "FinsROV ROS2 USD did not materialize as an OmniGraph "
                f"(graph_type={prim.GetTypeName()}, clock_type={node_prim.GetTypeName()})"
            )

        self._attributes = {
            name: og.Controller.attribute(f"{GRAPH_PATH}/{node}.{port}")
            for name, node, port in (
                ("clock_time", "Clock", "inputs:timeStamp"),
                ("imu_time", "Imu", "inputs:timeStamp"),
                ("imu_orientation", "Imu", "inputs:orientation"),
                ("imu_angular_velocity", "Imu", "inputs:angularVelocity"),
                ("imu_linear_acceleration", "Imu", "inputs:linearAcceleration"),
                ("odom_time", "Odom", "inputs:timeStamp"),
                ("odom_position", "Odom", "inputs:position"),
                ("odom_orientation", "Odom", "inputs:orientation"),
                ("odom_linear_velocity", "Odom", "inputs:linearVelocity"),
                ("odom_angular_velocity", "Odom", "inputs:angularVelocity"),
                ("pose_sec", "Pose", "inputs:header:stamp:sec"),
                ("pose_nsec", "Pose", "inputs:header:stamp:nanosec"),
                ("pose_x", "Pose", "inputs:pose:pose:position:x"),
                ("pose_y", "Pose", "inputs:pose:pose:position:y"),
                ("pose_z", "Pose", "inputs:pose:pose:position:z"),
                ("pose_qx", "Pose", "inputs:pose:pose:orientation:x"),
                ("pose_qy", "Pose", "inputs:pose:pose:orientation:y"),
                ("pose_qz", "Pose", "inputs:pose:pose:orientation:z"),
                ("pose_qw", "Pose", "inputs:pose:pose:orientation:w"),
                ("depth_sec", "Depth", "inputs:header:stamp:sec"),
                ("depth_nsec", "Depth", "inputs:header:stamp:nanosec"),
                ("depth_x", "Depth", "inputs:pose:pose:position:x"),
                ("depth_y", "Depth", "inputs:pose:pose:position:y"),
                ("depth_z", "Depth", "inputs:pose:pose:position:z"),
                ("dvl_sec", "Dvl", "inputs:header:stamp:sec"),
                ("dvl_nsec", "Dvl", "inputs:header:stamp:nanosec"),
                ("dvl_x", "Dvl", "inputs:twist:twist:linear:x"),
                ("dvl_y", "Dvl", "inputs:twist:twist:linear:y"),
                ("dvl_z", "Dvl", "inputs:twist:twist:linear:z"),
                ("status", "Status", "inputs:data"),
                ("thrusters", "Thrusters", "outputs:data"),
                ("reset", "Reset", "outputs:data"),
            )
        }
        invalid = [name for name, attribute in self._attributes.items() if not attribute.is_valid()]
        if invalid:
            raise RuntimeError(f"FinsROV ROS2 graph is missing attributes: {invalid}")

    def _set(self, name: str, value) -> None:
        self._og.Controller.set(self._attributes[name], value)

    def _get(self, name: str):
        return self._og.Controller.get(self._attributes[name])

    @staticmethod
    def _stamp(sim_time_sec: float) -> tuple[int, int]:
        nanoseconds = max(0, round(float(sim_time_sec) * 1_000_000_000.0))
        return nanoseconds // 1_000_000_000, nanoseconds % 1_000_000_000

    def publish_truth(self, sim_time_sec: float, state: TruthState, command_timeout_sec: float) -> None:
        sec, nanosec = self._stamp(sim_time_sec)
        orientation = tuple(float(value) for value in state.orientation_world_body_xyzw)
        position = tuple(float(value) for value in state.position_world)
        linear_velocity = tuple(float(value) for value in state.linear_velocity_body)
        angular_velocity = tuple(float(value) for value in state.angular_velocity_body)
        acceleration = tuple(float(value) for value in state.linear_acceleration_body)
        for name in ("clock_time", "imu_time", "odom_time"):
            self._set(name, float(sim_time_sec))
        self._set("imu_orientation", orientation)
        self._set("imu_angular_velocity", angular_velocity)
        self._set("imu_linear_acceleration", acceleration)
        self._set("odom_position", position)
        self._set("odom_orientation", orientation)
        self._set("odom_linear_velocity", linear_velocity)
        self._set("odom_angular_velocity", angular_velocity)
        self._set("pose_sec", sec)
        self._set("pose_nsec", nanosec)
        for name, value in zip(("pose_x", "pose_y", "pose_z"), position, strict=True):
            self._set(name, value)
        for name, value in zip(("pose_qx", "pose_qy", "pose_qz", "pose_qw"), orientation, strict=True):
            self._set(name, value)
        self._set("depth_sec", sec)
        self._set("depth_nsec", nanosec)
        self._set("depth_x", 0.0)
        self._set("depth_y", position[1])
        self._set("depth_z", 0.0)
        self._set("dvl_sec", sec)
        self._set("dvl_nsec", nanosec)
        for name, value in zip(("dvl_x", "dvl_y", "dvl_z"), linear_velocity, strict=True):
            self._set(name, value)
        self._set(
            "status",
            json.dumps(
                {
                    "ready": True,
                    "sim_time_sec": round(float(sim_time_sec), 6),
                    "command_timeout_sec": float(command_timeout_sec),
                    "coordinate_convention": "controller:x-forward,y-up,z-left",
                },
                separators=(",", ":"),
            ),
        )

    def command_messages(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        def _array(name: str) -> np.ndarray | None:
            value = self._get(name)
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            return array if array.size else None

        return _array("thrusters"), _array("reset")
