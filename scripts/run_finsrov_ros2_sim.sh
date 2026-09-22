#!/usr/bin/env bash
# Run Isaac's bundled ROS 2 Humble runtime without sourcing system ROS 2.
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:?Set ISAACLAB_PATH to the Isaac Lab installation directory.}"
FINSIM_ISAACLAB_ROOT="${FINSIM_ISAACLAB_ROOT:?Set FINSIM_ISAACLAB_ROOT to this repository.}"
ISAAC_ROS2_CORE="${ISAACLAB_PATH}/env_isaaclab/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core"
ISAAC_ROS2_LIB="${ISAAC_ROS2_CORE}/humble/lib"

if [[ ! -f "${ISAAC_ROS2_LIB}/librcutils.so" ]]; then
  echo "Isaac Sim bundled ROS 2 Humble libraries were not found: ${ISAAC_ROS2_LIB}" >&2
  exit 2
fi

# Do not let an externally sourced ROS 2 (normally Python 3.10) leak into
# Isaac's Python 3.12 process.  Its Python rclpy/message extensions and all
# native ROS 2 libraries must come from the same bundled Humble runtime: mixing
# those extensions with /opt/ros/humble (Python 3.10) libraries can abort in
# rosidl's generated C code during rclpy startup.
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${ISAAC_ROS2_LIB}"

exec "${ISAACLAB_PATH}/isaaclab.sh" -p \
  "${FINSIM_ISAACLAB_ROOT}/scripts/run_finsrov_ros2_sim.py" "$@"
