# FinsROV Isaac Lab ROS2 真值仿真链路

该链路将一个 Isaac Lab FinsROV 环境作为独立的 DDS 仿真端点：Isaac
进程发布 `/clock` 和 controller-compatible 真值状态，订阅八路推进器
force-N 命令；现有 FinsSim ROS2 控制器继续运行在自己的 Python 3.10/uv
环境中。两边不互相 import Python 包。

```text
Isaac Lab / Python 3.12 (Isaac bundled ROS 2 runtime)
  /clock, pose, imu, depth, dvl  ──DDS──>  motion_controller / Python 3.10
  <──DDS── /sim/finsrov/thrusters_out [Float32MultiArray, 8 x force N]
```

## 已实现的接口

| 方向 | Topic | 类型 | 语义 |
| --- | --- | --- | --- |
| Isaac → ROS2 | `/clock` | `rosgraph_msgs/Clock` | 仿真时间 |
| Isaac → ROS2 | `/sim/finsrov/controller/{pose,depth}` | `geometry_msgs/PoseWithCovarianceStamped` | `controller_world` 下的真值位置 |
| Isaac → ROS2 | `/sim/finsrov/controller/imu` | `sensor_msgs/Imu` | 姿态、机体系角速度、线加速度 |
| Isaac → ROS2 | `/sim/finsrov/controller/dvl` | `geometry_msgs/TwistWithCovarianceStamped` | 机体系线速度 |
| ROS2 → Isaac | `/sim/finsrov/thrusters_out` | `std_msgs/Float32MultiArray` | canonical 八路、单位 N、每路限制为 `[-7, 7]` |
| ROS2 → Isaac | `/sim/finsrov/reset` | `std_msgs/Float32MultiArray` | 现有 reset service 发布 `[1.0]` 即触发重置 |

坐标在 Isaac 端集中变换为 controller 约定：`x` 前、`y` 上、`z` 左；
角速度按轴向量规则变换。不要在 controller、hardware bridge 或 MCU 再做
一次轴交换。

## 安装与启动

先在**未 source ROS2** 的终端安装/更新 Isaac extension。不要把 ROS2
Python 3.10 的 `PYTHONPATH` 或 `LD_LIBRARY_PATH` 传入 Isaac。

```bash
export ISAACLAB_PATH=/home/fins/UnderwaterSim/IsaacLab
export FINSIM_ISAACLAB_ROOT=/home/fins/UnderwaterSim/Code/FinsSim/simulators/isaaclab/FinsSimIsaacLab
cd "$ISAACLAB_PATH"
./isaaclab.sh -p -m pip install -e "$FINSIM_ISAACLAB_ROOT/source/finssim_isaaclab_tasks"
```

在第一个 ROS2 终端启动控制器和 sim-truth readiness gate：

```bash
cd /home/fins/UnderwaterSim/Code/FinsSim/ros2_ws
./scripts/colcon_build_uv.sh --packages-select finssim_motion_control
./scripts/run_ros2_uv.sh ros2 launch finssim_motion_control motion_controller.launch.py \
  params_file:=src/finssim_motion_control/config/FinsROV/traditional_pid_position_yaw_isaaclab.yaml
```

在第二个干净终端启动 Isaac。将 `OMNI_KIT_ACCEPT_EULA=YES` 仅用于已经
接受 NVIDIA Omniverse 条款的账户/机器：

```bash
export ISAACLAB_PATH=/home/fins/UnderwaterSim/IsaacLab
export FINSIM_ISAACLAB_ROOT=/home/fins/UnderwaterSim/Code/FinsSim/simulators/isaaclab/FinsSimIsaacLab
cd "$ISAACLAB_PATH"
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_VERSION ROS_PYTHON_VERSION RMW_IMPLEMENTATION
export OMNI_KIT_ACCEPT_EULA=YES
"$FINSIM_ISAACLAB_ROOT/scripts/run_finsrov_ros2_sim.sh" --viz none
```

启动脚本会设置 `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` 并仅暴露
`/opt/ros/humble/lib` 的 C/C++ 运行库。它不会 source 系统 ROS2 的 Python
3.10 setup 脚本；Isaac 使用自己的 Python 3.12 `rclpy` 与消息包。当前本机
Isaac Sim 6.0.1 的 bundled-Linux fallback 路径构造异常，因此不能省略这条
`LD_LIBRARY_PATH` 设置。

## 打通验证

控制器启动后，另开 ROS2 shell：

```bash
cd /home/fins/UnderwaterSim/Code/FinsSim/ros2_ws
./scripts/run_ros2_uv.sh ros2 topic hz /clock
./scripts/run_ros2_uv.sh ros2 topic hz /sim/finsrov/controller/pose
./scripts/run_ros2_uv.sh ros2 topic echo /sim/finsrov/controller/state/status --once
./scripts/run_ros2_uv.sh ros2 topic info /sim/finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 run finssim_motion_control send_position_goal \
  --topic /sim/motion_controller/command/position_controller_world --x 0.2 --y 0.0 --z 0.0 --yaw 0
```

预期是 state status 中 `ready=true`，并且推进器 topic 同时出现 controller
publisher 和 Isaac subscriber。使用 reset 时需用同一 profile 的 service：

```bash
./scripts/run_ros2_uv.sh ros2 run finssim_motion_control reset_service --ros-args \
  --params-file src/finssim_motion_control/config/FinsROV/traditional_pid_position_yaw_isaaclab.yaml
```

本阶段是 truth-state 闭环；相机、IMU 噪声、DVL/深度传感器模型和 OmniGraph
相机发布是后续扩展，不能把它们当成当前已验证的能力。
