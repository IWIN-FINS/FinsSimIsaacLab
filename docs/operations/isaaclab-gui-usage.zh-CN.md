# IsaacLab 与 GUI 使用说明

本文说明 FinsSimIsaacLab 在本机上的 Isaac Sim / Isaac Lab GUI 启动方式。

## 1. 路径与环境

FinsSimIsaacLab 外部项目和 Isaac Lab 安装目录是分开的：

```text
Isaac Lab 安装目录：${ISAACLAB_PATH}
FinsSimIsaacLab 项目：${FINSIM_ISAACLAB_ROOT}
Isaac Lab Python 环境：${ISAACLAB_PATH}/env_isaaclab
```

在运行以下命令前设置实际位置：

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export FINSIM_ISAACLAB_ROOT=/path/to/FinsSimIsaacLab
```

每次启动前进入 Isaac Lab 安装目录并激活专用环境：

```bash
cd "${ISAACLAB_PATH}"
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
```

当前主机的 GPU 约定如下：

- 物理 GPU `0`：NVIDIA T1000，主要用于显示。
- 物理 GPU `1`：NVIDIA RTX 3090，主要用于 Isaac Lab 训练。

训练和无 GUI 任务检查默认使用 RTX 3090：

```bash
export CUDA_VISIBLE_DEVICES=1
```

原生 GUI 启动时不要强制设置 `CUDA_VISIBLE_DEVICES=1`。Isaac Sim 会通过 Vulkan 使用桌面显示 GPU；强制设置 CUDA 可见设备会触发 Isaac Sim 的设备枚举警告，并可能导致窗口或渲染异常。

另外，不要在已经 `source /opt/ros/humble/setup.bash` 或
`source ros2_ws/scripts/source_ros2_uv.sh` 的 ROS2 shell 中直接启动 Isaac Sim。
ROS2 的 Python 3.10 路径和动态库会污染 Isaac Sim 的 Python 3.12 / Kit 扩展环境。
如果当前 shell 已经加载 ROS2，启动前执行：

```bash
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION RMW_IMPLEMENTATION
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_LOCALHOST_ONLY
```

## 2. 只打开原生 Isaac Sim GUI

该方式只启动 Isaac Sim，不会自动创建 FinsSim 任务或 WarpAUV 环境：

```bash
cd ${ISAACLAB_PATH}
source env_isaaclab/bin/activate
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION RMW_IMPLEMENTATION
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_LOCALHOST_ONLY
export OMNI_KIT_ACCEPT_EULA=YES
unset CUDA_VISIBLE_DEVICES
isaacsim
```

如果命令找不到 `isaacsim`，确认已经执行了 `source env_isaaclab/bin/activate`，并检查：

```bash
which isaacsim
python --version
```

关闭窗口即可退出；也可以在启动命令的终端中按 `Ctrl-C`。

## 3. 启动 WarpAUV GUI 单步检查

该方式会启动 Isaac Sim GUI，注册 FinsSim 任务，创建一个 WarpAUV 环境，执行一次 reset 和 step，然后自动退出。适合验证安装和 GUI 是否正常：

```bash
cd ${ISAACLAB_PATH}
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=1

./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/smoke_step_warpauv.py \
  --viz kit
```

终端出现类似下面的输出，说明环境完成了一次有效 step：

```text
WarpAUV one-env reset/step: ok ...
```

该脚本自动退出是预期行为，不是 GUI 崩溃。

## 4. 只查看指定 task 的 scene

如果只是查看 task 的场景和资产，不需要启动训练器。使用 `view_task.py`，它会创建指定 task，持续执行零动作，并在 episode 结束后自动 reset：

```bash
cd ${ISAACLAB_PATH}
source env_isaaclab/bin/activate
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION RMW_IMPLEMENTATION
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_LOCALHOST_ONLY
export OMNI_KIT_ACCEPT_EULA=YES
unset CUDA_VISIBLE_DEVICES

./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/view_task.py \
  --task FinsSim-WarpAUV-Direct-v0 \
  --num-envs 1 \
  --viz kit
```

查看其他 task 时只替换 `--task`，例如：

```bash
--task FinsSim-FinsROV-Direct-v0
--task FinsSim-FinsROV-PoseHold-v0
--task FinsSim-WarpAUV-PoseHold-v0
```

窗口中可以直接观察 scene、ROV 资产、环境原点和物理状态；按 `Ctrl-C` 退出。该脚本不创建 RSL-RL trainer，也不会生成训练 checkpoint。

## 5. Isaac Sim 编辑器使用说明

### 5.1 编辑器主要区域

启动 task viewer 后，Isaac Sim 窗口中的常用区域如下：

- **Stage**：查看当前 USD 场景树，例如 `/World/ground`、`/World/Light` 和 `/World/envs/env_0/Robot`。
- **Viewport**：观察和操作三维场景，可以旋转视角、平移、缩放和聚焦选中对象。
- **Property**：查看或修改选中 Prim 的 Transform、材质、可见性、物理属性和其他 USD 属性。
- **Content Browser**：浏览本地或 Nucleus 资产；本项目主要使用本地 USD 文件。
- **Console / Script Editor**：查看 Kit 扩展日志，或执行 Isaac Sim Python API 调试代码。
- **Timeline**：控制仿真时间线。task viewer 会持续推进仿真，暂停后可以检查当前状态。

常用操作：

1. 在 Stage 中选择 Prim，在 Property 中修改属性。
2. 使用 Viewport 工具栏的选择、移动、旋转和缩放工具编辑对象。
3. 使用 Stage 的搜索功能定位 `/World/envs/env_0/Robot` 等 Prim。
4. 使用 `File > Save As` 将当前 USD stage 保存到新的文件，避免覆盖原始资产。

### 5.2 在编辑器中打开机器人资产

原生 Isaac Sim GUI 可以直接打开机器人 USD 资产：

```text
WarpAUV:
${FINSIM_ISAACLAB_ROOT}/source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/warpauv/data/warpauv.usd

FinsROV:
${FINSIM_ISAACLAB_ROOT}/source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/finsrov/data/finsrov.usd
```

适合在编辑器中修改的内容包括：

- 模型几何、部件相对位置和可见性；
- 材质、颜色、纹理和渲染属性；
- 视觉灯光、相机和调试标记；
- USD 中已经存在的碰撞体和刚体属性。

保存资产后，重新启动 task viewer 才能确认 task 加载到修改后的 USD。

### 5.3 运行时 task scene 的限制

FinsSim task 不是一个预先保存好的完整 USD 场景。以
`FinsSim-WarpAUV-Direct-v0` 为例：

- `warpauv.usd` 提供机器人资产；
- `env.py` 运行时创建地面和灯光；
- Isaac Lab 根据 `num_envs` 和 `env_spacing` 生成并复制环境；
- task reset 会重新设置机器人位置、速度、目标位置和目标姿态；
- 水体和水动力目前主要由 Python/Warp 物理模型计算，不是一个可在 Viewport 中编辑的水池场景。

因此，在 task viewer 中拖动 `/World/envs/env_0/Robot` 属于运行时修改。它可以用于检查效果，但重新 reset 或重启程序后可能被 Python 配置覆盖，也不会自动修改源代码或源 USD。

## 6. 用 Python 修改 task scene

需要持久化的 task 场景修改，应根据修改对象选择对应文件。

### 6.1 环境数量和布局

修改 `tasks/direct/warpauv/env_cfg.py`：

```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=1,
    env_spacing=4.0,
    replicate_physics=True,
)
```

`num_envs=1` 适合编辑和观察；训练时可以改成更大的数量。`env_spacing` 控制多个环境之间的间距。

### 6.2 地面和灯光

修改 `tasks/direct/warpauv/env.py` 的 `_setup_scene()`：

```python
spawn_ground_plane(
    prim_path="/World/ground",
    cfg=GroundPlaneCfg(),
)

light_cfg = sim_utils.DomeLightCfg(
    intensity=2000.0,
    color=(0.75, 0.75, 0.75),
)
light_cfg.func("/World/Light", light_cfg)
```

这里可以持久化修改地面类型、灯光强度和颜色。修改后重新运行 task viewer。

### 6.3 机器人资产和初始状态

机器人 USD 路径和初始状态位于：

```text
source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/warpauv/asset_cfg.py
source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/finsrov/asset_cfg.py
```

例如：

```python
init_state=RigidObjectCfg.InitialStateCfg(
    pos=(0.0, 0.0, 8.0),
)
```

但 DirectRLEnv 的 `_reset_idx()` 还会根据 `starting_depth`、目标采样半径和随机姿态重新设置状态。若要修改 task 每次 reset 后的初始位置，应同时检查对应的 `env.py`。

### 6.4 修改物理和 task 行为

以下内容应通过 Python 修改，而不是在编辑器中拖动物体：

- 时间步长和渲染频率：`sim.dt`、`render_interval`；
- 水密度、黏度、质量和惯量；
- 推进器位置、力矩和死区；
- episode 时长、越界条件和 reset 逻辑；
- observation、action、reward 和目标采样逻辑。

这些参数影响训练和评估语义，修改后应重新运行 smoke 检查，并确认 policy observation/action 接口没有被意外改变。

## 7. 编辑器修改与 Python 修改的选择

| 修改内容 | 推荐方式 | 是否自动持久化 |
| --- | --- | --- |
| 机器人网格、材质、纹理 | 编辑器修改对应 USD | 保存 USD 后持久化 |
| 机器人碰撞体 | 编辑器或 USD 配置 | 保存 USD 后持久化 |
| 地面、灯光、环境复制 | Python task scene 配置 | 修改代码后持久化 |
| 初始位置和 reset 行为 | Python `env_cfg.py` / `env.py` | 修改代码后持久化 |
| 水动力、推进器、reward | Python physics/task 代码 | 修改代码后持久化 |
| 运行中的 Prim 位姿 | 编辑器运行时修改 | 通常不会持久化 |

如果使用 `File > Save As` 保存了一个完整运行时 stage，需要在 Python task 中显式加载这个 USD；仅保存文件并不会让 Gymnasium task 自动使用它。

## 8. Isaac Sim 与 Unity 的区别

### 8.1 场景组织

Unity 通常以一个 `.unity` Scene 作为主要编辑单元，GameObject、Prefab、组件和脚本都可以在 Editor 中组织。Isaac Sim 以 USD Stage 和 Prim 组织场景，资产通常是独立 USD 文件，通过 Python 配置或引用关系加载。

### 8.2 运行模式

Unity 的 Play Mode 通常直接运行当前编辑器中的 Scene。IsaacLab task 则通常是：

```text
启动 Isaac Sim App
  -> 注册 Gymnasium task
  -> Python 创建 SimulationCfg 和 SceneCfg
  -> 加载 USD 资产并复制 env_0 ... env_N
  -> reset task 并开始仿真
```

因此 Isaac Sim 原生 GUI 和 FinsSim task viewer 是两个层次：原生 GUI 是编辑器/模拟器，task viewer 是在 GUI 上运行 Python task 的入口。

### 8.3 脚本和组件

- Unity 主要使用 C# MonoBehaviour、Prefab 和 Inspector。
- Isaac Sim / IsaacLab 主要使用 Python 配置类、USD Prim、Isaac Lab asset config 和 Gymnasium task 注册。
- Unity 中在 Inspector 修改组件参数，通常直接写入 Scene 或 Prefab。
- IsaacLab 中在 Property 面板修改运行时 Prim，不一定会回写 `env_cfg.py` 或 task 的 USD 资产。

### 8.4 物理和训练环境

Unity Play 场景可以作为一个完整的可视化世界持续运行。IsaacLab task 通常还要满足批量环境、reset、observation、action、reward 和 device 执行约定。为了支持并行训练，task scene 会被复制成多个环境，因此编辑器里的单个 `env_0` 修改不一定代表所有环境的最终生成规则。

### 8.5 当前 FinsSim 的实际边界

当前 FinsSim IsaacLab 集成更接近“Python task + USD 机器人资产 + 运行时场景生成”，而不是 Unity 那种完整的可视化关卡编辑工作流。推荐的工作方式是：

1. 用 Isaac Sim 编辑器编辑机器人 USD 资产；
2. 用 Python 修改 task 的地面、灯光、环境布局、reset 和物理；
3. 用 `view_task.py --num-envs 1 --viz kit` 查看最终运行效果；
4. 确认无误后再增加 `num_envs` 或进入训练流程。

## 9. 启动带 GUI 的训练

使用 FinsSim 的 WarpAUV 任务进行训练：

```bash
cd ${ISAACLAB_PATH}
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=1

./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/train.py \
  --task FinsSim-WarpAUV-Direct-v0 \
  --num_envs 64 \
  --viz kit
```

GUI 运行时可以观察 Isaac Sim 场景和环境状态。训练日志和 checkpoint 默认写入：

```text
${ISAACLAB_PATH}/logs/rsl_rl/finssim_warpauv_direct/
```

停止训练：

```text
Ctrl-C
```

实际使用时建议先用较小的环境数量验证 GUI：

```bash
--num_envs 1
```

## 10. 加载 checkpoint 并在 GUI 中回放

将 `<实验目录>` 替换为实际训练目录：

```bash
cd ${ISAACLAB_PATH}
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=1

./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/play.py \
  --task FinsSim-WarpAUV-Direct-v0 \
  --checkpoint ${ISAACLAB_PATH}/logs/rsl_rl/finssim_warpauv_direct/<实验目录>/model_0.pt \
  --num_envs 1 \
  --viz kit
```

例如：

```bash
--checkpoint ${ISAACLAB_PATH}/logs/rsl_rl/finssim_warpauv_direct/2026-08-14_22-00-57/model_0.pt
```

回放会持续运行，按 `Ctrl-C` 退出。

## 11. 无 GUI 模式

服务器或不需要画面时，将可视化模式设为 `none`：

```bash
./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/train.py \
  --task FinsSim-WarpAUV-Direct-v0 \
  --num_envs 64 \
  --viz none
```

`smoke_warpauv.py` 也用于无 GUI 的快速检查：

```bash
./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/smoke_warpauv.py
```

## 12. 常见问题

### 7.1 GUI 没有窗口或显示黑屏

先确认当前桌面会话和显示变量：

```bash
echo "$DISPLAY"
echo "$XDG_RUNTIME_DIR"
nvidia-smi
```

确认启动成功的标志是终端出现：

```text
Isaac Sim Full App is loaded.
app ready
```

如果看到 `PrimCaching`、`LayerEditMode` 或
`get_action_registry` 缺失，通常不是这些 API 需要手动安装，而是 Isaac Sim
扩展和 ROS2/Python 动态库版本混用了。关闭当前 Isaac Sim 进程后，在清理 ROS2
变量的 shell 中重新启动：

```bash
cd ${ISAACLAB_PATH}
source env_isaaclab/bin/activate
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION RMW_IMPLEMENTATION
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_LOCALHOST_ONLY
export OMNI_KIT_ACCEPT_EULA=YES
unset CUDA_VISIBLE_DEVICES
isaacsim
```

如果当前机器通过远程 SSH 登录，普通 SSH 会话通常没有可用的桌面显示。应在本机桌面终端执行，或正确配置图形转发和 X11 权限。

也可以尝试让 Isaac Sim 使用显示 GPU：

```bash
unset CUDA_VISIBLE_DEVICES
isaacsim
```

### 7.2 任务找不到或导入失败

确认使用的是 Isaac Lab 专用环境，而不是 FinsSim ROS2 的 Python 3.10 uv 环境：

```bash
which python
python --version
```

FinsSim 任务扩展应安装在 Isaac Lab 环境中。需要重新安装时：

```bash
./isaaclab.sh -p -m pip install -e \
  ${FINSIM_ISAACLAB_ROOT}/source/finssim_isaaclab_tasks
```

### 7.3 资源或 checkpoint 路径错误

检查文件是否存在：

```bash
test -f ${FINSIM_ISAACLAB_ROOT}/scripts/play.py && echo ok
find ${ISAACLAB_PATH}/logs/rsl_rl/finssim_warpauv_direct \
  -name 'model_0.pt' -type f
```

### 7.4 GUI 启动后很快退出

`smoke_step_warpauv.py` 只执行一次 step，执行成功后会主动关闭 Isaac Sim。需要持续显示时使用 `train.py` 或 `play.py`。

## 13. 版本和接口注意事项

- Isaac Lab 安装和运行目录由 `${ISAACLAB_PATH}` 配置。
- FinsSim 任务名为 `FinsSim-WarpAUV-Direct-v0`。
- 当前 Lab 3 任务使用 `xyzw` 四元数。
- 历史 `legacy_warpauv_v2` checkpoint 使用旧的 `wxyz` observation 约定，不要直接按当前 Lab 3 任务加载。
- ROS2 workspace 使用独立的 Python 3.10 uv 环境，不要用它启动 Isaac Sim GUI。

相关文档：

- [仓库 README](../../README.zh-CN.md)
- [兼容性锁定](../reference/compatibility.md)
- [策略 schema](../reference/policy-schemas.md)
