[English](README.md) | [中文](README.zh-CN.md)

# FinsSimIsaacLab

[![许可证：Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![引用 FinsSimIsaacLab](https://img.shields.io/badge/Citation-CITATION.cff-0A7BBB)](CITATION.cff)

**FinsSimIsaacLab** 是 FinsSim 面向 Isaac Lab 3 的 GPU 并行水下机器人学习项目，
提供 FinsROV 与 WarpAUV task、共享水动力学、策略契约，以及供 FinsSim 其他组件使用
的轻量 runtime adapter。

> 本仓库是 Isaac Lab extension 项目，不是 Isaac Sim 或 Isaac Lab 的安装包。二者必须
> 单独安装，并与 FinsSim 的 ROS 2 Python 环境隔离。

## 主要内容

- **FinsROV task**：Unity-compatible 定点、physical-wrench 定点、T3 移动目标和轨迹跟踪。
- **WarpAUV baseline**：用于回归、对比和水下 RL 研究的上游 WarpAUV Lab 3 适配。
- **共享物理**：GPU batch 水动力学、推进器动态、水体效应和可选 mesh-force 计算。
- **策略边界**：显式 observation/action schema 与 manifest；同名 task 或同维度 checkpoint 不代表可互换。
- **Runtime adapter**：不导入 Isaac Sim 的纯 PyTorch 包，服务于受支持的 ROS 2 策略加载链路。

## 目录

```text
runtime/                         Python 3.10–3.12 policy adapter；不导入 Isaac Sim
source/finssim_isaaclab_tasks/   Isaac Lab extension、资产、物理和 task 定义
scripts/                         训练、回放、场景查看、转换和 smoke 脚本
third_party/isaac-auv-env/       固定版本的 BSD-3-Clause WarpAUV 上游参考实现
docs/                            架构、操作、契约与许可证审计
```

## 依赖与环境边界

extension 锁定于 **Isaac Lab release/3.0.0-beta2**、Isaac Sim `6.0.1.0` 与 Python `3.12`；详见[兼容性锁定](docs/reference/compatibility.md)。Isaac Sim、Isaac Lab 和 SKRL 必须安装在独立的 Python 3.12 环境。`runtime/` adapter 则支持 Python `>=3.10,<3.13`，供 FinsSim/ROS 2 工具使用。

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export FINSIM_ISAACLAB_ROOT=/path/to/FinsSimIsaacLab
cd "$ISAACLAB_PATH"
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

# 在 Isaac Lab 的 Python 环境中安装 external extension。
./isaaclab.sh -p -m pip install -e \
  "$FINSIM_ISAACLAB_ROOT/source/finssim_isaaclab_tasks"
```

使用 SKRL task 时，也要在同一个环境中安装额外依赖：

```bash
env -u VIRTUAL_ENV -u CONDA_PREFIX ./isaaclab.sh -p -m pip install -r \
  "$FINSIM_ISAACLAB_ROOT/requirements-isaaclab.txt"
```

## 快速开始

运行一个 FinsROV GUI 单环境 smoke check：

```bash
cd "$ISAACLAB_PATH"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
  "$FINSIM_ISAACLAB_ROOT/scripts/smoke_step.py" \
  --task FinsSim-FinsROV-HoldForPosition-v0 --num-envs 1 --steps 300 --viz kit
```

运行 headless 原生 Isaac Lab 训练：

```bash
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
  "$FINSIM_ISAACLAB_ROOT/scripts/train.py" \
  --task FinsSim-FinsROV-HoldForPosition-v0 \
  --num_envs 2048 --max_iterations 400 --viz none
```

`--viz kit` 用于打开 Isaac Sim 检查；无图形运行请使用 `--viz none`。在将 checkpoint 接入 ROS 2 或硬件前，必须读取同目录的 `policy_manifest.json`，并选择完全匹配的 deployment backend 与 action schema。

## Task 总览

| Task ID | 策略契约 | 主要用途 |
| --- | --- | --- |
| `FinsSim-FinsROV-HoldForPosition-v0` | Unity-compatible 16D / 8 路直接推进器 | 三维位置与 yaw 定点 |
| `FinsSim-FinsROV-HoldForPosition-Wrench-v0` | 16D / 6 路有界 wrench action | Physical-wrench allocator 实验 |
| `FinsSim-FinsROV-T3-MovingTarget-v0` | 13D / 8 路直接推进器 | 移动目标控制 |
| `FinsSim-FinsROV-TrajectoryTracking-v0` | Unity T2-compatible 30D / 8 路直接推进器 | 轨迹跟踪 |
| `FinsSim-FinsROV-Direct-v0` | 旧 WarpAUV-style direct interface | 仅回归 |
| `FinsSim-FinsROV-PoseHold-v0` | Manager-based pose-hold | Manager-based 实验 |
| `FinsSim-WarpAUV-Direct-v0` | Lab 3 17D `xyzw` / 6 action | 上游派生 baseline |
| `FinsSim-WarpAUV-PoseHold-v0` | Manager-based pose-hold | Manager-based baseline |

FinsROV 定点和 wrench task 的 README 说明当前可部署的契约。轨迹 task 需要专用 Isaac Lab runtime loader，不能交给 Unity checkpoint backend。

## 文档

- [文档总索引](docs/README.zh-CN.md)
- [系统与物理架构](docs/architecture/underwater-framework.md)
- [兼容性锁定](docs/reference/compatibility.md)
- [策略 schema](docs/reference/policy-schemas.md)
- [Isaac Sim GUI 与 task 工作流](docs/operations/isaaclab-gui-usage.zh-CN.md)
- [Runtime adapter](runtime/README.md)
- [Task extension](source/finssim_isaaclab_tasks/README.md)

跨后端控制契约、ROS 2 集成和实验编排请从 [FinsSim 主仓库](https://github.com/IWIN-FINS/FinsSim) 开始阅读。

## 引用

如果本仓库对您的学术工作有帮助，请引用 [`CITATION.cff`](CITATION.cff) 中的软件记录。WarpAUV 上游参考是独立工作；如使用其派生 baseline，也请引用[上游 README](https://github.com/warplab/isaac-auv-env) 中给出的论文。

```bibtex
@software{finssim_isaaclab_2026,
  author  = {{IWIN-FINS}},
  title   = {FinsSimIsaacLab: Isaac Lab Tasks for Underwater Robot Learning},
  year    = {2026},
  version = {0.1.0}
}
```

目前没有本项目专属的归档论文或 DOI；未来发表后，应将其作为 preferred citation 加入 `CITATION.cff`。

## 许可证、资产与商业条款

FinsSim 自有的源代码和文档采用 [Apache License 2.0](LICENSE)。商业支持、担保或其他公开许可证未提供的条款可通过独立书面协议取得，见 [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)。

本仓库并非整体均为 Apache-2.0：固定版本的 WarpAUV 参考实现为 BSD-3-Clause；Isaac Lab 与 Isaac Sim 分别适用其独立许可证；FinsROV FBX/USD 资产和历史 checkpoint 是否可公开再分发，还必须满足[许可证审计](docs/governance/license-audit.zh-CN.md)列出的权利链确认。重新分发前请阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
