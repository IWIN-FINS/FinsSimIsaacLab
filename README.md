[English](README.md) | [中文](README.zh-CN.md)

# FinsSimIsaacLab

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Cite FinsSimIsaacLab](https://img.shields.io/badge/Citation-CITATION.cff-0A7BBB)](CITATION.cff)

**FinsSimIsaacLab** is FinsSim's Isaac Lab 3 project for GPU-parallel
underwater-robot learning. It provides FinsROV and WarpAUV tasks, shared
underwater physics, policy contracts, and a lightweight runtime adapter for
deploying supported policies through the wider FinsSim stack.

> This repository is an Isaac Lab extension project, not an Isaac Sim or Isaac
> Lab distribution. Install those dependencies separately and keep their
> Python 3.12 environment isolated from FinsSim's ROS 2 Python environment.

## What is here

- **FinsROV tasks** — Unity-compatible pose hold, physical-wrench pose hold,
  moving-target (T3), and trajectory tracking tasks.
- **WarpAUV baseline** — a Lab 3 adaptation of the upstream WarpAUV task for
  regression, comparison, and underwater-RL research.
- **Shared physics** — GPU-batched hydrodynamics, actuator dynamics, water
  effects, and optional mesh-force calculation.
- **Policy boundaries** — explicit observation/action schemas and manifests;
  task IDs alone never make checkpoints interchangeable.
- **Runtime adapter** — a pure-PyTorch, Isaac-Sim-free package for the
  supported ROS 2 policy-loading path.

## Repository map

```text
runtime/                         Python 3.10–3.12 policy adapter; no Isaac Sim import
source/finssim_isaaclab_tasks/   Isaac Lab extension, assets, physics, and task definitions
scripts/                         Training, play, task-viewing, conversion, and smoke entry points
third_party/isaac-auv-env/       Pinned BSD-3-Clause upstream WarpAUV reference
docs/                            Architecture, operations, contracts, and license audit
```

## Requirements and environment boundary

The extension is locked to **Isaac Lab release/3.0.0-beta2**, Isaac Sim
`6.0.1.0`, and Python `3.12`; see the
[compatibility lock](docs/reference/compatibility.md). Isaac Sim/Isaac Lab and
SKRL belong in their own Python 3.12 environment. The `runtime/` adapter is
deliberately separate and supports Python `>=3.10,<3.13` for FinsSim/ROS 2
tooling.

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export FINSIM_ISAACLAB_ROOT=/path/to/FinsSimIsaacLab
cd "$ISAACLAB_PATH"
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

# Install the external extension into Isaac Lab's Python environment.
./isaaclab.sh -p -m pip install -e \
  "$FINSIM_ISAACLAB_ROOT/source/finssim_isaaclab_tasks"
```

For SKRL-based tasks, install the extra requirement into the same environment:

```bash
env -u VIRTUAL_ENV -u CONDA_PREFIX ./isaaclab.sh -p -m pip install -r \
  "$FINSIM_ISAACLAB_ROOT/requirements-isaaclab.txt"
```

## Quick start

Run a one-environment FinsROV GUI smoke check:

```bash
cd "$ISAACLAB_PATH"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
  "$FINSIM_ISAACLAB_ROOT/scripts/smoke_step.py" \
  --task FinsSim-FinsROV-HoldForPosition-v0 --num-envs 1 --steps 300 --viz kit
```

Run headless training with the native Isaac Lab entry point:

```bash
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p \
  "$FINSIM_ISAACLAB_ROOT/scripts/train.py" \
  --task FinsSim-FinsROV-HoldForPosition-v0 \
  --num_envs 2048 --max_iterations 400 --viz none
```

`--viz kit` opens Isaac Sim for inspection; prefer `--viz none` for
non-visual runs. Before connecting a trained policy to ROS 2 or hardware,
read its adjacent `policy_manifest.json` and use only the matching deployment
backend and action schema.

## Task catalog

| Task ID | Policy contract | Primary use |
| --- | --- | --- |
| `FinsSim-FinsROV-HoldForPosition-v0` | Unity-compatible 16D observation / 8 direct thrusters | 3D position and yaw hold |
| `FinsSim-FinsROV-HoldForPosition-Wrench-v0` | 16D observation / 6 bounded wrench actions | Physical-wrench allocator experiments |
| `FinsSim-FinsROV-T3-MovingTarget-v0` | 13D observation / 8 direct thrusters | Moving-target control |
| `FinsSim-FinsROV-TrajectoryTracking-v0` | Unity T2-compatible 30D observation / 8 direct thrusters | Trajectory tracking |
| `FinsSim-FinsROV-Direct-v0` | Legacy WarpAUV-style direct interface | Regression only |
| `FinsSim-FinsROV-PoseHold-v0` | Manager-based pose-hold interface | Manager-based experiments |
| `FinsSim-WarpAUV-Direct-v0` | Lab 3 17D `xyzw` schema / 6 actions | Upstream-derived baseline |
| `FinsSim-WarpAUV-PoseHold-v0` | Manager-based pose-hold interface | Manager-based baseline |

The FinsROV HoldForPosition and wrench task documents describe the deployable
contracts in detail; the trajectory task requires a dedicated Isaac Lab
runtime loader and must not be passed to a Unity checkpoint backend.

## Documentation

- [Documentation index](docs/README.md)
- [System and physics architecture](docs/architecture/underwater-framework.md)
- [Compatibility lock](docs/reference/compatibility.md)
- [Policy schemas](docs/reference/policy-schemas.md)
- [Isaac Sim GUI and task workflow](docs/operations/isaaclab-gui-usage.zh-CN.md)
- [Runtime adapter](runtime/README.md)
- [Task extension](source/finssim_isaaclab_tasks/README.md)

For the cross-backend control contract, ROS 2 integration, and experiment
orchestration, see the [FinsSim workspace](https://github.com/IWIN-FINS/FinsSim).

## Citation

If this repository contributes to academic work, cite its software record in
[`CITATION.cff`](CITATION.cff). The upstream WarpAUV reference is a separate
work and should also be cited where applicable; its citation is retained in
[the upstream README](https://github.com/warplab/isaac-auv-env).

```bibtex
@software{finssim_isaaclab_2026,
  author  = {{IWIN-FINS}},
  title   = {FinsSimIsaacLab: Isaac Lab Tasks for Underwater Robot Learning},
  year    = {2026},
  version = {0.1.0}
}
```

There is no project-specific archival paper or DOI yet. Add it to
`CITATION.cff` as the preferred citation when one is published.

## License, assets, and commercial terms

FinsSim-owned source code and documentation are available under the
[Apache License 2.0](LICENSE). A separate commercial agreement may provide
support, warranties, or other terms; see
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

This repository is not uniformly Apache-2.0. The pinned WarpAUV reference is
BSD-3-Clause, Isaac Lab is separately licensed, Isaac Sim has separate NVIDIA
terms, and the redistributability of FinsROV mesh/USD assets and the historical
checkpoint requires the rights-chain confirmation described in the
[license audit](docs/governance/license-audit.zh-CN.md). Read
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.
