# Compatibility lock

| Component | Locked value |
| --- | --- |
| Isaac Lab | `release/3.0.0-beta2`, commit `2e44ddb2e19536579140496023b5ccb060bc4152` |
| Isaac Sim pip package | `6.0.1.0` |
| Python | `3.12` |
| Training GPU | RTX 3090 (`CUDA_VISIBLE_DEVICES=1`) |
| Runtime adapter | Python `>=3.10,<3.13`; no Isaac Sim import |
| WarpAUV source | `isaac-auv-env` commit `7c5ebe7f7a08acd2570b5fba328e92b7f59f6794` |

The Lab 3 task uses `xyzw` quaternions. `runtime/finssim_isaaclab` retains the
historical WarpAUV `wxyz` policy observation only under the explicit
`legacy_warpauv_v2` schema. Never load a Lab 3 artifact through that legacy
schema.

## Installation boundary

Install Isaac Sim and Isaac Lab only from `${ISAACLAB_PATH}`.
The FinsSim ROS2 workspace remains a separate Python 3.10 uv environment.
Set `ISAACLAB_PATH` and `FINSIM_ISAACLAB_ROOT` as shown in the repository
README before running the commands below.

## Host baseline and verification status

Captured on this host during initial integration:

| Item | Value |
| --- | --- |
| NVIDIA driver | `580.173.02` |
| Training GPU | NVIDIA RTX 3090, physical GPU index `1` |
| Display GPU | NVIDIA T1000, physical GPU index `0` |
| IsaacLab venv | `${ISAACLAB_PATH}/env_isaaclab`, CPython `3.12.12` |
| PyTorch / CUDA runtime | `2.10.0+cu129` / `12.9` |

Isaac Sim `6.0.1.0` and the Lab external extension are installed in the
dedicated Python 3.12 environment. The verified runtime is PyTorch
`2.10.0+cu129`, CUDA `12.9`, on `CUDA_VISIBLE_DEVICES=1` (RTX 3090).

The following validation was run successfully on 2026-08-14:

- `FinsSim-WarpAUV-Direct-v0` registered through the external callback;
- one headless environment reset and zero-action step: 17-dimensional policy
  observations and a finite reward (`0.446526`);
- 64 headless environments, one fresh RSL-RL iteration: 1,536 steps,
  mean reward `2.06`, iteration time `1.07 s`, and `model_0.pt` checkpoint.

The smoke run artifacts are under
`${ISAACLAB_PATH}/logs/rsl_rl/finssim_warpauv_direct/2026-08-14_22-00-57/`.

To reproduce the installation and smoke checks in the dedicated Python 3.12
venv:

```bash
cd "${ISAACLAB_PATH}"
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES  # required for non-interactive Isaac Sim use
uv pip install 'isaacsim[all,extscache]==6.0.1.0' \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow
./isaaclab.sh --install 'rl[rsl-rl]'
./isaaclab.sh -p -m pip install -e \
  "${FINSIM_ISAACLAB_ROOT}/source/finssim_isaaclab_tasks"
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p \
  "${FINSIM_ISAACLAB_ROOT}/scripts/smoke_warpauv.py"
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p \
  "${FINSIM_ISAACLAB_ROOT}/scripts/smoke_step_warpauv.py"
```
