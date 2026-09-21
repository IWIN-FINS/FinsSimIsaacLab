# FinsROV HoldForPosition Wrench (Unity 16D / six wrench actions)

Task ID: `FinsSim-FinsROV-HoldForPosition-Wrench-v0`.

This task inherits the 16D Unity-compatible observation, equivalent-box Learning-to-Swim hydrodynamics, reset distribution, reward structure, episode length, PPO settings, FinsROV asset, and delayed calibrated-thruster dynamics from the neighbouring `finsrov_hold_for_position` task. It changes only the policy action contract: six normalized wrench actions are allocated to the same eight physical thrusters.

## Action contract

The normalized policy output is ordered as follows:

```text
[surge=Fx, sway=Fz, heave=Fy, roll=Mx, pitch=Mz, yaw=My]
```

It is expressed in the Unity FinsROV controller body axes (`x` forward, `y` up, `z` left). `+/-1` maps to the calibrated pure-axis physical capabilities:

```text
[19.528527 N, 18.415027 N, 22.501886 N,
  3.863995 N*m, 3.079985 N*m, 7.080833 N*m]
```

The bounded physical allocator uses the calibrated force directions and CoM-referenced moment arms with symmetric `+/-7 N` limits for every thruster. A feasible wrench is reproduced exactly; a coupled request outside the feasible set is projected to the closest bounded wrench after normalization by the six limits. The allocated force is then normalized per thruster (`u_i = F_i / 7 N`) before entering the inherited actuator. That actuator reconstructs the same `+/-7 N` request and applies its `20 ms` delay, `10 ms` first-order response, and `1000 N/s` slew limit.

This is the same force and wrench-limit convention as ROS
`ppo_wrench_for_pose_physical_wrench_allocator_sim.yaml`; the hardware
deployment profile intentionally has separate asymmetric calibrated limits.

The reward's action-energy term remains the Learning-to-Swim form but now acts on the six normalized wrench actions. No checkpoint is interchangeable with the 8D direct-thruster task.

## Disable roll and pitch control

The 6D action/checkpoint shape is retained, but roll (`Mx`) and pitch (`Mz`)
policy requests are masked to zero by default because this task controls
position and yaw. The mask only disables the **policy-requested actuator
moments**; it intentionally does not remove hydrostatic or hydrodynamic body
moments.

Set it for an entire training run through Hydra:

```bash
... --task FinsSim-FinsROV-HoldForPosition-Wrench-v0 env.enable_roll_pitch_moments=true
```

For a running Python environment, toggle it directly:

```python
env.unwrapped.set_roll_pitch_moments_enabled(False)
```

## ROS2 deployment

Use
`ros2_ws/src/finssim_motion_control/config/FinsROV/ppo_wrench_isaaclab_finsrov_hold_for_position.yaml`.
Set `checkpoint_status: ready` and `checkpoint_path` to a checkpoint trained
with this task's `finsrov_physical_wrench_sim7n_v2` schema. The profile sends
the policy's 6D action through `physical_wrench_allocator` with symmetric
`+/-7 N` limits and masks roll/pitch requests.

## Run

GUI smoke test:

```bash
cd ${ISAACLAB_PATH}
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/smoke_step.py \
  --task FinsSim-FinsROV-HoldForPosition-Wrench-v0 --num-envs 1 --steps 300
```

2048-environment PPO training:

```bash
cd ${ISAACLAB_PATH}
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/train.py \
  --task FinsSim-FinsROV-HoldForPosition-Wrench-v0 --num_envs 2048 --max_iterations 400
```
