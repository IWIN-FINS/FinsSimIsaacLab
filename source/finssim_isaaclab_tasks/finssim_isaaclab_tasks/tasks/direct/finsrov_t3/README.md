# FinsROV T3 moving-target task

Task ID: `FinsSim-FinsROV-T3-MovingTarget-v0`.

This is a separate task. It does not change
`FinsSim-FinsROV-HoldForPosition-v0`. The T3 task adapts Unity
`ControlForMovingTargetReward` to the FinsROV Isaac hydrodynamics and eight
independent calibrated thruster actions.

The 13D observation is:

```text
target offset in body frame       3
target velocity in body frame     3
vehicle linear velocity           3
vehicle angular velocity          3
target distance                   1
```

All vectors are converted from Isaac `[x forward, y left, z up]` to Unity
`[x forward, y up, z left]`; angular velocity uses the axial-vector sign.
Actions remain `[V_LF,V_LB,V_RB,V_RF,H_LF,H_LB,H_RB,H_RF]`.

The target is a kinematic object. Its velocity is moved toward a randomly
retargeted desired velocity with a configured acceleration limit, while its
position is integrated every physics step and bounded within the one-metre
pool. `target_acceleration_w` is the acceleration of the target object, not a
desired vehicle acceleration.

Run a smoke test with:

```bash
cd ${ISAACLAB_PATH}
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/smoke_step.py \
  --task FinsSim-FinsROV-T3-MovingTarget-v0 --num-envs 1 --steps 300
```
