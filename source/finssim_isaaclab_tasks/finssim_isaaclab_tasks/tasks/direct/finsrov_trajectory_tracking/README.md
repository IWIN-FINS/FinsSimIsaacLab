# FinsROV T2-style trajectory tracking

Task ID: `FinsSim-FinsROV-TrajectoryTracking-v0`.

This is the Scheme-A IsaacLab migration of the Unity T2 task. It keeps only
straight-line and circular references, uses the current Isaac FinsROV asset,
Learning-to-Swim equivalent-box hydrodynamics, and eight independent
calibrated thruster actions. It does not modify the HoldForPosition tasks.

## Contract

The action order is:

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

Each clipped action maps directly to its calibrated signed force limit:

```text
F_i = action_i * Fmax_i
```

The 30D observation is, in order:

```text
four body-frame preview offsets (12)
reference velocity (3)
current linear velocity (3)
current angular velocity (3)
body-up direction (3)
reference tangent yaw sin/cos (2)
progress sin/cos at 1x and 2x (4)
```

Position is normalized by `3 m`; linear and angular velocities by `1`; all
normalized values are clipped to `[-2, 2]`. The preview spacing is `0.1 s`.
Unity/controller vectors use `[forward, up, left]`, while Isaac task vectors
use `[forward, left, up]`. Angular velocity also receives the axial-vector
reflection sign when building the Unity-compatible observation.

## Curriculum and reset

The configured curriculum stages are:

```text
stage 0: straight line only
stage 1: straight line or circle, intermediate ranges
stage 2: straight line or circle, full pool-safe ranges
```

The vehicle starts at the reference position at `t=0` with independent
position jitter of at most `0.05 m` per axis. Roll and pitch are randomized in
`[-20, 20]` degrees and yaw in `[-180, 180]` degrees. Initial linear/angular
velocity and thruster force are zero.

The physical pool is represented as a 2 m (Isaac x) by 1 m (Isaac y) tank;
the task/environment origin is at the centre of the water surface (`z=0`).
The trajectory depth is sampled in `[-0.78, -0.22] m`, with `0.03 m` reset
jitter. The boundary check uses a conservative FinsROV envelope of about
`0.60 m x 0.44 m x 0.26 m` and a 5 cm wall/surface/floor clearance. The pool
floor is at `z=-1 m`; crossing the safe hull envelope ends the episode.

## Reward

The Unity T2 position, upright, yaw-rate, and effort terms are retained. An
additional tangent-yaw alignment term is included so that the vehicle is
encouraged to point along the horizontal trajectory tangent, not merely pass
through the correct positions. The yaw term is masked when the reference
horizontal speed is near zero.

## PPO

The RSL-RL configuration keeps the current Learning-to-Swim settings:

```text
2048 environments, 24 rollout steps, 400 iterations,
64-64 ELU actor/critic, learning rate 5e-4,
gamma 0.99, GAE lambda 0.95, 5 epochs, 4 mini-batches.
```

Physics is `120 Hz`, decimation is `2`, and the episode duration is `30 s`.

## Current deployment status

The ROS workspace already contains a Unity T2 `trajectory30` runtime contract,
but this Isaac task's `.pt` checkpoint needs a dedicated IsaacLab trajectory
backend before deployment. Do not load this checkpoint through the existing
Unity `.zip` trajectory backend.
