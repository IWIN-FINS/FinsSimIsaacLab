# FinsROV HoldForPosition (3D position + yaw / Unity 16D / eight thrusters)

Task ID: `FinsSim-FinsROV-HoldForPosition-v0`.

This is the standalone FinsROV task for end-to-end eight-thruster training. Its objective is 3D position plus heading (yaw); roll and pitch are not commanded target variables. It does not replace `FinsSim-FinsROV-Direct-v0` (legacy 17D WarpAUV interface) or `FinsSim-FinsROV-PoseHold-v0` (manager-based 13D interface).

## Policy contract

The action is the Unity `FinsROVAgentRuntime.DefaultThrusterOrder`:

```text
[Vertical1=V_LF, Vertical2=V_LB, Vertical3=V_RB, Vertical4=V_RF,
 Horizontal1=H_LF, Horizontal2=H_LB, Horizontal3=H_RB, Horizontal4=H_RF]
```

Each policy action is clipped to `[-1, 1]` and directly maps to its calibrated signed thrust authority: `F_i = a_i * Fmax_i`. Forward limits are `[8.474877, 8.797188, 8.797188, 8.474877, 8.797188, 8.474877, 8.797188, 8.474877] N`; reverse magnitudes are `[7.974983, 8.272793, 8.272793, 7.974983, 8.272793, 7.974983, 8.272793, 7.974983] N`, in canonical action order. The task retains 20 ms delay, 10 ms first-order response, and 1000 N/s slew limit. Force directions and moment arms are from `FinsSim/tools/finsrov_wrench_geometry.yaml`, converted about the Unity Rigidbody centre of mass.

The 16D observation is exactly the field order in `HoldForPosition.CollectObservations`:

```text
0:3   Unity-body target offset / 3 m
3:9   relative target rotation 6D: [m00,m10,m20,m01,m11,m21]
9:12  Unity-body linear velocity / 1 m/s, clipped to [-2,2]
12:15 Unity-body angular velocity / 1 rad/s, clipped to [-2,2]
15    clamp01(||Unity-body target offset|| / 3 m)
```

Unity body coordinates are `[x forward, y up, z left]`; Isaac body coordinates are `[x forward, y left, z up]`. Polar quantities use `[x,z,y]`; angular velocity additionally changes sign because it is an axial vector. Relative rotations use `R_unity = B^T R_isaac B` before extracting Unity's two matrix columns.

The numerical task frame is verified from the explicit transformed thruster positions/directions and does not infer forces from the render mesh. The checked-in generic FBX-to-USD render asset still declares `Y-up` and `0.01 m/unit`; when referenced into Isaac Lab's `Z-up`, metre-based stage it is currently not reoriented or rescaled. Its GUI body axes and roughly 100x visual scale are therefore **not** authoritative for judging the task's forward/up axes. This asset-import issue does not alter the policy observation, thrust, or hydrodynamic coordinate calculations described above, but must be fixed before using the GUI for visual/collision-frame validation.

## Dynamics, reset, and reward

The water model is the Learning to Swim equivalent-inertia rectangular-body model, not the existing Fossen model: fully submerged hydrostatic buoyancy plus its quadratic and linear-viscous drag equations. It uses FinsROV's `12.11 kg` mass, Isaac-axis inertia `(0.13154832, 0.23185866, 0.24651341) kg m^2`, `0.0118 m^3` volume, `1027 kg/m^3` water density, and CoM-to-CoB `(0,0,0.11) m`.

Learning to Swim's reset/training settings are retained where compatible with the physical pool: 120 Hz physics, decimation 2, **10 s episodes**, a 2 m solid horizontal reset disk, independent random reset and target depths in `[-0.95,-0.05] m`, random **yaw-only** target heading, 10% guided starts, 7 m horizontal bounds, CoB randomization radius `0.05 m`, and relative-volume randomization `[0.0103, 0.0133] m^3` (the FinsROV equivalent of Learning to Swim's approximately +/-13.2% range). With water surface `z=0` and pool bottom `z=-1 m`, both the target and reset CoM remain inside the water column. The ground/collision plane is at the pool bottom, and an episode terminates if the vehicle CoM crosses either 5 cm safety boundary. The Learning-to-Swim PPO configuration is also retained: 2048 environments, 24 rollout steps, 400 iterations, 64-64 ELU actor/critic, and the same PPO hyperparameters.

The reward is evaluated from the hidden 3D-position/yaw goal state and 8D action:

```text
0.2 * exp(-||position error||^2)
+ 0.5 * exp(-absolute yaw error in rad)
+ 0.2 * exp(-||eight actions||^2)
```

The action shape remains eight direct Unity thruster commands. Existing checkpoints must not be resumed because the reward has changed.

## Run

Start a one-environment GUI smoke test from the Isaac Lab checkout:

```bash
cd ${ISAACLAB_PATH}
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/smoke_step.py \
  --task FinsSim-FinsROV-HoldForPosition-v0 --num-envs 1 --steps 300
```

For the Learning-to-Swim-scale headless training run:

```bash
cd ${ISAACLAB_PATH}
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
  ${FINSIM_ISAACLAB_ROOT}/scripts/train.py \
  --task FinsSim-FinsROV-HoldForPosition-v0 --num_envs 2048 --max_iterations 400
```

Copy `policy_manifest.json` beside a trained checkpoint. The existing ROS runtime adapter still only accepts the legacy WarpAUV 17D/6D format; deployment of this 16D/8D checkpoint needs a dedicated runtime loader.
