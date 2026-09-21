# FinsSim Isaac Lab runtime adapter

This package is deliberately independent of Isaac Sim and supports Python
`>=3.10,<3.13`.  It is installed in the ROS2 uv environment and retains the
historical `finssim_isaaclab` Python import API.

Supported ROS2 policy contracts are deliberately explicit:

- `legacy_warpauv_v2` is the historical WarpAUV checkpoint format.  Its
  17-dimensional observation uses the Lab 2-era `wxyz` convention.  Do not
  load `FinsSim-WarpAUV-Direct-v0` artifacts through this loader.  The Lab 3
  task uses `warpauv_lab3_v0`, whose 17-value order is documented in
  [`../docs/reference/policy-schemas.md`](../docs/reference/policy-schemas.md) and whose artifacts
  must carry that schema identifier in their manifest.
- `FinsSim-FinsROV-HoldForPosition-v0` is loaded by the ROS2 backend
  `isaaclab_finsrov_hold_for_position`.  It accepts the RSL-RL `.pt` checkpoint
  directly (`actor_state_dict`, 16 observations, 8 actions) when it was trained
  with action schema `finsrov_calibrated_normalized_thrust_v2`. The 16 values
  use the Unity HoldForPosition pose-16/rotation-6D contract and targets are
  yaw-only. Its actions are direct FinsROV thruster actions in the fixed order
  `[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]`; the ROS profile converts
  each action directly to its calibrated signed force limit, using
  `F_i = clip(action_i, -1, 1) * Fmax_i` and the per-thruster positive and
  negative limits in the deployment YAML.
- `FinsSim-FinsROV-HoldForPosition-Wrench-v0` is loaded by
  `isaaclab_finsrov_hold_for_position_wrench`.  It accepts an RSL-RL `.pt`
  checkpoint with 16 observations and 6 actions under
  `finsrov_physical_wrench_sim7n_v2`.  ROS maps its action order
  `[surge, sway, heave, roll, pitch, yaw]` through the bounded physical
  allocator to eight symmetric `+/-7 N` thruster commands; roll and pitch are
  masked by the provided profile.

The checkpoint loader contains no Isaac Sim dependency.  A deployment profile
must still select the matching backend and output mode; a `.pt` file alone
does not encode the observation schema, action ordering, or safety limits.

`legacy_checkpoints/warpauv_v2_poshold_dr_2024-09-13.pt` is retained only for
best-effort regression playback. Its SHA-256 is
`6fdc46b49f1e533e84779fce2f5cb3ceb524e2e92c7309c48f97be01b66c3c8a`.
