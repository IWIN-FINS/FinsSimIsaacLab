# WarpAUV policy schemas

The two 17-value schemas are deliberately incompatible.  A checkpoint
manifest must set exactly one `observation_schema` value before it can be
loaded or deployed.

| Schema | Quaternion order | Observation order | Consumer |
| --- | --- | --- | --- |
| `legacy_warpauv_v2` | `wxyz` | goal quaternion, body goal offset, body quaternion, body linear velocity, body angular velocity | `runtime/finssim_isaaclab`, historical checkpoints only |
| `warpauv_lab3_v0` | `xyzw` | goal quaternion, body goal offset, body quaternion, body linear velocity, body angular velocity | `FinsSim-WarpAUV-Direct-v0`, new Lab 3 checkpoints |

The numerical field count alone is not sufficient for compatibility.  The
runtime package rejects neither format by inspecting arbitrary historical
`.pt` files, so deployment tooling must read a neighboring manifest such as:

```json
{
  "task_id": "FinsSim-WarpAUV-Direct-v0",
  "observation_schema": "warpauv_lab3_v0",
  "quaternion_order": "xyzw"
}
```

Only `legacy_warpauv_v2` is supported by the ROS2 runtime adapter in this
milestone.  Lab 3 hardware deployment is intentionally out of scope.
