# Isaac 水下仿真框架

`physics/` 是所有 FinsSim Isaac task 的唯一水动力入口，单位为 SI，四元数为 Lab 3 `xyzw`，wrench 为 body-frame `[u,v,w,p,q,r]`。

支持 Unity/MARUS 对应模式：`off`、`hydrostatic_only`、`simplified_fossen`、`fossen`、`mesh`、`fossen_mesh_residual`。参数化模式、推进器、海流和状态传感器均使用 GPU batch；mesh 同时提供张量参考实现和 Warp 1.13 `environment × triangle` kernel，缺少 mesh wrench 时会明确报错，不能静默退回参数化模型。

相机和 LiDAR 使用 Isaac Lab 原生 `CameraCfg` / `RayCasterCfg`。2D/3D 声呐使用 RayCaster 几何距离并通过 FinsSim 后处理生成距离衰减、Gaussian、speckle 和 Rayleigh 噪声。传感器按 task 组合，默认不发布 ROS2。

FinsROV 从 Unity `FinsROV_Fossen` 进行一次性独立导入。资产 manifest 记录源 FBX、物理 hull、哈希、单位和转换入口；Unity 不读取也不依赖 Isaac 文件。

`scripts/physics_bench.py --vehicle finsrov|warpauv [--backend ...]` 可在 CUDA 上输出固定状态的水动力与推进器 wrench，供后续 Unity fixture 对照使用。ROS2 transport 不在 extension 内实现。
