"""Factories binding FinsSim sensor semantics to Isaac Lab native sensors."""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

from .state import CameraSensorCfg, LidarSensorCfg, SonarSensorCfg


def make_camera_cfg(cfg: CameraSensorCfg, prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=cfg.sample_period_s,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 100.0)),
        width=cfg.width,
        height=cfg.height,
    )


def make_lidar_cfg(cfg: LidarSensorCfg, prim_path: str, mesh_prim_paths: list[str]) -> RayCasterCfg:
    horizontal_res_deg = 360.0 / max(1, cfg.horizontal_resolution)
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=mesh_prim_paths,
        update_period=cfg.sample_period_s,
        ray_alignment="base",
        max_distance=cfg.max_range_m,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=cfg.vertical_resolution,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=horizontal_res_deg,
        ),
    )


def make_sonar_cfg(cfg: SonarSensorCfg, prim_path: str, mesh_prim_paths: list[str]) -> RayCasterCfg:
    # Aperture is derived from MARUS' angular FoV. RayCaster supplies distances;
    # ``sonar_intensity`` supplies attenuation and stochastic intensity semantics.
    focal_length = 24.0
    horizontal_aperture = 2.0 * focal_length * math.tan(math.radians(cfg.horizontal_fov_deg) * 0.5)
    vertical_aperture = 2.0 * focal_length * math.tan(math.radians(cfg.vertical_fov_deg) * 0.5)
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=mesh_prim_paths,
        update_period=cfg.sample_period_s,
        ray_alignment="base",
        max_distance=cfg.max_range_m,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            width=cfg.width,
            height=cfg.height if cfg.dimensions == 3 else 1,
        ),
    )


def sonar_intensity(distance_m: torch.Tensor, valid: torch.Tensor, cfg: SonarSensorCfg,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """MARUS-style geometric return with range, Gaussian, speckle and Rayleigh terms."""
    normalized = (1.0 - distance_m / cfg.max_range_m).clamp(0.0, 1.0)
    intensity = normalized * valid.to(normalized.dtype)
    if cfg.gaussian_noise_std > 0:
        intensity = intensity + torch.randn(intensity.shape, device=intensity.device, generator=generator) * cfg.gaussian_noise_std
    if cfg.speckle_level > 0:
        intensity = intensity * (1.0 + torch.randn(intensity.shape, device=intensity.device, generator=generator) * cfg.speckle_level)
    if cfg.rayleigh_scale > 0:
        uniform = torch.rand(intensity.shape, device=intensity.device, generator=generator).clamp_min(1.0e-7)
        intensity = intensity + cfg.rayleigh_scale * torch.sqrt(-2.0 * torch.log(uniform))
    return intensity.clamp(0.0, 1.0)

