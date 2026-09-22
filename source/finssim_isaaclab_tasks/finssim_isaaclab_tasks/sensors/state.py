"""MARUS-compatible sensor semantics implemented as batched Isaac-side tensors.

Visual devices are configuration records here: tasks bind ``CameraSensorCfg``
and ``LidarSensorCfg`` to Isaac Lab's native sensor objects, while sonar is
bound to the GPU ray-query implementation.  None publish ROS messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SensorCfg:
    sample_period_s: float = 0.02
    latency_s: float = 0.0
    noise_std: float = 0.0


@dataclass(frozen=True)
class ImuSensorCfg(SensorCfg):
    with_gravity: bool = True


@dataclass(frozen=True)
class DepthSensorCfg(SensorCfg):
    water_surface_z_m: float = 0.0


@dataclass(frozen=True)
class DvlSensorCfg(SensorCfg):
    max_range_m: float = 120.0


@dataclass(frozen=True)
class PoseSensorCfg(SensorCfg):
    pass


@dataclass(frozen=True)
class CameraSensorCfg(SensorCfg):
    width: int = 1920
    height: int = 1080
    prim_path: str = ""


@dataclass(frozen=True)
class LidarSensorCfg(SensorCfg):
    horizontal_resolution: int = 1024
    vertical_resolution: int = 16
    min_range_m: float = 0.2
    max_range_m: float = 100.0


@dataclass(frozen=True)
class SonarSensorCfg(SensorCfg):
    dimensions: int = 2
    width: int = 256
    height: int = 1
    min_range_m: float = 0.6
    max_range_m: float = 30.0
    horizontal_fov_deg: float = 60.0
    vertical_fov_deg: float = 30.0
    gaussian_noise_std: float = 0.0
    speckle_level: float = 0.0
    rayleigh_scale: float = 0.0


@dataclass(frozen=True)
class AisSensorCfg(SensorCfg):
    pass


class StateSensorSuite:
    """Computes IMU/depth/DVL/pose/AIS semantic values from batched rigid state."""

    def __init__(self, num_envs: int, device: torch.device | str, seed: int = 0):
        self.num_envs, self.device = num_envs, torch.device(device)
        self._previous_velocity_w = torch.zeros(num_envs, 3, device=self.device)
        self._has_previous_velocity = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._previous_velocity_w[ids] = 0.0
        self._has_previous_velocity[ids] = False

    def sample(
        self,
        position_w: torch.Tensor,
        quat_w_xyzw: torch.Tensor,
        linear_velocity_w: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        dt: float,
        depth_cfg: DepthSensorCfg = DepthSensorCfg(),
        imu_cfg: ImuSensorCfg = ImuSensorCfg(),
        dvl_cfg: DvlSensorCfg = DvlSensorCfg(),
    ) -> dict[str, torch.Tensor]:
        acceleration_w = torch.zeros_like(linear_velocity_w)
        if dt > 1e-6:
            raw = (linear_velocity_w - self._previous_velocity_w) / dt
            acceleration_w = torch.where(self._has_previous_velocity[:, None], raw, acceleration_w)
        self._previous_velocity_w.copy_(linear_velocity_w)
        self._has_previous_velocity.fill_(True)
        if imu_cfg.with_gravity:
            acceleration_w[:, 2] += -9.81
        depth = (depth_cfg.water_surface_z_m - position_w[:, 2:3])
        return {
            "imu_linear_acceleration_w": self._noise(acceleration_w, imu_cfg.noise_std),
            "imu_angular_velocity_b": self._noise(angular_velocity_b, imu_cfg.noise_std),
            "imu_orientation_xyzw": quat_w_xyzw,
            "depth_m": self._noise(depth, depth_cfg.noise_std),
            "dvl_ground_velocity_w": self._noise(linear_velocity_w, dvl_cfg.noise_std),
            "pose_position_w": position_w,
            "pose_orientation_xyzw": quat_w_xyzw,
            "ais_sog_m_s": linear_velocity_w[:, :2].norm(dim=-1, keepdim=True),
        }

    def _noise(self, value: torch.Tensor, std: float) -> torch.Tensor:
        return value if std <= 0 else value + torch.randn(value.shape, device=self.device, dtype=value.dtype, generator=self.generator) * std
