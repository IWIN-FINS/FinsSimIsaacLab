"""GPU-batched surface-geometry hydrodynamics.

The implementation vectorizes over ``environment x triangle`` and never loops
over environments.  It mirrors MARUS' clipped wetted-area, Stonefish form drag,
skin drag, force-at-centroid torque, and closed-hull volume semantics.  The
tensor implementation is the reference path; a Warp kernel can consume the same
preprocessed topology without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import HydrodynamicsProfileCfg


@dataclass(frozen=True)
class SurfaceMeshData:
    vertices_b: torch.Tensor
    triangles: torch.Tensor
    signed_volume: float
    center_of_buoyancy_b: tuple[float, float, float]
    closed: bool

    @classmethod
    def preprocess(cls, vertices_b: torch.Tensor, triangles: torch.Tensor) -> "SurfaceMeshData":
        vertices = vertices_b.detach().to(dtype=torch.float32)
        faces = triangles.detach().to(dtype=torch.long)
        tri = vertices[faces]
        tetra = torch.sum(tri[:, 0] * torch.cross(tri[:, 1], tri[:, 2], dim=-1), dim=-1) / 6.0
        signed_volume = float(tetra.sum().item())
        if abs(signed_volume) > 1.0e-9:
            centroid = ((tri.sum(dim=1) * 0.25) * tetra[:, None]).sum(dim=0) / signed_volume
        else:
            centroid = torch.zeros(3, dtype=vertices.dtype, device=vertices.device)
        edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0)
        edges = torch.sort(edges, dim=-1).values
        _, counts = torch.unique(edges.cpu(), dim=0, return_counts=True)
        return cls(vertices, faces, signed_volume, tuple(float(x) for x in centroid), bool(torch.all(counts == 2)))


@dataclass
class SurfaceMeshResult:
    wrench_b: torch.Tensor
    form_wrench_b: torch.Tensor
    skin_wrench_b: torch.Tensor
    submerged_volume: torch.Tensor
    wetted_area: torch.Tensor


class BatchedSurfaceGeometryHydro:
    def __init__(self, mesh: SurfaceMeshData, profile: HydrodynamicsProfileCfg, device: torch.device | str):
        self.device = torch.device(device)
        self.profile = profile
        self.vertices = mesh.vertices_b.to(self.device)
        self.triangles = mesh.triangles.to(self.device)
        self.closed = mesh.closed
        self.full_volume = abs(mesh.signed_volume) * profile.mesh_buoyancy_volume_scale

    def compute(
        self,
        root_position_w: torch.Tensor,
        root_quat_w_xyzw: torch.Tensor,
        linear_velocity_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        water_height_w: torch.Tensor,
        water_velocity_b: torch.Tensor,
    ) -> SurfaceMeshResult:
        tri_b = self.vertices[self.triangles]
        tri_b = tri_b.unsqueeze(0).expand(root_position_w.shape[0], -1, -1, -1)
        tri_w = self._rotate(root_quat_w_xyzw[:, None, None, :], tri_b) + root_position_w[:, None, None, :]
        depth = water_height_w[:, None, None] - tri_w[..., 2]
        fraction, centroid_b = self._clipped_area_fraction_and_centroid(tri_b, depth)
        edge_a = tri_b[:, :, 1] - tri_b[:, :, 0]
        edge_b = tri_b[:, :, 2] - tri_b[:, :, 0]
        cross = torch.cross(edge_a, edge_b, dim=-1)
        double_area = cross.norm(dim=-1)
        area = 0.5 * double_area * fraction
        valid = area >= self.profile.min_triangle_area
        area = torch.where(valid, area, torch.zeros_like(area))
        normal = cross / double_area.clamp_min(1.0e-9).unsqueeze(-1)
        point_velocity = linear_velocity_b[:, None, :] + torch.cross(
            angular_velocity_b[:, None, :].expand_as(centroid_b), centroid_b, dim=-1
        )
        relative = water_velocity_b[:, None, :] - point_velocity
        normal_velocity = (relative * normal).sum(dim=-1)
        speed = relative.norm(dim=-1)
        form = 0.5 * self.profile.water_density * self.profile.form_drag_coefficient
        form = form * relative * speed.unsqueeze(-1) * (-normal_velocity).clamp_min(0.0).unsqueeze(-1) * area.unsqueeze(-1)
        tangent = relative - normal_velocity.unsqueeze(-1) * normal
        skin = self.profile.water_density * self.profile.skin_drag_coefficient * tangent * area.unsqueeze(-1)
        form_force, skin_force = form.sum(dim=1), skin.sum(dim=1)
        form_torque = torch.cross(centroid_b, form, dim=-1).sum(dim=1)
        skin_torque = torch.cross(centroid_b, skin, dim=-1).sum(dim=1)
        form_wrench = self._scaled_wrench(form_force, form_torque, self.profile.form_drag_axis_scale, self.profile.form_drag_torque_axis_scale)
        skin_wrench = self._scaled_wrench(skin_force, skin_torque, self.profile.skin_drag_axis_scale, self.profile.skin_drag_torque_axis_scale)
        wetted_area = area.sum(dim=1)
        total_area = (0.5 * double_area).sum(dim=1).clamp_min(1.0e-9)
        volume = self.full_volume * (wetted_area / total_area).clamp(0.0, 1.0) if self.closed else torch.zeros_like(wetted_area)
        return SurfaceMeshResult(form_wrench + skin_wrench, form_wrench, skin_wrench, volume, wetted_area)

    @staticmethod
    def _rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        xyz, w = quat[..., :3], quat[..., 3:4]
        return vector + 2.0 * torch.cross(xyz, torch.cross(xyz, vector, dim=-1) + w * vector, dim=-1)

    @staticmethod
    def _clipped_area_fraction_and_centroid(tri: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inside = depth >= 0.0
        count = inside.sum(dim=-1)
        fraction = torch.zeros_like(depth[..., 0])
        centroid = tri.mean(dim=-2)
        fraction = torch.where(count == 3, torch.ones_like(fraction), fraction)
        # The three cyclic cases are shape-static and remain a single batched GPU graph.
        for pivot in range(3):
            other_a, other_b = (pivot + 1) % 3, (pivot + 2) % 3
            one = (count == 1) & inside[..., pivot]
            da, db, dc = depth[..., pivot], depth[..., other_a], depth[..., other_b]
            ta = (da / (da - db).clamp_min(1.0e-9)).clamp(0.0, 1.0)
            tb = (da / (da - dc).clamp_min(1.0e-9)).clamp(0.0, 1.0)
            pa = tri[..., pivot, :] + ta.unsqueeze(-1) * (tri[..., other_a, :] - tri[..., pivot, :])
            pb = tri[..., pivot, :] + tb.unsqueeze(-1) * (tri[..., other_b, :] - tri[..., pivot, :])
            fraction = torch.where(one, ta * tb, fraction)
            centroid = torch.where(one.unsqueeze(-1), (tri[..., pivot, :] + pa + pb) / 3.0, centroid)
            two = (count == 2) & ~inside[..., pivot]
            ia, ib = tri[..., other_a, :], tri[..., other_b, :]
            da2, db2 = depth[..., other_a], depth[..., other_b]
            ta2 = (da2 / (da2 - depth[..., pivot]).clamp_min(1.0e-9)).clamp(0.0, 1.0)
            tb2 = (db2 / (db2 - depth[..., pivot]).clamp_min(1.0e-9)).clamp(0.0, 1.0)
            pa2 = ia + ta2.unsqueeze(-1) * (tri[..., pivot, :] - ia)
            pb2 = ib + tb2.unsqueeze(-1) * (tri[..., pivot, :] - ib)
            fraction = torch.where(two, 1.0 - (1.0 - ta2) * (1.0 - tb2), fraction)
            centroid = torch.where(two.unsqueeze(-1), (ia + ib + pa2 + pb2) * 0.25, centroid)
        return fraction, centroid

    @staticmethod
    def _scaled_wrench(force: torch.Tensor, torque: torch.Tensor, force_scale, torque_scale) -> torch.Tensor:
        return torch.cat((force * torch.as_tensor(force_scale, device=force.device), torque * torch.as_tensor(torque_scale, device=torque.device)), dim=-1)
