"""Warp 1.13 kernel for large-batch surface-geometry hydrodynamics."""

from __future__ import annotations

import torch
import warp as wp

from .core import HydrodynamicsProfileCfg
from .mesh import SurfaceMeshData, SurfaceMeshResult


@wp.kernel
def _surface_hydro_kernel(
    vertices: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    triangle_count: int,
    root_position: wp.array(dtype=wp.vec3),
    root_quat: wp.array(dtype=wp.quat),
    linear_velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    water_height: wp.array(dtype=wp.float32),
    water_velocity: wp.array(dtype=wp.vec3),
    density: float,
    form_coefficient: float,
    skin_coefficient: float,
    min_area: float,
    output: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_id = tid // triangle_count
    face_id = tid - env_id * triangle_count
    i0 = triangle_indices[face_id * 3]
    i1 = triangle_indices[face_id * 3 + 1]
    i2 = triangle_indices[face_id * 3 + 2]
    p0 = vertices[i0]
    p1 = vertices[i1]
    p2 = vertices[i2]
    q = root_quat[env_id]
    origin = root_position[env_id]
    w0 = wp.quat_rotate(q, p0) + origin
    w1 = wp.quat_rotate(q, p1) + origin
    w2 = wp.quat_rotate(q, p2) + origin
    d0 = water_height[env_id] - w0[2]
    d1 = water_height[env_id] - w1[2]
    d2 = water_height[env_id] - w2[2]
    n_inside = int(d0 >= 0.0) + int(d1 >= 0.0) + int(d2 >= 0.0)
    fraction = float(0.0)
    center = (p0 + p1 + p2) / 3.0
    if n_inside == 3:
        fraction = 1.0
    elif n_inside == 1:
        pin = p0
        pa = p1
        pb = p2
        din = d0
        da = d1
        db = d2
        if d1 >= 0.0:
            pin, pa, pb, din, da, db = p1, p2, p0, d1, d2, d0
        elif d2 >= 0.0:
            pin, pa, pb, din, da, db = p2, p0, p1, d2, d0, d1
        ta = wp.clamp(din / (din - da), 0.0, 1.0)
        tb = wp.clamp(din / (din - db), 0.0, 1.0)
        ia = pin + ta * (pa - pin)
        ib = pin + tb * (pb - pin)
        fraction = ta * tb
        center = (pin + ia + ib) / 3.0
    elif n_inside == 2:
        pout = p0
        pa = p1
        pb = p2
        dout = d0
        da = d1
        db = d2
        if d1 < 0.0:
            pout, pa, pb, dout, da, db = p1, p2, p0, d1, d2, d0
        elif d2 < 0.0:
            pout, pa, pb, dout, da, db = p2, p0, p1, d2, d0, d1
        ta = wp.clamp(da / (da - dout), 0.0, 1.0)
        tb = wp.clamp(db / (db - dout), 0.0, 1.0)
        ia = pa + ta * (pout - pa)
        ib = pb + tb * (pout - pb)
        fraction = 1.0 - (1.0 - ta) * (1.0 - tb)
        center = (pa + pb + ia + ib) * 0.25
    cross = wp.cross(p1 - p0, p2 - p0)
    double_area = wp.length(cross)
    area = 0.5 * double_area * fraction
    if area < min_area:
        return
    normal = cross / double_area
    relative = water_velocity[env_id] - (linear_velocity[env_id] + wp.cross(angular_velocity[env_id], center))
    normal_velocity = wp.dot(relative, normal)
    form = wp.vec3(0.0, 0.0, 0.0)
    if normal_velocity < -1.0e-5:
        form = 0.5 * density * form_coefficient * relative * wp.length(relative) * (-normal_velocity) * area
    tangent = relative - normal_velocity * normal
    skin = density * skin_coefficient * tangent * area
    force = form + skin
    torque = wp.cross(center, force)
    base = env_id * 8
    wp.atomic_add(output, base, force[0])
    wp.atomic_add(output, base + 1, force[1])
    wp.atomic_add(output, base + 2, force[2])
    wp.atomic_add(output, base + 3, torque[0])
    wp.atomic_add(output, base + 4, torque[1])
    wp.atomic_add(output, base + 5, torque[2])
    wp.atomic_add(output, base + 6, area)
    wp.atomic_add(output, base + 7, 0.5 * double_area)


class WarpSurfaceGeometryHydro:
    """Runs one Warp thread per environment/triangle and reduces per-env wrench."""

    def __init__(self, mesh: SurfaceMeshData, profile: HydrodynamicsProfileCfg, device: str):
        wp.init()
        self.profile = profile
        self.device = device
        self.vertices = mesh.vertices_b.to(dtype=torch.float32, device=device).contiguous()
        self.triangles = mesh.triangles.to(dtype=torch.int32, device=device).contiguous().view(-1)
        self.triangle_count = mesh.triangles.shape[0]
        self.closed = mesh.closed
        self.full_volume = abs(mesh.signed_volume) * profile.mesh_buoyancy_volume_scale

    def compute(self, root_position_w, root_quat_w_xyzw, linear_velocity_b, angular_velocity_b,
                water_height_w, water_velocity_b) -> SurfaceMeshResult:
        count = root_position_w.shape[0]
        output = torch.zeros(count, 8, dtype=torch.float32, device=root_position_w.device)
        wp.launch(
            _surface_hydro_kernel,
            dim=count * self.triangle_count,
            inputs=[
                wp.from_torch(self.vertices, dtype=wp.vec3), wp.from_torch(self.triangles), self.triangle_count,
                wp.from_torch(root_position_w.contiguous(), dtype=wp.vec3),
                wp.from_torch(root_quat_w_xyzw.contiguous(), dtype=wp.quat),
                wp.from_torch(linear_velocity_b.contiguous(), dtype=wp.vec3),
                wp.from_torch(angular_velocity_b.contiguous(), dtype=wp.vec3),
                wp.from_torch(water_height_w.contiguous()), wp.from_torch(water_velocity_b.contiguous(), dtype=wp.vec3),
                self.profile.water_density, self.profile.form_drag_coefficient, self.profile.skin_drag_coefficient,
                self.profile.min_triangle_area, wp.from_torch(output.view(-1)),
            ],
            device=self.device,
        )
        wrench, wetted, total = output[:, :6], output[:, 6], output[:, 7].clamp_min(1.0e-9)
        volume = self.full_volume * (wetted / total).clamp(0.0, 1.0) if self.closed else torch.zeros_like(wetted)
        zeros = torch.zeros_like(wrench)
        return SurfaceMeshResult(wrench, zeros, zeros, volume, wetted)
