"""Bounded physical 6D-wrench allocation for the FinsROV eight-thruster layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.contract import (
    THRUSTER_DIRECTIONS_B,
    THRUSTER_POSITIONS_FROM_COM_B,
)


# The policy uses the established FinsROV/ROS wrench checkpoint order.  In the
# Unity body it means [Fx, Fz, Fy, Mx, Mz, My].  Since Isaac body axes are
# [forward, left, up], its translational entries already have the same order;
# the three rotational entries gain a minus sign under the frame reflection.
POLICY_WRENCH_NAMES: Final[tuple[str, ...]] = (
    "surge=Fx",
    "sway=Fz",
    "heave=Fy",
    "roll=Mx",
    "pitch=Mz",
    "yaw=My",
)

# Pure-axis capability limits of the calibrated FinsROV_Fossen layout with all
# eight channels constrained to +/-7 N.  Units follow POLICY_WRENCH_NAMES.
POLICY_WRENCH_LIMITS: Final[tuple[float, ...]] = (
    19.528527,
    18.415027,
    22.501886,
    3.863995,
    3.079985,
    7.080833,
)


@dataclass(frozen=True)
class FinsROVPhysicalWrenchAllocatorCfg:
    """Physical wrench and +/-7 N actuator limits used by the allocator."""

    wrench_limits: tuple[float, float, float, float, float, float] = POLICY_WRENCH_LIMITS
    max_forward_force_n: tuple[float, ...] = (7.0,) * 8
    max_reverse_force_n: tuple[float, ...] = (7.0,) * 8
    enable_roll_pitch_moments: bool = False


class FinsROVPhysicalWrenchAllocator:
    """Map normalized six-axis policy actions to normalized eight-thruster commands.

    The solver is the batched bounded weighted least-squares active-set method
    used by the FinsSim physical wrench allocator.  Feasible targets recreate
    the requested wrench; combined requests outside the eight-thruster feasible
    set return the closest bounded wrench, weighted by the pure-axis limits.
    """

    def __init__(self, cfg: FinsROVPhysicalWrenchAllocatorCfg, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.wrench_limits = torch.tensor(cfg.wrench_limits, dtype=torch.float32, device=self.device)
        self.lower_force_n = -self._validate_force_limits(cfg.max_reverse_force_n, "max_reverse_force_n")
        self.upper_force_n = self._validate_force_limits(cfg.max_forward_force_n, "max_forward_force_n")
        self.roll_pitch_moments_enabled = cfg.enable_roll_pitch_moments
        self.wrench_matrix_i = self._build_wrench_matrix()
        self.free_thruster_masks, self.free_mask_pseudoinverses = self._build_free_mask_pseudoinverses()

    def allocate(self, normalized_wrench_actions: torch.Tensor) -> torch.Tensor:
        """Return normalized 8D commands for the inherited calibrated actuator.

        Inputs are normalized ``[-1, 1]`` policy actions in
        ``[surge, sway, heave, roll, pitch, yaw]`` order.  The bounded solver
        works in N/N*m and returns 8D actions in ``[-1, 1]``.  Passing those
        actions to ``CalibratedNormalizedThrusters`` reconstructs the same
        allocated force using its per-thruster +/-7 N limits.
        """

        actions = torch.clamp(normalized_wrench_actions, -1.0, 1.0)
        if not self.roll_pitch_moments_enabled:
            actions = actions.clone()
            actions[:, 3:5] = 0.0
        target_wrench_i = self.policy_actions_to_wrench_i(actions)
        allocated_force_n = self._allocate_bounded_force(target_wrench_i)
        return torch.where(
            allocated_force_n >= 0.0,
            allocated_force_n / self.upper_force_n.to(allocated_force_n.dtype),
            allocated_force_n / (-self.lower_force_n).to(allocated_force_n.dtype),
        )

    def set_roll_pitch_moments_enabled(self, enabled: bool) -> None:
        """Enable or mask the policy's roll (Mx) and pitch (Mz) requests.

        Masking preserves the 6D checkpoint/action interface while ensuring
        that policy roll and pitch commands request zero moment from the
        eight-thruster allocator.  It does not disable physical hydrostatic or
        hydrodynamic moments that arise from the vehicle state itself.
        """

        self.roll_pitch_moments_enabled = bool(enabled)

    def policy_actions_to_wrench_i(self, normalized_wrench_actions: torch.Tensor) -> torch.Tensor:
        """Scale policy actions and transform their moments into Isaac axes."""

        physical_policy_wrench = normalized_wrench_actions * self.wrench_limits.to(normalized_wrench_actions.dtype)
        return torch.cat((physical_policy_wrench[:, :3], -physical_policy_wrench[:, 3:]), dim=-1)

    def forces_to_policy_wrench(self, force_n: torch.Tensor) -> torch.Tensor:
        """Reconstruct the realized wrench in policy order for tests/diagnostics."""

        wrench_i = force_n @ self.wrench_matrix_i.to(force_n.dtype).transpose(0, 1)
        return torch.cat((wrench_i[:, :3], -wrench_i[:, 3:]), dim=-1)

    def _validate_force_limits(self, values: tuple[float, ...], name: str) -> torch.Tensor:
        limits = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1)
        if limits.numel() != 8:
            raise ValueError(f"{name} must contain 8 force magnitudes, got {limits.numel()}")
        if not bool(torch.isfinite(limits).all()) or bool(torch.any(limits <= 0.0)):
            raise ValueError(f"{name} must contain finite positive force magnitudes")
        return limits

    def _build_wrench_matrix(self) -> torch.Tensor:
        positions_b = torch.tensor(THRUSTER_POSITIONS_FROM_COM_B, dtype=torch.float32, device=self.device)
        directions_b = torch.tensor(THRUSTER_DIRECTIONS_B, dtype=torch.float32, device=self.device)
        directions_b /= torch.linalg.vector_norm(directions_b, dim=-1, keepdim=True).clamp_min(1.0e-6)
        moments_b = torch.cross(positions_b, directions_b, dim=-1)
        return torch.cat((directions_b.transpose(0, 1), moments_b.transpose(0, 1)), dim=0)

    def _build_free_mask_pseudoinverses(self) -> tuple[torch.Tensor, torch.Tensor]:
        mask_ids = torch.arange(1 << 8, device=self.device, dtype=torch.long)
        thruster_bits = 1 << torch.arange(8, device=self.device, dtype=torch.long)
        free_thruster_masks = (mask_ids.unsqueeze(1) & thruster_bits.unsqueeze(0)) != 0
        weighted_wrench_matrix = self.wrench_matrix_i / self.wrench_limits.view(6, 1)
        masked_weighted_matrices = weighted_wrench_matrix.unsqueeze(0) * free_thruster_masks.unsqueeze(1)
        return free_thruster_masks, torch.linalg.pinv(masked_weighted_matrices)

    def _allocate_bounded_force(self, target_wrench_i: torch.Tensor) -> torch.Tensor:
        weights = torch.reciprocal(self.wrench_limits).to(target_wrench_i.dtype)
        lower = self.lower_force_n.to(target_wrench_i.dtype)
        upper = self.upper_force_n.to(target_wrench_i.dtype)
        weighted_target = target_wrench_i * weights.view(1, 6)
        batch_size = target_wrench_i.shape[0]
        force_n = torch.zeros(batch_size, 8, dtype=target_wrench_i.dtype, device=target_wrench_i.device)
        free_mask_ids = torch.full((batch_size,), (1 << 8) - 1, dtype=torch.long, device=target_wrench_i.device)
        wrench_matrix_i = self.wrench_matrix_i.to(target_wrench_i.dtype)

        for _ in range(8):
            free = self.free_thruster_masks[free_mask_ids]
            pseudoinverse = self.free_mask_pseudoinverses[free_mask_ids].to(target_wrench_i.dtype)
            fixed_force_n = torch.where(free, torch.zeros_like(force_n), force_n)
            weighted_fixed_wrench = (fixed_force_n @ wrench_matrix_i.transpose(0, 1)) * weights.view(1, 6)
            free_solution = torch.bmm(pseudoinverse, (weighted_target - weighted_fixed_wrench).unsqueeze(2)).squeeze(2)
            force_n = torch.where(free, free_solution, force_n)

            violation = torch.maximum(lower.view(1, -1) - force_n, force_n - upper.view(1, -1))
            violation = torch.where(free, violation, torch.full_like(violation, float("-inf")))
            worst_violation, worst_index = torch.max(violation, dim=1)
            needs_clamp = worst_violation > 0.0
            selected_force = force_n.gather(1, worst_index.unsqueeze(1))
            selected_bound = torch.minimum(
                torch.maximum(selected_force, lower[worst_index].unsqueeze(1)),
                upper[worst_index].unsqueeze(1),
            )
            force_n = force_n.scatter(
                1,
                worst_index.unsqueeze(1),
                torch.where(needs_clamp.unsqueeze(1), selected_bound, selected_force),
            )
            free_mask_ids = free_mask_ids & ~(needs_clamp.to(torch.long) << worst_index)

        return torch.minimum(torch.maximum(force_n, lower.view(1, -1)), upper.view(1, -1))
