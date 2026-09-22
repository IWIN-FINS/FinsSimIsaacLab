"""Non-episodic FinsROV environment used by a ROS 2 simulator session."""

from __future__ import annotations

import torch

from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.env import (
    FinsROVHoldForPositionEnv,
)


class FinsROVRos2Env(FinsROVHoldForPositionEnv):
    """Disable training termination; reset is initiated explicitly through ROS 2."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        # Install the authored graph while DirectRLEnv is still composing the
        # stage.  Doing this from the runner after construction is too late for
        # OmniGraph to materialize node handles in this Isaac Lab lifecycle.
        from .action_graph import load_authored_graph

        load_authored_graph()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros

    def reset_from_ros(self) -> None:
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
