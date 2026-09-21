from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class IsaacLabRslRlActor(nn.Module):
    """Two-layer ELU actor shared by the supported RSL-RL checkpoints."""

    def __init__(self, observation_dim: int = 17, action_dim: int = 6) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)


class IsaacLabRslRlPolicy:
    """Load and run the historical WarpAUV actor from an IsaacLab RSL-RL checkpoint."""

    observation_dim = 17
    action_dim = 6

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        requested_device = str(device).strip().lower()
        if requested_device in {"", "auto"}:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"IsaacLab checkpoint does not exist: {self.checkpoint_path}")

        self.actor = IsaacLabRslRlActor(self.observation_dim, self.action_dim).to(self.device)
        try:
            checkpoint: Any = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:  # torch versions before the weights_only keyword
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        actor_state = self._extract_actor_state(checkpoint)
        try:
            self.actor.actor.load_state_dict(actor_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"IsaacLab actor architecture does not match the expected {self.observation_dim}->64->64->{self.action_dim} network"
            ) from exc
        self.actor.eval()
        self.iteration = int(checkpoint.get("iter", -1))

    def predict_raw(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size != self.observation_dim:
            raise ValueError(f"IsaacLab policy expects {self.observation_dim} observation values, got {obs.size}")
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            output = self.actor(tensor).squeeze(0)
        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        return np.clip(self.predict_raw(observation), -1.0, 1.0).astype(np.float32, copy=False), None

    def reset(self) -> None:
        """Match the stateless deterministic actor API used by the ROS loader."""

    def _extract_actor_state(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        """Extract the Lab 2-style actor state used by the legacy WarpAUV export."""

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Unsupported IsaacLab checkpoint format: expected model_state_dict in {self.checkpoint_path}"
            )
        model_state = checkpoint["model_state_dict"]
        if not isinstance(model_state, dict):
            raise ValueError(f"model_state_dict is not a state dictionary in {self.checkpoint_path}")
        actor_state = {
            key[len("actor.") :]: value
            for key, value in model_state.items()
            if str(key).startswith("actor.")
        }
        if not actor_state:
            raise ValueError(f"No actor.* parameters found in {self.checkpoint_path}")
        return actor_state


class FinsROVHoldForPositionRslRlPolicy(IsaacLabRslRlPolicy):
    """Run the FinsROV 16D-observation, eight-thruster RSL-RL actor.

    Isaac Lab 3 RSL-RL checkpoints store this actor under ``actor_state_dict``
    using ``mlp.*`` names, unlike the historical WarpAUV ``model_state_dict``
    export handled by :class:`IsaacLabRslRlPolicy`.
    """

    observation_dim = 16
    action_dim = 8

    def _extract_actor_state(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
            raise ValueError(
                f"Unsupported FinsROV IsaacLab checkpoint format: expected actor_state_dict in {self.checkpoint_path}"
            )
        actor_state_dict = checkpoint["actor_state_dict"]
        if not isinstance(actor_state_dict, dict):
            raise ValueError(f"actor_state_dict is not a state dictionary in {self.checkpoint_path}")

        # RSL-RL also serializes distribution.std_param.  Deterministic
        # deployment needs only the actor MLP, whose keys are ``mlp.0.*`` etc.
        actor_state = {
            key[len("mlp.") :]: value
            for key, value in actor_state_dict.items()
            if str(key).startswith("mlp.")
        }
        if not actor_state:
            raise ValueError(f"No actor_state_dict mlp.* parameters found in {self.checkpoint_path}")
        return actor_state


class FinsROVHoldForPositionWrenchRslRlPolicy(FinsROVHoldForPositionRslRlPolicy):
    """Run the FinsROV 16D-observation, six-wrench-action RSL-RL actor."""

    action_dim = 6


def load_isaaclab_backend(checkpoint_path: str, *, device: str = "auto") -> IsaacLabRslRlPolicy:
    return IsaacLabRslRlPolicy(checkpoint_path, device=device)


def load_finsrov_hold_for_position_backend(
    checkpoint_path: str, *, device: str = "auto"
) -> FinsROVHoldForPositionRslRlPolicy:
    """Load the current FinsROV HoldForPosition 16D/8D RSL-RL checkpoint."""

    return FinsROVHoldForPositionRslRlPolicy(checkpoint_path, device=device)


def load_finsrov_hold_for_position_wrench_backend(
    checkpoint_path: str, *, device: str = "auto"
) -> FinsROVHoldForPositionWrenchRslRlPolicy:
    """Load the FinsROV HoldForPosition 16D/6D wrench RSL-RL checkpoint."""

    return FinsROVHoldForPositionWrenchRslRlPolicy(checkpoint_path, device=device)
