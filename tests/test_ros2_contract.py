"""Tests that do not need Isaac Sim or a ROS 2 installation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "finssim_isaaclab_tasks"
    / "finssim_isaaclab_tasks"
    / "ros2"
    / "contract.py"
)
SPEC = importlib.util.spec_from_file_location("finsrov_ros2_contract", MODULE_PATH)
assert SPEC and SPEC.loader
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def test_truth_mapping_preserves_polar_vectors_and_corrects_axial_vectors() -> None:
    truth = contract.isaac_truth_to_fins(
        position_world_isaac=(1.0, 2.0, 3.0),
        orientation_world_body_isaac_xyzw=(0.0, 0.0, 0.0, 1.0),
        linear_velocity_body_isaac=(4.0, 5.0, 6.0),
        angular_velocity_body_isaac=(7.0, 8.0, 9.0),
        linear_acceleration_body_isaac=(10.0, 11.0, 12.0),
    )
    assert np.allclose(truth.position_world, (1.0, 3.0, 2.0))
    assert np.allclose(truth.orientation_world_body_xyzw, (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(truth.linear_velocity_body, (4.0, 6.0, 5.0))
    assert np.allclose(truth.angular_velocity_body, (-7.0, -9.0, -8.0))
    assert np.allclose(truth.linear_acceleration_body, (10.0, 12.0, 11.0))


def test_action_buffer_clamps_rejects_invalid_commands_and_expires() -> None:
    commands = contract.ActionCommandBuffer(command_timeout_sec=0.5)
    assert np.array_equal(commands.command_force_n(0.0), np.zeros(8, dtype=np.float32))
    assert not commands.receive_force_n((1.0,) * 7, 0.0)
    assert commands.receive_force_n((-9.0, -7.0, -1.0, 0.0, 1.0, 7.0, 9.0, np.nan), 0.0) is False
    assert commands.receive_force_n((-9.0, -7.0, -1.0, 0.0, 1.0, 7.0, 9.0, 3.0), 0.0)
    assert np.allclose(commands.command_force_n(0.4), (-7.0, -7.0, -1.0, 0.0, 1.0, 7.0, 7.0, 3.0))
    assert np.array_equal(commands.command_force_n(0.5001), np.zeros(8, dtype=np.float32))
    commands.request_reset()
    assert commands.take_reset_request()
    assert not commands.take_reset_request()
