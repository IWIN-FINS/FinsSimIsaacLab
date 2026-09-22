import numpy as np
import torch

from finssim_isaaclab.adapter import IsaacLabToFinsRovAdapter
from finssim_isaaclab.frame import convert_quaternion_xyzw_to_wxyz, transform_body_angular_vector
from finssim_isaaclab.isaac_thruster_model import IsaacWarpAUVThrustModel
from finssim_isaaclab.observation import build_warpauv_observation
from finssim_isaaclab.policy import (
    IsaacLabRslRlActor,
    load_finsrov_hold_for_position_backend,
    load_finsrov_hold_for_position_wrench_backend,
)


def test_isaac_force_curve_and_deadzone() -> None:
    model = IsaacWarpAUVThrustModel()
    force = model.action_to_force([1.0, -1.0, 0.0, 0.079, 0.08, 0.2])
    np.testing.assert_allclose(force[:2], [136.368, -131.464], rtol=1e-5, atol=1e-5)
    assert force[2] == 0.0
    assert force[3] == 0.0
    assert force[4] > 0.0
    assert force[5] > 0.0


def test_auto_scale_maps_envelope_to_fins_reference() -> None:
    adapter = IsaacLabToFinsRovAdapter()
    # The reference is a per-axis envelope, so no single corner necessarily
    # reaches all six maxima at the same time.
    corners = np.asarray(
        [adapter.action_to_fins_wrench([1.0 if (mask >> i) & 1 else -1.0 for i in range(6)]) for mask in range(64)]
    )
    np.testing.assert_allclose(
        np.max(np.abs(corners), axis=0), adapter.fins_reference_wrench, rtol=1e-5, atol=1e-5
    )
    assert np.all(np.isfinite(adapter.wrench_scale))


def test_axis_mapping_applies_axial_sign_to_torque() -> None:
    model = IsaacWarpAUVThrustModel()
    adapter = IsaacLabToFinsRovAdapter(
        thrust_model=model,
        auto_scale_from_envelope=False,
        force_axis_signs=(1.0, 1.0, 1.0),
    )
    virtual = model.action_to_wrench([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    fins = adapter.action_to_fins_wrench([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fins[:3], [virtual[0], virtual[2], virtual[1]], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fins[3:], [-virtual[3], -virtual[5], -virtual[4]], rtol=1e-6, atol=1e-6)


def test_fins_yaw_rate_maps_to_negative_isaac_z_rate() -> None:
    # Fins yaw is +y; Isaac yaw is +z. The bases differ by a reflection, so
    # an axial vector gains the determinant sign: +y_F -> -z_I.
    np.testing.assert_allclose(transform_body_angular_vector([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0])


def test_warpauv_observation_uses_exact_body_frame_axis_mapping() -> None:
    observation = build_warpauv_observation(
        target_position_fins=[1.0, 2.0, 3.0],
        target_quaternion_fins_xyzw=[0.0, 0.0, 0.0, 1.0],
        position_fins=[0.0, 0.0, 0.0],
        orientation_fins_xyzw=[0.0, 0.0, 0.0, 1.0],
        linear_velocity_body_fins=[4.0, 5.0, 6.0],
        angular_velocity_body_fins=[7.0, 8.0, 9.0],
    )
    np.testing.assert_allclose(
        observation,
        [1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 4.0, 6.0, 5.0, -7.0, -9.0, -8.0],
    )


def test_fins_positive_yaw_goal_becomes_negative_isaac_z_rotation() -> None:
    fins_yaw_90_xyzw = [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)]
    quaternion_wxyz = convert_quaternion_xyzw_to_wxyz(fins_yaw_90_xyzw)
    np.testing.assert_allclose(quaternion_wxyz, [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)], atol=1e-6)


def test_finsrov_hold_for_position_loader_reads_lab3_actor_state_dict(tmp_path) -> None:
    actor = IsaacLabRslRlActor(observation_dim=16, action_dim=8)
    actor_state_dict = {
        f"mlp.{key}": value.detach().clone()
        for key, value in actor.actor.state_dict().items()
    }
    actor_state_dict["distribution.std_param"] = torch.zeros(8)
    checkpoint_path = tmp_path / "finsrov.pt"
    torch.save({"actor_state_dict": actor_state_dict, "iter": 123}, checkpoint_path)

    policy = load_finsrov_hold_for_position_backend(checkpoint_path, device="cpu")
    action, _ = policy.predict(np.zeros(16, dtype=np.float32))

    assert policy.iteration == 123
    assert action.shape == (8,)
    assert np.all(np.abs(action) <= 1.0)


def test_finsrov_hold_for_position_wrench_loader_reads_six_action_actor(tmp_path) -> None:
    actor = IsaacLabRslRlActor(observation_dim=16, action_dim=6)
    actor_state_dict = {
        f"mlp.{key}": value.detach().clone()
        for key, value in actor.actor.state_dict().items()
    }
    actor_state_dict["distribution.std_param"] = torch.zeros(6)
    checkpoint_path = tmp_path / "finsrov_wrench.pt"
    torch.save({"actor_state_dict": actor_state_dict, "iter": 456}, checkpoint_path)

    policy = load_finsrov_hold_for_position_wrench_backend(checkpoint_path, device="cpu")
    action, _ = policy.predict(np.zeros(16, dtype=np.float32))

    assert policy.iteration == 456
    assert action.shape == (6,)
    assert np.all(np.abs(action) <= 1.0)
