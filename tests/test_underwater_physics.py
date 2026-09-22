from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import SceneEntityCfg
from pxr import Usd, UsdGeom, UsdPhysics
from finssim_isaaclab_tasks.physics.actuators import ThrusterModelCfg, ThrusterSystem
from finssim_isaaclab_tasks.physics.core import HydrodynamicsProfileCfg, HydrodynamicsState, UnderwaterHydrodynamics
from finssim_isaaclab_tasks.physics.water import WaterKinematics, WaterKinematicsCfg
from finssim_isaaclab_tasks.physics.mesh import BatchedSurfaceGeometryHydro, SurfaceMeshData
from finssim_isaaclab_tasks.physics.mesh_warp import WarpSurfaceGeometryHydro
from finssim_isaaclab_tasks.sensors import SonarSensorCfg, sonar_intensity
from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.contract import (
    OBSERVATION_FIELDS,
    THRUSTER_DIRECTIONS_B,
    THRUSTER_NAMES,
    THRUSTER_POSITIONS_FROM_COM_B,
    build_hold_for_position_observation,
    isaac_to_unity_angular,
    isaac_to_unity_polar,
    normalized_action_to_calibrated_force_request,
    yaw_error_magnitude_xyzw,
)
from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position.dynamics import (
    EquivalentBoxHydrodynamics,
    EquivalentBoxHydrodynamicsCfg,
)
from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position_wrench.allocator import (
    FinsROVPhysicalWrenchAllocator,
    FinsROVPhysicalWrenchAllocatorCfg,
    POLICY_WRENCH_LIMITS,
)
from finssim_isaaclab_tasks.tasks.direct.finsrov_hold_for_position_wrench.env_cfg import (
    FinsROVHoldForPositionWrenchEnvCfg,
)
from finssim_isaaclab_tasks.tasks.manager_based.pose_hold import mdp


def _state(count=2):
    return HydrodynamicsState(
        quat_w_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(count, 1),
        linear_velocity_b=torch.zeros(count, 3), angular_velocity_b=torch.zeros(count, 3),
        water_linear_velocity_b=torch.zeros(count, 3), water_angular_velocity_b=torch.zeros(count, 3),
    )


def test_hydrostatic_and_off_modes():
    state = _state()
    hydro = UnderwaterHydrodynamics(HydrodynamicsProfileCfg(mode="hydrostatic_only", displaced_volume=1.0, water_density=1000.0), 2, "cpu")
    assert torch.allclose(hydro.compute(state, 0.01).wrench_b[:, 2], torch.full((2,), 9810.0))
    off = UnderwaterHydrodynamics(HydrodynamicsProfileCfg(mode="off", displaced_volume=1.0), 2, "cpu")
    assert torch.count_nonzero(off.compute(state, 0.01).wrench_b) == 0


def test_simplified_and_fossen_reset_state():
    state = _state(1)
    state.linear_velocity_b[:] = torch.tensor([[2.0, 0.0, 0.0]])
    cfg = HydrodynamicsProfileCfg(mode="fossen", linear_damping=(2.0, 0, 0, 0, 0, 0), added_mass_diagonal=(1.0, 0, 0, 0, 0, 0))
    hydro = UnderwaterHydrodynamics(cfg, 1, "cpu")
    assert hydro.compute(state, 0.1).damping_b[0, 0] == -4.0
    hydro.reset()
    assert hydro.compute(state, 0.1).added_mass_acceleration_b[0, 0] == 0.0


def test_thruster_and_water_batching():
    thrusters = ThrusterSystem(ThrusterModelCfg(positions_b=((0, 1, 0),), directions_b=((1, 0, 0),), max_forward_force_n=10.0, max_reverse_force_n=10.0), 4, "cpu")
    force, torque, applied = thrusters.compute(torch.ones(4, 1), 0.01)
    assert torch.allclose(force[:, 0], torch.full((4,), 10.0))
    assert torch.allclose(torque[:, 2], torch.full((4,), -10.0))
    water = WaterKinematics(WaterKinematicsCfg(current_velocity_w=(1.0, 2.0, 3.0)), 4, "cpu")
    _, current = water.sample(torch.zeros(4, 3), 0.01)
    assert torch.allclose(current[0], torch.tensor([1.0, 2.0, 3.0]))


def test_mesh_backend_is_batched_and_clips_waterline():
    # Closed tetrahedron with one face crossing z=0.
    vertices = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-0.5, 0.8, 1.0], [-0.5, -0.8, 1.0]])
    triangles = torch.tensor([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]])
    mesh = SurfaceMeshData.preprocess(vertices, triangles)
    assert mesh.closed
    backend = BatchedSurfaceGeometryHydro(mesh, HydrodynamicsProfileCfg(mode="mesh"), "cpu")
    result = backend.compute(
        root_position_w=torch.zeros(3, 3),
        root_quat_w_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1),
        linear_velocity_b=torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1),
        angular_velocity_b=torch.zeros(3, 3),
        water_height_w=torch.zeros(3),
        water_velocity_b=torch.zeros(3, 3),
    )
    assert result.wrench_b.shape == (3, 6)
    assert torch.all(result.wetted_area > 0)
    assert torch.all(result.submerged_volume > 0)
    assert torch.all(result.submerged_volume < backend.full_volume)
    warp_backend = WarpSurfaceGeometryHydro(mesh, HydrodynamicsProfileCfg(mode="mesh"), "cpu")
    warp_result = warp_backend.compute(
        torch.zeros(3, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1),
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1), torch.zeros(3, 3), torch.zeros(3), torch.zeros(3, 3),
    )
    assert warp_result.wrench_b.shape == (3, 6)
    assert torch.all(warp_result.wetted_area > 0)


def test_mesh_mode_requires_mesh_wrench():
    hydro = UnderwaterHydrodynamics(HydrodynamicsProfileCfg(mode="mesh"), 2, "cpu")
    try:
        hydro.compute(_state(), 0.01)
    except ValueError as error:
        assert "requires mesh_wrench_b" in str(error)
    else:
        raise AssertionError("mesh mode accepted an absent geometry wrench")


def test_sonar_intensity_is_bounded_and_masks_invalid_returns():
    cfg = SonarSensorCfg(max_range_m=10.0)
    intensity = sonar_intensity(torch.tensor([0.0, 5.0, 10.0]), torch.tensor([True, True, False]), cfg)
    assert torch.allclose(intensity, torch.tensor([1.0, 0.5, 0.0]))


def test_pose_hold_uses_default_depth_as_target():
    root_position = torch.tensor([[10.25, -1.0, 8.25], [-3.0, 4.0, 7.75]])
    env_origins = torch.tensor([[10.0, -1.0, 0.0], [-3.0, 4.0, 0.0]])
    default_pose = torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    proxy = lambda value: SimpleNamespace(torch=value)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=proxy(root_position),
            default_root_pose=proxy(default_pose),
            root_quat_w=proxy(default_pose[:, 3:]),
            root_lin_vel_b=proxy(torch.zeros(2, 3)),
            root_ang_vel_b=proxy(torch.zeros(2, 3)),
        )
    )
    class Scene:
        def __init__(self):
            self.env_origins = env_origins

        def __getitem__(self, key):
            assert key == "robot"
            return robot

    env = SimpleNamespace(scene=Scene())
    asset_cfg = SceneEntityCfg("robot")
    observation = mdp.pose_hold_observation(env, asset_cfg)
    assert torch.allclose(observation[:, :3], torch.tensor([[0.25, 0.0, 0.25], [0.0, 0.0, -0.25]]))
    assert not mdp.out_of_bounds(env, 4.0, asset_cfg).any()


def test_finsrov_hold_for_position_unity_16d_contract():
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    observation = build_hold_for_position_observation(
        body_goal_offset_i=torch.tensor([[1.0, 2.0, 3.0]]),
        root_quat_w_i=root_quat,
        goal_quat_w_i=root_quat,
        linear_velocity_b_i=torch.tensor([[3.0, 2.0, 1.0]]),
        angular_velocity_b_i=torch.tensor([[3.0, 2.0, 1.0]]),
        position_scale=3.0,
        linear_velocity_scale=1.0,
        angular_velocity_scale=1.0,
        velocity_clip=2.0,
    )
    assert len(OBSERVATION_FIELDS) == 16
    assert observation.shape == (1, 16)
    assert torch.allclose(observation[0, :3], torch.tensor([1.0 / 3.0, 1.0, 2.0 / 3.0]))
    assert torch.allclose(observation[0, 3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(observation[0, 9:12], torch.tensor([2.0, 1.0, 2.0]))
    assert torch.allclose(observation[0, 12:15], torch.tensor([-2.0, -1.0, -2.0]))
    assert torch.allclose(observation[0, 15:], torch.tensor([1.0]))
    assert THRUSTER_NAMES == (
        "Vertical1=V_LF",
        "Vertical2=V_LB",
        "Vertical3=V_RB",
        "Vertical4=V_RF",
        "Horizontal1=H_LF",
        "Horizontal2=H_LB",
        "Horizontal3=H_RB",
        "Horizontal4=H_RF",
    )
    requests = normalized_action_to_calibrated_force_request(
        torch.tensor([[-2.0, -0.5, 0.0, 0.25, 2.0]]),
        max_forward_force_n=(2.0, 4.0, 6.0, 8.0, 10.0),
        max_reverse_force_n=(1.0, 3.0, 5.0, 7.0, 9.0),
    )
    assert torch.allclose(requests, torch.tensor([[-1.0, -1.5, 0.0, 2.0, 10.0]]))


def test_finsrov_yaw_error_ignores_roll_and_pitch():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    yaw_quarter_turn = torch.tensor([[0.0, 0.0, 2.0**-0.5, 2.0**-0.5]])
    roll_quarter_turn = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    assert torch.allclose(yaw_error_magnitude_xyzw(yaw_quarter_turn, identity), torch.tensor([torch.pi / 2]))
    assert torch.allclose(yaw_error_magnitude_xyzw(identity, roll_quarter_turn), torch.zeros(1), atol=1.0e-6)


def test_finsrov_thruster_wrench_uses_the_correct_reflected_frame():
    """A transformed Unity r-cross-F wrench must agree with Isaac's wrench."""

    positions_i = torch.tensor(THRUSTER_POSITIONS_FROM_COM_B)
    directions_i = torch.tensor(THRUSTER_DIRECTIONS_B)
    force_n = torch.tensor([1.2, -0.7, 2.4, -1.1, 0.9, -1.8, 0.3, 2.1])
    forces_i = directions_i * force_n.unsqueeze(-1)
    torque_i = torch.cross(positions_i, forces_i, dim=-1).sum(dim=0)

    positions_u = isaac_to_unity_polar(positions_i)
    forces_u = isaac_to_unity_polar(forces_i)
    torque_u = torch.cross(positions_u, forces_u, dim=-1).sum(dim=0)

    assert torch.allclose(torque_u, isaac_to_unity_angular(torque_i), atol=1.0e-6)
    assert torch.allclose(isaac_to_unity_polar(directions_i[4]), torch.tensor([1.0, 0.0, -1.0]))


def test_finsrov_task_frame_asset_has_one_physical_frame_and_unity_com():
    """The visual USD must not silently introduce an axis, unit, or CoM mismatch."""

    assets_dir = (
        Path(__file__).resolve().parents[1]
        / "source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/finsrov/data"
    )
    task_stage = Usd.Stage.Open(str(assets_dir / "finsrov_task_frame.usda"))
    raw_stage = Usd.Stage.Open(str(assets_dir / "finsrov.usd"))
    assert task_stage is not None and raw_stage is not None
    assert UsdGeom.GetStageUpAxis(task_stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(task_stage) == 1.0

    root = task_stage.GetDefaultPrim()
    mass_api = UsdPhysics.MassAPI(root)
    assert tuple(mass_api.GetCenterOfMassAttr().Get()) == pytest.approx((0.0, 0.0, -0.08), abs=1.0e-7)
    assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        (0.13154832, 0.23185866, 0.24651341), abs=1.0e-7
    )
    rigid_bodies = [prim.GetPath() for prim in task_stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert rigid_bodies == [root.GetPath()]

    # The asset import's raw [right, up, back] centimetre frame is mapped to
    # task [forward, left, up] metres: task = [-raw_z, -raw_x, raw_y] * .01.
    raw_thruster = raw_stage.GetPrimAtPath("/World/Vertical1")
    task_thruster = task_stage.GetPrimAtPath("/FinsROV/visual/raw/Vertical1")
    raw_position = UsdGeom.XformCache().GetLocalToWorldTransform(raw_thruster).ExtractTranslation()
    task_position = UsdGeom.XformCache().GetLocalToWorldTransform(task_thruster).ExtractTranslation()
    expected_task_position = (-raw_position[2] * 0.01, -raw_position[0] * 0.01, raw_position[1] * 0.01)
    assert tuple(task_position) == pytest.approx(expected_task_position, abs=1.0e-7)

    # The eight explicit action wrenches are referenced to the same physical
    # CoM as the visual model.  Check a vertical thruster whose three offsets
    # exercise all three axes.
    com_b = mass_api.GetCenterOfMassAttr().Get()
    visual_position_from_com = tuple(task_position[index] - com_b[index] for index in range(3))
    assert visual_position_from_com == pytest.approx(THRUSTER_POSITIONS_FROM_COM_B[0], abs=1.0e-7)


def test_finsrov_equivalent_box_hydrodynamics_matches_learning_to_swim_formula():
    hydro = EquivalentBoxHydrodynamics(
        EquivalentBoxHydrodynamicsCfg(
            water_density=1000.0,
            water_dynamic_viscosity=0.001,
            gravity_magnitude=9.81,
            mass=12.11,
            inertia_b=(0.13154832, 0.23185866, 0.24651341),
            displaced_volume=0.01,
            com_to_cob_b=(0.0, 0.0, 0.1),
            com_to_cob_randomization_radius=0.0,
            volume_range=(0.01, 0.01),
            enable_domain_randomization=False,
        ),
        1,
        "cpu",
    )
    result = hydro.compute(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros(1, 3),
    )
    inertia_sum_without_axis = torch.tensor(
        [
            0.23185866 + 0.24651341 - 0.13154832,
            0.24651341 + 0.13154832 - 0.23185866,
            0.13154832 + 0.23185866 - 0.24651341,
        ]
    )
    half_dimensions = torch.sqrt(3.0 / (2.0 * 12.11) * inertia_sum_without_axis)
    expected_drag_x = -2.0 * 1000.0 * half_dimensions[1] * half_dimensions[2]
    assert torch.allclose(result.buoyancy_b[0, :3], torch.tensor([0.0, 0.0, 98.1]))
    assert torch.allclose(result.buoyancy_b[0, 3:], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(result.quadratic_drag_b[0, 0], expected_drag_x)


def test_finsrov_wrench_allocator_reconstructs_feasible_pure_axes():
    allocator = FinsROVPhysicalWrenchAllocator(
        FinsROVPhysicalWrenchAllocatorCfg(enable_roll_pitch_moments=True), "cpu"
    )
    policy_actions = 0.5 * torch.eye(6)
    normalized_thruster_actions = allocator.allocate(policy_actions)
    force_n = normalized_action_to_calibrated_force_request(
        normalized_thruster_actions,
        max_forward_force_n=(7.0,) * 8,
        max_reverse_force_n=(7.0,) * 8,
    )
    reconstructed_wrench = allocator.forces_to_policy_wrench(force_n)
    expected_wrench = policy_actions * torch.tensor(POLICY_WRENCH_LIMITS)
    assert torch.all(torch.abs(normalized_thruster_actions) <= 1.0 + 1.0e-6)
    assert torch.all(torch.abs(force_n) <= 7.0 + 1.0e-6)
    assert torch.allclose(reconstructed_wrench, expected_wrench, atol=2.0e-4, rtol=2.0e-5)


def test_finsrov_wrench_allocator_can_mask_roll_and_pitch_requests():
    allocator = FinsROVPhysicalWrenchAllocator(
        FinsROVPhysicalWrenchAllocatorCfg(enable_roll_pitch_moments=False), "cpu"
    )
    actions = torch.tensor([[0.2, -0.3, 0.1, 0.8, -0.7, 0.25]])
    masked_actions = actions.clone()
    masked_actions[:, 3:5] = 0.0

    assert torch.allclose(allocator.allocate(actions), allocator.allocate(masked_actions), atol=1.0e-6)


def test_finsrov_wrench_task_uses_sim7n_limits_and_masks_roll_pitch_by_default():
    cfg = FinsROVHoldForPositionWrenchEnvCfg()

    assert cfg.max_forward_force_n == (7.0,) * 8
    assert cfg.max_reverse_force_n == (7.0,) * 8
    assert cfg.enable_roll_pitch_moments is False
