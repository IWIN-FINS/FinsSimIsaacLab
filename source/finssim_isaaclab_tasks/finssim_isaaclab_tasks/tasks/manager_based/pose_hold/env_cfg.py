"""Isaac Lab Manager-Based pose-hold environments."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from finssim_isaaclab_tasks.assets.finsrov import FINSROV_CFG
from finssim_isaaclab_tasks.assets.warpauv import WARPAUV_CFG
from . import mdp
from .actions import UnderwaterThrusterActionCfg


@configclass
class PoseHoldSceneCfg(InteractiveSceneCfg):
    robot = FINSROV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class ActionsCfg:
    thrusters = UnderwaterThrusterActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=mdp.pose_hold_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    position = RewTerm(func=mdp.position_hold_reward, weight=1.0)
    orientation = RewTerm(func=mdp.orientation_hold_reward, weight=0.5)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds, params={"limit_m": 4.0})


@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)}, "velocity_range": {}},
    )


@configclass
class FinsROVPoseHoldEnvCfg(ManagerBasedRLEnvCfg):
    scene: PoseHoldSceneCfg = PoseHoldSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands = None
    curriculum = None
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    decimation = 2
    episode_length_s = 5.0


@configclass
class WarpAUVPoseHoldEnvCfg(FinsROVPoseHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WARPAUV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.thrusters.vehicle = "warpauv"
