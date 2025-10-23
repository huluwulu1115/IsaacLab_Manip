import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform, quat_mul, quat_conjugate
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.utils.spaces import spec_to_gym_space, sample_space
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors import Camera, CameraCfg


ASSET_PATH = Path(__file__).parent / "../../../../../assets"

@configclass
class EventCfg:
    """Configuration for randomization."""
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )


@configclass
class BinaryJointPositionActionCfg:
    """Configuration for binary joint position control."""
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    open_command_expr: dict[str, float] = MISSING
    close_command_expr: dict[str, float] = MISSING

    def __init__(self, asset_name: str, joint_names: list[str], 
                 open_command_expr: dict[str, float], close_command_expr: dict[str, float]):
        self.asset_name = asset_name
        self.joint_names = joint_names
        self.open_command_expr = open_command_expr
        self.close_command_expr = close_command_expr


@configclass
class JointPositionActionCfg:
    """Configuration for joint position control."""
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float = MISSING
    use_default_offset: bool = True

    def __init__(self, asset_name: str, joint_names: list[str], 
                 scale: float, use_default_offset: bool = True):
        self.asset_name = asset_name
        self.joint_names = joint_names
        self.scale = scale
        self.use_default_offset = use_default_offset

@configclass
class FrankaDroidEnvCfg(DirectRLEnvCfg):
    # Match manager-based environment exactly
    episode_length_s = 5.0
    decimation = 2
    action_space = 8
    observation_space = 0
    state_space = 0
    
    # Sim2Real Configuration
    use_vision_observations = False  # True for student (vision), False for teacher (state)

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_PATH / "franka_robotiq_2f_85_flattened.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175, 
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61, 
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=20.0,
                velocity_limit_sim=1.0, 
                stiffness=200.0,   
                damping=40.0,      
            ),
        },
    )

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi/4},
    )

    objects = [
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object_0",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0, 0.055],
                rot=[1, 0, 0, 0]
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        ),
    ]
    
    object = objects[0]
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Wrist camera with RGBD for sim2real transfer
    wrist_cam = CameraCfg(
        prim_path="/World/envs/env_.*/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=224,  # Reduced from 720 for better performance (still sufficient for manipulation)
        width=224,   # Reduced from 1280 for better performance
        data_types=["rgb", "distance_to_camera"],  # RGBD: RGB + Depth
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=3.024,  # Adjusted to match 1:1 aspect ratio (224x224)
            vertical_aperture=3.024,
            clipping_range=(0.01, 10.0),  # Important for depth accuracy
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), 
            rot=(-0.420, 0.570, 0.576, -0.409), 
            convention="opengl"
        ),
    )
    
    events: EventCfg = EventCfg()

    dof_velocity_scale = 1.0
    large_reach_threshold = 3.0


class FrankaDroidEnv(DirectRLEnv):
    """
    DROID grasp and lift environment - Eureka-compatible.
    
    A manipulation environment featuring a Franka Panda robot with Robotiq 2F-85 gripper.
    The robot must grasp a cube and lift it to randomly sampled goal positions.
    
    Key Features:
    - 8-DOF action space (7 arm joints + 1 gripper)
    - Object position randomization on reset
    - Dynamic goal resampling every 5 seconds
    - Curriculum-based action/velocity penalties
    - Visual markers for end-effector pose debugging
    
    Observation Space (41-dim):
    - Joint positions (8) and velocities (8)
    - Object pose in robot frame (7: position + quaternion)
    - End-effector pose in robot frame (7: position + quaternion)
    - Goal position in robot frame (3)
    - Previous actions (8)
    
    Oracle Reward Components:
    - Reaching: Encourages end-effector to approach object
    - Lifting: Rewards raising object above minimal height
    - Goal tracking: Guides lifted object to goal position
    - Regularization: Action smoothness and velocity penalties
    
    Eureka Integration:
    This environment is compatible with Eureka for automatic reward function design.
    Key attributes available for custom reward functions (see _get_rewards docstring):
    - self._robot.data.ee_pos_w: End-effector position
    - self._robot.data.obj_pos_w: Object position
    - self._robot.data.dist_ee_obj: EE-to-object distance
    - self._robot.data.dist_obj_goal: Object-to-goal distance
    - self.goal_pos_w: Goal positions
    - self.minimal_height: Lift height threshold (0.04m)
    - self.actions, self.prev_actions, self.action_delta: Action tracking
    """

    cfg: FrankaDroidEnvCfg
    
    def __init__(self, cfg: FrankaDroidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Configure observation and action spaces
        self._compute_observation_space_size()
        self._reconfigure_gym_spaces()
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # ============================================================================
        # Robot Configuration
        # ============================================================================
        # Joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_range = self.robot_dof_upper_limits - self.robot_dof_lower_limits
        self.robot_dof_range = torch.where(
             self.robot_dof_range < 1e-6, torch.ones_like(self.robot_dof_range), self.robot_dof_range
        )
        
        # Joint control targets
        self.robot_dof_targets = self._robot.data.default_joint_pos.clone()

        # Find joint indices for arm and gripper
        self.arm_joint_indices = [
            self._robot.find_joints(f"panda_joint{i}")[0][0] 
            for i in range(1, 8)
        ]
        self.gripper_joint_index = self._robot.find_joints("finger_joint")[0][0]

        # ============================================================================
        # Environment State Buffers
        # ============================================================================
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.action_delta = torch.zeros_like(self.prev_actions)
        self.prev_gripper_joint_pos = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Curriculum learning counter
        self.step_count = 0

        # Intermediate values for reward computation (accessible by Eureka)
        self._robot.data.ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._robot.data.obj_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._robot.data.dist_ee_obj = torch.zeros(self.num_envs, device=self.device)
        self._robot.data.dist_obj_goal = torch.zeros(self.num_envs, device=self.device)

        # ============================================================================
        # End-Effector Configuration
        # ============================================================================
        # Offset from panda_link8 to gripper tip in link frame
        self.ee_offset = torch.tensor([0.0, 0.0, 0.16], device=self.device)
        self.gripper_action_history = torch.zeros(self.num_envs, 20, device=self.device)
        self.minimal_height = 0.04
        
        # Get body indices for end-effector computation
        panda_link8_bodies = self._robot.find_bodies("panda_link8")
        self.hand_body_idx = panda_link8_bodies[0][0]
        
        # Sample initial goals
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goal(all_env_ids)

        # ============================================================================
        # Visualization Markers for End-Effector and Goal
        # ============================================================================
        # Frame marker showing robot base orientation (above Franka)
        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_frame_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/ee_frame"))
        
        # Sphere marker at EE position
        ee_dot_cfg = VisualizationMarkersCfg(
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
            prim_path="/Visuals/ee_dot"
        )
        self.ee_dot_marker = VisualizationMarkers(ee_dot_cfg)
        
        # Goal position marker (coordinate frame showing target position)
        goal_frame_cfg = FRAME_MARKER_CFG.copy()
        goal_frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)  # Slightly smaller than EE frame
        self.goal_marker = VisualizationMarkers(goal_frame_cfg.replace(prim_path="/Visuals/goal_frame"))


    def _sample_goal(self, env_ids: torch.Tensor):
        """Sample goal positions: x∈[0.4,0.6], y∈[-0.25,0.25], z∈[0.25,0.5]."""
        # Sample goals in local coordinates relative to each environment's origin
        n_envs = len(env_ids)
        local_goals = torch.zeros(n_envs, 3, device=self.device)
        local_goals[:, 0] = sample_uniform(0.4, 0.6, (n_envs,), device=self.device)
        local_goals[:, 1] = sample_uniform(-0.25, 0.25, (n_envs,), device=self.device)
        local_goals[:, 2] = sample_uniform(0.25, 0.5, (n_envs,), device=self.device)
        
        # Transform to world frame by adding environment origins
        self.goal_pos_w[env_ids] = local_goals + self.scene.env_origins[env_ids]

    def _configure_gym_env_spaces(self):
        """Override to use placeholder spaces initially."""
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.state_space = None
        if self.cfg.state_space:
            self.single_observation_space["critic"] = spec_to_gym_space(self.cfg.state_space)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

    def _reconfigure_gym_spaces(self):
        """Reconfigure gym spaces with computed observation space size."""
        self.cfg.observation_space = self.computed_obs_size
        
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.state_space = None
        if self.cfg.state_space:
            self.single_observation_space["critic"] = spec_to_gym_space(self.cfg.state_space)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

    def _compute_observation_space_size(self):
        """Compute observation space size based on observation mode."""
        if self.cfg.use_vision_observations:
            # Student mode: proprioception only (for use with RGBD camera images)
            # 8 joint pos + 8 joint vel + 8 actions = 24
            self.computed_obs_size = 8 + 8 + self.cfg.action_space
            print(f"[STUDENT MODE - Vision] Observation space: {self.computed_obs_size}D proprioception")
            print(f"                        Camera: {self.cfg.wrist_cam.height}x{self.cfg.wrist_cam.width} RGBD")
        else:
            # Teacher mode: full privileged state for faster training
            # 8 joint pos + 8 joint vel + 7 obj pose + 7 ee pose + 3 goal + 8 actions = 41
            self.computed_obs_size = 8 + 8 + 3 + 4 + 3 + 4 + 3 + self.cfg.action_space
            print(f"[TEACHER MODE - State] Observation space: {self.computed_obs_size}D privileged state")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        
        # Only create camera if in vision mode
        if self.cfg.use_vision_observations:
            self._wrist_cam = Camera(self.cfg.wrist_cam)
            self.scene.sensors["wrist_cam"] = self._wrist_cam
        
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_0/Table",
            table_cfg,
            translation=(0.525, 0.0, 0.0),
            orientation=(0.707, 0.0, 0.0, 0.707),  # Match manager-based rotation
        )
        
        # Multi-object setup - check if objects are RigidObject or Articulation
        if hasattr(self.cfg, 'objects') and len(self.cfg.objects) > 0:
            for i, obj_cfg in enumerate(self.cfg.objects):
                # Determine if it's an ArticulationCfg or RigidObjectCfg
                if isinstance(obj_cfg, ArticulationCfg):
                    obj_instance = Articulation(obj_cfg)
                    self.scene.articulations[f"object_{i}"] = obj_instance
                    if i == 0:
                        self._object = obj_instance
                        self.scene.articulations["object"] = obj_instance
                else:  # RigidObjectCfg
                    obj_instance = RigidObject(obj_cfg)
                    self.scene.rigid_objects[f"object_{i}"] = obj_instance
                    if i == 0:
                        self._object = obj_instance
                        self.scene.rigid_objects["object"] = obj_instance
            self.num_objects = len(self.cfg.objects)
            self._objects = []
            for i in range(self.num_objects):
                # Try to get from articulations first, then rigid_objects
                if f"object_{i}" in self.scene.articulations:
                    self._objects.append(self.scene.articulations[f"object_{i}"])
                else:
                    self._objects.append(self.scene.rigid_objects[f"object_{i}"])
        else:
            # Fallback to single object (backward compatibility)
            if isinstance(self.cfg.object, ArticulationCfg):
                self._object = Articulation(self.cfg.object)
                self.scene.articulations["object"] = self._object
            else:
                self._object = RigidObject(self.cfg.object)
                self.scene.rigid_objects["object"] = self._object
            self.num_objects = 1
            self._objects = [self._object]
        
        # Register robot with scene BEFORE cloning
        self.scene.articulations["robot"] = self._robot
        

        # Spawn ground plane at manager-based height (-1.05 m)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func(
            "/World/GroundPlane", ground_cfg, translation=(0.0, 0.0, -1.05)
        )

        # Clone environments - this must happen AFTER all entities are registered
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Initialize camera AFTER cloning (it needs the cloned prims to exist)
        if self.cfg.use_vision_observations and hasattr(self, '_wrist_cam'):
            # The scene will handle initialization during update
            pass

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before applying physics step."""
        # Clip actions to match RL Games config (clip_actions: 100.0)
        self.actions = actions.clone().clamp(-100.0, 100.0)

        # Squash gripper action for smoother control
        self.actions[:, 7] = 0.04 * torch.tanh(self.actions[:, 7])
        
        # Compute action rate for penalty calculation
        self.action_delta = self.actions - self.prev_actions
        self.prev_actions[:] = self.actions

        # Update gripper action history (rolling window)
        self.gripper_action_history = torch.roll(self.gripper_action_history, shifts=-1, dims=1)
        self.gripper_action_history[:, -1] = self.actions[:, 7]

        # Resample goals every 5 seconds
        self.goal_timer += self.dt
        envs_to_resample = torch.nonzero(self.goal_timer >= 5.0, as_tuple=False).flatten()
        if envs_to_resample.numel() > 0:
            self._sample_goal(envs_to_resample)
            self.goal_timer[envs_to_resample] -= 5.0

    def _apply_action(self):
        """Apply actions to robot joints with position control."""
        # Ensure default joint positions have batch dimension
        default_joint_pos = self._robot.data.default_joint_pos
        if default_joint_pos.dim() == 1:
            default_joint_pos = default_joint_pos.unsqueeze(0).expand(self.num_envs, -1)
        elif default_joint_pos.shape[0] == 1:
            default_joint_pos = default_joint_pos.expand(self.num_envs, -1)

        # Start with default positions
        targets = default_joint_pos.clone()
        
        # Arm control: Apply scaled actions to arm joints (scale=0.5)
        for i, joint_idx in enumerate(self.arm_joint_indices):
            targets[:, joint_idx] = default_joint_pos[:, joint_idx] + self.cfg.arm_action.scale * self.actions[:, i]

        # Gripper control: Binary open/close based on action sign (negative = close)
        is_closing = self.actions[:, 7] < 0.0
        close_pos = self.cfg.gripper_action.close_command_expr["finger_joint"]
        open_pos = self.cfg.gripper_action.open_command_expr["finger_joint"]
        targets[:, self.gripper_joint_index] = torch.where(
            is_closing, 
            torch.full((self.num_envs,), close_pos, device=self.device),
            torch.full((self.num_envs,), open_pos, device=self.device)
        )

        # Clamp to joint limits and apply
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation conditions."""
        self._compute_intermediate_values()
        
        # Terminate if object drops below minimum height
        obj_pos = self._objects[0].data.root_pos_w
        terminated = self.terminated_buf.clone() | (obj_pos[:, 2] < -0.05)
        
        # Truncate on episode timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        self.terminated_buf[:] = False
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """
        Oracle reward function for Eureka.
        
        This function will be automatically renamed to _get_rewards_oracle() by Eureka,
        and compared against LLM-generated reward functions in _get_rewards_eureka().
        
        Available data for Eureka reward functions:
        - self._robot.data.ee_pos_w: End-effector position in world frame (N, 3)
        - self._robot.data.obj_pos_w: Object position in world frame (N, 3)
        - self._robot.data.dist_ee_obj: Distance between EE and object (N,)
        - self._robot.data.dist_obj_goal: Distance between object and goal (N,)
        - self.goal_pos_w: Goal position in world frame (N, 3)
        - self._robot.data.joint_pos: Robot joint positions (N, num_joints)
        - self._robot.data.joint_vel: Robot joint velocities (N, num_joints)
        - self._objects[0].data.root_pos_w: Object position (N, 3)
        - self._objects[0].data.root_quat_w: Object orientation (N, 4)
        - self.actions: Current actions (N, 8)
        - self.prev_actions: Previous actions (N, 8)
        - self.action_delta: Action rate of change (N, 8)
        - self.minimal_height: Minimum lift height threshold (float)
        - self.step_count: Global curriculum step counter (int)
        """
        self._compute_intermediate_values()  # Update intermediate values
        
        # Get hand position and orientation
        hand_pos = self._robot.data.body_pos_w[:, self.hand_body_idx]
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx]
        offset = self.ee_offset.unsqueeze(0).repeat(hand_pos.shape[0], 1)
        
        self._robot.data.ee_pos_w = hand_pos + quat_apply(hand_quat, offset)
        self._robot.data.obj_pos_w = self._objects[0].data.root_pos_w
        
        # Visualize EE orientation frame above Franka base (for better visibility)
        robot_base_pos, _ = self._squeeze_robot_frame(self._robot.data.root_pos_w, self._robot.data.root_quat_w)
        frame_pos = robot_base_pos.clone()
        frame_pos[:, 2] += 1.0  # Position 1.0m above robot base
        self.ee_frame_marker.visualize(frame_pos, hand_quat)
        
        # Visualize EE position
        self.ee_dot_marker.visualize(self._robot.data.ee_pos_w)
        
        # Visualize goal position (coordinate frame showing target location)
        # Use identity quaternion (no rotation) for the goal frame
        goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        goal_quat[:, 0] = 1.0  # w=1, x=y=z=0 (identity quaternion)
        self.goal_marker.visualize(self.goal_pos_w, goal_quat)

        # ============================================================================
        # Reward Components
        # ============================================================================
        
        # 1. Reaching: Encourage EE to approach object (weight=1.0, std=0.1)
        self._robot.data.dist_ee_obj = torch.norm(self._robot.data.ee_pos_w - self._robot.data.obj_pos_w, dim=-1)
        reaching_reward = (1.0 - torch.tanh(self._robot.data.dist_ee_obj / 0.1)) * 1.0
        
        # 2. Lifting: Reward for raising object above minimal height (weight=15.0)
        lifting_reward = (self._robot.data.obj_pos_w[:, 2] > self.minimal_height).float() * 15.0

        # 3. Goal tracking: Guide lifted object to goal position (weight=16.0, std=0.3)
        self._robot.data.dist_obj_goal = torch.norm(self._robot.data.obj_pos_w - self.goal_pos_w, dim=-1)
        lifted_mask = (self._robot.data.obj_pos_w[:, 2] > self.minimal_height).float()
        goal_tracking = lifted_mask * (1.0 - torch.tanh(self._robot.data.dist_obj_goal / 0.3)) * 16.0

        # 4. Fine-grained goal tracking: Precise positioning near goal (weight=5.0, std=0.05)
        goal_tracking_fine = lifted_mask * (1.0 - torch.tanh(self._robot.data.dist_obj_goal / 0.05)) * 5.0

        # 5. Regularization: Action smoothness and velocity penalties (curriculum-based)
        # Penalty weight fades from -1e-4 to -1e-7 over 10k steps
        curr_factor = min(self.step_count / 10000.0, 1.0)
        penalty_w = (-1e-3) * (1.0 - curr_factor) + (-1e-6) * curr_factor
        action_penalty = torch.sum(self.action_delta ** 2, dim=-1) * penalty_w
        joint_vel_penalty = torch.sum(self._robot.data.joint_vel ** 2, dim=-1) * penalty_w

        # Total reward computation
        total_reward = (
            reaching_reward +
            lifting_reward +
            goal_tracking +
            goal_tracking_fine +
            action_penalty +
            joint_vel_penalty
        )

        # Safety check: replace any NaN or Inf values with zeros
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Log reward components for monitoring
        if hasattr(self, "extras"):
            self.extras.setdefault("log", {}).update({
                "Episode/Episode_Reward/reaching_reward":     reaching_reward.mean(),
                "Episode/Episode_Reward/lifting_reward":      lifting_reward.mean(),
                "Episode/Episode_Reward/goal_tracking":       goal_tracking.mean(),
                "Episode/Episode_Reward/goal_tracking_fine":  goal_tracking_fine.mean(),
                "Episode/Episode_Reward/action_penalty":      action_penalty.mean(),
                "Episode/Episode_Reward/joint_vel_penalty":   joint_vel_penalty.mean(),
                "Episode/Episode_Reward/total_reward":        total_reward.mean(),
            })

        return total_reward

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Compute useful intermediate values derived from simulation state."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Ensure env_ids is a tensor
        if not isinstance(env_ids, torch.Tensor):
            return

        # Get hand position and orientation
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_body_idx]
        hand_quat = self._robot.data.body_quat_w[env_ids, self.hand_body_idx]
        offset = self.ee_offset.unsqueeze(0).repeat(env_ids.shape[0], 1)
        
        # Compute end-effector position
        ee_pos = hand_pos + quat_apply(hand_quat, offset)
        
        # Get object position
        obj_pos = self._objects[0].data.root_pos_w[env_ids]
        
        # Compute distances
        reach_dist = torch.norm(ee_pos - obj_pos, p=2, dim=-1)
        
        # Check for bad states
        large_mask = reach_dist > self.cfg.large_reach_threshold
        nan_mask = torch.isnan(obj_pos).any(dim=1) | torch.isnan(ee_pos).any(dim=1)
        
        # Log bad state ratio if computing for all environments
        if env_ids.shape[0] == self.num_envs:
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"]["state/large_reach_ratio"] = large_mask.float().mean()
        
        # Combine bad masks
        bad_mask = large_mask | nan_mask
        
        # Reset bad environments
        if bad_mask.any():
            bad_ids = env_ids[bad_mask]
            self.terminated_buf[bad_ids] = True
            self._reset_idx(bad_ids, call_compute_iv=True)

    def get_success_metric(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute success metric as combination of reaching and lifting.
        
        Returns:
            Success metric in [0, 1] where:
            - 0.0-0.5: Reaching component (proximity to object)
            - 0.0-0.5: Lifting component (object height)
        """
        hand_pos = self._robot.data.body_pos_w[:, self.hand_body_idx][env_ids]
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx][env_ids]
        obj_pos = self._objects[0].data.root_pos_w[env_ids]
        offset = self.ee_offset.unsqueeze(0).repeat(env_ids.shape[0], 1)
        ee_pos = hand_pos + quat_apply(hand_quat, offset)

        reaching_metric = (1 - torch.tanh(torch.norm(ee_pos - obj_pos, dim=-1))) * 0.5
        lifting_metric = torch.clamp(obj_pos[:, 2] / 0.2, 0, 1.0) * 0.5

        return reaching_metric + lifting_metric

    def post_physics_step(self):
        """Post-physics step hook to update curriculum counter."""
        super().post_physics_step()
        self.step_count += 1

    def _reset_idx(self, env_ids: torch.Tensor, *, call_compute_iv: bool = True):
        """Reset specified environments to initial state."""
        super()._reset_idx(env_ids)

        # ============================================================================
        # Reset Robot State
        # ============================================================================
        robot_root = self._robot.data.default_root_state[env_ids].clone()
        robot_root[:, 0:3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(robot_root[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids=env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = default_joint_pos

        # ============================================================================
        # Reset Object State with Randomization
        # ============================================================================
        obj_root = self._object.data.default_root_state[env_ids].clone()
        obj_root[:, 0:3] += self.scene.env_origins[env_ids]

        # Randomize object position: x ∈ (-0.1, 0.1), y ∈ (-0.25, 0.25)
        n_envs = env_ids.shape[0]
        obj_root[:, 0] += sample_uniform(-0.1, 0.1, (n_envs,), device=self.device)
        obj_root[:, 1] += sample_uniform(-0.25, 0.25, (n_envs,), device=self.device)
        obj_root[:, 7:] = 0.0

        self._object.write_root_pose_to_sim(obj_root[:, :7], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(obj_root[:, 7:], env_ids=env_ids)

        # ============================================================================
        # Reset Buffers
        # ============================================================================
        self.goal_timer[env_ids] = 0.0
        self._sample_goal(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.prev_gripper_joint_pos[env_ids] = self._robot.data.joint_pos[env_ids, -2:]

    def _get_observations(self) -> dict:
        """Construct observation vector for policy (teacher or student mode)."""
        self._compute_intermediate_values()

        obs_components = []

        # === PROPRIOCEPTION (ALWAYS INCLUDED) ===
        # 1. Relative joint positions (7 arm + 1 gripper)
        joint_pos_list = [self._robot.data.joint_pos[:, idx:idx+1] for idx in self.arm_joint_indices]
        joint_pos_list.append(self._robot.data.joint_pos[:, self.gripper_joint_index:self.gripper_joint_index+1])
        
        default_joint_pos_list = [self._robot.data.default_joint_pos[:, idx:idx+1] for idx in self.arm_joint_indices]
        default_joint_pos_list.append(self._robot.data.default_joint_pos[:, self.gripper_joint_index:self.gripper_joint_index+1])
        
        joint_pos = torch.cat(joint_pos_list, dim=-1)
        default_joint_pos = torch.cat(default_joint_pos_list, dim=-1)
        obs_components.append(joint_pos - default_joint_pos)
        
        # 2. Joint velocities (7 arm + 1 gripper)
        joint_vel_list = [self._robot.data.joint_vel[:, idx:idx+1] for idx in self.arm_joint_indices]
        joint_vel_list.append(self._robot.data.joint_vel[:, self.gripper_joint_index:self.gripper_joint_index+1])
        obs_components.append(torch.cat(joint_vel_list, dim=-1))
        
        # === PRIVILEGED STATE (FOR TEACHER LABELING) ===
        # Always compute full state for teacher, regardless of mode
        # Get robot root frame for transformations
        robot_pos_w, robot_quat_w = self._squeeze_robot_frame(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w
        )
        
        # 3. Object position in robot frame (PRIVILEGED)
        obj_pos_w, obj_quat_w = self._squeeze_robot_frame(
            self._objects[0].data.root_pos_w, self._objects[0].data.root_quat_w
        )
        privileged_obs = [
            self._transform_world_to_robot_frame(obj_pos_w, robot_pos_w, robot_quat_w),
            # 4. Object orientation in robot frame (PRIVILEGED)
            self._transform_world_to_robot_frame_quat(obj_quat_w, robot_quat_w),
        ]
        
        # 5. End-effector position in robot frame (PRIVILEGED)
        hand_pos = self._robot.data.body_pos_w[:, self.hand_body_idx]
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx]
        offset = self.ee_offset.unsqueeze(0).repeat(hand_pos.shape[0], 1)
        ee_pos_w = hand_pos + quat_apply(hand_quat, offset)
        privileged_obs.extend([
            self._transform_world_to_robot_frame(ee_pos_w, robot_pos_w, robot_quat_w),
            # 6. End-effector orientation in robot frame (PRIVILEGED)
            self._transform_world_to_robot_frame_quat(hand_quat, robot_quat_w),
            # 7. Goal position in robot frame (PRIVILEGED)
            self._transform_world_to_robot_frame(self.goal_pos_w, robot_pos_w, robot_quat_w),
        ])

        # 8. Previous actions (ALWAYS INCLUDED)
        obs_components.append(self.actions)

        # Build observations based on mode
        if self.cfg.use_vision_observations:
            # Student mode: proprio only (24D)
            student_obs = torch.cat(obs_components, dim=-1)
            student_obs = torch.clamp(student_obs, -100.0, 100.0)
            student_obs = torch.nan_to_num(student_obs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Teacher obs: full state (41D) - for labeling only
            # CRITICAL: Must match the order used in teacher mode!
            # Teacher mode order: [joint_pos, joint_vel, prev_actions, privileged_obs]
            teacher_obs_components = obs_components + privileged_obs  # Don't reorder!
            teacher_obs = torch.cat(teacher_obs_components, dim=-1)
            teacher_obs = torch.clamp(teacher_obs, -100.0, 100.0)
            teacher_obs = torch.nan_to_num(teacher_obs, nan=0.0, posinf=0.0, neginf=0.0)
            
            result = {"policy": student_obs, "teacher_obs": teacher_obs}
        else:
            # Teacher mode: full state (41D)
            obs_components.extend(privileged_obs)
            obs = torch.cat(obs_components, dim=-1)
            obs = torch.clamp(obs, -100.0, 100.0)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            result = {"policy": obs}
        
        # === VISION (STUDENT MODE ONLY) ===
        if self.cfg.use_vision_observations:
            # Get RGBD data (scene automatically updates sensors)
            # RGB: (num_envs, height, width, 3), range [0, 255]
            result["wrist_cam_rgb"] = self._wrist_cam.data.output["rgb"]
            # Depth: (num_envs, height, width, 1), range [0.01, 10.0] meters
            result["wrist_cam_depth"] = self._wrist_cam.data.output["distance_to_camera"]
        
        return result

    
    def _squeeze_robot_frame(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper to squeeze robot root pose dimensions if needed."""
        if pos.dim() == 3:
            return pos[:, 0, :], quat[:, 0, :]
        return pos, quat

    def _transform_world_to_robot_frame(self, pos_w: torch.Tensor, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform world position to robot root frame."""
        return quat_apply_inverse(robot_quat_w, pos_w - robot_pos_w)
    
    def _transform_world_to_robot_frame_quat(self, quat_w: torch.Tensor, robot_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform world quaternion to robot root frame."""
        return quat_mul(quat_conjugate(robot_quat_w), quat_w)
