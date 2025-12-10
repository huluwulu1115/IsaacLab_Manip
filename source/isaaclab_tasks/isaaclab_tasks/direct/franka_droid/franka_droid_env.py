import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

# MISSING import removed - not needed with proper @configclass usage

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform, quat_mul, quat_conjugate
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.utils.spaces import spec_to_gym_space, sample_space
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg


# Assets folder is now in the same directory as this file
ASSET_PATH = Path(__file__).parent / "assets"


@configclass
class FrankaDroidEnvCfg(DirectRLEnvCfg):
    """
    Franka DROID environment v0
    Old version of the environment, using the old Franka Panda robotiq 2f-85 gripper usd
    There are some issue with the old usd.
    """

    episode_length_s = 10.0  # 10 seconds for exploration
    decimation = 7  # Control frequency: 100Hz / 7 ≈ 14.3Hz (matches real hardware ~15Hz)
    action_space = 8
    observation_space = 0
    state_space = 0

    # Sim2Real Configuration
    use_vision_observations = False  # True for student (vision), False for teacher (state)

    # Early Termination Configuration
    enable_early_termination = True  # Terminate early when task is completed
    early_termination_hold_time = 0.5  # Time (seconds) object must stay at goal before terminating
    success_distance_threshold = 0.05  # Distance threshold (meters) to consider object at goal

    # Visualization Configuration
    enable_visualization_markers = True  # Set to False to disable EE/goal markers (saves GPU resources)

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,  # Simulation frequency: 100Hz (physics stability)
        render_interval=7,  # Match decimation for rendering
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

    # Action configuration (simplified)
    arm_action_scale: float = 0.5
    """Scale factor for arm joint position actions."""
    gripper_open_pos: float = 0.0
    """Gripper open position in radians (0°)."""
    gripper_close_pos: float = np.pi / 4
    """Gripper close position in radians (45°)."""

    # Goal position configuration (local coordinates relative to environment origin)
    goal_position: tuple = (0.4, 0.0, 0.25)
    """Fixed goal position (x, y, z) in local coordinates. Object should be lifted here after grasping."""
    goal_position_noise: tuple = (0.0, 0.0, 0.0)
    """Random noise range (±) added to goal position for domain randomization."""

    # Robot: Let USD define physics parameters (v1 feature!)
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
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
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
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
        },
    )

    objects = [
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object_0",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
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

    # Wrist camera with RGBD for sim2real transfer
    # Using TiledCamera (like DEXTRAH) for better multi-environment support
    # This solves descriptor allocation issues in headless mode with many environments
    wrist_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
        data_types=["rgb", "distance_to_camera"],  # RGB + Depth (like DEXTRAH uses ["rgb", "depth"])
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
            clipping_range=(0.01, 10.0),
        ),
        width=320,
        height=240,
    )


class FrankaDroidEnv(DirectRLEnv):
    """Franka + Robotiq gripper pick-and-place environment.

    Obs: Teacher(41D) = joint_pos/vel(16) + obj_pose(7) + ee_pose(7) + goal(3) + actions(8)
         Student(24D) = joint_pos/vel(16) + actions(8) + RGBD camera
    """

    cfg: FrankaDroidEnvCfg

    def __init__(self, cfg: FrankaDroidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._compute_observation_space_size()
        self._reconfigure_gym_spaces()
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Robot configuration
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = self._robot.data.default_joint_pos.clone()
        self.arm_joint_indices = [self._robot.find_joints(f"panda_joint{i}")[0][0] for i in range(1, 8)]
        self.gripper_joint_index = self._robot.find_joints("finger_joint")[0][0]

        # State buffers
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.action_delta = torch.zeros_like(self.prev_actions)
        self.step_count = 0

        # Early termination: track how long object has been at goal position
        self.success_hold_timer = torch.zeros(self.num_envs, device=self.device)  # Accumulates time at goal

        # Cached values for reward/obs computation
        self._robot.data.ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._robot.data.obj_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._robot.data.dist_ee_obj = torch.zeros(self.num_envs, device=self.device)
        self._robot.data.dist_obj_goal = torch.zeros(self.num_envs, device=self.device)

        # End-effector config
        self.ee_offset = torch.tensor([0.0, 0.0, 0.16], device=self.device)
        self.minimal_height = 0.04
        self.hand_body_idx = self._robot.find_bodies("panda_link8")[0][0]

        # Goal position configuration (in local coordinates)
        self.goal_pos_local = torch.tensor(self.cfg.goal_position, device=self.device, dtype=torch.float32)
        self.goal_pos_noise = torch.tensor(self.cfg.goal_position_noise, device=self.device, dtype=torch.float32)

        # Initialize goals
        self._sample_goal(torch.arange(self.num_envs, device=self.device))

        # Visualization markers (optional)
        if self.cfg.enable_visualization_markers:
            frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ee_frame")
            frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.ee_frame_marker = VisualizationMarkers(frame_cfg)

            ee_dot_cfg = VisualizationMarkersCfg(
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.01, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                    )
                },
                prim_path="/Visuals/ee_dot",
            )
            self.ee_dot_marker = VisualizationMarkers(ee_dot_cfg)

            goal_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/goal_frame")
            goal_frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
            self.goal_marker = VisualizationMarkers(goal_frame_cfg)
        else:
            self.ee_frame_marker = self.ee_dot_marker = self.goal_marker = None

        # RL Games compatibility
        self.num_actions = self.cfg.action_space
        self.num_observations = 24
        self.num_teacher_observations = 41

    def _sample_goal(self, env_ids: torch.Tensor):
        """Set goal position for object placement after grasping.

        Goal is a fixed position (configurable via cfg.goal_position) with optional noise.
        """
        n = len(env_ids)
        # Start with fixed goal position (in local coordinates)
        goal_local = self.goal_pos_local.unsqueeze(0).expand(n, -1).clone()

        # Add optional randomization noise
        if self.goal_pos_noise.abs().sum() > 0:
            noise = (torch.rand(n, 3, device=self.device) * 2 - 1) * self.goal_pos_noise
            goal_local = goal_local + noise

        # Convert to world coordinates
        self.goal_pos_w[env_ids] = goal_local + self.scene.env_origins[env_ids]

    def _configure_gym_env_spaces(self):
        """Placeholder - will be reconfigured after computing obs size."""
        pass

    def _reconfigure_gym_spaces(self):
        """Configure gym spaces with computed observation size."""
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
        """Compute observation space size: Student=24D, Teacher=41D."""
        action_dim = int(self.cfg.action_space) if not isinstance(self.cfg.action_space, int) else self.cfg.action_space
        if self.cfg.use_vision_observations:
            self.computed_obs_size = 8 + 8 + action_dim  # 24D
        else:
            self.computed_obs_size = 8 + 8 + 7 + 7 + 3 + action_dim  # 41D (goal is 3D xyz)

    def _setup_scene(self):
        """Setup robot, object, table, and optional camera."""
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)
        self._objects = [self._object]

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        if self.cfg.use_vision_observations:
            self._wrist_cam = TiledCamera(self.cfg.wrist_cam)
            self.scene.sensors["wrist_cam"] = self._wrist_cam

        # Table and ground
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_0/Table", table_cfg, translation=(0.525, 0.0, 0.0), orientation=(0.707, 0.0, 0.0, 0.707)
        )
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg, translation=(0.0, 0.0, -1.05))

        self.scene.clone_environments(copy_from_source=False)
        sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func("/World/Light")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions: squash gripper, compute action delta."""
        self.actions = actions.clone()
        self.actions[:, 7] = 0.04 * torch.tanh(self.actions[:, 7])  # Squash gripper
        self.action_delta = self.actions - self.prev_actions
        self.prev_actions[:] = self.actions

    def _apply_action(self):
        """Apply position control to arm joints and binary gripper control."""
        default_pos = self._robot.data.default_joint_pos
        if default_pos.dim() == 1:
            default_pos = default_pos.unsqueeze(0).expand(self.num_envs, -1)
        elif default_pos.shape[0] == 1:
            default_pos = default_pos.expand(self.num_envs, -1)

        targets = default_pos.clone()
        for i, idx in enumerate(self.arm_joint_indices):
            targets[:, idx] = default_pos[:, idx] + self.cfg.arm_action_scale * self.actions[:, i]

        # Binary gripper: close if action < 0
        targets[:, self.gripper_joint_index] = torch.where(
            self.actions[:, 7] < 0.0,
            torch.full((self.num_envs,), self.cfg.gripper_close_pos, device=self.device),
            torch.full((self.num_envs,), self.cfg.gripper_open_pos, device=self.device),
        )

        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation conditions."""
        # Compute ee_pos and obj_pos (cached for reuse in _get_rewards and _get_observations)
        hand_pos = self._robot.data.body_pos_w[:, self.hand_body_idx]
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx]
        offset = self.ee_offset.unsqueeze(0).repeat(hand_pos.shape[0], 1)

        self._robot.data.ee_pos_w = hand_pos + quat_apply(hand_quat, offset)
        self._robot.data.obj_pos_w = self._objects[0].data.root_pos_w

        # Terminate if object is out of bounds (dropped below table or thrown too high)
        # IMPORTANT: Use height relative to environment origin, not world coordinates!
        obj_pos = self._objects[0].data.root_pos_w
        obj_height_local = obj_pos[:, 2] - self.scene.env_origins[:, 2]
        obj_dropped = obj_height_local < -0.05
        obj_too_high = obj_height_local > 1.0  # Object thrown/flew too high
        obj_out_of_bounds = obj_dropped | obj_too_high

        # Compute success metric (once per step, with component logging)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        success_metric = self.get_success_metric(all_env_ids, log_components=True)

        # Check if object is at goal position (within threshold distance)
        dist_obj_goal = torch.norm(self._robot.data.obj_pos_w - self.goal_pos_w, dim=-1)
        at_goal = dist_obj_goal < self.cfg.success_distance_threshold

        # Update success hold timer: increment if at goal, reset if not
        self.success_hold_timer = torch.where(
            at_goal,
            self.success_hold_timer + self.dt,  # Accumulate time
            torch.zeros_like(self.success_hold_timer),  # Reset timer
        )

        # Log success metrics (always, regardless of enable_early_termination)
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Success/metric_mean"] = success_metric.mean()
        self.extras["log"]["Success/at_goal_count"] = at_goal.float().sum()
        self.extras["log"]["Success/at_goal_rate"] = at_goal.float().mean()
        self.extras["log"]["Success/hold_time_max"] = self.success_hold_timer.max()

        # Early termination: object stayed at goal for required time
        success_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.enable_early_termination:
            success_timeout = self.success_hold_timer >= self.cfg.early_termination_hold_time

        terminated = self.terminated_buf.clone() | obj_out_of_bounds | success_timeout

        # Truncate on episode timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        self.terminated_buf[:] = False
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards: reaching + grasping + lifting + goal_tracking - penalties."""
        # Visualize markers (only if enabled)
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx]
        if self.cfg.enable_visualization_markers:
            robot_base_pos, _ = self._squeeze_robot_frame(self._robot.data.root_pos_w, self._robot.data.root_quat_w)
            frame_pos = robot_base_pos.clone()
            frame_pos[:, 2] += 1.0
            self.ee_frame_marker.visualize(frame_pos, hand_quat)
            self.ee_dot_marker.visualize(self._robot.data.ee_pos_w)

            # Visualize goal position
            goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
            goal_quat[:, 0] = 1.0
            self.goal_marker.visualize(self.goal_pos_w, goal_quat)

        # ============================================================================
        # Reward Components
        # ============================================================================

        # 1. Reaching: Encourage EE to approach object (weight=5.0)
        self._robot.data.dist_ee_obj = torch.norm(self._robot.data.ee_pos_w - self._robot.data.obj_pos_w, dim=-1)
        reaching_reward = (1.0 - torch.tanh(self._robot.data.dist_ee_obj / 0.1)) * 5.0

        # 2. Grasping: Reward closing gripper near object (weight=3.0)
        gripper_closed = (self.actions[:, 7] < 0.0).float()
        near_object = (self._robot.data.dist_ee_obj < 0.08).float()
        grasp_reward = gripper_closed * near_object * 3.0

        # 3. Lifting: Continuous reward for raising object toward goal height (weight=8.0)
        # IMPORTANT: Use height relative to environment origin, not world coordinates!
        obj_height_local = self._robot.data.obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        goal_height = self.goal_pos_local[2]  # Target height from config
        # Smooth reward from table (0.02m) to goal height, with small margin above
        lifting_reward = torch.clamp((obj_height_local - 0.02) / (goal_height - 0.02 + 0.01), 0, 1) * 8.0

        # 4. Goal tracking: Guide lifted object to goal position (weight=9.0)
        self._robot.data.dist_obj_goal = torch.norm(self._robot.data.obj_pos_w - self.goal_pos_w, dim=-1)
        lifted_mask = (obj_height_local > self.minimal_height).float()

        # Adaptive precision: coarse early, fine later
        goal_sigma = 0.3 if self.step_count < 5000 else 0.1
        goal_tracking = lifted_mask * (1.0 - torch.tanh(self._robot.data.dist_obj_goal / goal_sigma)) * 9.0

        # 5. Regularization: Action smoothness and velocity penalties (curriculum-based)
        # Penalty weight fades from -1e-3 to -1e-6 over 10k steps (linear decay)
        # Early: strict constraints for stability | Late: relaxed for performance
        curr_factor = min(self.step_count / 10000.0, 1.0)
        penalty_w = (-1e-3) * (1.0 - curr_factor) + (-1e-6) * curr_factor
        action_penalty = torch.sum(self.action_delta**2, dim=-1) * penalty_w
        joint_vel_penalty = torch.sum(self._robot.data.joint_vel**2, dim=-1) * penalty_w

        # Total reward computation
        total_reward = (
            reaching_reward  # 5.0
            + grasp_reward  # 3.0
            + lifting_reward  # 8.0
            + goal_tracking  # 9.0
            + action_penalty  # ~-0.001
            + joint_vel_penalty  # ~-0.001
        )
        # Maximum possible: 25.0 (more balanced than previous 37.0)

        # Safety check: replace any NaN or Inf values with zeros
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Log key metrics only
        if hasattr(self, "extras"):
            self.extras.setdefault("log", {}).update(
                {
                    "Reward/total": total_reward.mean(),
                    "State/dist_ee_obj": self._robot.data.dist_ee_obj.mean(),
                    "State/dist_obj_goal": self._robot.data.dist_obj_goal.mean(),
                    "State/obj_height": obj_height_local.mean(),  # Use local height
                }
            )

        return total_reward

    def get_success_metric(
        self, env_ids: torch.Tensor, log_components: bool = False, use_cache: bool = False
    ) -> torch.Tensor:
        """Compute success metric [0,1]: reaching(0.33) + lifting(0.33) + goal(0.34)."""
        ee_pos = self._robot.data.ee_pos_w[env_ids]
        obj_pos = self._robot.data.obj_pos_w[env_ids]

        # IMPORTANT: Use height relative to environment origin!
        obj_height_local = obj_pos[:, 2] - self.scene.env_origins[env_ids, 2]
        goal_height = self.goal_pos_local[2]  # Target height from config

        reaching = (1 - torch.tanh(torch.norm(ee_pos - obj_pos, dim=-1))) * 0.33
        lifting = torch.clamp(obj_height_local / goal_height, 0, 1.0) * 0.33
        goal = (1 - torch.tanh(torch.norm(obj_pos - self.goal_pos_w[env_ids], dim=-1) / 0.15)) * 0.34

        return reaching + lifting + goal

    def post_physics_step(self):
        """Post-physics step hook to update curriculum counter."""
        super().post_physics_step()
        self.step_count += 1

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset robot and object to initial state with randomization."""
        super()._reset_idx(env_ids)
        n = env_ids.shape[0]

        # Reset robot
        robot_root = self._robot.data.default_root_state[env_ids].clone()
        robot_root[:, 0:3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(robot_root[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids=env_ids)
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        self._robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), env_ids=env_ids)
        self.robot_dof_targets[env_ids] = default_joint_pos

        # Reset object with position randomization
        obj_root = self._object.data.default_root_state[env_ids].clone()
        obj_root[:, 0:3] += self.scene.env_origins[env_ids]
        obj_root[:, 0] += sample_uniform(-0.1, 0.1, (n,), device=self.device)
        obj_root[:, 1] += sample_uniform(-0.25, 0.25, (n,), device=self.device)
        obj_root[:, 7:] = 0.0
        self._object.write_root_pose_to_sim(obj_root[:, :7], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(obj_root[:, 7:], env_ids=env_ids)

        # Reset buffers
        self._sample_goal(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.success_hold_timer[env_ids] = 0.0  # Reset early termination timer

    def _get_observations(self) -> dict:
        """Build observation: Student(24D) or Teacher(41D)."""
        # Proprioception: relative joint pos + vel (16D)
        joint_indices = self.arm_joint_indices + [self.gripper_joint_index]
        joint_pos = torch.cat([self._robot.data.joint_pos[:, i : i + 1] for i in joint_indices], dim=-1)
        default_pos = torch.cat([self._robot.data.default_joint_pos[:, i : i + 1] for i in joint_indices], dim=-1)
        joint_vel = torch.cat([self._robot.data.joint_vel[:, i : i + 1] for i in joint_indices], dim=-1)

        proprio = [joint_pos - default_pos, joint_vel, self.actions]

        # Privileged state (17D): obj_pose(7) + ee_pose(7) + goal(3)
        robot_pos, robot_quat = self._squeeze_robot_frame(self._robot.data.root_pos_w, self._robot.data.root_quat_w)
        obj_pos, obj_quat = self._squeeze_robot_frame(
            self._objects[0].data.root_pos_w, self._objects[0].data.root_quat_w
        )
        hand_quat = self._robot.data.body_quat_w[:, self.hand_body_idx]

        privileged = [
            self._transform_world_to_robot_frame(obj_pos, robot_pos, robot_quat),
            self._transform_world_to_robot_frame_quat(obj_quat, robot_quat),
            self._transform_world_to_robot_frame(self._robot.data.ee_pos_w, robot_pos, robot_quat),
            self._transform_world_to_robot_frame_quat(hand_quat, robot_quat),
            self._transform_world_to_robot_frame(self.goal_pos_w, robot_pos, robot_quat),
        ]

        def sanitize(t):
            return torch.nan_to_num(torch.clamp(t, -100, 100), nan=0.0)

        if self.cfg.use_vision_observations:
            result = {
                "policy": sanitize(torch.cat(proprio, dim=-1)),
                "teacher_obs": sanitize(torch.cat(proprio + privileged, dim=-1)),
                "wrist_cam_rgb": self._wrist_cam.data.output["rgb"],
                "wrist_cam_depth": self._wrist_cam.data.output["distance_to_camera"],
            }
        else:
            result = {"policy": sanitize(torch.cat(proprio + privileged, dim=-1))}

        return result

    def _squeeze_robot_frame(self, pos: torch.Tensor, quat: torch.Tensor):
        """Squeeze robot root pose if needed."""
        return (pos[:, 0, :], quat[:, 0, :]) if pos.dim() == 3 else (pos, quat)

    def _transform_world_to_robot_frame(self, pos_w, robot_pos_w, robot_quat_w):
        """Transform position to robot frame."""
        return quat_apply_inverse(robot_quat_w, pos_w - robot_pos_w)

    def _transform_world_to_robot_frame_quat(self, quat_w, robot_quat_w):
        """Transform quaternion to robot frame."""
        return quat_mul(quat_conjugate(robot_quat_w), quat_w)
