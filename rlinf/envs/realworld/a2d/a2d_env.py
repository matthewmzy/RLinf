# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.scheduler import (
    A2DHWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger

from .a2d_robot_state import A2DRobotState


@dataclass
class A2DRobotConfig:
    """Configuration for A2D real-world rollout via the official controller docker."""

    controller_host: Optional[str] = None
    controller_port: int = 12321
    grpc_timeout_s: float = 5.0
    ready_timeout_s: float = 60.0
    auto_start_server: bool = False
    container_name: Optional[str] = None
    grpc_config_file: Optional[str] = None
    server_command: Optional[str] = None

    is_dummy: bool = False
    step_frequency: float = 10.0
    max_num_steps: int = 100
    reset_on_env_reset: bool = True
    task_name: str = "A2D teleoperation"
    policy_action_dim: int = 28
    waist_action: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # RLinf policies usually act in [-1, 1]; map that range into controller action values.
    normalize_actions: bool = True
    clip_policy_actions: bool = True
    action_low: list[float] = field(
        default_factory=lambda: [-1.0] * 16 + [0.0] * 12
    )
    action_high: list[float] = field(default_factory=lambda: [1.0] * 28)
    model_control_modes: list[int] = field(default_factory=lambda: [0])
    teleop_control_modes: list[int] = field(default_factory=lambda: [1])
    idle_control_modes: list[int] = field(default_factory=lambda: [99])
    expose_intervention_from_control_mode: bool = True
    filter_idle_transitions: bool = True

    image_keys: list[str] = field(
        default_factory=lambda: ["rgb_head", "rgb_left_hand", "rgb_right_hand"]
    )
    image_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "rgb_head": [720, 1080, 3],
            "rgb_left_hand": [720, 1080, 3],
            "rgb_right_hand": [720, 1080, 3],
        }
    )
    state_keys: list[str] = field(
        default_factory=lambda: [
            "arm_joint_states",
            "left_hand_states",
            "right_hand_states",
            "waist_joints_states",
        ]
    )
    state_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "arm_joint_states": [14],
            "left_hand_states": [6],
            "right_hand_states": [6],
            "waist_joints_states": [2],
            "qrcode_detected": [1],
        }
    )

    reward_state_key: Optional[str] = None
    reward_scale: float = 1.0
    reward_offset: float = 0.0
    reward_clip_min: Optional[float] = None
    reward_clip_max: Optional[float] = None
    success_state_key: Optional[str] = None
    success_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.controller_port = int(self.controller_port)
        self.grpc_timeout_s = float(self.grpc_timeout_s)
        self.ready_timeout_s = float(self.ready_timeout_s)
        self.step_frequency = float(self.step_frequency)
        self.max_num_steps = int(self.max_num_steps)
        self.policy_action_dim = int(self.policy_action_dim)
        self.clip_policy_actions = bool(self.clip_policy_actions)
        self.waist_action = np.asarray(self.waist_action, dtype=np.float32).tolist()
        self.action_low = np.asarray(self.action_low, dtype=np.float32).tolist()
        self.action_high = np.asarray(self.action_high, dtype=np.float32).tolist()
        self.model_control_modes = [int(value) for value in self.model_control_modes]
        self.teleop_control_modes = [int(value) for value in self.teleop_control_modes]
        self.idle_control_modes = [int(value) for value in self.idle_control_modes]
        self.expose_intervention_from_control_mode = bool(
            self.expose_intervention_from_control_mode
        )
        self.filter_idle_transitions = bool(self.filter_idle_transitions)
        if len(self.action_low) != len(self.action_high):
            raise ValueError("action_low and action_high must have the same length.")
        if len(self.action_low) != 28:
            raise ValueError(
                f"A2D expects 28 action dimensions, got {len(self.action_low)}."
            )
        if self.policy_action_dim not in (26, 28):
            raise ValueError(
                f"A2D only supports policy_action_dim 26 or 28, got {self.policy_action_dim}."
            )
        if self.policy_action_dim == 26 and len(self.waist_action) != 2:
            raise ValueError(
                "waist_action must contain exactly 2 values when policy_action_dim is 26."
            )
        self.image_shapes = {
            key: list(value) for key, value in dict(self.image_shapes).items()
        }
        self.state_shapes = {
            key: list(value) for key, value in dict(self.state_shapes).items()
        }


class A2DEnv(gym.Env):
    """Gymnasium environment that talks to a running A2D controller container."""

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        self._logger = get_logger()
        self.config = A2DRobotConfig(**override_cfg)
        self.hardware_info = hardware_info
        self.worker_info: Optional[WorkerInfo] = worker_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._num_steps = 0
        self._robot_state = A2DRobotState()

        self._init_action_obs_spaces()

        if not self.config.is_dummy:
            self._setup_hardware()
            self._robot_state = self._controller.reset().wait()[0]

    def _setup_hardware(self) -> None:
        from .a2d_controller import A2DController

        assert isinstance(self.hardware_info, A2DHWInfo), (
            f"hardware_info must be A2DHWInfo, but got {type(self.hardware_info)}."
        )
        if self.config.controller_host is None:
            self.config.controller_host = self.hardware_info.config.controller_host
        if self.config.container_name is None:
            self.config.container_name = self.hardware_info.config.container_name
        if self.config.grpc_config_file is None:
            self.config.grpc_config_file = self.hardware_info.config.grpc_config_file
        if self.config.server_command is None:
            self.config.server_command = self.hardware_info.config.server_command
        if self.config.auto_start_server is False:
            self.config.auto_start_server = self.hardware_info.config.auto_start_server
        if self.config.controller_port == 12321:
            self.config.controller_port = self.hardware_info.config.grpc_port

        self._controller = A2DController.launch_controller(
            controller_host=self.config.controller_host,
            controller_port=self.config.controller_port,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
            grpc_timeout_s=self.config.grpc_timeout_s,
            ready_timeout_s=self.config.ready_timeout_s,
            container_name=self.config.container_name,
            grpc_config_file=self.config.grpc_config_file,
            auto_start_server=self.config.auto_start_server,
            server_command=self.config.server_command,
        )

    def _init_action_obs_spaces(self) -> None:
        action_low = np.full((self.config.policy_action_dim,), -1.0, dtype=np.float32)
        action_high = np.full((self.config.policy_action_dim,), 1.0, dtype=np.float32)
        if not self.config.normalize_actions:
            controller_low = np.asarray(self.config.action_low, dtype=np.float32)
            controller_high = np.asarray(self.config.action_high, dtype=np.float32)
            if self.config.policy_action_dim == 26:
                action_low = controller_low[2:]
                action_high = controller_high[2:]
            else:
                action_low = controller_low
                action_high = controller_high
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        frames_space = {
            key: gym.spaces.Box(
                low=0,
                high=255,
                shape=tuple(shape),
                dtype=np.uint8,
            )
            for key, shape in self.config.image_shapes.items()
            if key in self.config.image_keys
        }
        states_space = {
            key: gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(shape),
                dtype=np.float32,
            )
            for key, shape in self.config.state_shapes.items()
            if key in self.config.state_keys
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(states_space),
                "frames": gym.spaces.Dict(frames_space),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _expand_policy_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.config.policy_action_dim:
            raise ValueError(
                f"Expected {self.config.policy_action_dim} policy action dims, got {action.size}."
            )
        if self.config.policy_action_dim == 28:
            return action
        # The A2D controller uses waist + arm + hands, while psi-policy rgb_state
        # predicts only arm + hands. Fill waist from config and prepend it.
        waist_action = np.asarray(self.config.waist_action, dtype=np.float32)
        return np.concatenate([waist_action, action], axis=0)

    def _map_policy_action_to_controller(self, action: np.ndarray) -> np.ndarray:
        controller_action = self._expand_policy_action(action)
        if not self.config.normalize_actions:
            return controller_action.astype(np.float32)
        low = np.asarray(self.config.action_low, dtype=np.float32)
        high = np.asarray(self.config.action_high, dtype=np.float32)
        controller_action = np.clip(controller_action, -1.0, 1.0)
        return (
            (((controller_action + 1.0) * 0.5) * (high - low) + low).astype(np.float32)
        )

    def _map_controller_action_to_policy(self, controller_action: np.ndarray) -> np.ndarray:
        controller_action = np.asarray(controller_action, dtype=np.float32).reshape(-1)
        if controller_action.size != 28:
            raise ValueError(
                f"Expected 28 controller action dims, got {controller_action.size}."
            )

        policy_action = (
            controller_action[2:]
            if self.config.policy_action_dim == 26
            else controller_action.copy()
        )
        if not self.config.normalize_actions:
            return policy_action.astype(np.float32)

        low = np.asarray(self.config.action_low, dtype=np.float32)
        high = np.asarray(self.config.action_high, dtype=np.float32)
        normalized = 2.0 * (controller_action - low) / np.maximum(high - low, 1e-6) - 1.0
        if self.config.policy_action_dim == 26:
            normalized = normalized[2:]
        return normalized.astype(np.float32)

    def _get_controller_state_action(self, robot_state: A2DRobotState) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    robot_state.states["waist_joints_states"], dtype=np.float32
                ).reshape(-1),
                np.asarray(
                    robot_state.states["arm_joint_states"], dtype=np.float32
                ).reshape(-1),
                np.asarray(
                    robot_state.states["left_hand_states"], dtype=np.float32
                ).reshape(-1),
                np.asarray(
                    robot_state.states["right_hand_states"], dtype=np.float32
                ).reshape(-1),
            ],
            axis=0,
        )

    def _get_policy_space_action(self, robot_state: A2DRobotState) -> np.ndarray:
        return self._map_controller_action_to_policy(
            self._get_controller_state_action(robot_state)
        )

    def _get_control_mode_name(self, control_mode: Optional[int]) -> str:
        if control_mode in self.config.model_control_modes:
            return "model"
        if control_mode in self.config.teleop_control_modes:
            return "teleop"
        if control_mode in self.config.idle_control_modes:
            return "idle"
        if control_mode is None:
            return "unknown"
        return f"mode_{control_mode}"

    def _extract_observation(self, robot_state: A2DRobotState) -> dict:
        frames = {}
        for key in self.config.image_keys:
            if key not in robot_state.images:
                raise KeyError(
                    f"Image key '{key}' not found in controller output. "
                    f"Available keys: {sorted(robot_state.images.keys())}"
                )
            frames[key] = robot_state.images[key].copy()

        state = {}
        for key in self.config.state_keys:
            if key not in robot_state.states:
                raise KeyError(
                    f"State key '{key}' not found in controller output. "
                    f"Available keys: {sorted(robot_state.states.keys())}"
                )
            state[key] = robot_state.states[key].astype(np.float32).copy()

        return {"state": state, "frames": frames}

    def _build_info(self, robot_state: A2DRobotState) -> dict:
        control_mode = robot_state.control_mode
        control_mode_name = self._get_control_mode_name(control_mode)
        is_idle = (
            self.config.filter_idle_transitions
            and control_mode in self.config.idle_control_modes
        )
        info = {
            "timestamps": robot_state.timestamps.copy(),
            "control_mode": control_mode,
            "control_mode_name": control_mode_name,
            "trajectory_label": robot_state.trajectory_label,
            "is_switch_mode": robot_state.is_switch_mode,
            "data_valid": not is_idle,
        }
        if (
            self.config.expose_intervention_from_control_mode
            and control_mode in self.config.teleop_control_modes
        ):
            info["intervene_action"] = self._get_policy_space_action(robot_state)
        return info

    def _calc_reward(self, robot_state: A2DRobotState) -> float:
        reward_key = self.config.reward_state_key
        if reward_key is None:
            return 0.0
        reward_value = robot_state.states.get(reward_key)
        if reward_value is None:
            raise KeyError(
                f"reward_state_key '{reward_key}' not found in controller state."
            )
        reward = float(np.asarray(reward_value, dtype=np.float32).reshape(-1)[0])
        reward = reward * self.config.reward_scale + self.config.reward_offset
        if self.config.reward_clip_min is not None:
            reward = max(reward, self.config.reward_clip_min)
        if self.config.reward_clip_max is not None:
            reward = min(reward, self.config.reward_clip_max)
        return reward

    def _check_success(self, robot_state: A2DRobotState) -> bool:
        success_key = self.config.success_state_key
        if success_key is None:
            return False
        success_value = robot_state.states.get(success_key)
        if success_value is None:
            raise KeyError(
                f"success_state_key '{success_key}' not found in controller state."
            )
        scalar = float(np.asarray(success_value, dtype=np.float32).reshape(-1)[0])
        return scalar >= self.config.success_threshold

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.config.clip_policy_actions:
            action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            controller_action = self._map_policy_action_to_controller(action)
            self._controller.set_action(controller_action).wait()
            self._num_steps += 1

            step_time = time.time() - start_time
            time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
            self._robot_state = self._controller.get_state().wait()[0]
            observation = self._extract_observation(self._robot_state)
            reward = self._calc_reward(self._robot_state)
            terminated = self._check_success(self._robot_state)
            truncated = self._num_steps >= self.config.max_num_steps
            info = self._build_info(self._robot_state)
            return observation, reward, terminated, truncated, info

        self._num_steps += 1
        observation = self._base_observation_space.sample()
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, 0.0, False, truncated, {}

    def reset(self, seed=None, options=None):
        del options
        self._num_steps = 0
        if self.config.is_dummy:
            return self._base_observation_space.sample(), {}

        if self.config.reset_on_env_reset:
            self._robot_state = self._controller.reset(seed=seed).wait()[0]
        else:
            self._robot_state = self._controller.get_state().wait()[0]
        observation = self._extract_observation(self._robot_state)
        return observation, self._build_info(self._robot_state)

    @property
    def task_description(self) -> str:
        return self.config.task_name
