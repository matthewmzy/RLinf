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

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
DictConfig = pytest.importorskip("omegaconf").DictConfig

try:
    import ray
    from packaging import version as vs
except ImportError:
    ray = None
    vs = None


def test_cluster_config_parses_a2d_hardware():
    if ray is None or vs is None or vs.parse(ray.__version__) < vs.parse("2.47.0"):
        pytest.skip("Ray>=2.47.0 is required for scheduler config tests.")
    from rlinf.scheduler.cluster.config import ClusterConfig
    from rlinf.scheduler.hardware.robots.a2d import A2DConfig

    config = DictConfig(
        {
            "num_nodes": 2,
            "component_placement": {"env": {"node_group": "a2d", "placement": "0"}},
            "node_groups": [
                {
                    "label": "a2d",
                    "node_ranks": "1",
                    "hardware": {
                        "type": "A2D",
                        "configs": [
                            {
                                "node_rank": 1,
                                "controller_host": "127.0.0.1",
                                "grpc_port": 12321,
                                "container_name": "a2d-runtime",
                            }
                        ],
                    },
                }
            ],
        }
    )

    cluster_cfg = ClusterConfig.from_dict_cfg(config)
    a2d_group = cluster_cfg.node_groups[0]
    assert a2d_group.hardware_type == "A2D"
    node_hw = cluster_cfg.get_node_hw_configs_by_rank(1)
    assert len(node_hw) == 1
    assert isinstance(node_hw[0], A2DConfig)
    assert node_hw[0].controller_host == "127.0.0.1"
    assert node_hw[0].grpc_port == 12321


def test_cluster_config_supports_single_node_gpu_and_a2d_groups():
    if ray is None or vs is None or vs.parse(ray.__version__) < vs.parse("2.47.0"):
        pytest.skip("Ray>=2.47.0 is required for scheduler config tests.")
    from rlinf.scheduler.cluster.config import ClusterConfig

    config = DictConfig(
        {
            "num_nodes": 1,
            "component_placement": {
                "actor": {"node_group": "4090", "placement": "0"},
                "env": {"node_group": "a2d", "placement": "0"},
                "rollout": {"node_group": "4090", "placement": "0"},
            },
            "node_groups": [
                {
                    "label": "4090",
                    "node_ranks": "0",
                },
                {
                    "label": "a2d",
                    "node_ranks": "0",
                    "hardware": {
                        "type": "A2D",
                        "configs": [
                            {
                                "node_rank": 0,
                                "controller_host": "127.0.0.1",
                                "grpc_port": 12321,
                                "container_name": "a2d-runtime",
                            }
                        ],
                    },
                },
            ],
        }
    )

    cluster_cfg = ClusterConfig.from_dict_cfg(config)
    assert cluster_cfg.node_groups[0].label == "4090"
    assert cluster_cfg.node_groups[1].label == "a2d"
    node_hw = cluster_cfg.get_node_hw_configs_by_rank(0)
    assert len(node_hw) == 1
    assert node_hw[0].controller_host == "127.0.0.1"


def test_a2d_env_dummy_reset_and_step():
    pytest.importorskip("gymnasium")
    from rlinf.envs.realworld.a2d import A2DEnv

    env = A2DEnv(
        override_cfg={
            "is_dummy": True,
            "task_name": "dummy a2d",
            "image_shapes": {
                "rgb_head": [128, 128, 3],
                "rgb_left_hand": [128, 128, 3],
                "rgb_right_hand": [128, 128, 3],
            },
        }
    )
    obs, info = env.reset()
    assert info == {}
    assert obs["frames"]["rgb_head"].shape == (128, 128, 3)
    assert obs["state"]["arm_joint_states"].shape == (14,)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs["frames"]["rgb_left_hand"].shape == (128, 128, 3)
    assert reward == 0.0
    assert terminated is False
    assert isinstance(truncated, bool)
    assert step_info == {}


def test_a2d_env_builds_intervention_info_from_control_mode():
    pytest.importorskip("gymnasium")
    from rlinf.envs.realworld.a2d import A2DEnv
    from rlinf.envs.realworld.a2d.a2d_robot_state import A2DRobotState

    env = A2DEnv(
        override_cfg={
            "is_dummy": True,
            "policy_action_dim": 26,
            "normalize_actions": False,
            "clip_policy_actions": False,
            "model_control_modes": [0],
            "teleop_control_modes": [1],
            "idle_control_modes": [99],
        }
    )
    robot_state = A2DRobotState(
        states={
            "arm_joint_states": np.arange(14, dtype=np.float32),
            "left_hand_states": np.arange(6, dtype=np.float32) + 100.0,
            "right_hand_states": np.arange(6, dtype=np.float32) + 200.0,
            "waist_joints_states": np.array([300.0, 301.0], dtype=np.float32),
        },
        control_mode=1,
    )

    info = env._build_info(robot_state)
    expected_action = np.concatenate(
        [
            np.arange(14, dtype=np.float32),
            np.arange(6, dtype=np.float32) + 100.0,
            np.arange(6, dtype=np.float32) + 200.0,
        ]
    )
    np.testing.assert_allclose(info["intervene_action"], expected_action)
    assert info["control_mode_name"] == "teleop"
    assert info["data_valid"] is True


def test_a2d_env_marks_idle_frames_invalid():
    pytest.importorskip("gymnasium")
    from rlinf.envs.realworld.a2d import A2DEnv
    from rlinf.envs.realworld.a2d.a2d_robot_state import A2DRobotState

    env = A2DEnv(
        override_cfg={
            "is_dummy": True,
            "idle_control_modes": [99],
        }
    )
    info = env._build_info(
        A2DRobotState(states={}, control_mode=99, trajectory_label=2, is_switch_mode=True)
    )

    assert info["control_mode_name"] == "idle"
    assert info["data_valid"] is False
    assert "intervene_action" not in info


class _Waitable:
    def __init__(self, value):
        self._value = value

    def wait(self):
        return self._value


def test_a2d_env_disables_step_limit_after_intervention():
    pytest.importorskip("gymnasium")
    from rlinf.envs.realworld.a2d import A2DEnv

    env = object.__new__(A2DEnv)
    env.config = SimpleNamespace(
        is_dummy=False,
        clip_policy_actions=False,
        max_num_steps=1,
        step_frequency=1e9,
    )
    env._num_steps = 0
    env._episode_intervened = False
    env._robot_state = object()
    env._map_policy_action_to_controller = lambda action: action
    env._extract_observation = lambda robot_state: {"state": {}, "frames": {}}
    env._calc_reward = lambda robot_state: 0.0
    env._check_success = lambda robot_state: False

    info_sequence = [
        {"intervene_action": np.array([0.0, 0.0], dtype=np.float32)},
        {},
    ]
    env._build_info = lambda robot_state: info_sequence.pop(0)
    env._controller = SimpleNamespace(
        set_action=lambda action: _Waitable(None),
        get_state=lambda: _Waitable([object()]),
    )

    _, _, terminated_1, truncated_1, _ = A2DEnv.step(
        env, np.zeros((2,), dtype=np.float32)
    )
    _, _, terminated_2, truncated_2, _ = A2DEnv.step(
        env, np.zeros((2,), dtype=np.float32)
    )

    assert terminated_1 is False
    assert truncated_1 is False
    assert env._episode_intervened is True
    assert terminated_2 is False
    assert truncated_2 is False
