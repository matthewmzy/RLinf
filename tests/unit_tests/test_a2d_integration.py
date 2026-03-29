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

import pytest
from omegaconf import DictConfig

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
