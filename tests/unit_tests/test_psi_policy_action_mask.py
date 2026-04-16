# Copyright 2026 The RLinf Authors.
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

torch = pytest.importorskip("torch")
import torch.nn as nn
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
from types import SimpleNamespace

from rlinf.models.embodiment.psi_policy.psi_policy_for_rl import (
    PsiPolicyForRL,
    PsiPolicyConfig,
    _build_shape_meta,
    _build_rl_action_mask,
    _normalize_rl_action_mask_cfg,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def test_normalize_rl_action_mask_cfg_accepts_group_and_indices():
    cfg = _normalize_rl_action_mask_cfg(
        {"enabled": True, "active_groups": "arms", "active_indices": [14, 20]}
    )

    assert cfg["enabled"] is True
    assert cfg["active_groups"] == ["arms"]
    assert cfg["active_indices"] == [14, 20]


def test_build_rl_action_mask_for_arms_and_extra_hand_joints():
    cfg = PsiPolicyConfig()
    mask, active_indices = _build_rl_action_mask(
        cfg.action_dim,
        cfg.state_split,
        _normalize_rl_action_mask_cfg(
            {"enabled": True, "active_groups": ["arms"], "active_indices": [14, 20]}
        ),
    )

    expected_indices = list(range(14)) + [14, 20]
    assert torch.equal(mask.nonzero(as_tuple=False).squeeze(-1), torch.tensor(expected_indices))
    assert active_indices == expected_indices


def test_build_rl_action_mask_returns_all_dims_when_disabled():
    cfg = PsiPolicyConfig()
    mask, active_indices = _build_rl_action_mask(
        cfg.action_dim,
        cfg.state_split,
        _normalize_rl_action_mask_cfg({"enabled": False}),
    )

    assert mask.dtype == torch.bool
    assert mask.all()
    assert active_indices == list(range(cfg.action_dim))


def test_apply_reset_pose_mask_overrides_frozen_dims():
    policy = object.__new__(PsiPolicyForRL)
    nn.Module.__init__(policy)
    policy.action_dim = 26
    policy.active_rl_action_dim = 14
    policy.rl_action_mask = torch.tensor(
        [True] * 14 + [False] * 12, dtype=torch.bool
    )
    policy._get_rl_action_mask = PsiPolicyForRL._get_rl_action_mask.__get__(
        policy, PsiPolicyForRL
    )
    policy._extract_reset_action_reference = (
        PsiPolicyForRL._extract_reset_action_reference.__get__(policy, PsiPolicyForRL)
    )

    action = torch.arange(26, dtype=torch.float32).unsqueeze(0) + 1000.0
    reset_states = torch.arange(28, dtype=torch.float32).unsqueeze(0)
    masked_action = PsiPolicyForRL._apply_reset_pose_mask(
        policy,
        action,
        {"reset_states": reset_states},
    )

    expected = action.clone()
    expected[:, 14:26] = reset_states[:, 14:26]
    assert torch.allclose(masked_action, expected)


def test_critic_projects_masked_batch_actions_to_reset_pose():
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker._cached_rl_action_mask = None
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "psi_policy",
                    "action_dim": 26,
                    "state_split": {
                        "left_arm": [0, 7],
                        "right_arm": [7, 14],
                        "left_hand": [14, 20],
                        "right_hand": [20, 26],
                    },
                    "rl_action_mask": {
                        "enabled": True,
                        "active_groups": ["arms"],
                        "active_indices": [],
                    },
                }
            }
        }
    )

    actions = torch.arange(26, dtype=torch.float32).unsqueeze(0) + 500.0
    reset_states = torch.arange(28, dtype=torch.float32).unsqueeze(0)

    projected = EmbodiedSACFSDPPolicy._project_actions_to_reset_pose(
        worker,
        actions,
        {"reset_states": reset_states},
    )

    expected = actions.clone()
    expected[:, 14:26] = reset_states[:, 14:26]
    assert torch.allclose(projected, expected)


class _IdentityFieldNormalizer:
    def normalize(self, tensor):
        return tensor

    def unnormalize(self, tensor):
        return tensor


class _IdentityNormalizer(dict):
    def normalize(self, obs):
        return obs


def _make_minimal_policy(cfg: PsiPolicyConfig | None = None) -> PsiPolicyForRL:
    policy = object.__new__(PsiPolicyForRL)
    nn.Module.__init__(policy)
    policy.cfg = cfg or PsiPolicyConfig()
    policy.shape_meta = _build_shape_meta(policy.cfg)
    policy.action_dim = policy.cfg.action_dim
    policy.action_horizon = policy.cfg.action_horizon
    policy.active_rl_action_dim = policy.action_dim
    policy.register_parameter("_dummy", nn.Parameter(torch.zeros(1), requires_grad=False))
    policy.register_buffer(
        "rl_action_mask", torch.ones(policy.action_dim, dtype=torch.bool), persistent=False
    )
    policy.normalizer = _IdentityNormalizer(action=_IdentityFieldNormalizer())
    return policy


def test_preprocess_env_obs_keeps_images_in_zero_one_and_pads_history():
    cfg = PsiPolicyConfig(obs_horizon=2, image_size=4)
    policy = _make_minimal_policy(cfg)

    main_images = torch.tensor(
        [
            [
                [[0, 255, 128], [255, 128, 0]],
                [[64, 32, 16], [8, 4, 2]],
            ]
        ],
        dtype=torch.uint8,
    )
    extra_view_images = torch.stack([main_images, main_images], dim=1)
    states = torch.arange(28, dtype=torch.float32).unsqueeze(0)

    processed = policy.preprocess_env_obs(
        {
            "main_images": main_images,
            "extra_view_images": extra_view_images,
            "states": states,
        }
    )

    assert processed["rgb_head"].shape == (1, 2, 3, 4, 4)
    assert processed["rgb_left_hand"].shape == (1, 2, 3, 4, 4)
    assert processed["rgb_right_hand"].shape == (1, 2, 3, 4, 4)
    assert processed["rgb_head"].min() >= 0.0
    assert processed["rgb_head"].max() <= 1.0
    assert torch.allclose(processed["rgb_head"][:, 0], processed["rgb_head"][:, 1])
    assert processed["right_arm_states"].shape == (1, 2, 7)
    assert torch.allclose(
        processed["left_hand_states"][:, 0], processed["left_hand_states"][:, 1]
    )


def test_move_obs_dict_to_encoder_device_matches_encoder_dtype():
    policy = _make_minimal_policy()
    policy.obs_encoder = nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)

    moved = policy._move_obs_dict_to_encoder_device(
        {
            "rgb_head": torch.zeros(1, 1, 3, 4, 4, dtype=torch.float32),
            "right_arm_states": torch.zeros(1, 1, 7, dtype=torch.float32),
        }
    )

    assert moved["rgb_head"].dtype == torch.bfloat16
    assert moved["right_arm_states"].dtype == torch.bfloat16


def test_predict_action_batch_stores_full_chunk_actions_and_zero_noise_logprobs():
    cfg = PsiPolicyConfig(action_horizon=4, num_action_chunks=2)
    policy = _make_minimal_policy(cfg)
    action_chunks = torch.arange(4 * cfg.action_dim, dtype=torch.float32).reshape(
        1, 4, cfg.action_dim
    )
    obs_feature = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    policy.preprocess_env_obs = lambda env_obs: env_obs
    policy.encode_obs = lambda obs: (torch.zeros(1, 1, 8), obs_feature)
    policy._sample_action_chunks = lambda cond_tokens, training: action_chunks
    policy._sample_masked_action_chunks = lambda mean, noise_std: mean
    policy._apply_reset_pose_mask = lambda action, env_obs: action
    policy.cfg.noise_std_rollout = 0.0

    env_obs = {
        "main_images": torch.zeros(1, 2, 2, 3),
        "states": torch.zeros(1, 28),
    }
    chunk_actions, result = policy.predict_action_batch(
        env_obs, return_obs=True, return_shared_feature=True
    )

    expected_chunk_actions = action_chunks[:, : cfg.num_action_chunks]
    assert torch.allclose(chunk_actions, expected_chunk_actions)
    assert torch.allclose(
        result["forward_inputs"]["action"],
        expected_chunk_actions.reshape(1, -1),
    )
    assert torch.allclose(
        result["forward_inputs"]["model_action"],
        expected_chunk_actions.reshape(1, -1),
    )
    assert result["prev_logprobs"].shape == (1, cfg.num_action_chunks, cfg.action_dim)
    assert torch.count_nonzero(result["prev_logprobs"]) == 0
    assert torch.allclose(result["shared_feature"], obs_feature)


def test_sync_runtime_cfg_from_checkpoint_updates_horizon_and_image_size():
    cfg = PsiPolicyConfig(obs_horizon=1, action_horizon=16, num_action_chunks=4, image_size=224)
    policy = _make_minimal_policy(cfg)
    policy.shape_meta = {
        "obs": {
            "h264_head": {"shape": [3, 128, 128], "type": "rgb", "horizon": 2},
            "h264_left_hand": {"shape": [3, 128, 128], "type": "rgb", "horizon": 2},
            "h264_right_hand": {"shape": [3, 128, 128], "type": "rgb", "horizon": 2},
        },
        "action": {"shape": [26], "horizon": 8},
    }

    policy._sync_runtime_cfg_from_checkpoint(
        SimpleNamespace(n_obs_steps=2, n_action_steps=8),
        SimpleNamespace(action_horizon=8),
    )

    assert policy.cfg.obs_horizon == 2
    assert policy.cfg.action_horizon == 8
    assert policy.cfg.image_size == 128


def test_default_forward_uses_recorded_actions_when_recomputing_logprobs():
    cfg = PsiPolicyConfig(action_horizon=4, num_action_chunks=2)
    policy = _make_minimal_policy(cfg)
    mean_chunks = torch.ones(1, 4, cfg.action_dim, dtype=torch.float32)

    policy.preprocess_env_obs = lambda env_obs: env_obs
    policy.encode_obs = lambda obs: (torch.zeros(1, 1, 8), torch.zeros(1, 8))
    policy._sample_action_chunks = lambda cond_tokens, training: mean_chunks
    policy.cfg.noise_std_rollout = 0.0

    forward_inputs = {
        "main_images": torch.zeros(1, 2, 2, 3),
        "states": torch.zeros(1, 28),
        "action": mean_chunks[:, : cfg.num_action_chunks].reshape(1, -1),
    }
    output = policy.default_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
    )

    assert output["logprobs"].shape == (1, cfg.num_action_chunks, cfg.action_dim)
    assert output["entropy"].shape == (1, cfg.num_action_chunks, cfg.action_dim)
    assert torch.count_nonzero(output["logprobs"]) == 0
