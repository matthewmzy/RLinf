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

from rlinf.models.embodiment.psi_policy.psi_policy_for_rl import (
    PsiPolicyForRL,
    PsiPolicyConfig,
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
