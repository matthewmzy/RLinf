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

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.psi_policy.psi_policy_for_rl import (
    PsiPolicyForRL,
    PsiPolicyConfig,
    _build_shape_meta,
    _build_rl_action_mask,
    _normalize_rl_action_mask_cfg,
)
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy
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


def _make_chunk_tensor(step_values: list[list[float]], action_dim: int) -> torch.Tensor:
    chunks = []
    for step in step_values:
        step_chunks = [
            torch.full((action_dim,), float(value), dtype=torch.float32)
            for value in step
        ]
        chunks.append(torch.stack(step_chunks, dim=0).reshape(-1))
    return torch.stack(chunks, dim=0).unsqueeze(1)


def _make_flag_tensor(step_flags: list[list[bool]], action_dim: int) -> torch.Tensor:
    chunks = []
    for step in step_flags:
        step_chunks = [
            torch.full((action_dim,), bool(flag), dtype=torch.bool) for flag in step
        ]
        chunks.append(torch.stack(step_chunks, dim=0).reshape(-1))
    return torch.stack(chunks, dim=0).unsqueeze(1)


def _make_psi_dagger_trajectory(cfg: PsiPolicyConfig) -> Trajectory:
    traj_len = 3
    image_size = cfg.image_size

    actions = _make_chunk_tensor([[0, 1], [10, 11], [20, 21]], cfg.action_dim)
    intervene_flags = _make_flag_tensor(
        [[True, True], [True, False], [True, True]],
        cfg.action_dim,
    )
    transition_valids = torch.tensor(
        [[[True, True]], [[True, True]], [[True, True]]], dtype=torch.bool
    )
    dones = torch.tensor([[[False, False]], [[False, False]], [[False, False]]], dtype=torch.bool)
    rewards = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], dtype=torch.float32)

    main_images = torch.stack(
        [
            torch.full((1, image_size, image_size, 3), fill_value=step_idx, dtype=torch.uint8)
            for step_idx in range(traj_len)
        ],
        dim=0,
    )
    extra_view_images = torch.stack(
        [
            torch.full(
                (1, 2, image_size, image_size, 3),
                fill_value=step_idx + 1,
                dtype=torch.uint8,
            )
            for step_idx in range(traj_len)
        ],
        dim=0,
    )
    states = torch.stack(
        [
            torch.full((1, 28), fill_value=float(step_idx), dtype=torch.float32)
            for step_idx in range(traj_len)
        ],
        dim=0,
    )
    reset_states = torch.stack(
        [
            torch.full((1, 28), fill_value=100.0 + step_idx, dtype=torch.float32)
            for step_idx in range(traj_len)
        ],
        dim=0,
    )
    model_action = torch.full_like(actions, fill_value=-99.0)

    curr_obs = {
        "rgb_head": main_images.clone(),
        "rgb_left_hand": torch.zeros(traj_len, 1, image_size, image_size, 3, dtype=torch.uint8),
        "rgb_right_hand": torch.zeros(traj_len, 1, image_size, image_size, 3, dtype=torch.uint8),
        "right_arm_states": torch.zeros(traj_len, 1, 7, dtype=torch.float32),
        "left_arm_states": torch.zeros(traj_len, 1, 7, dtype=torch.float32),
        "right_hand_states": torch.zeros(traj_len, 1, 6, dtype=torch.float32),
        "left_hand_states": torch.zeros(traj_len, 1, 6, dtype=torch.float32),
    }
    next_obs = {
        key: value + 1 if value.dtype != torch.bool else value.clone()
        for key, value in curr_obs.items()
    }

    return Trajectory(
        max_episode_length=200,
        model_weights_id="psi-test",
        actions=actions,
        intervene_flags=intervene_flags,
        transition_valids=transition_valids,
        rewards=rewards,
        dones=dones,
        forward_inputs={
            "main_images": main_images,
            "extra_view_images": extra_view_images,
            "states": states,
            "reset_states": reset_states,
            "model_action": model_action,
        },
        curr_obs=curr_obs,
        next_obs=next_obs,
    )


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


def test_preprocess_env_obs_derives_action_joint_obs_from_states():
    cfg = PsiPolicyConfig(obs_horizon=1, image_size=4)
    policy = _make_minimal_policy(cfg)
    policy.shape_meta = {
        "obs": {
            **policy.shape_meta["obs"],
            "action_arm_joints": {"shape": [14], "type": "low_dim", "horizon": 1},
            "action_left_hand_joints": {"shape": [6], "type": "low_dim", "horizon": 1},
            "action_right_hand_joints": {"shape": [6], "type": "low_dim", "horizon": 1},
        },
        "action": policy.shape_meta["action"],
    }

    states = torch.arange(28, dtype=torch.float32).unsqueeze(0)
    processed = policy.preprocess_env_obs(
        {
            "main_images": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(1, 2, 2, 2, 3, dtype=torch.uint8),
            "states": states,
        }
    )

    expected_arm = torch.cat([states[:, :7], states[:, 7:14]], dim=-1).unsqueeze(1)
    expected_left_hand = states[:, 14:20].unsqueeze(1)
    expected_right_hand = states[:, 20:26].unsqueeze(1)
    assert torch.allclose(processed["action_arm_joints"], expected_arm)
    assert torch.allclose(processed["action_left_hand_joints"], expected_left_hand)
    assert torch.allclose(processed["action_right_hand_joints"], expected_right_hand)


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


def test_predict_action_horizon_batch_returns_full_horizon_for_execution_adapter():
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

    rollout_prediction = policy.predict_action_horizon_batch(
        {
            "main_images": torch.zeros(1, 2, 2, 3),
            "states": torch.zeros(1, 28),
        },
        return_shared_feature=True,
    )

    assert torch.allclose(rollout_prediction["action_horizon"], action_chunks)
    assert torch.allclose(rollout_prediction["model_action_horizon"], action_chunks)
    assert torch.allclose(
        rollout_prediction["action_chunks_mean_norm"], action_chunks
    )
    assert torch.allclose(rollout_prediction["action_chunks_norm"], action_chunks)
    assert torch.allclose(rollout_prediction["shared_feature"], obs_feature)


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


def test_prepare_dagger_sft_batch_prefers_model_action_and_reshapes_flat_chunks():
    cfg = PsiPolicyConfig(obs_horizon=1, action_horizon=16, num_action_chunks=2, image_size=4)
    policy = _make_minimal_policy(cfg)

    model_action = torch.arange(2 * 2 * cfg.action_dim, dtype=torch.float32).reshape(2, -1)
    prepared = policy.prepare_dagger_sft_batch(
        {
            "main_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(2, 2, 4, 4, 3, dtype=torch.uint8),
            "states": torch.arange(2 * 28, dtype=torch.float32).reshape(2, 28),
            "action": torch.full((2, cfg.action_dim), -1.0),
            "model_action": model_action,
        }
    )

    assert prepared["action"].shape == (2, 2, cfg.action_dim)
    assert torch.allclose(
        prepared["action"],
        model_action.reshape(2, 2, cfg.action_dim),
    )
    assert prepared["obs"]["rgb_head"].shape == (2, 1, 3, 4, 4)
    assert prepared["obs"]["rgb_left_hand"].shape == (2, 1, 3, 4, 4)
    assert prepared["obs"]["rgb_right_hand"].shape == (2, 1, 3, 4, 4)


def test_prepare_dagger_sft_batch_accepts_shape_meta_named_obs_and_3d_actions():
    cfg = PsiPolicyConfig(obs_horizon=1, action_horizon=16, num_action_chunks=2, image_size=4)
    policy = _make_minimal_policy(cfg)

    action_chunks = torch.arange(2 * cfg.action_dim, dtype=torch.float32).reshape(
        1, 2, cfg.action_dim
    )
    prepared = policy.prepare_dagger_sft_batch(
        {
            "rgb_head": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
            "rgb_left_hand": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
            "rgb_right_hand": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
            "right_arm_states": torch.zeros(1, 7),
            "left_arm_states": torch.zeros(1, 7),
            "right_hand_states": torch.zeros(1, 6),
            "left_hand_states": torch.zeros(1, 6),
            "action": action_chunks,
        }
    )

    assert prepared["action"].shape == (1, 2, cfg.action_dim)
    assert torch.allclose(prepared["action"], action_chunks)
    assert prepared["obs"]["rgb_head"].shape == (1, 1, 3, 4, 4)
    assert prepared["obs"]["right_arm_states"].shape == (1, 1, 7)


def test_forward_sft_dispatches_to_underlying_psi_policy_loss():
    cfg = PsiPolicyConfig()
    policy = _make_minimal_policy(cfg)

    class _RecordingPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, batch, training=True):
            self.calls.append((batch, training))
            return torch.tensor(1.25)

    recording_policy = _RecordingPolicy()
    policy.policy = recording_policy
    policy.train()

    data = {
        "obs": {"rgb_head": torch.zeros(1, 1, 3, 4, 4)},
        "action": torch.zeros(1, 1, cfg.action_dim),
    }
    output = policy.forward(forward_type=ForwardType.SFT, data=data)

    assert output.item() == pytest.approx(1.25)
    assert len(recording_policy.calls) == 1
    recorded_batch, recorded_training = recording_policy.calls[0]
    assert set(recorded_batch.keys()) == {"obs", "action"}
    assert set(recorded_batch["obs"].keys()) == {"rgb_head"}
    assert torch.equal(recorded_batch["obs"]["rgb_head"], data["obs"]["rgb_head"])
    assert torch.equal(recorded_batch["action"], data["action"])
    assert recorded_training is True


def test_prepare_dagger_replay_trajectories_builds_future_windows_from_executed_actions():
    cfg = PsiPolicyConfig(obs_horizon=1, action_horizon=4, num_action_chunks=2, image_size=4)
    policy = _make_minimal_policy(cfg)
    trajectory = _make_psi_dagger_trajectory(cfg)

    prepared = policy.prepare_dagger_replay_trajectories(trajectory)

    assert prepared is not None
    assert len(prepared) == 1
    replay_traj = prepared[0]
    assert "model_action" not in replay_traj.forward_inputs
    assert replay_traj.forward_inputs["action"].shape == (1, 1, 4, cfg.action_dim)
    assert replay_traj.actions.shape == (1, 1, 4, cfg.action_dim)
    assert torch.equal(
        replay_traj.forward_inputs["action"][0, 0, :, 0],
        torch.tensor([0.0, 1.0, 10.0, 11.0]),
    )
    assert torch.equal(
        replay_traj.forward_inputs["main_images"][0, 0],
        trajectory.forward_inputs["main_images"][0, 0],
    )
    assert torch.equal(
        replay_traj.forward_inputs["states"][0, 0],
        trajectory.forward_inputs["states"][0, 0],
    )
    assert replay_traj.rewards.shape == (1, 1)
    assert replay_traj.rewards[0, 0].item() == pytest.approx(3.0)


def test_prepare_dagger_replay_trajectories_drops_partial_and_short_tail_anchors():
    cfg = PsiPolicyConfig(obs_horizon=1, action_horizon=4, num_action_chunks=2, image_size=4)
    policy = _make_minimal_policy(cfg)
    trajectory = _make_psi_dagger_trajectory(cfg)

    prepared = policy.prepare_dagger_replay_trajectories(trajectory)

    assert prepared is not None
    replay_traj = prepared[0]
    assert replay_traj.forward_inputs["action"].shape[0] == 1
    assert torch.equal(
        replay_traj.forward_inputs["action"][0, 0, :, 0],
        torch.tensor([0.0, 1.0, 10.0, 11.0]),
    )


class _FallbackDaggerPolicy(BasePolicy):
    def default_forward(self, **kwargs):
        raise NotImplementedError

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError


def test_dagger_worker_uses_policy_replay_hook_and_default_fallback():
    cfg = PsiPolicyConfig(obs_horizon=1, action_horizon=4, num_action_chunks=2, image_size=4)
    psi_policy = _make_minimal_policy(cfg)
    psi_trajectory = _make_psi_dagger_trajectory(cfg)
    worker = object.__new__(EmbodiedDAGGERFSDPPolicy)
    worker.model = psi_policy

    psi_prepared = worker._prepare_replay_trajectories(psi_trajectory)

    assert len(psi_prepared) == 1
    assert psi_prepared[0].forward_inputs["action"].shape == (1, 1, 4, cfg.action_dim)

    fallback_policy = _FallbackDaggerPolicy()
    worker.model = fallback_policy
    fallback_trajectory = Trajectory(
        max_episode_length=8,
        model_weights_id="fallback",
        actions=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
        intervene_flags=torch.tensor([[[True, True]]], dtype=torch.bool),
        rewards=torch.tensor([[[1.0]]], dtype=torch.float32),
        transition_valids=torch.tensor([[[True]]], dtype=torch.bool),
        forward_inputs={"action": torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)},
    )

    fallback_prepared = worker._prepare_replay_trajectories(fallback_trajectory)

    assert len(fallback_prepared) == 1
    assert torch.equal(
        fallback_prepared[0].forward_inputs["action"],
        fallback_trajectory.extract_intervene_traj(mode="all")[0].forward_inputs["action"],
    )
