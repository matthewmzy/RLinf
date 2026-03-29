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

"""Psi-policy wrapper with RLinf SAC interfaces."""

import copy
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import dill
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiQHead

logger = logging.getLogger(__name__)

_PSI_POLICY_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../psi-policy")
)
if _PSI_POLICY_ROOT not in sys.path:
    sys.path.insert(0, _PSI_POLICY_ROOT)

from psi_policy.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from psi_policy.model.diffusion.transformer_for_action import TransformerForAction  # noqa: E402
from psi_policy.policy.psi_policy import PsiPolicy  # noqa: E402


@dataclass
class PsiPolicyConfig:
    action_dim: int = 26
    action_horizon: int = 16
    n_layer: int = 6
    n_head: int = 4
    p_drop_attn: float = 0.1
    use_attn_mask: bool = False

    add_q_head: bool = True
    num_q_heads: int = 2
    q_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])

    noise_std_train: float = 0.3
    noise_std_rollout: float = 0.02
    num_action_chunks: int = 4

    image_size: int = 224
    obs_horizon: int = 1
    fallback_hidden_size: int = 256

    state_split: dict[str, list[int]] = field(
        default_factory=lambda: {
            "right_arm": [0, 7],
            "left_arm": [7, 14],
            "left_hand": [14, 20],
            "right_hand": [20, 26],
        }
    )

    model_path: Optional[str] = None
    normalizer_path: Optional[str] = None
    checkpoint_model_key: str = "model"
    precision: str = "bfloat16"
    is_lora: bool = False
    lora_rank: int = 32
    model_type: str = "psi_policy"
    sharding_strategy: str = "no_shard"

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        self.action_dim = int(self.action_dim)
        self.action_horizon = int(self.action_horizon)
        self.num_action_chunks = int(self.num_action_chunks)
        self.num_q_heads = int(self.num_q_heads)
        self.obs_horizon = int(self.obs_horizon)
        self.fallback_hidden_size = int(self.fallback_hidden_size)
        if self.action_dim != 26:
            raise ValueError(
                f"PsiPolicyForRL currently only supports 26-dim rgb_state policy, got {self.action_dim}."
            )
        if not 1 <= self.num_action_chunks <= self.action_horizon:
            raise ValueError(
                "num_action_chunks must be in [1, action_horizon] for psi-policy."
            )
        if self.checkpoint_model_key not in {"model", "ema_model"}:
            raise ValueError(
                "checkpoint_model_key must be either 'model' or 'ema_model'."
            )


def _build_shape_meta(cfg: PsiPolicyConfig) -> dict[str, Any]:
    ss = cfg.state_split
    return {
        "obs": {
            "rgb_head": {
                "shape": [3, cfg.image_size, cfg.image_size],
                "type": "rgb",
                "horizon": cfg.obs_horizon,
            },
            "rgb_left_hand": {
                "shape": [3, cfg.image_size, cfg.image_size],
                "type": "rgb",
                "horizon": cfg.obs_horizon,
            },
            "rgb_right_hand": {
                "shape": [3, cfg.image_size, cfg.image_size],
                "type": "rgb",
                "horizon": cfg.obs_horizon,
            },
            "right_arm_states": {
                "shape": [ss["right_arm"][1] - ss["right_arm"][0]],
                "type": "low_dim",
                "horizon": cfg.obs_horizon,
            },
            "left_arm_states": {
                "shape": [ss["left_arm"][1] - ss["left_arm"][0]],
                "type": "low_dim",
                "horizon": cfg.obs_horizon,
            },
            "right_hand_states": {
                "shape": [ss["right_hand"][1] - ss["right_hand"][0]],
                "type": "low_dim",
                "horizon": cfg.obs_horizon,
            },
            "left_hand_states": {
                "shape": [ss["left_hand"][1] - ss["left_hand"][0]],
                "type": "low_dim",
                "horizon": cfg.obs_horizon,
            },
        },
        "action": {
            "shape": [cfg.action_dim],
            "horizon": cfg.action_horizon,
        },
    }


class SimplePsiObsEncoder(nn.Module):
    """Checkpoint-free fallback encoder for tests and local smoke runs."""

    def __init__(self, shape_meta: dict[str, Any], hidden_size: int):
        super().__init__()
        self.shape_meta = shape_meta
        self.hidden_size = hidden_size
        state_dim = 0
        for key, meta in shape_meta["obs"].items():
            if meta["type"] == "low_dim":
                state_dim += int(meta["shape"][0])
        input_dim = 3 * 3 + state_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def output_shape(self):
        return (1, 1, self.hidden_size), [1]

    def _encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] -> [B, 3]
        return tensor.mean(dim=(-1, -2)).mean(dim=1)

    def forward(self, obs_dict, training=True):
        del training
        features = [
            self._encode_image(obs_dict["rgb_head"]),
            self._encode_image(obs_dict["rgb_left_hand"]),
            self._encode_image(obs_dict["rgb_right_hand"]),
            obs_dict["right_arm_states"].mean(dim=1),
            obs_dict["left_arm_states"].mean(dim=1),
            obs_dict["right_hand_states"].mean(dim=1),
            obs_dict["left_hand_states"].mean(dim=1),
        ]
        obs_feature = torch.cat(features, dim=-1)
        return self.proj(obs_feature).unsqueeze(1)


def _create_identity_normalizer(shape_meta: dict[str, Any], action_dim: int) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    for key, meta in shape_meta["obs"].items():
        size = int(meta["shape"][0])
        normalizer[key] = SingleFieldLinearNormalizer.create_identity().to(torch.float32)
        normalizer[key].params_dict["scale"] = nn.Parameter(
            torch.ones(size, dtype=torch.float32), requires_grad=False
        )
        normalizer[key].params_dict["offset"] = nn.Parameter(
            torch.zeros(size, dtype=torch.float32), requires_grad=False
        )
        normalizer[key].params_dict["input_stats"]["min"] = nn.Parameter(
            -torch.ones(size, dtype=torch.float32), requires_grad=False
        )
        normalizer[key].params_dict["input_stats"]["max"] = nn.Parameter(
            torch.ones(size, dtype=torch.float32), requires_grad=False
        )
        normalizer[key].params_dict["input_stats"]["mean"] = nn.Parameter(
            torch.zeros(size, dtype=torch.float32), requires_grad=False
        )
        normalizer[key].params_dict["input_stats"]["std"] = nn.Parameter(
            torch.ones(size, dtype=torch.float32), requires_grad=False
        )
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity().to(torch.float32)
    normalizer["action"].params_dict["scale"] = nn.Parameter(
        torch.ones(action_dim, dtype=torch.float32), requires_grad=False
    )
    normalizer["action"].params_dict["offset"] = nn.Parameter(
        torch.zeros(action_dim, dtype=torch.float32), requires_grad=False
    )
    normalizer["action"].params_dict["input_stats"]["min"] = nn.Parameter(
        -torch.ones(action_dim, dtype=torch.float32), requires_grad=False
    )
    normalizer["action"].params_dict["input_stats"]["max"] = nn.Parameter(
        torch.ones(action_dim, dtype=torch.float32), requires_grad=False
    )
    normalizer["action"].params_dict["input_stats"]["mean"] = nn.Parameter(
        torch.zeros(action_dim, dtype=torch.float32), requires_grad=False
    )
    normalizer["action"].params_dict["input_stats"]["std"] = nn.Parameter(
        torch.ones(action_dim, dtype=torch.float32), requires_grad=False
    )
    return normalizer


class PsiPolicyForRL(nn.Module, BasePolicy):
    """RLinf wrapper around psi-policy with SAC actor/critic APIs."""

    def __init__(self, cfg: PsiPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = _build_shape_meta(cfg)

        self.policy = self._build_or_load_policy()
        self.obs_encoder = self.policy.obs_encoder
        self.model = self.policy.model
        self.normalizer = self.policy.normalizer
        self.action_dim = int(self.policy.action_dim)
        self.action_horizon = int(self.policy.action_horizon)

        obs_shape, _ = self.obs_encoder.output_shape()
        self.n_emb = int(obs_shape[-1])

        if cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.n_emb,
                action_feature_dim=self.action_dim,
                hidden_dims=cfg.q_hidden_dims,
                num_q_heads=cfg.num_q_heads,
            )

        total_params = sum(p.numel() for p in self.parameters())
        q_params = sum(p.numel() for p in self.q_head.parameters()) if cfg.add_q_head else 0
        logger.info(
            "PsiPolicyForRL ready: total=%.1fM, q_head=%.1fM, action_horizon=%s, num_action_chunks=%s",
            total_params / 1e6,
            q_params / 1e6,
            self.action_horizon,
            self.cfg.num_action_chunks,
        )

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_or_load_policy(self) -> PsiPolicy:
        if self.cfg.model_path:
            return self._load_workspace_policy(self.cfg.model_path)
        return self._build_fallback_policy()

    def _build_fallback_policy(self) -> PsiPolicy:
        logger.warning(
            "Psi-policy checkpoint is not provided; using lightweight fallback policy for smoke tests."
        )
        obs_encoder = SimplePsiObsEncoder(
            shape_meta=self.shape_meta,
            hidden_size=self.cfg.fallback_hidden_size,
        )
        policy = PsiPolicy(
            shape_meta=self.shape_meta,
            obs_encoder=obs_encoder,
            n_layer=self.cfg.n_layer,
            n_head=self.cfg.n_head,
            p_drop_attn=self.cfg.p_drop_attn,
            use_attn_mask=self.cfg.use_attn_mask,
        )
        policy.normalizer = _create_identity_normalizer(
            self.shape_meta, action_dim=self.cfg.action_dim
        )
        return policy

    def _load_workspace_policy(self, ckpt_path: str) -> PsiPolicy:
        logger.info("Loading psi-policy checkpoint via workspace payload: %s", ckpt_path)
        with open(ckpt_path, "rb") as fp:
            payload = torch.load(fp, map_location="cpu", pickle_module=dill, weights_only=False)

        cfg = copy.deepcopy(payload["cfg"])
        cls = hydra.utils.get_class(cfg._target_)
        cfg.optimizer.start_ckpt_path = None
        if hasattr(cfg.optimizer, "start_normalizer_path"):
            cfg.optimizer.start_normalizer_path = None
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        if (
            self.cfg.checkpoint_model_key == "ema_model"
            and getattr(workspace, "ema_model", None) is not None
        ):
            policy = workspace.ema_model
        else:
            policy = workspace.model

        if self.cfg.normalizer_path:
            self._override_policy_normalizer(policy, self.cfg.normalizer_path)

        if int(policy.action_dim) != self.cfg.action_dim:
            raise ValueError(
                f"Checkpoint action_dim={policy.action_dim} does not match config action_dim={self.cfg.action_dim}."
            )

        self.shape_meta = cfg.shape_meta
        logger.info(
            "Loaded psi-policy checkpoint with obs_horizon=%s, action_horizon=%s, encoder_target=%s",
            cfg.n_obs_steps,
            cfg.n_action_steps,
            cfg.policy.obs_encoder._target_,
        )
        return policy

    def _override_policy_normalizer(self, policy: PsiPolicy, normalizer_path: str) -> None:
        logger.info("Overriding psi-policy normalizer from: %s", normalizer_path)
        with open(normalizer_path, "rb") as fp:
            normalizer_state = pickle.load(fp)
        if isinstance(normalizer_state, LinearNormalizer):
            policy.normalizer = normalizer_state
            return
        if isinstance(normalizer_state, dict) and "params_dict" in normalizer_state:
            policy.normalizer.load_state_dict(normalizer_state)
            return
        raise TypeError(
            f"Unsupported normalizer payload in {normalizer_path}: {type(normalizer_state)}"
        )

    def preprocess_env_obs(self, env_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.device

        def _to_btchw(image: torch.Tensor) -> torch.Tensor:
            if image.ndim == 4:
                image = image.unsqueeze(1)
            if image.shape[-1] == 3:
                image = image.permute(0, 1, 4, 2, 3)
            image = image.to(device=device, dtype=torch.float32)
            if image.max() > 1.0:
                image = image / 255.0
            target_size = tuple(self.shape_meta["obs"]["rgb_head"]["shape"][1:])
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
            image = F.interpolate(
                image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            image = image * 2.0 - 1.0
            return image.reshape(
                -1,
                self.cfg.obs_horizon,
                self.shape_meta["obs"]["rgb_head"]["shape"][0],
                target_size[0],
                target_size[1],
            )

        processed = {"rgb_head": _to_btchw(env_obs["main_images"])}
        extra_view_images = env_obs.get("extra_view_images")
        if extra_view_images is not None:
            processed["rgb_left_hand"] = _to_btchw(extra_view_images[:, 0])
            processed["rgb_right_hand"] = _to_btchw(extra_view_images[:, 1])
        else:
            processed["rgb_left_hand"] = processed["rgb_head"].clone()
            processed["rgb_right_hand"] = processed["rgb_head"].clone()

        states = env_obs["states"].to(device=device, dtype=torch.float32)
        if states.shape[-1] < self.cfg.action_dim:
            raise ValueError(
                f"psi-policy expects at least {self.cfg.action_dim} state dims from A2D, got {states.shape[-1]}."
            )
        ss = self.cfg.state_split
        processed["right_arm_states"] = states[:, ss["right_arm"][0] : ss["right_arm"][1]].unsqueeze(1)
        processed["left_arm_states"] = states[:, ss["left_arm"][0] : ss["left_arm"][1]].unsqueeze(1)
        processed["left_hand_states"] = states[:, ss["left_hand"][0] : ss["left_hand"][1]].unsqueeze(1)
        processed["right_hand_states"] = states[:, ss["right_hand"][0] : ss["right_hand"][1]].unsqueeze(1)
        return processed

    def encode_obs(self, obs: dict[str, torch.Tensor], detach_encoder: bool = False):
        nobs = self.normalizer.normalize(obs)
        cond_tokens = self.obs_encoder(nobs, training=self.training)
        obs_feature = cond_tokens.mean(dim=1)
        if detach_encoder:
            cond_tokens = cond_tokens.detach()
            obs_feature = obs_feature.detach()
        return cond_tokens, obs_feature

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"ForwardType {forward_type} not supported")

    def _sample_action_chunks(self, cond_tokens: torch.Tensor, training: bool):
        batch_size = cond_tokens.shape[0]
        z = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=cond_tokens.device,
            dtype=cond_tokens.dtype,
        )
        return self.model(
            z,
            cond=cond_tokens,
            training=training,
            gen_attn_map=False,
        )[0]

    def sac_forward(self, obs, **kwargs):
        del kwargs
        cond_tokens, obs_feature = self.encode_obs(obs)
        action_chunks_mean_norm = self._sample_action_chunks(cond_tokens, training=True)
        action_mean_norm = action_chunks_mean_norm[:, 0, :]
        noise_std = self.cfg.noise_std_train
        eps = torch.randn_like(action_mean_norm)
        action_norm = action_mean_norm + noise_std * eps
        log_prob = (
            Normal(action_mean_norm.detach(), noise_std)
            .log_prob(action_norm.detach())
            .sum(dim=-1)
        )
        action = self.normalizer["action"].unnormalize(action_norm.unsqueeze(1)).squeeze(1)
        return action, log_prob, obs_feature

    def sac_q_forward(
        self, obs, actions, shared_feature=None, detach_encoder=False, **kwargs
    ):
        del kwargs
        if shared_feature is None:
            _, obs_feature = self.encode_obs(obs, detach_encoder=detach_encoder)
        else:
            obs_feature = shared_feature.detach() if detach_encoder else shared_feature
        return self.q_head(obs_feature, actions)

    def default_forward(self, forward_inputs=None, **kwargs):
        del kwargs
        if forward_inputs is None:
            raise ValueError("forward_inputs is required for psi-policy rollout forward.")
        obs = {
            "main_images": forward_inputs["main_images"],
            "states": forward_inputs["states"],
        }
        if "extra_view_images" in forward_inputs:
            obs["extra_view_images"] = forward_inputs["extra_view_images"]
        obs = self.preprocess_env_obs(obs)
        cond_tokens, _ = self.encode_obs(obs)
        action_chunks_mean_norm = self._sample_action_chunks(cond_tokens, training=False)
        noise_std = self.cfg.noise_std_rollout
        action_chunks_norm = action_chunks_mean_norm + noise_std * torch.randn_like(
            action_chunks_mean_norm
        )
        action_norm = action_chunks_norm[:, 0, :]
        action = self.normalizer["action"].unnormalize(action_norm.unsqueeze(1)).squeeze(1)
        log_prob = (
            Normal(action_chunks_mean_norm[:, 0, :].detach(), noise_std)
            .log_prob(action_norm.detach())
            .sum(dim=-1)
        )
        return {"action": action, "log_prob": log_prob}

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        return_shared_feature=False,
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        processed_obs = self.preprocess_env_obs(env_obs)
        cond_tokens, obs_feature = self.encode_obs(processed_obs)
        action_chunks_mean_norm = self._sample_action_chunks(cond_tokens, training=False)
        noise_std = self.cfg.noise_std_rollout
        action_chunks_norm = action_chunks_mean_norm + noise_std * torch.randn_like(
            action_chunks_mean_norm
        )
        chunks_norm = action_chunks_norm[:, : self.cfg.num_action_chunks, :]
        chunk_actions = self.normalizer["action"].unnormalize(chunks_norm)
        log_prob = (
            Normal(action_chunks_mean_norm[:, 0, :].detach(), noise_std)
            .log_prob(chunks_norm[:, 0, :].detach())
            .sum(dim=-1)
        )
        chunk_values = torch.zeros_like(log_prob[..., :1])
        forward_inputs = {"action": chunk_actions[:, 0, :]}
        if return_obs:
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["states"] = env_obs["states"]
            if "extra_view_images" in env_obs:
                forward_inputs["extra_view_images"] = env_obs["extra_view_images"]
        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = obs_feature
        return chunk_actions, result
