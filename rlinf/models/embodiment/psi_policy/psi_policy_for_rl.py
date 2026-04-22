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
import contextlib
import logging
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import dill
import hydra
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiQHead

logger = logging.getLogger(__name__)

def _resolve_psi_policy_root() -> str:
    current_file = Path(__file__).resolve()
    candidates: list[Path] = []

    env_override = os.environ.get("PSI_POLICY_ROOT")
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())

    candidates.extend(
        [
            current_file.parents[4] / "psi-policy",
            current_file.parents[5] / "psi-policy",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


_PSI_POLICY_ROOT = _resolve_psi_policy_root()
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

    noise_std_train: float = 0.05
    noise_std_rollout: float = 0.005
    num_action_chunks: int = 4
    use_chunk_rl: bool = False
    rollout_rtc_enabled: bool = False
    rollout_rtc_execute_step: Optional[int] = None
    rollout_rtc_merge_weight_base: float = 0.8

    image_size: int = 224
    obs_horizon: int = 1
    fallback_hidden_size: int = 256
    train_obs_encoder: bool = False

    state_split: dict[str, list[int]] = field(
        default_factory=lambda: {
            "right_arm": [7, 14],
            "left_arm": [0, 7],
            "left_hand": [14, 20],
            "right_hand": [20, 26],
        }
    )
    rl_action_mask: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "active_groups": ["all"],
            "active_indices": [],
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
        self.use_chunk_rl = bool(self.use_chunk_rl)
        self.rollout_rtc_enabled = bool(self.rollout_rtc_enabled)
        if self.rollout_rtc_execute_step is None:
            self.rollout_rtc_execute_step = self.num_action_chunks
        self.rollout_rtc_execute_step = int(self.rollout_rtc_execute_step)
        self.rollout_rtc_merge_weight_base = float(self.rollout_rtc_merge_weight_base)
        self.num_q_heads = int(self.num_q_heads)
        self.obs_horizon = int(self.obs_horizon)
        self.fallback_hidden_size = int(self.fallback_hidden_size)
        self.rl_action_mask = _normalize_rl_action_mask_cfg(self.rl_action_mask)
        if self.action_dim != 26:
            raise ValueError(
                f"PsiPolicyForRL currently only supports 26-dim rgb_state policy, got {self.action_dim}."
            )
        if not 1 <= self.num_action_chunks <= self.action_horizon:
            raise ValueError(
                "num_action_chunks must be in [1, action_horizon] for psi-policy."
            )
        if self.rollout_rtc_execute_step <= 0:
            raise ValueError("rollout_rtc_execute_step must be > 0.")
        if self.rollout_rtc_enabled and self.rollout_rtc_execute_step != self.num_action_chunks:
            raise ValueError(
                "rollout_rtc_enabled requires rollout_rtc_execute_step to match "
                "num_action_chunks so RLinf pops the same number of actions per inference "
                "as psi-policy client execute_step."
            )
        if self.rollout_rtc_merge_weight_base <= 0:
            raise ValueError("rollout_rtc_merge_weight_base must be > 0.")
        if self.checkpoint_model_key not in {"model", "ema_model"}:
            raise ValueError(
                "checkpoint_model_key must be either 'model' or 'ema_model'."
            )


def _normalize_rl_action_mask_cfg(mask_cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    default_cfg = {
        "enabled": False,
        "active_groups": ["all"],
        "active_indices": [],
    }
    if mask_cfg is None:
        return default_cfg

    if hasattr(mask_cfg, "items"):
        raw_cfg = dict(mask_cfg)
    else:
        raise TypeError(
            f"rl_action_mask must be a mapping or None, got {type(mask_cfg)}"
        )

    cfg = default_cfg.copy()
    cfg.update(raw_cfg)
    cfg["enabled"] = bool(cfg.get("enabled", False))

    active_groups = cfg.get("active_groups", ["all"])
    if active_groups is None:
        active_groups = []
    elif isinstance(active_groups, str):
        active_groups = [active_groups]
    else:
        active_groups = list(active_groups)
    cfg["active_groups"] = [str(group) for group in active_groups]

    active_indices = cfg.get("active_indices", [])
    if active_indices is None:
        active_indices = []
    elif isinstance(active_indices, int):
        active_indices = [active_indices]
    else:
        active_indices = list(active_indices)
    cfg["active_indices"] = [int(index) for index in active_indices]
    return cfg


def _build_rl_action_mask(
    action_dim: int, state_split: dict[str, list[int]], mask_cfg: dict[str, Any]
) -> tuple[torch.Tensor, list[int]]:
    if not mask_cfg.get("enabled", False):
        all_indices = list(range(action_dim))
        return torch.ones(action_dim, dtype=torch.bool), all_indices

    group_ranges = {
        "all": [(0, action_dim)],
        "arms": [tuple(state_split["left_arm"]), tuple(state_split["right_arm"])],
        "hands": [tuple(state_split["left_hand"]), tuple(state_split["right_hand"])],
        "left_arm": [tuple(state_split["left_arm"])],
        "right_arm": [tuple(state_split["right_arm"])],
        "left_hand": [tuple(state_split["left_hand"])],
        "right_hand": [tuple(state_split["right_hand"])],
    }
    mask = torch.zeros(action_dim, dtype=torch.bool)

    for group_name in mask_cfg.get("active_groups", []):
        if group_name not in group_ranges:
            valid_names = ", ".join(sorted(group_ranges))
            raise ValueError(
                f"Unsupported rl_action_mask active group '{group_name}'. "
                f"Valid groups: {valid_names}"
            )
        for start, end in group_ranges[group_name]:
            mask[start:end] = True

    for index in mask_cfg.get("active_indices", []):
        if not 0 <= index < action_dim:
            raise ValueError(
                f"rl_action_mask index {index} is out of range for action_dim={action_dim}."
            )
        mask[index] = True

    active_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
    if len(active_indices) == 0:
        raise ValueError(
            "rl_action_mask enabled=True but no action dimensions were selected."
        )
    return mask, active_indices


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


@contextlib.contextmanager
def _disable_timm_pretrained_download():
    original_create_model = timm.create_model

    def _create_model_without_remote_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = _create_model_without_remote_pretrained
    try:
        yield
    finally:
        timm.create_model = original_create_model


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
        self.obs_encoder_target: Optional[str] = None

        self.policy = self._build_or_load_policy()
        self.obs_encoder = self.policy.obs_encoder
        self.model = self.policy.model
        self.normalizer = self.policy.normalizer
        if not self.cfg.train_obs_encoder:
            self.obs_encoder.requires_grad_(False)
        self.action_dim = int(self.policy.action_dim)
        self.action_horizon = int(self.policy.action_horizon)
        self._rollout_rtc_states: list[_RolloutRTCState] = []
        rl_action_mask, active_rl_action_indices = _build_rl_action_mask(
            self.action_dim, self.cfg.state_split, self.cfg.rl_action_mask
        )
        self.register_buffer("rl_action_mask", rl_action_mask, persistent=False)
        self.active_rl_action_indices = active_rl_action_indices
        self.active_rl_action_dim = len(active_rl_action_indices)
        self.sac_action_feature_dim = self.action_dim * (
            self.cfg.num_action_chunks if self.cfg.use_chunk_rl else 1
        )

        obs_shape, _ = self.obs_encoder.output_shape()
        self.n_emb = int(obs_shape[-1])

        if cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.n_emb,
                action_feature_dim=self.sac_action_feature_dim,
                hidden_dims=cfg.q_hidden_dims,
                num_q_heads=cfg.num_q_heads,
            )

        total_params = sum(p.numel() for p in self.parameters())
        q_params = sum(p.numel() for p in self.q_head.parameters()) if cfg.add_q_head else 0
        logger.info(
            "PsiPolicyForRL ready: total=%.1fM, q_head=%.1fM, action_horizon=%s, "
            "num_action_chunks=%s, rl_action_dims=%s/%s, rl_action_indices=%s",
            total_params / 1e6,
            q_params / 1e6,
            self.action_horizon,
            self.cfg.num_action_chunks,
            self.active_rl_action_dim,
            self.action_dim,
            self.active_rl_action_indices,
        )
        logger.info(
            "PsiPolicyForRL obs_encoder training is %s.",
            "enabled" if self.cfg.train_obs_encoder else "disabled (frozen)",
        )
        if self.cfg.use_chunk_rl:
            logger.info(
                "PsiPolicyForRL SAC chunk-RL enabled: critic_action_dim=%s (%s x %s).",
                self.sac_action_feature_dim,
                self.cfg.num_action_chunks,
                self.action_dim,
            )
        if self.cfg.rollout_rtc_enabled:
            logger.info(
                "PsiPolicyForRL rollout RTC reproduction enabled: execute_step=%s, "
                "merge_weight_base=%.3f.",
                self.cfg.rollout_rtc_execute_step,
                self.cfg.rollout_rtc_merge_weight_base,
            )

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_obs_encoder_device(self) -> torch.device:
        if not hasattr(self, "obs_encoder"):
            return self.device
        try:
            return next(self.obs_encoder.parameters()).device
        except StopIteration:
            return self.device

    def _get_obs_encoder_dtype(self) -> torch.dtype:
        if not hasattr(self, "obs_encoder"):
            return next(self.parameters()).dtype
        try:
            return next(self.obs_encoder.parameters()).dtype
        except StopIteration:
            return next(self.parameters()).dtype

    def _move_obs_dict_to_encoder_device(
        self, obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        obs_device = self._get_obs_encoder_device()
        obs_dtype = self._get_obs_encoder_dtype()
        return {
            key: value.to(device=obs_device, dtype=obs_dtype).contiguous()
            for key, value in obs.items()
        }

    def get_sac_target_entropy(self) -> float:
        entropy_dim = self.active_rl_action_dim
        if self.cfg.use_chunk_rl:
            entropy_dim *= self.cfg.num_action_chunks
        return -float(entropy_dim)

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
        self.obs_encoder_target = (
            f"{type(obs_encoder).__module__}.{type(obs_encoder).__name__}"
        )
        return policy

    def _load_workspace_policy(self, ckpt_path: str) -> PsiPolicy:
        logger.info("Loading psi-policy checkpoint via workspace payload: %s", ckpt_path)
        with open(ckpt_path, "rb") as fp:
            payload = torch.load(fp, map_location="cpu", pickle_module=dill, weights_only=False)

        cfg = copy.deepcopy(payload["cfg"])
        policy_state_key = self.cfg.checkpoint_model_key
        if policy_state_key not in payload["state_dicts"]:
            available_keys = sorted(payload["state_dicts"].keys())
            if policy_state_key == "ema_model" and "model" in payload["state_dicts"]:
                logger.warning(
                    "psi-policy checkpoint does not contain '%s'; falling back to 'model'. "
                    "Available state_dicts=%s",
                    policy_state_key,
                    available_keys,
                )
                policy_state_key = "model"
            else:
                raise KeyError(
                    f"psi-policy checkpoint missing state_dict '{policy_state_key}'. "
                    f"Available state_dicts: {available_keys}"
                )

        # Instantiate only the policy graph instead of the full training workspace.
        # This avoids importing training-only modules that may drift from the runtime env.
        with _disable_timm_pretrained_download():
            policy = hydra.utils.instantiate(copy.deepcopy(cfg.policy))
        load_result = policy.load_state_dict(payload["state_dicts"][policy_state_key], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "psi-policy checkpoint load is not strict: "
                f"missing_keys={load_result.missing_keys}, "
                f"unexpected_keys={load_result.unexpected_keys}"
            )

        if self.cfg.normalizer_path:
            self._override_policy_normalizer(policy, self.cfg.normalizer_path)

        if int(policy.action_dim) != self.cfg.action_dim:
            raise ValueError(
                f"Checkpoint action_dim={policy.action_dim} does not match config action_dim={self.cfg.action_dim}."
            )

        self.shape_meta = copy.deepcopy(cfg.shape_meta)
        self._sync_runtime_cfg_from_checkpoint(cfg, policy)
        self.obs_encoder_target = getattr(
            cfg.policy.obs_encoder,
            "_target_",
            f"{type(policy.obs_encoder).__module__}.{type(policy.obs_encoder).__name__}",
        )
        logger.info(
            "Loaded psi-policy checkpoint with obs_horizon=%s, action_horizon=%s, "
            "encoder_target=%s, state_dict_key=%s",
            cfg.n_obs_steps,
            cfg.n_action_steps,
            self.obs_encoder_target,
            policy_state_key,
        )
        return policy

    def _sync_runtime_cfg_from_checkpoint(self, workspace_cfg, policy: PsiPolicy) -> None:
        self.cfg.obs_horizon = int(getattr(workspace_cfg, "n_obs_steps", self.cfg.obs_horizon))
        self.cfg.action_horizon = int(
            getattr(workspace_cfg, "n_action_steps", policy.action_horizon)
        )

        obs_meta = self.shape_meta.get("obs", {})
        image_key = next(
            (
                key
                for key in ("rgb_head", "h264_head", "rgb_left_hand", "h264_left_hand")
                if key in obs_meta
            ),
            None,
        )
        if image_key is not None:
            image_shape = obs_meta[image_key].get("shape", [])
            if len(image_shape) >= 3:
                self.cfg.image_size = int(image_shape[1])

        self.cfg._update_info()

    def _get_runtime_obs_horizon(self) -> int:
        obs_meta = self.shape_meta.get("obs", {})
        for key in (
            "rgb_head",
            "h264_head",
            "rgb_left_hand",
            "h264_left_hand",
            "right_arm_states",
            "left_arm_states",
        ):
            if key in obs_meta and "horizon" in obs_meta[key]:
                return int(obs_meta[key]["horizon"])
        return int(self.cfg.obs_horizon)

    def _get_image_obs_keys(self) -> dict[str, str]:
        obs_meta = self.shape_meta.get("obs", {})

        def _pick(*candidates: str) -> str:
            for key in candidates:
                if key in obs_meta:
                    return key
            return candidates[0]

        return {
            "main": _pick("rgb_head", "h264_head"),
            "left": _pick("rgb_left_hand", "h264_left_hand"),
            "right": _pick("rgb_right_hand", "h264_right_hand"),
        }

    def _get_image_target_size(self, obs_key: str) -> tuple[int, int]:
        image_shape = self.shape_meta["obs"][obs_key]["shape"]
        return int(image_shape[1]), int(image_shape[2])

    def _pad_or_truncate_time(
        self, tensor: torch.Tensor, target_horizon: int
    ) -> torch.Tensor:
        current_horizon = int(tensor.shape[1])
        if current_horizon == target_horizon:
            return tensor
        if current_horizon > target_horizon:
            return tensor[:, -target_horizon:]

        pad_size = target_horizon - current_horizon
        padding = tensor[:, :1].expand(-1, pad_size, *tensor.shape[2:])
        return torch.cat([padding, tensor], dim=1)

    def _to_btd(self, states: torch.Tensor, target_horizon: int) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.ndim != 3:
            raise ValueError(
                f"psi-policy states must have shape [B, D] or [B, T, D], got {tuple(states.shape)}."
            )
        states = self._pad_or_truncate_time(states, target_horizon)
        return states.to(
            device=self._get_obs_encoder_device(), dtype=torch.float32
        ).contiguous()

    def _to_btchw(
        self, image: torch.Tensor, target_horizon: int, target_size: tuple[int, int]
    ) -> torch.Tensor:
        if image.ndim == 4:
            image = image.unsqueeze(1)
        if image.ndim != 5:
            raise ValueError(
                "psi-policy images must have shape [B, H, W, C], [B, T, H, W, C], "
                f"[B, C, H, W], or [B, T, C, H, W], got {tuple(image.shape)}."
            )

        if image.shape[-1] in (3, 4):
            image = image[..., :3]
            image = image.permute(0, 1, 4, 2, 3)
        elif image.shape[-3] in (3, 4):
            image = image[..., :3, :, :]
        else:
            raise ValueError(
                "psi-policy images must have 3 or 4 channels in the last or third-last dim, "
                f"got shape {tuple(image.shape)}."
            )

        image = self._pad_or_truncate_time(image, target_horizon)
        image = image.to(
            device=self._get_obs_encoder_device(), dtype=torch.float32
        )
        if torch.any(image > 1.0):
            image = image / 255.0

        batch_size, horizon = image.shape[:2]
        image = image.reshape(batch_size * horizon, *image.shape[2:])
        if tuple(image.shape[-2:]) != target_size:
            image = F.interpolate(
                image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return image.reshape(batch_size, horizon, image.shape[1], *target_size).contiguous()

    def _select_extra_view(self, extra_view_images: torch.Tensor, view_index: int) -> torch.Tensor:
        if extra_view_images.ndim == 5:
            return extra_view_images[:, view_index]
        if extra_view_images.ndim == 6:
            return extra_view_images[:, :, view_index]
        raise ValueError(
            "psi-policy extra_view_images must have shape [B, N, H, W, C] or "
            f"[B, T, N, H, W, C], got {tuple(extra_view_images.shape)}."
        )

    def _flatten_chunk_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[0], -1).contiguous()

    def _reshape_actions_for_q_head(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 3:
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"psi-policy critic expects action dim {self.action_dim}, got {tuple(actions.shape)}."
                )
            if self.cfg.use_chunk_rl:
                if actions.shape[1] < self.cfg.num_action_chunks:
                    raise ValueError(
                        f"psi-policy chunk RL expects at least {self.cfg.num_action_chunks} chunks, "
                        f"got {tuple(actions.shape)}."
                    )
                return self._flatten_chunk_tensor(actions[:, : self.cfg.num_action_chunks, :])
            return actions[:, 0, :].contiguous()

        if actions.dim() != 2:
            raise ValueError(
                f"psi-policy critic expects 2D or 3D action tensors, got {tuple(actions.shape)}."
            )

        if self.cfg.use_chunk_rl:
            expected_dim = self.cfg.num_action_chunks * self.action_dim
            if actions.shape[-1] != expected_dim:
                raise ValueError(
                    f"psi-policy chunk RL expects flattened action dim {expected_dim}, "
                    f"got {tuple(actions.shape)}."
                )
            return actions.contiguous()

        if actions.shape[-1] == self.action_dim:
            return actions.contiguous()

        if actions.shape[-1] % self.action_dim == 0:
            reshaped = actions.reshape(actions.shape[0], -1, self.action_dim)
            return reshaped[:, 0, :].contiguous()

        raise ValueError(
            f"psi-policy critic cannot reshape actions with shape {tuple(actions.shape)}."
        )

    def _compute_chunk_logprobs(
        self,
        action_chunks_mean_norm: torch.Tensor,
        action_chunks_norm: torch.Tensor,
        noise_std: float,
    ) -> torch.Tensor:
        if noise_std <= 0:
            return torch.zeros_like(action_chunks_norm)

        rl_action_mask = self._get_rl_action_mask(action_chunks_norm).view(1, 1, -1)
        return (
            Normal(action_chunks_mean_norm.detach(), noise_std)
            .log_prob(action_chunks_norm.detach())
            .mul(rl_action_mask)
        )

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

    def _get_known_low_dim_state_slices(self) -> dict[str, list[slice]]:
        ss = self.cfg.state_split
        state_slices = {
            "left_arm_states": [slice(ss["left_arm"][0], ss["left_arm"][1])],
            "right_arm_states": [slice(ss["right_arm"][0], ss["right_arm"][1])],
            "arm_joint_states": [slice(ss["left_arm"][0], ss["right_arm"][1])],
            "left_hand_states": [slice(ss["left_hand"][0], ss["left_hand"][1])],
            "right_hand_states": [slice(ss["right_hand"][0], ss["right_hand"][1])],
            "action_arm_joints": [
                slice(ss["left_arm"][0], ss["left_arm"][1]),
                slice(ss["right_arm"][0], ss["right_arm"][1]),
            ],
            "action_left_hand_joints": [
                slice(ss["left_hand"][0], ss["left_hand"][1])
            ],
            "action_right_hand_joints": [
                slice(ss["right_hand"][0], ss["right_hand"][1])
            ],
        }

        waist_slice = slice(self.cfg.action_dim, self.cfg.action_dim + 2)
        state_slices["waist_joints_states"] = [waist_slice]
        state_slices["action_waist_joints"] = [waist_slice]
        return state_slices

    def _concat_state_slices(
        self,
        states: torch.Tensor,
        state_slices: list[slice],
        obs_key: str,
    ) -> torch.Tensor:
        required_dim = max(state_slice.stop for state_slice in state_slices)
        if states.shape[-1] < required_dim:
            raise ValueError(
                f"psi-policy expects env states to include dims up to {required_dim} for '{obs_key}', "
                f"got {states.shape[-1]}."
            )
        parts = [states[:, :, state_slice] for state_slice in state_slices]
        if len(parts) == 1:
            return parts[0].contiguous()
        return torch.cat(parts, dim=-1).contiguous()

    def _build_low_dim_obs(
        self,
        env_obs: dict[str, torch.Tensor],
        states: Optional[torch.Tensor],
        target_horizon: int,
    ) -> dict[str, torch.Tensor]:
        processed: dict[str, torch.Tensor] = {}
        obs_meta = self.shape_meta.get("obs", {})
        state_slices = self._get_known_low_dim_state_slices()

        for key, meta in obs_meta.items():
            if meta.get("type") != "low_dim":
                continue

            value: Optional[torch.Tensor] = None
            if key in env_obs and key != "states":
                value = self._to_btd(
                    torch.as_tensor(env_obs[key]), target_horizon=target_horizon
                )
            elif key in state_slices:
                if states is None:
                    raise KeyError(
                        f"psi-policy checkpoint expects low-dim obs '{key}', but env_obs does not contain "
                        "'states' to derive it."
                    )
                value = self._concat_state_slices(states, state_slices[key], obs_key=key)

            if value is None:
                continue

            expected_shape = meta.get("shape", [])
            if len(expected_shape) == 1 and value.shape[-1] != int(expected_shape[0]):
                raise ValueError(
                    f"psi-policy obs '{key}' expects dim {int(expected_shape[0])}, got {value.shape[-1]}."
                )
            processed[key] = value

        required_state_keys = [
            key
            for key in getattr(getattr(self, "obs_encoder", None), "state_keys", [])
            if key in obs_meta
        ]
        missing_required_keys = [
            key for key in required_state_keys if key not in processed
        ]
        if missing_required_keys:
            available_keys = sorted(env_obs.keys())
            raise KeyError(
                "psi-policy checkpoint requires low-dim obs keys "
                f"{missing_required_keys}, but RLinf could not build them from env_obs "
                f"keys {available_keys}."
            )

        return processed

    def preprocess_env_obs(self, env_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_horizon = self._get_runtime_obs_horizon()
        image_obs_keys = self._get_image_obs_keys()

        main_images = torch.as_tensor(env_obs["main_images"])
        processed = {
            image_obs_keys["main"]: self._to_btchw(
                main_images,
                target_horizon=obs_horizon,
                target_size=self._get_image_target_size(image_obs_keys["main"]),
            )
        }

        extra_view_images = env_obs.get("extra_view_images")
        if extra_view_images is not None:
            extra_view_images = torch.as_tensor(extra_view_images)
            processed[image_obs_keys["left"]] = self._to_btchw(
                self._select_extra_view(extra_view_images, 0),
                target_horizon=obs_horizon,
                target_size=self._get_image_target_size(image_obs_keys["left"]),
            )
            processed[image_obs_keys["right"]] = self._to_btchw(
                self._select_extra_view(extra_view_images, 1),
                target_horizon=obs_horizon,
                target_size=self._get_image_target_size(image_obs_keys["right"]),
            )
        else:
            processed[image_obs_keys["left"]] = processed[image_obs_keys["main"]].clone()
            processed[image_obs_keys["right"]] = processed[image_obs_keys["main"]].clone()

        states = None
        if "states" in env_obs:
            states = self._to_btd(torch.as_tensor(env_obs["states"]), target_horizon=obs_horizon)
            if states.shape[-1] < self.cfg.action_dim:
                raise ValueError(
                    f"psi-policy expects at least {self.cfg.action_dim} state dims from A2D, got {states.shape[-1]}."
                )

        processed.update(
            self._build_low_dim_obs(
                env_obs,
                states=states,
                target_horizon=obs_horizon,
            )
        )
        return processed

    def encode_obs(self, obs: dict[str, torch.Tensor], detach_encoder: bool = False):
        nobs = self.normalizer.normalize(obs)
        nobs = self._move_obs_dict_to_encoder_device(nobs)
        train_obs_encoder = bool(
            self.cfg.train_obs_encoder and self.training and not detach_encoder
        )
        with torch.set_grad_enabled(train_obs_encoder):
            cond_tokens = self.obs_encoder(nobs, training=train_obs_encoder)
        obs_feature = cond_tokens.mean(dim=1)
        if detach_encoder or not train_obs_encoder:
            cond_tokens = cond_tokens.detach()
            obs_feature = obs_feature.detach()
        return cond_tokens, obs_feature

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["raw_obs"] = obs
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

    def _get_rl_action_mask(self, reference: torch.Tensor) -> torch.Tensor:
        return self.rl_action_mask.to(device=reference.device, dtype=reference.dtype)

    def _sample_masked_action(self, action_mean_norm: torch.Tensor, noise_std: float):
        rl_action_mask = self._get_rl_action_mask(action_mean_norm)
        if noise_std <= 0:
            zero_log_prob = torch.zeros(
                action_mean_norm.shape[0],
                device=action_mean_norm.device,
                dtype=action_mean_norm.dtype,
            )
            return action_mean_norm, action_mean_norm, zero_log_prob
        sampled_action_norm = action_mean_norm + noise_std * torch.randn_like(
            action_mean_norm
        ) * rl_action_mask
        detached_frozen_action_norm = (
            sampled_action_norm * rl_action_mask
            + sampled_action_norm.detach() * (1.0 - rl_action_mask)
        )
        log_prob = (
            Normal(action_mean_norm.detach(), noise_std)
            .log_prob(sampled_action_norm.detach())
            .mul(rl_action_mask)
            .sum(dim=-1)
        )
        return sampled_action_norm, detached_frozen_action_norm, log_prob

    def _sample_masked_action_chunks(
        self, action_chunks_mean_norm: torch.Tensor, noise_std: float
    ) -> torch.Tensor:
        if noise_std <= 0:
            return action_chunks_mean_norm
        rl_action_mask = self._get_rl_action_mask(action_chunks_mean_norm).view(1, 1, -1)
        return action_chunks_mean_norm + noise_std * torch.randn_like(
            action_chunks_mean_norm
        ) * rl_action_mask

    def _extract_reset_action_reference(
        self, env_obs: Optional[dict[str, torch.Tensor]], reference: torch.Tensor
    ) -> torch.Tensor:
        if env_obs is None:
            raise ValueError(
                "rl_action_mask reset-pose override requires raw env_obs, but got None."
            )
        reset_states = env_obs.get("reset_states")
        if reset_states is None:
            raise ValueError(
                "rl_action_mask reset-pose override requires env_obs['reset_states']."
            )
        if reset_states.dim() == 1:
            reset_states = reset_states.unsqueeze(0)
        reset_action = reset_states[..., : self.action_dim].to(
            device=reference.device, dtype=reference.dtype
        )
        return reset_action

    def _apply_reset_pose_mask(
        self, action: torch.Tensor, env_obs: Optional[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if self.active_rl_action_dim == self.action_dim:
            return action
        rl_action_mask = self._get_rl_action_mask(action)
        reset_action = self._extract_reset_action_reference(env_obs, action)
        while reset_action.dim() < action.dim():
            reset_action = reset_action.unsqueeze(1)
        return action * rl_action_mask + reset_action * (1.0 - rl_action_mask)

    def reset_rollout_rtc_state(self, batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            self._rollout_rtc_states = []
            return
        self._rollout_rtc_states = [
            _RolloutRTCState() for _ in range(int(batch_size))
        ]

    def _ensure_rollout_rtc_state(self, batch_size: int) -> None:
        current_states = getattr(self, "_rollout_rtc_states", None)
        if current_states is None or len(current_states) != batch_size:
            self.reset_rollout_rtc_state(batch_size=batch_size)

    def _rtc_find_nearest_point_index(
        self, predictions: torch.Tensor, point: torch.Tensor
    ) -> int:
        search_steps = min(self.cfg.rollout_rtc_execute_step, predictions.shape[0])
        if search_steps <= 0:
            raise RuntimeError("RTC search_steps must be positive.")
        distances = torch.linalg.vector_norm(
            predictions[:search_steps] - point.unsqueeze(0), dim=-1
        )
        return int(torch.argmin(distances).item())

    def _rtc_merge_prediction_queue(
        self, rtc_state: "_RolloutRTCState", predictions: torch.Tensor
    ) -> None:
        if predictions.ndim != 2:
            raise ValueError(
                f"RTC predictions must have shape [horizon, action_dim], got {tuple(predictions.shape)}."
            )
        if not rtc_state.action_queue:
            rtc_state.action_queue = deque(
                predictions[idx].clone() for idx in range(predictions.shape[0])
            )
            return

        reference_point = (
            rtc_state.last_action
            if rtc_state.last_action is not None
            else rtc_state.action_queue[-1]
        )
        start_index = self._rtc_find_nearest_point_index(predictions, reference_point)
        predictions_to_merge = [
            predictions[idx].clone() for idx in range(start_index, predictions.shape[0])
        ]
        if len(predictions_to_merge) < self.cfg.num_action_chunks:
            raise RuntimeError(
                "RTC reproduction could not keep enough future actions after alignment: "
                f"start_index={start_index}, horizon={predictions.shape[0]}, "
                f"required_chunk={self.cfg.num_action_chunks}."
            )

        min_length = min(len(rtc_state.action_queue), len(predictions_to_merge))
        for idx in range(min_length):
            weight = self.cfg.rollout_rtc_merge_weight_base / (idx**2 + 1)
            predictions_to_merge[idx] = (
                weight * rtc_state.action_queue[idx]
                + (1.0 - weight) * predictions_to_merge[idx]
            )
        rtc_state.action_queue = deque(predictions_to_merge)

    def _rtc_pop_action_chunk(
        self, rtc_state: "_RolloutRTCState", chunk_size: int
    ) -> torch.Tensor:
        if len(rtc_state.action_queue) < chunk_size:
            raise RuntimeError(
                "RTC reproduction queue underflow: "
                f"requested {chunk_size} actions, only {len(rtc_state.action_queue)} left."
            )

        chunk_actions = []
        for _ in range(chunk_size):
            action = rtc_state.action_queue.popleft().clone()
            rtc_state.last_action = action
            chunk_actions.append(action)
        return torch.stack(chunk_actions, dim=0)

    def _apply_rollout_rtc(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        if predicted_actions.ndim != 3:
            raise ValueError(
                "RTC reproduction expects predicted actions with shape "
                f"[B, horizon, action_dim], got {tuple(predicted_actions.shape)}."
            )

        batch_size = predicted_actions.shape[0]
        self._ensure_rollout_rtc_state(batch_size)
        predicted_actions_cpu = predicted_actions.detach().to(
            device="cpu", dtype=torch.float32
        )

        merged_chunks = []
        for env_idx in range(batch_size):
            rtc_state = self._rollout_rtc_states[env_idx]
            self._rtc_merge_prediction_queue(rtc_state, predicted_actions_cpu[env_idx])
            merged_chunks.append(
                self._rtc_pop_action_chunk(rtc_state, self.cfg.num_action_chunks)
            )

        merged_chunk_actions = torch.stack(merged_chunks, dim=0)
        return merged_chunk_actions.to(
            device=predicted_actions.device, dtype=predicted_actions.dtype
        )

    def sac_forward(self, obs, raw_obs=None, **kwargs):
        del kwargs
        cond_tokens, obs_feature = self.encode_obs(obs)
        num_chunks = self.cfg.num_action_chunks if self.cfg.use_chunk_rl else 1
        action_chunks_mean_norm = self._sample_action_chunks(
            cond_tokens, training=True
        )[:, :num_chunks, :]
        noise_std = self.cfg.noise_std_train
        if self.cfg.use_chunk_rl:
            action_chunks_norm = self._sample_masked_action_chunks(
                action_chunks_mean_norm, noise_std
            )
            action_chunks = self.normalizer["action"].unnormalize(action_chunks_norm)
            action_chunks = self._apply_reset_pose_mask(action_chunks, raw_obs)
            log_prob = self._compute_chunk_logprobs(
                action_chunks_mean_norm,
                action_chunks_norm,
                noise_std,
            ).sum(dim=-1)
            return self._flatten_chunk_tensor(action_chunks), log_prob, obs_feature

        action_mean_norm = action_chunks_mean_norm[:, 0, :]
        _, action_norm_for_q, log_prob = self._sample_masked_action(
            action_mean_norm, noise_std
        )
        action = self.normalizer["action"].unnormalize(
            action_norm_for_q.unsqueeze(1)
        ).squeeze(1)
        action = self._apply_reset_pose_mask(action, raw_obs)
        return action, log_prob, obs_feature

    def sac_q_forward(
        self, obs, actions, shared_feature=None, detach_encoder=False, **kwargs
    ):
        del kwargs
        if shared_feature is None:
            _, obs_feature = self.encode_obs(obs, detach_encoder=detach_encoder)
        else:
            obs_feature = shared_feature.detach() if detach_encoder else shared_feature
        return self.q_head(obs_feature, self._reshape_actions_for_q_head(actions))

    def default_forward(
        self,
        forward_inputs=None,
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=False,
        **kwargs,
    ):
        del kwargs
        if forward_inputs is None:
            raise ValueError("forward_inputs is required for psi-policy rollout forward.")
        obs = {
            "main_images": forward_inputs["main_images"],
            "states": forward_inputs["states"],
        }
        if "extra_view_images" in forward_inputs:
            obs["extra_view_images"] = forward_inputs["extra_view_images"]
        if "reset_states" in forward_inputs:
            obs["reset_states"] = forward_inputs["reset_states"]
        obs = self.preprocess_env_obs(obs)
        cond_tokens, _ = self.encode_obs(obs)
        action_chunks_mean_norm = self._sample_action_chunks(cond_tokens, training=False)
        noise_std = self.cfg.noise_std_rollout
        provided_action = forward_inputs.get("action")
        if provided_action is not None:
            provided_action = provided_action.to(device=self.device, dtype=torch.float32)
            if provided_action.ndim == 2:
                provided_action = provided_action.reshape(provided_action.shape[0], -1, self.action_dim)
            if provided_action.ndim != 3 or provided_action.shape[-1] != self.action_dim:
                raise ValueError(
                    "psi-policy forward_inputs['action'] must have shape [B, chunk*action_dim] or "
                    f"[B, chunk, action_dim], got {tuple(provided_action.shape)}."
                )
            num_chunks = int(provided_action.shape[1])
            action_chunks = provided_action[:, :num_chunks].contiguous()
            action_chunks_norm = self.normalizer["action"].normalize(action_chunks)
        else:
            num_chunks = self.cfg.num_action_chunks
            action_chunks_mean_norm = action_chunks_mean_norm[:, :num_chunks]
            action_chunks_norm = self._sample_masked_action_chunks(
                action_chunks_mean_norm, noise_std
            )
            action_chunks = self.normalizer["action"].unnormalize(action_chunks_norm)
            action_chunks = self._apply_reset_pose_mask(action_chunks, forward_inputs)

        action_chunks_mean_norm = action_chunks_mean_norm[:, :num_chunks]
        logprobs = self._compute_chunk_logprobs(
            action_chunks_mean_norm,
            action_chunks_norm,
            noise_std,
        )

        output = {
            "action": action_chunks[:, 0, :],
            "log_prob": logprobs[:, 0, :].sum(dim=-1),
        }
        if compute_logprobs:
            output["logprobs"] = logprobs
        if compute_entropy:
            if noise_std <= 0:
                output["entropy"] = torch.zeros_like(logprobs)
            else:
                rl_action_mask = self._get_rl_action_mask(action_chunks_mean_norm).view(1, 1, -1)
                output["entropy"] = (
                    Normal(action_chunks_mean_norm.detach(), noise_std).entropy().mul(rl_action_mask)
                )
        if compute_values:
            raise NotImplementedError("psi-policy does not expose a value head for default_forward.")
        return output

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
        action_chunks_norm = self._sample_masked_action_chunks(
            action_chunks_mean_norm, noise_std
        )
        all_model_chunk_actions = self.normalizer["action"].unnormalize(action_chunks_norm)
        all_chunk_actions = self._apply_reset_pose_mask(all_model_chunk_actions, env_obs)

        if self.cfg.rollout_rtc_enabled:
            if noise_std > 0:
                raise ValueError(
                    "rollout_rtc_enabled currently requires noise_std_rollout=0.0 so "
                    "RTC-selected actions keep exact psi-policy client semantics."
                )
            chunk_actions = self._apply_rollout_rtc(all_chunk_actions)
            model_chunk_actions = chunk_actions
            chunk_logprobs = torch.zeros_like(chunk_actions)
        else:
            chunks_norm = action_chunks_norm[:, : self.cfg.num_action_chunks, :]
            model_chunk_actions = all_model_chunk_actions[:, : self.cfg.num_action_chunks, :]
            chunk_actions = all_chunk_actions[:, : self.cfg.num_action_chunks, :]
            chunk_logprobs = self._compute_chunk_logprobs(
                action_chunks_mean_norm[:, : self.cfg.num_action_chunks, :],
                chunks_norm,
                noise_std,
            )
        chunk_values = torch.zeros(
            (chunk_actions.shape[0], 1),
            device=chunk_actions.device,
            dtype=chunk_actions.dtype,
        )
        forward_inputs = {
            "action": self._flatten_chunk_tensor(chunk_actions),
            "model_action": self._flatten_chunk_tensor(model_chunk_actions),
        }
        if return_obs:
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["states"] = env_obs["states"]
            if "reset_states" in env_obs:
                forward_inputs["reset_states"] = env_obs["reset_states"]
            if "extra_view_images" in env_obs:
                forward_inputs["extra_view_images"] = env_obs["extra_view_images"]
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = obs_feature
        return chunk_actions, result


@dataclass
class _RolloutRTCState:
    action_queue: deque[torch.Tensor] = field(default_factory=deque)
    last_action: Optional[torch.Tensor] = None
