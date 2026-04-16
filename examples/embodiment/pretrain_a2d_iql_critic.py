#!/usr/bin/env python
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

"""Offline IQL-style critic pretraining for A2D psi-policy SAC."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("EMBODIED_PATH", str(Path(__file__).resolve().parent))

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models import get_model
from rlinf.utils.nested_dict_process import put_tensor_device


class ValueHead(nn.Module):
    """Small V(s) head used by offline IQL critic pretraining."""

    def __init__(self, hidden_size: int, hidden_dims: list[int]):
        super().__init__()
        dims = [hidden_size, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers = [layer for layer in self.net if isinstance(layer, nn.Linear)]
        last_linear = linear_layers[-1]
        gain = nn.init.calculate_gain("tanh")
        for layer in linear_layers:
            if layer is last_linear:
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain A2D SAC critic with an IQL-style offline objective."
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="realworld_a2d_sac_psi",
        help="Embodiment config used to instantiate PsiPolicyForRL.",
    )
    parser.add_argument(
        "--replay-buffer-path",
        type=Path,
        required=True,
        help="Path produced by convert_a2d_zarr_to_replay_buffer.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save IQL artifacts.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional override for actor.model.model_path.",
    )
    parser.add_argument(
        "--normalizer-path",
        type=Path,
        default=None,
        help="Optional override for actor.model.normalizer_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Training device. Use "auto", "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--amp",
        type=str,
        choices=["none", "bf16", "fp16"],
        default="bf16",
        help="Autocast dtype used on CUDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=2000,
        help="Number of IQL optimization steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of transitions per optimizer step.",
    )
    parser.add_argument(
        "--sample-window-size",
        type=int,
        default=4096,
        help="Replay sampling window size. Keep it >= total offline episodes.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=8,
        help="Number of recent trajectories to keep in the in-memory replay cache.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3.0e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-5,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Global grad clip.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor. Defaults to algorithm.gamma from config.",
    )
    parser.add_argument(
        "--expectile",
        type=float,
        default=0.9,
        help="IQL expectile used by V(s) regression.",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for the V-head expectile loss.",
    )
    parser.add_argument(
        "--train-encoder",
        action="store_true",
        help="Also update the psi-policy observation encoder. Default is q_head-only.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output dir first if it already exists.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compose_config(config_name: str):
    config_dir = Path(__file__).resolve().parent / "config"
    with hydra.initialize_config_dir(
        version_base="1.1",
        config_dir=str(config_dir),
        job_name="a2d_iql_critic_pretrain",
    ):
        return hydra.compose(config_name=config_name)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def autocast_context(device: torch.device, amp_mode: str):
    if device.type != "cuda" or amp_mode == "none":
        return nullcontext()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    return torch.autocast(device_type="cuda", dtype=dtype_map[amp_mode])


def expectile_regression_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.square()


def scalar_metric(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.sample_window_size < 1:
        raise ValueError("--sample-window-size must be >= 1.")
    if not (0.0 < args.expectile < 1.0):
        raise ValueError("--expectile must be in (0, 1).")
    if args.num_updates < 1:
        raise ValueError("--num-updates must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_config(args.config_name)
    if args.model_path is not None:
        cfg.actor.model.model_path = str(args.model_path.expanduser().resolve())
    if args.normalizer_path is not None:
        cfg.actor.model.normalizer_path = str(args.normalizer_path.expanduser().resolve())

    gamma = float(args.gamma if args.gamma is not None else cfg.algorithm.gamma)
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    replay_buffer_runtime_dir = output_dir / "_buffer_runtime"
    replay_buffer_runtime_dir.mkdir(parents=True, exist_ok=True)

    replay_buffer = TrajectoryReplayBuffer(
        seed=args.seed,
        enable_cache=True,
        cache_size=args.cache_size,
        sample_window_size=args.sample_window_size,
        auto_save=True,
        auto_save_path=str(replay_buffer_runtime_dir),
        trajectory_format="pt",
    )
    replay_buffer.load_checkpoint(str(args.replay_buffer_path.expanduser().resolve()))
    dataset_stats = replay_buffer.get_stats()

    try:
        model = get_model(cfg.actor.model)
    except Exception as exc:
        raise RuntimeError(
            "Failed to build PsiPolicyForRL from the provided checkpoint. "
            "Please verify that the checkpoint path, normalizer path, and "
            "psi-policy obs encoder implementation are mutually compatible."
        ) from exc

    model = model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.q_head.parameters():
        param.requires_grad_(True)
    if args.train_encoder:
        for param in model.obs_encoder.parameters():
            param.requires_grad_(True)

    value_head = ValueHead(
        hidden_size=int(model.n_emb),
        hidden_dims=list(cfg.actor.model.q_hidden_dims),
    ).to(device)

    if args.train_encoder:
        model.train()
    else:
        model.eval()
    model.q_head.train()
    value_head.train()

    trainable_params = list(model.q_head.parameters()) + list(value_head.parameters())
    if args.train_encoder:
        trainable_params += list(model.obs_encoder.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = {
        "config_name": args.config_name,
        "replay_buffer_path": str(args.replay_buffer_path.expanduser().resolve()),
        "output_dir": str(output_dir),
        "model_path": cfg.actor.model.model_path,
        "normalizer_path": cfg.actor.model.normalizer_path,
        "device": str(device),
        "amp": args.amp if device.type == "cuda" else "none",
        "seed": args.seed,
        "num_updates": args.num_updates,
        "batch_size": args.batch_size,
        "sample_window_size": args.sample_window_size,
        "cache_size": args.cache_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "gamma": gamma,
        "expectile": args.expectile,
        "value_loss_weight": args.value_loss_weight,
        "train_encoder": args.train_encoder,
        "dataset_stats": dataset_stats,
    }
    save_json(run_config, output_dir / "run_config.json")

    metrics_path = output_dir / "metrics.jsonl"
    latest_metrics: dict[str, float] = {}
    start_time = time.time()
    interrupted = False

    try:
        for update in range(1, args.num_updates + 1):
            batch = replay_buffer.sample(args.batch_size)
            batch = put_tensor_device(batch, device=device)
            actions = batch["actions"].to(dtype=torch.float32)
            rewards = batch["rewards"].to(dtype=torch.float32).reshape(-1, 1)
            terminations = batch["terminations"].to(dtype=torch.float32).reshape(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, args.amp):
                curr_obs = model.preprocess_env_obs(batch["curr_obs"])
                next_obs = model.preprocess_env_obs(batch["next_obs"])
                _, curr_features = model.encode_obs(
                    curr_obs, detach_encoder=not args.train_encoder
                )
                with torch.no_grad():
                    _, next_features = model.encode_obs(next_obs, detach_encoder=True)

                q_values = model.q_head(curr_features, actions)
                min_q = torch.min(q_values.detach(), dim=-1, keepdim=True).values
                v_values = value_head(curr_features)
                with torch.no_grad():
                    next_v = value_head(next_features)
                    q_target = rewards + (1.0 - terminations) * gamma * next_v

                q_loss = F.mse_loss(q_values, q_target.expand_as(q_values))
                value_loss = expectile_regression_loss(
                    min_q - v_values, args.expectile
                ).mean()
                loss = q_loss + args.value_loss_weight * value_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            latest_metrics = {
                "update": float(update),
                "loss": scalar_metric(loss),
                "q_loss": scalar_metric(q_loss),
                "value_loss": scalar_metric(value_loss),
                "q_mean": scalar_metric(q_values.mean()),
                "q_target_mean": scalar_metric(q_target.mean()),
                "v_mean": scalar_metric(v_values.mean()),
                "adv_mean": scalar_metric((min_q - v_values).mean()),
                "reward_mean": scalar_metric(rewards.mean()),
                "terminal_fraction": scalar_metric(terminations.mean()),
                "grad_norm": scalar_metric(grad_norm),
                "elapsed_sec": time.time() - start_time,
            }

            with open(metrics_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(latest_metrics, ensure_ascii=False) + "\n")

            if update == 1 or update % args.log_interval == 0 or update == args.num_updates:
                print(
                    "[IQL] "
                    f"step={update}/{args.num_updates} "
                    f"loss={latest_metrics['loss']:.6f} "
                    f"q_loss={latest_metrics['q_loss']:.6f} "
                    f"v_loss={latest_metrics['value_loss']:.6f} "
                    f"q_mean={latest_metrics['q_mean']:.6f} "
                    f"v_mean={latest_metrics['v_mean']:.6f} "
                    f"adv_mean={latest_metrics['adv_mean']:.6f}"
                )
    except KeyboardInterrupt:
        interrupted = True
        print("[IQL] Interrupted by user. Saving current weights before exit.")

    actor_model_state_path = output_dir / "actor_model_state_dict.pt"
    lightweight_package_path = output_dir / "critic_init_package.pt"
    training_state_path = output_dir / "iql_training_state.pt"

    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        actor_model_state_path,
    )
    torch.save(
        {
            "q_head_state_dict": {
                key: value.detach().cpu()
                for key, value in model.q_head.state_dict().items()
            },
            "value_head_state_dict": {
                key: value.detach().cpu()
                for key, value in value_head.state_dict().items()
            },
            "metadata": {
                **run_config,
                "interrupted": interrupted,
                "latest_metrics": latest_metrics,
            },
        },
        lightweight_package_path,
    )
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "value_head_state_dict": {
                key: value.detach().cpu()
                for key, value in value_head.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                **run_config,
                "interrupted": interrupted,
                "latest_metrics": latest_metrics,
            },
        },
        training_state_path,
    )
    save_json(
        {
            **run_config,
            "interrupted": interrupted,
            "latest_metrics": latest_metrics,
            "actor_model_state_dict": str(actor_model_state_path),
            "critic_init_package": str(lightweight_package_path),
            "training_state": str(training_state_path),
        },
        output_dir / "summary.json",
    )
    replay_buffer.close(wait=True)

    print(
        "Saved IQL critic warm start artifacts to "
        f"{output_dir}. Use runner.ckpt_path={actor_model_state_path} "
        "to bootstrap online SAC."
    )


if __name__ == "__main__":
    main()
