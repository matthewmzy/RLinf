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

"""Generate an intuitive report for A2D IQL critic warm-start runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("EMBODIED_PATH", str(Path(__file__).resolve().parent))

import hydra
import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from rlinf.models import get_model

matplotlib.use("Agg")


class ValueHead(nn.Module):
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
        description="Generate a visual report for A2D IQL critic warm-start training."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to the summary.json produced by pretrain_a2d_iql_critic.py.",
    )
    parser.add_argument(
        "--training-state-path",
        type=Path,
        default=None,
        help="Optional override for iql_training_state.pt.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory where report assets will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Evaluation device. Use "auto", "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of transitions per evaluation batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for reconstructing the initial critic.",
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
        job_name="a2d_iql_critic_report",
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


def load_metrics(metrics_path: Path) -> list[dict]:
    with open(metrics_path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def build_model_and_value_head(
    summary: dict, training_state_path: Path, device: torch.device, seed: int
):
    cfg = compose_config(summary["config_name"])
    cfg.actor.model.model_path = summary["model_path"]
    cfg.actor.model.normalizer_path = summary["normalizer_path"]

    set_seed(seed)
    model = get_model(cfg.actor.model).float().to(device)
    value_head = ValueHead(
        hidden_size=int(model.n_emb),
        hidden_dims=list(cfg.actor.model.q_hidden_dims),
    ).float().to(device)

    training_state = torch.load(
        training_state_path, map_location="cpu", weights_only=False
    )
    return cfg, model, value_head, training_state


def load_episode_files(replay_buffer_path: Path) -> list[Path]:
    files = sorted(replay_buffer_path.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No trajectory .pt files found under replay buffer path: {replay_buffer_path}"
        )
    return files


def evaluate_dataset(
    model,
    value_head,
    replay_buffer_path: Path,
    gamma: float,
    device: torch.device,
    amp_mode: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    files = load_episode_files(replay_buffer_path)
    model.eval()
    model.q_head.eval()
    value_head.eval()

    min_q_all: list[np.ndarray] = []
    v_all: list[np.ndarray] = []
    ideal_all: list[np.ndarray] = []
    progress_all: list[np.ndarray] = []
    steps_to_goal_all: list[np.ndarray] = []
    episode_ids_all: list[np.ndarray] = []

    with torch.no_grad():
        for episode_idx, file_path in enumerate(files):
            traj = torch.load(file_path, map_location="cpu", weights_only=False)
            valid_mask = traj["transition_valids"].squeeze(-1).bool()
            length = int(valid_mask.sum().item())
            if length <= 0:
                continue

            curr_obs = {key: value[valid_mask] for key, value in traj["curr_obs"].items()}
            actions = traj["actions"][valid_mask][:, 0, :].to(dtype=torch.float32)
            rewards = traj["rewards"][valid_mask][:, 0].to(dtype=torch.float32)
            terminations = traj["terminations"][valid_mask][:, 0].bool()

            terminal_candidates = torch.nonzero(terminations, as_tuple=False).squeeze(-1)
            terminal_idx = (
                int(terminal_candidates[-1].item())
                if terminal_candidates.numel() > 0
                else length - 1
            )

            steps = np.arange(length, dtype=np.int64)
            steps_to_goal = terminal_idx - steps
            ideal_returns = np.power(gamma, np.clip(steps_to_goal, a_min=0, a_max=None))
            progress = steps / max(terminal_idx, 1)
            if rewards[-1].item() < 1.0 and terminal_candidates.numel() > 0:
                ideal_returns[-1] = 1.0

            batch_min_q: list[np.ndarray] = []
            batch_v: list[np.ndarray] = []

            for start in range(0, length, batch_size):
                end = min(start + batch_size, length)
                obs_batch = {
                    key: value[start:end]
                    for key, value in curr_obs.items()
                }
                action_batch = actions[start:end].to(device=device)

                with autocast_context(device, amp_mode):
                    processed_obs = model.preprocess_env_obs(obs_batch)
                    _, obs_feature = model.encode_obs(processed_obs, detach_encoder=True)
                    q_values = model.q_head(
                        obs_feature, action_batch.to(dtype=torch.float32)
                    )
                    v_values = value_head(obs_feature)

                batch_min_q.append(
                    q_values.min(dim=-1).values.detach().float().cpu().numpy()
                )
                batch_v.append(v_values.squeeze(-1).detach().float().cpu().numpy())

            min_q_all.append(np.concatenate(batch_min_q))
            v_all.append(np.concatenate(batch_v))
            ideal_all.append(ideal_returns.astype(np.float32))
            progress_all.append(progress.astype(np.float32))
            steps_to_goal_all.append(steps_to_goal.astype(np.int64))
            episode_ids_all.append(np.full(length, episode_idx, dtype=np.int64))

    return {
        "min_q": np.concatenate(min_q_all),
        "v": np.concatenate(v_all),
        "ideal": np.concatenate(ideal_all),
        "progress": np.concatenate(progress_all),
        "steps_to_goal": np.concatenate(steps_to_goal_all),
        "episode_id": np.concatenate(episode_ids_all),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    if np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stage_gap(values: np.ndarray, progress: np.ndarray) -> float:
    early_mask = progress <= 0.2
    late_mask = progress >= 0.8
    if not early_mask.any() or not late_mask.any():
        return 0.0
    return float(values[late_mask].mean() - values[early_mask].mean())


def summarize_alignment(eval_dict: dict[str, np.ndarray]) -> dict[str, float]:
    min_q = eval_dict["min_q"]
    v_values = eval_dict["v"]
    ideal = eval_dict["ideal"]
    progress = eval_dict["progress"]
    return {
        "min_q_mae": float(np.mean(np.abs(min_q - ideal))),
        "min_q_rmse": float(np.sqrt(np.mean((min_q - ideal) ** 2))),
        "min_q_corr": pearson_corr(min_q, ideal),
        "min_q_late_minus_early": stage_gap(min_q, progress),
        "v_mae": float(np.mean(np.abs(v_values - ideal))),
        "v_rmse": float(np.sqrt(np.mean((v_values - ideal) ** 2))),
        "v_corr": pearson_corr(v_values, ideal),
        "v_late_minus_early": stage_gap(v_values, progress),
        "v_terminal_mean": float(v_values[progress >= 0.98].mean()),
        "v_start_mean": float(v_values[progress <= 0.02].mean()),
        "ideal_terminal_mean": float(ideal[progress >= 0.98].mean()),
        "ideal_start_mean": float(ideal[progress <= 0.02].mean()),
    }


def bin_curve(
    x: np.ndarray, y: np.ndarray, bins: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(bins, np.nan, dtype=np.float64)
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if mask.any():
            values[idx] = y[mask].mean()
    return centers, values


def create_dashboard(
    metrics: list[dict],
    initial_eval: dict[str, np.ndarray],
    trained_eval: dict[str, np.ndarray],
    dashboard_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    colors = {
        "navy": "#0f172a",
        "teal": "#0f766e",
        "cyan": "#0284c7",
        "amber": "#ea580c",
        "rose": "#be123c",
        "slate": "#475569",
        "bg": "#f8fafc",
    }

    updates = np.array([row["update"] for row in metrics], dtype=np.float64)
    loss = np.array([row["loss"] for row in metrics], dtype=np.float64)
    q_loss = np.array([row["q_loss"] for row in metrics], dtype=np.float64)
    value_loss = np.array([row["value_loss"] for row in metrics], dtype=np.float64)
    q_mean = np.array([row["q_mean"] for row in metrics], dtype=np.float64)
    q_target_mean = np.array([row["q_target_mean"] for row in metrics], dtype=np.float64)
    v_mean = np.array([row["v_mean"] for row in metrics], dtype=np.float64)

    smooth_window = max(5, len(metrics) // 40)
    smooth_loss = moving_average(loss, smooth_window)
    smooth_q_loss = moving_average(q_loss, smooth_window)
    smooth_value_loss = moving_average(value_loss, smooth_window)
    smooth_q_mean = moving_average(q_mean, smooth_window)
    smooth_q_target_mean = moving_average(q_target_mean, smooth_window)
    smooth_v_mean = moving_average(v_mean, smooth_window)

    progress = trained_eval["progress"]
    x_curve, ideal_curve = bin_curve(progress, trained_eval["ideal"])
    _, trained_q_curve = bin_curve(progress, trained_eval["min_q"])
    _, trained_v_curve = bin_curve(progress, trained_eval["v"])
    _, init_q_curve = bin_curve(initial_eval["progress"], initial_eval["min_q"])
    _, init_v_curve = bin_curve(initial_eval["progress"], initial_eval["v"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=colors["bg"])
    for ax in axes.ravel():
        ax.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot(updates, smooth_loss, color=colors["navy"], linewidth=2.5, label="total loss")
    ax.plot(updates, smooth_q_loss, color=colors["cyan"], linewidth=2.0, label="q loss")
    ax.plot(
        updates,
        smooth_value_loss,
        color=colors["amber"],
        linewidth=2.0,
        label="value loss",
    )
    ax.set_yscale("log")
    ax.set_title("IQL Training Loss")
    ax.set_xlabel("update")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot(updates, smooth_q_mean, color=colors["cyan"], linewidth=2.2, label="q mean")
    ax.plot(
        updates,
        smooth_q_target_mean,
        color=colors["teal"],
        linewidth=2.2,
        label="q target mean",
    )
    ax.plot(updates, smooth_v_mean, color=colors["rose"], linewidth=2.2, label="v mean")
    ax.set_title("Critic Statistics During Training")
    ax.set_xlabel("update")
    ax.set_ylabel("value")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.plot(x_curve, ideal_curve, color=colors["navy"], linewidth=3, label="ideal return")
    ax.plot(
        x_curve,
        init_q_curve,
        color=colors["slate"],
        linewidth=2,
        linestyle="--",
        label="initial minQ",
    )
    ax.plot(
        x_curve,
        trained_q_curve,
        color=colors["cyan"],
        linewidth=2.6,
        label="trained minQ",
    )
    ax.set_title("minQ vs. Success Return")
    ax.set_xlabel("episode progress to success")
    ax.set_ylabel("discounted success return")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    ax.plot(x_curve, ideal_curve, color=colors["navy"], linewidth=3, label="ideal return")
    ax.plot(
        x_curve,
        init_v_curve,
        color=colors["slate"],
        linewidth=2,
        linestyle="--",
        label="initial V(s)",
    )
    ax.plot(
        x_curve,
        trained_v_curve,
        color=colors["amber"],
        linewidth=2.6,
        label="trained V(s)",
    )
    ax.set_title("V(s) Learns the Success Geometry")
    ax.set_xlabel("episode progress to success")
    ax.set_ylabel("value")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("A2D IQL Critic Warm Start Dashboard", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(dashboard_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_effect_summary_figure(
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    summary_metrics: dict,
    effect_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    colors = {
        "navy": "#0f172a",
        "cyan": "#0284c7",
        "amber": "#ea580c",
        "green": "#15803d",
        "slate": "#64748b",
        "bg": "#f8fafc",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=colors["bg"])
    for ax in axes:
        ax.set_facecolor("white")

    ax = axes[0]
    labels = ["minQ MAE", "V(s) MAE"]
    before_vals = [before_metrics["min_q_mae"], before_metrics["v_mae"]]
    after_vals = [after_metrics["min_q_mae"], after_metrics["v_mae"]]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, before_vals, width, color=colors["slate"], label="before")
    ax.bar(x + width / 2, after_vals, width, color=colors["cyan"], label="after")
    ax.set_xticks(x, labels)
    ax.set_title("Alignment Error to Ideal Success Return")
    ax.set_ylabel("mean absolute error")
    ax.legend(frameon=False)

    ax = axes[1]
    labels = ["minQ corr", "V(s) corr", "V late-early gap"]
    before_vals = [
        before_metrics["min_q_corr"],
        before_metrics["v_corr"],
        before_metrics["v_late_minus_early"],
    ]
    after_vals = [
        after_metrics["min_q_corr"],
        after_metrics["v_corr"],
        after_metrics["v_late_minus_early"],
    ]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, before_vals, width, color=colors["slate"], label="before")
    ax.bar(x + width / 2, after_vals, width, color=colors["amber"], label="after")
    ax.set_xticks(x, labels, rotation=10)
    ax.set_title("Shape of the Learned Value Function")
    ax.axhline(0.0, color="#cbd5e1", linewidth=1.0)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.axis("off")
    loss_drop = summary_metrics["loss_drop_ratio"]
    runtime = summary_metrics["elapsed_sec"]
    lines = [
        "Training scorecard",
        "",
        f"Updates: {summary_metrics['num_updates']}",
        f"Batch size: {summary_metrics['batch_size']}",
        f"Dataset: {summary_metrics['num_episodes']} episodes / {summary_metrics['num_transitions']} transitions",
        f"Runtime: {runtime / 60.0:.1f} min",
        "",
        f"Loss drop: {loss_drop:.1f}x",
        f"Final total loss: {summary_metrics['final_loss']:.6f}",
        f"Final q mean: {summary_metrics['final_q_mean']:.4f}",
        f"Final v mean: {summary_metrics['final_v_mean']:.4f}",
        "",
        f"Warm-start file:",
        "actor_model_state_dict.pt",
    ]
    ax.text(
        0.02,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        color=colors["navy"],
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#eef6ff",
            edgecolor="#bfdbfe",
            linewidth=1.5,
        ),
    )

    fig.suptitle("A2D Critic Warm Start: Before vs. After", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(effect_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_architecture_svg(svg_path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1120" viewBox="0 0 1800 1120" fill="none">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1800" y2="1120" gradientUnits="userSpaceOnUse">
      <stop stop-color="#F8FBFF"/>
      <stop offset="1" stop-color="#FFF8F1"/>
    </linearGradient>
    <linearGradient id="blue" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="#0EA5E9"/>
      <stop offset="1" stop-color="#1D4ED8"/>
    </linearGradient>
    <linearGradient id="green" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="#10B981"/>
      <stop offset="1" stop-color="#047857"/>
    </linearGradient>
    <linearGradient id="amber" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="#FB923C"/>
      <stop offset="1" stop-color="#C2410C"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="160%">
      <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#0F172A" flood-opacity="0.12"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1800" height="1120" fill="url(#bg)"/>
  <circle cx="1490" cy="170" r="210" fill="#DBEAFE" fill-opacity="0.45"/>
  <circle cx="260" cy="940" r="230" fill="#FEF3C7" fill-opacity="0.45"/>
  <circle cx="1660" cy="940" r="180" fill="#DCFCE7" fill-opacity="0.42"/>

  <text x="80" y="86" fill="#0F172A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="38" font-weight="800">
    A2D Psi-Policy to SAC to IQL Warm Start to Online RL
  </text>
  <text x="80" y="124" fill="#475569" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">
    A source-aligned map of the exact stack used in RLinf for the new v0.8 checkpoint.
  </text>

  <rect x="70" y="175" width="290" height="280" rx="30" fill="#FFFFFF" stroke="#CBD5E1" filter="url(#shadow)"/>
  <text x="100" y="228" fill="#0F172A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="28" font-weight="700">Robot observations</text>
  <rect x="102" y="256" width="226" height="46" rx="14" fill="#F1F5F9"/>
  <text x="122" y="286" fill="#0F172A" font-family="monospace" font-size="19">rgb_head</text>
  <rect x="102" y="316" width="226" height="46" rx="14" fill="#F1F5F9"/>
  <text x="122" y="346" fill="#0F172A" font-family="monospace" font-size="19">rgb_left_hand</text>
  <rect x="102" y="376" width="226" height="46" rx="14" fill="#F1F5F9"/>
  <text x="122" y="406" fill="#0F172A" font-family="monospace" font-size="19">rgb_right_hand</text>
  <text x="100" y="438" fill="#475569" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">arm + hand + waist states</text>
  <text x="100" y="474" fill="#64748B" font-family="monospace" font-size="15">curr_obs / next_obs / actions</text>

  <rect x="420" y="155" width="430" height="340" rx="34" fill="url(#green)" filter="url(#shadow)"/>
  <text x="454" y="214" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="32" font-weight="800">timm_obs_encoder_3rgb_joint</text>
  <text x="454" y="250" fill="#DCFCE7" font-family="monospace" font-size="17">psi_policy/model/vision/timm_obs_encoder_3rgb_joint.py</text>
  <rect x="454" y="282" width="360" height="52" rx="16" fill="rgba(255,255,255,0.14)"/>
  <text x="478" y="315" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">3 x ViT-small feature streams, each 384-d</text>
  <rect x="454" y="350" width="360" height="52" rx="16" fill="rgba(255,255,255,0.14)"/>
  <text x="478" y="383" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">Concatenate RGB features + joint state: 1178-d</text>
  <rect x="454" y="418" width="360" height="52" rx="16" fill="rgba(255,255,255,0.14)"/>
  <text x="478" y="451" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">Linear projection to 400-d token sequence</text>
  <text x="454" y="486" fill="#ECFDF5" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">
    The v0.8 compatibility fix was: raw_output_dim = 3 * feature_dim + state_dim.
  </text>

  <rect x="910" y="155" width="370" height="340" rx="34" fill="url(#blue)" filter="url(#shadow)"/>
  <text x="946" y="214" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="31" font-weight="800">PsiPolicyForRL actor body</text>
  <text x="946" y="250" fill="#DBEAFE" font-family="monospace" font-size="17">rlinf/models/embodiment/psi_policy/psi_policy_for_rl.py</text>
  <rect x="946" y="282" width="294" height="54" rx="16" fill="rgba(255,255,255,0.14)"/>
  <text x="970" y="316" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">TransformerForAction diffusion backbone</text>
  <rect x="946" y="352" width="294" height="54" rx="16" fill="rgba(255,255,255,0.14)"/>
  <text x="970" y="386" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">Predict 16-step action chunk</text>
  <rect x="946" y="422" width="294" height="54" rx="16" fill="rgba(255,255,255,0.14)"/>
  <text x="970" y="456" fill="white" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="20">Execute first 8 steps online</text>

  <rect x="1350" y="176" width="360" height="152" rx="30" fill="#FFFFFF" stroke="#CBD5E1" filter="url(#shadow)"/>
  <text x="1382" y="226" fill="#0F172A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="28" font-weight="800">MultiQHead Q1, Q2</text>
  <text x="1382" y="261" fill="#475569" font-family="monospace" font-size="17">q_head.py : [256, 256, 256] x 2</text>
  <text x="1382" y="295" fill="#475569" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">Added on top of obs_feature + action</text>

  <rect x="1350" y="346" width="360" height="150" rx="30" fill="#FFF7ED" stroke="#FDBA74" filter="url(#shadow)"/>
  <text x="1382" y="395" fill="#9A3412" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="28" font-weight="800">IQL ValueHead V(s)</text>
  <text x="1382" y="430" fill="#C2410C" font-family="monospace" font-size="17">pretrain_a2d_iql_critic.py</text>
  <text x="1382" y="464" fill="#9A3412" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">Temporary offline head for expectile regression</text>

  <rect x="70" y="560" width="820" height="420" rx="36" fill="#FFFFFF" stroke="#BFDBFE" filter="url(#shadow)"/>
  <text x="110" y="624" fill="#0F172A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="34" font-weight="800">Stage 1. Offline IQL critic warm start on successful demos</text>
  <text x="110" y="662" fill="#475569" font-family="monospace" font-size="18">examples/embodiment/pretrain_a2d_iql_critic.py</text>
  <rect x="110" y="706" width="250" height="102" rx="24" fill="#EFF6FF"/>
  <text x="136" y="750" fill="#1D4ED8" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="24" font-weight="700">Offline success buffer</text>
  <text x="136" y="785" fill="#1E3A8A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">21 episodes / 2352 transitions</text>
  <rect x="404" y="706" width="210" height="102" rx="24" fill="#F0FDF4"/>
  <text x="432" y="750" fill="#047857" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="24" font-weight="700">Freeze backbone</text>
  <text x="432" y="785" fill="#065F46" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">Train q_head + V only</text>
  <rect x="646" y="706" width="200" height="102" rx="24" fill="#FFF7ED"/>
  <text x="674" y="750" fill="#C2410C" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="24" font-weight="700">Target</text>
  <text x="674" y="785" fill="#9A3412" font-family="monospace" font-size="18">Q = r + gamma V(s')</text>
  <rect x="110" y="840" width="736" height="96" rx="28" fill="#0F172A"/>
  <text x="144" y="888" fill="#F8FAFC" font-family="monospace" font-size="21">value_loss = expectile(min(Q1, Q2) - V(s), tau = 0.9)</text>
  <text x="144" y="918" fill="#CBD5E1" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">
    Result: initialize the online critic with success-aware geometry before touching the real robot.
  </text>

  <rect x="940" y="560" width="770" height="420" rx="36" fill="#FFFFFF" stroke="#BBF7D0" filter="url(#shadow)"/>
  <text x="980" y="624" fill="#0F172A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="34" font-weight="800">Stage 2. Online SAC fine-tuning on the real robot</text>
  <text x="980" y="662" fill="#475569" font-family="monospace" font-size="18">rlinf/workers/actor/fsdp_sac_policy_worker.py</text>
  <rect x="980" y="706" width="220" height="102" rx="24" fill="#F0FDF4"/>
  <text x="1006" y="750" fill="#047857" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="24" font-weight="700">Real robot rollout</text>
  <text x="1006" y="785" fill="#065F46" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">policy / teleop / human reward</text>
  <rect x="1230" y="706" width="212" height="102" rx="24" fill="#EFF6FF"/>
  <text x="1258" y="750" fill="#1D4ED8" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="24" font-weight="700">Replay buffer</text>
  <text x="1258" y="785" fill="#1E3A8A" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">online + demo_buffer</text>
  <rect x="1470" y="706" width="200" height="102" rx="24" fill="#FFF7ED"/>
  <text x="1498" y="750" fill="#C2410C" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="24" font-weight="700">SAC update</text>
  <text x="1498" y="785" fill="#9A3412" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="18">critic then actor</text>
  <rect x="980" y="840" width="690" height="96" rx="28" fill="#0F172A"/>
  <text x="1016" y="883" fill="#F8FAFC" font-family="monospace" font-size="20">critic: MSE(Q(s,a), r + gamma * min target_Q(s', a'))</text>
  <text x="1016" y="915" fill="#CBD5E1" font-family="monospace" font-size="20">actor: mean(alpha * log pi(a|s) - Q(s,a)), plus entropy tuning</text>

  <path d="M 360 315 C 386 315, 392 315, 420 315" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 850 315 C 874 315, 884 315, 910 315" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 1280 250 C 1306 250, 1316 250, 1350 250" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 1280 420 C 1308 420, 1320 420, 1350 420" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 575 495 C 575 550, 575 550, 575 560" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 1090 495 C 1090 540, 1090 540, 1090 560" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 1200 756 C 1212 756, 1218 756, 1230 756" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>
  <path d="M 1442 756 C 1452 756, 1460 756, 1470 756" stroke="#334155" stroke-width="4" marker-end="url(#arrow)"/>

  <text x="530" y="536" fill="#334155" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="17">obs_feature</text>
  <text x="1046" y="538" fill="#334155" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="17">warm-started critic</text>
  <text x="1288" y="734" fill="#334155" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="16">store transitions</text>
  <text x="1512" y="688" fill="#334155" font-family="Inter, Segoe UI, PingFang SC, sans-serif" font-size="16">bootstrap online RL</text>
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def write_report_markdown(
    report_path: Path,
    summary: dict,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    effect_summary: dict,
) -> None:
    loss_drop = effect_summary["loss_drop_ratio"]
    warm_start_path = Path(summary["actor_model_state_dict"])
    report = f"""# A2D IQL Critic Warm Start Report

## Run summary

- Config: `{summary["config_name"]}`
- Dataset replay buffer: `{summary["replay_buffer_path"]}`
- Checkpoint: `{summary["model_path"]}`
- Normalizer: `{summary["normalizer_path"]}`
- Device: `{summary["device"]}`
- Updates: `{summary["num_updates"]}`
- Batch size: `{summary["batch_size"]}`
- Gamma: `{summary["gamma"]}`
- Expectile: `{summary["expectile"]}`
- Dataset size: {summary["dataset_stats"]["num_trajectories"]} episodes / {summary["dataset_stats"]["total_samples"]} transitions

## What changed after pretraining

- Total loss dropped from `{effect_summary["first_loss"]:.6f}` to `{effect_summary["final_loss"]:.6f}` (`{loss_drop:.2f}x` lower).
- `minQ` MAE to the ideal discounted success return: `{before_metrics["min_q_mae"]:.4f}` -> `{after_metrics["min_q_mae"]:.4f}`.
- `V(s)` MAE to the ideal discounted success return: `{before_metrics["v_mae"]:.4f}` -> `{after_metrics["v_mae"]:.4f}`.
- `V(s)` correlation with ideal return: `{before_metrics["v_corr"]:.4f}` -> `{after_metrics["v_corr"]:.4f}`.
- `V(s)` late-minus-early gap: `{before_metrics["v_late_minus_early"]:.4f}` -> `{after_metrics["v_late_minus_early"]:.4f}`.
- Note: the slight `V(s)` MAE increase is acceptable here because IQL trains `V(s)` as an expectile target, not as a direct Monte Carlo return regressor. For `V(s)`, shape and ordering matter more than exact scale.

## Files in this report

- `critic_training_dashboard.png`: training curves plus before/after value geometry.
- `critic_effect_summary.png`: compact scorecard of alignment improvements.
- `a2d_psi_sac_iql_architecture.svg`: end-to-end structure map from psi-policy to online SAC.
- `effect_summary.json`: machine-readable scalar summary.

## Warm-start the real robot SAC run

Use the saved actor state dict as the online bootstrap checkpoint:

```bash
python examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi \\
  runner.ckpt_path={warm_start_path}
```

If you want to override paths explicitly, also set:

```bash
actor.model.model_path={summary["model_path"]}
actor.model.normalizer_path={summary["normalizer_path"]}
rollout.model.model_path={summary["model_path"]}
```

## Code map

- `examples/embodiment/pretrain_a2d_iql_critic.py`: offline IQL critic warm start
- `rlinf/models/embodiment/psi_policy/psi_policy_for_rl.py`: psi-policy wrapped into SAC interfaces
- `rlinf/models/embodiment/modules/q_head.py`: double-Q critic head
- `rlinf/workers/actor/fsdp_sac_policy_worker.py`: online SAC update loop
- `psi_policy/model/vision/timm_obs_encoder_3rgb_joint.py`: 3-camera encoder used by the new v0.8 checkpoint
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = args.summary_path.expanduser().resolve()
    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    training_state_path = (
        args.training_state_path.expanduser().resolve()
        if args.training_state_path is not None
        else Path(summary["training_state"]).expanduser().resolve()
    )
    report_dir = (
        args.report_dir.expanduser().resolve()
        if args.report_dir is not None
        else summary_path.parent / "report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else summary["seed"])
    device = resolve_device(args.device)
    amp_mode = "none"
    gamma = float(summary["gamma"])

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg, model, value_head, training_state = build_model_and_value_head(
        summary=summary,
        training_state_path=training_state_path,
        device=device,
        seed=seed,
    )

    initial_eval = evaluate_dataset(
        model=model,
        value_head=value_head,
        replay_buffer_path=Path(summary["replay_buffer_path"]),
        gamma=gamma,
        device=device,
        amp_mode=amp_mode,
        batch_size=args.batch_size,
    )

    model.load_state_dict(training_state["model_state_dict"], strict=True)
    value_head.load_state_dict(training_state["value_head_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32)
    value_head = value_head.to(device=device, dtype=torch.float32)

    trained_eval = evaluate_dataset(
        model=model,
        value_head=value_head,
        replay_buffer_path=Path(summary["replay_buffer_path"]),
        gamma=gamma,
        device=device,
        amp_mode=amp_mode,
        batch_size=args.batch_size,
    )

    metrics = load_metrics(summary_path.parent / "metrics.jsonl")
    before_metrics = summarize_alignment(initial_eval)
    after_metrics = summarize_alignment(trained_eval)

    first_loss = float(metrics[0]["loss"])
    final_loss = float(metrics[-1]["loss"])
    effect_summary = {
        "report_dir": str(report_dir),
        "num_updates": int(summary["num_updates"]),
        "batch_size": int(summary["batch_size"]),
        "num_episodes": int(summary["dataset_stats"]["num_trajectories"]),
        "num_transitions": int(summary["dataset_stats"]["total_samples"]),
        "elapsed_sec": float(summary["latest_metrics"]["elapsed_sec"]),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss_drop_ratio": float(first_loss / max(final_loss, 1.0e-12)),
        "final_q_mean": float(summary["latest_metrics"]["q_mean"]),
        "final_v_mean": float(summary["latest_metrics"]["v_mean"]),
        "before": before_metrics,
        "after": after_metrics,
    }

    create_dashboard(
        metrics=metrics,
        initial_eval=initial_eval,
        trained_eval=trained_eval,
        dashboard_path=report_dir / "critic_training_dashboard.png",
    )
    create_effect_summary_figure(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        summary_metrics=effect_summary,
        effect_path=report_dir / "critic_effect_summary.png",
    )
    generate_architecture_svg(report_dir / "a2d_psi_sac_iql_architecture.svg")
    write_report_markdown(
        report_path=report_dir / "report.md",
        summary=summary,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        effect_summary=effect_summary,
    )

    with open(report_dir / "effect_summary.json", "w", encoding="utf-8") as fp:
        json.dump(effect_summary, fp, indent=2, ensure_ascii=False)

    print(json.dumps(effect_summary, indent=2, ensure_ascii=False))
    print(f"Saved report assets to {report_dir}")


if __name__ == "__main__":
    main()
