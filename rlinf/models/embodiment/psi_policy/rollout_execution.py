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

"""Psi-policy rollout-side action execution adapters."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class PsiPolicyActionExecutionResult:
    executed_actions: torch.Tensor
    model_actions: torch.Tensor
    exact_logprobs: bool


@dataclass
class _RTCExecutionState:
    action_queue: deque[torch.Tensor] = field(default_factory=deque)
    last_action: Optional[torch.Tensor] = None


@dataclass
class _WindowBlendExecutionState:
    prev_overlap_tail: Optional[torch.Tensor] = None


def resolve_action_execution_cfg(
    raw_cfg: Any,
    *,
    execute_step: int,
    action_horizon: int,
    legacy_rollout_rtc_enabled: bool = False,
    legacy_rollout_rtc_search_window: Optional[int] = None,
    legacy_rollout_rtc_merge_weight_base: float = 0.8,
) -> dict[str, Any]:
    cfg = {
        "mode": "direct",
        "noise_stage": "pre_smooth",
        "reset_on_episode_end": True,
        "rtc": {
            "search_window": None,
            "merge_weight_base": 0.8,
        },
        "window": {
            "window_size": None,
            "overlap_steps": None,
        },
    }

    if raw_cfg is not None:
        if hasattr(raw_cfg, "items"):
            raw_cfg = copy.deepcopy(dict(raw_cfg))
        else:
            raise TypeError(
                f"rollout.action_execution must be a mapping or None, got {type(raw_cfg)}."
            )
        for key, value in raw_cfg.items():
            if key in {"rtc", "window"} and value is not None:
                cfg[key].update(dict(value))
            else:
                cfg[key] = value
    elif legacy_rollout_rtc_enabled:
        cfg["mode"] = "rtc"
        cfg["rtc"]["search_window"] = legacy_rollout_rtc_search_window
        cfg["rtc"]["merge_weight_base"] = legacy_rollout_rtc_merge_weight_base

    cfg["mode"] = str(cfg.get("mode", "direct"))
    cfg["noise_stage"] = str(cfg.get("noise_stage", "pre_smooth"))
    cfg["reset_on_episode_end"] = bool(cfg.get("reset_on_episode_end", True))
    cfg["rtc"]["merge_weight_base"] = float(
        cfg["rtc"].get("merge_weight_base", legacy_rollout_rtc_merge_weight_base)
    )
    search_window = cfg["rtc"].get("search_window")
    cfg["rtc"]["search_window"] = (
        execute_step if search_window is None else int(search_window)
    )

    window_size = cfg["window"].get("window_size")
    overlap_steps = cfg["window"].get("overlap_steps")
    cfg["window"]["window_size"] = (
        min(action_horizon, execute_step * 2)
        if window_size is None
        else int(window_size)
    )
    cfg["window"]["overlap_steps"] = (
        execute_step if overlap_steps is None else int(overlap_steps)
    )

    valid_modes = {"direct", "rtc", "window_blend"}
    if cfg["mode"] not in valid_modes:
        raise ValueError(
            f"Unsupported rollout.action_execution.mode '{cfg['mode']}'. "
            f"Valid modes: {sorted(valid_modes)}"
        )
    if cfg["noise_stage"] != "pre_smooth":
        raise ValueError(
            "Psi-policy rollout.action_execution only supports noise_stage='pre_smooth'."
        )
    if execute_step <= 0:
        raise ValueError("Psi-policy action execution execute_step must be > 0.")
    if execute_step > action_horizon:
        raise ValueError(
            f"execute_step={execute_step} cannot exceed action_horizon={action_horizon}."
        )
    if cfg["rtc"]["search_window"] <= 0:
        raise ValueError("rollout.action_execution.rtc.search_window must be > 0.")
    if cfg["rtc"]["merge_weight_base"] <= 0:
        raise ValueError(
            "rollout.action_execution.rtc.merge_weight_base must be > 0."
        )
    if cfg["window"]["window_size"] < execute_step:
        raise ValueError(
            "rollout.action_execution.window.window_size must be >= execute_step."
        )
    if cfg["window"]["window_size"] > action_horizon:
        raise ValueError(
            "rollout.action_execution.window.window_size cannot exceed action_horizon."
        )
    if not 0 <= cfg["window"]["overlap_steps"] <= execute_step:
        raise ValueError(
            "rollout.action_execution.window.overlap_steps must be in "
            f"[0, execute_step], got {cfg['window']['overlap_steps']}."
        )
    if execute_step + cfg["window"]["overlap_steps"] > cfg["window"]["window_size"]:
        raise ValueError(
            "rollout.action_execution.window requires execute_step + overlap_steps "
            "<= window_size."
        )
    return cfg


class PsiPolicyActionExecutionAdapter:
    def __init__(
        self,
        *,
        execute_step: int,
        action_horizon: int,
        cfg: dict[str, Any],
    ) -> None:
        self.execute_step = int(execute_step)
        self.action_horizon = int(action_horizon)
        self.cfg = cfg
        self.mode = cfg["mode"]
        self.reset_on_episode_end = bool(cfg["reset_on_episode_end"])
        self.search_window = int(cfg["rtc"]["search_window"])
        self.merge_weight_base = float(cfg["rtc"]["merge_weight_base"])
        self.window_size = int(cfg["window"]["window_size"])
        self.overlap_steps = int(cfg["window"]["overlap_steps"])
        self._states: list[_RTCExecutionState | _WindowBlendExecutionState] = []

    @property
    def is_stateful(self) -> bool:
        return self.mode in {"rtc", "window_blend"}

    def reset(self, reset_mask: Any = None) -> None:
        if reset_mask is None:
            self._states = []
            return
        if not self.reset_on_episode_end or not self.is_stateful:
            return
        reset_mask = torch.as_tensor(reset_mask, dtype=torch.bool).reshape(-1)
        self._ensure_states(int(reset_mask.shape[0]))
        for env_idx, should_reset in enumerate(reset_mask.tolist()):
            if should_reset:
                self._states[env_idx] = self._new_state()

    def apply(
        self,
        predicted_actions: torch.Tensor,
        model_predicted_actions: Optional[torch.Tensor] = None,
    ) -> PsiPolicyActionExecutionResult:
        if predicted_actions.ndim != 3:
            raise ValueError(
                "Psi-policy action execution expects [B, horizon, action_dim], got "
                f"{tuple(predicted_actions.shape)}."
            )
        if predicted_actions.shape[1] < self.execute_step:
            raise ValueError(
                f"Need at least {self.execute_step} actions to execute, got "
                f"horizon={predicted_actions.shape[1]}."
            )

        if self.mode == "direct":
            model_actions = (
                model_predicted_actions[:, : self.execute_step].contiguous()
                if model_predicted_actions is not None
                else predicted_actions[:, : self.execute_step].contiguous()
            )
            return PsiPolicyActionExecutionResult(
                executed_actions=predicted_actions[:, : self.execute_step].contiguous(),
                model_actions=model_actions,
                exact_logprobs=True,
            )
        if self.mode == "rtc":
            executed_actions = self._apply_rtc(predicted_actions)
            return PsiPolicyActionExecutionResult(
                executed_actions=executed_actions,
                model_actions=executed_actions,
                exact_logprobs=False,
            )
        if self.mode == "window_blend":
            executed_actions = self._apply_window_blend(predicted_actions)
            return PsiPolicyActionExecutionResult(
                executed_actions=executed_actions,
                model_actions=executed_actions,
                exact_logprobs=False,
            )
        raise RuntimeError(f"Unexpected psi-policy execution mode: {self.mode}")

    def _new_state(self):
        if self.mode == "rtc":
            return _RTCExecutionState()
        if self.mode == "window_blend":
            return _WindowBlendExecutionState()
        return None

    def _ensure_states(self, batch_size: int) -> None:
        if not self.is_stateful:
            return
        if len(self._states) != batch_size:
            self._states = [self._new_state() for _ in range(batch_size)]

    def _rtc_find_nearest_point_index(
        self, predictions: torch.Tensor, reference_point: torch.Tensor
    ) -> int:
        search_steps = min(self.search_window, int(predictions.shape[0]))
        distances = torch.linalg.vector_norm(
            predictions[:search_steps] - reference_point.unsqueeze(0),
            dim=-1,
        )
        return int(torch.argmin(distances).item())

    def _rtc_merge_prediction_queue(
        self, rtc_state: _RTCExecutionState, predictions: torch.Tensor
    ) -> None:
        if len(rtc_state.action_queue) == 0:
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
        if len(predictions_to_merge) < self.execute_step:
            raise RuntimeError(
                "RTC action execution cannot provide enough future actions after "
                f"alignment: start_index={start_index}, horizon={predictions.shape[0]}, "
                f"execute_step={self.execute_step}."
            )

        merge_length = min(len(rtc_state.action_queue), len(predictions_to_merge))
        for idx in range(merge_length):
            weight = self.merge_weight_base / (idx**2 + 1)
            predictions_to_merge[idx] = (
                weight * rtc_state.action_queue[idx]
                + (1.0 - weight) * predictions_to_merge[idx]
            )
        rtc_state.action_queue = deque(predictions_to_merge)

    def _rtc_pop_action_chunk(
        self, rtc_state: _RTCExecutionState, chunk_size: int
    ) -> torch.Tensor:
        if len(rtc_state.action_queue) < chunk_size:
            raise RuntimeError(
                "RTC action execution queue underflow: "
                f"requested {chunk_size}, available {len(rtc_state.action_queue)}."
            )
        chunk_actions = []
        for _ in range(chunk_size):
            action = rtc_state.action_queue.popleft().clone()
            rtc_state.last_action = action
            chunk_actions.append(action)
        return torch.stack(chunk_actions, dim=0)

    def _apply_rtc(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        batch_size = int(predicted_actions.shape[0])
        self._ensure_states(batch_size)
        merged_chunks = []
        for env_idx in range(batch_size):
            rtc_state = self._states[env_idx]
            assert isinstance(rtc_state, _RTCExecutionState)
            self._rtc_merge_prediction_queue(rtc_state, predicted_actions[env_idx])
            merged_chunks.append(self._rtc_pop_action_chunk(rtc_state, self.execute_step))
        return torch.stack(merged_chunks, dim=0)

    def _store_window_overlap(
        self,
        state: _WindowBlendExecutionState,
        prediction_window: torch.Tensor,
    ) -> None:
        if self.overlap_steps <= 0:
            state.prev_overlap_tail = None
            return
        overlap_start = self.execute_step
        overlap_end = overlap_start + self.overlap_steps
        state.prev_overlap_tail = prediction_window[overlap_start:overlap_end].clone()

    def _apply_window_blend(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        batch_size = int(predicted_actions.shape[0])
        self._ensure_states(batch_size)
        current_windows = predicted_actions[:, : self.window_size].contiguous()
        blended_chunks = []

        for env_idx in range(batch_size):
            state = self._states[env_idx]
            assert isinstance(state, _WindowBlendExecutionState)
            prediction_window = current_windows[env_idx]

            if (
                self.overlap_steps > 0
                and state.prev_overlap_tail is not None
                and state.prev_overlap_tail.shape[0] == self.overlap_steps
            ):
                blended_overlap = 0.5 * state.prev_overlap_tail + 0.5 * prediction_window[
                    : self.overlap_steps
                ]
                if self.execute_step > self.overlap_steps:
                    executed_actions = torch.cat(
                        [
                            blended_overlap,
                            prediction_window[
                                self.overlap_steps : self.execute_step
                            ].clone(),
                        ],
                        dim=0,
                    )
                else:
                    executed_actions = blended_overlap[: self.execute_step].clone()
            else:
                executed_actions = prediction_window[: self.execute_step].clone()

            self._store_window_overlap(state, prediction_window)
            blended_chunks.append(executed_actions)

        return torch.stack(blended_chunks, dim=0)
