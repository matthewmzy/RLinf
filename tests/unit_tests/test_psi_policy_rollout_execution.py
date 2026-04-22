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
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from rlinf.models.embodiment.psi_policy.rollout_execution import (
    PsiPolicyActionExecutionAdapter,
    resolve_action_execution_cfg,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _make_cfg(mode: str, **overrides):
    cfg = {
        "mode": mode,
        "noise_stage": "pre_smooth",
        "reset_on_episode_end": True,
        "rtc": {
            "search_window": 2,
            "merge_weight_base": 0.8,
        },
        "window": {
            "window_size": 8,
            "overlap_steps": 4,
        },
    }
    for key, value in overrides.items():
        if key in {"rtc", "window"}:
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def test_direct_action_execution_keeps_first_macro_chunk():
    adapter = PsiPolicyActionExecutionAdapter(
        execute_step=2,
        action_horizon=6,
        cfg=_make_cfg("direct"),
    )
    predictions = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    result = adapter.apply(predictions, predictions + 100.0)

    assert result.exact_logprobs is True
    assert torch.equal(result.executed_actions, predictions[:, :2])
    assert torch.equal(result.model_actions, (predictions + 100.0)[:, :2])


def test_rtc_action_execution_matches_old_client_semantics_and_resets():
    adapter = PsiPolicyActionExecutionAdapter(
        execute_step=2,
        action_horizon=6,
        cfg=_make_cfg("rtc"),
    )
    first_prediction = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    second_prediction = first_prediction + 1000.0

    first = adapter.apply(first_prediction)
    second = adapter.apply(second_prediction)

    expected_second = torch.stack(
        [
            0.8 * first_prediction[0, 2] + 0.2 * second_prediction[0, 0],
            0.4 * first_prediction[0, 3] + 0.6 * second_prediction[0, 1],
        ],
        dim=0,
    ).unsqueeze(0)

    assert torch.equal(first.executed_actions, first_prediction[:, :2])
    assert second.exact_logprobs is False
    assert torch.allclose(second.executed_actions, expected_second)

    adapter.reset(torch.tensor([True]))
    after_reset = adapter.apply(second_prediction)
    assert torch.equal(after_reset.executed_actions, second_prediction[:, :2])


def test_window_blend_two_step_example_and_reset():
    adapter = PsiPolicyActionExecutionAdapter(
        execute_step=4,
        action_horizon=8,
        cfg=_make_cfg("window_blend"),
    )
    first_prediction = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    second_prediction = first_prediction + 100.0

    first = adapter.apply(first_prediction)
    second = adapter.apply(second_prediction)

    expected_second = 0.5 * first_prediction[:, 4:8] + 0.5 * second_prediction[:, :4]

    assert torch.equal(first.executed_actions, first_prediction[:, :4])
    assert second.exact_logprobs is False
    assert torch.allclose(second.executed_actions, expected_second)

    adapter.reset(torch.tensor([True]))
    after_reset = adapter.apply(second_prediction)
    assert torch.equal(after_reset.executed_actions, second_prediction[:, :4])


def test_resolve_action_execution_cfg_uses_legacy_rtc_alias():
    cfg = resolve_action_execution_cfg(
        None,
        execute_step=4,
        action_horizon=16,
        legacy_rollout_rtc_enabled=True,
        legacy_rollout_rtc_search_window=4,
        legacy_rollout_rtc_merge_weight_base=0.7,
    )

    assert cfg["mode"] == "rtc"
    assert cfg["rtc"]["search_window"] == 4
    assert cfg["rtc"]["merge_weight_base"] == 0.7


def test_rollout_model_config_deep_merge_reads_rollout_only_overrides():
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "psi_policy",
                    "model_path": "/actor/model.ckpt",
                    "normalizer_path": "/actor/normalizer.pkl",
                    "noise_std_rollout": 0.0,
                    "precision": "bf16",
                }
            },
            "rollout": {
                "model": {
                    "model_path": "/rollout/model.ckpt",
                    "normalizer_path": "/rollout/normalizer.pkl",
                    "noise_std_rollout": 0.02,
                    "precision": "fp16",
                }
            },
        }
    )

    rollout_model_cfg = MultiStepRolloutWorker._build_rollout_model_config(worker)

    assert rollout_model_cfg.model_path == "/rollout/model.ckpt"
    assert rollout_model_cfg.normalizer_path == "/rollout/normalizer.pkl"
    assert rollout_model_cfg.noise_std_rollout == 0.02
    assert rollout_model_cfg.precision == "fp16"
