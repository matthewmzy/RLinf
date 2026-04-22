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

import os
from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
EMBODIED_PATH = str(CONFIG_DIR.parent)


def _compose(config_name: str):
    previous = os.environ.get("EMBODIED_PATH")
    os.environ["EMBODIED_PATH"] = EMBODIED_PATH
    try:
        with hydra.initialize_config_dir(
            version_base="1.1",
            config_dir=str(CONFIG_DIR),
        ):
            return hydra.compose(config_name=config_name)
    finally:
        if previous is None:
            os.environ.pop("EMBODIED_PATH", None)
        else:
            os.environ["EMBODIED_PATH"] = previous


@pytest.mark.parametrize(
    ("config_name", "mode", "safe_enabled"),
    [
        ("realworld_a2d_sac_psi_rtc_repro", "rtc", False),
        ("realworld_a2d_sac_psi_rtc_async", "rtc", True),
        ("realworld_a2d_sac_psi_direct_repro", "direct", False),
        ("realworld_a2d_sac_psi_direct_async", "direct", True),
        ("realworld_a2d_sac_psi_window_repro", "window_blend", False),
        ("realworld_a2d_sac_psi_window_async", "window_blend", True),
    ],
)
def test_new_psi_policy_presets_compose(config_name, mode, safe_enabled):
    cfg = _compose(config_name)

    assert cfg.rollout.action_execution.mode == mode
    assert cfg.rollout.action_execution.noise_stage == "pre_smooth"
    assert cfg.actor.model.num_action_chunks == 4
    assert cfg.env.train.override_cfg.safe_box.enabled is safe_enabled
    assert cfg.env.eval.override_cfg.safe_box.enabled is safe_enabled


@pytest.mark.parametrize(
    ("alias_name", "expected_mode"),
    [
        ("realworld_a2d_sac_psi_async", "direct"),
        ("realworld_a2d_sac_psi_async_repro", "rtc"),
        ("realworld_a2d_sac_psi_safe_explore", "direct"),
        ("realworld_a2d_sac_psi", "direct"),
    ],
)
def test_legacy_alias_configs_still_compose(alias_name, expected_mode):
    cfg = _compose(alias_name)

    assert cfg.rollout.action_execution.mode == expected_mode


def test_async_preset_keeps_rollout_model_noise_override():
    cfg = _compose("realworld_a2d_sac_psi_direct_async")

    assert cfg.actor.model.noise_std_train == 0.05
    assert cfg.rollout.model.noise_std_rollout == 0.005
    assert (
        OmegaConf.to_container(cfg.rollout.model, resolve=True)["normalizer_path"]
        == "/home/psibot/workspace_zhiyuan/psi-policy/data/outputs/2026.04.10/04.54_zjxc_v0.8_vit-small_rgb-state/normalizer.pkl"
    )
