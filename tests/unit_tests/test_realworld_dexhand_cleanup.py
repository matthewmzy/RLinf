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

from dataclasses import fields
import importlib
import sys
from types import SimpleNamespace
import types

import pytest

if "rlinf_dexhand" not in sys.modules:
    glove_module = types.ModuleType("rlinf_dexhand.glove")

    class _DummyGloveExpert:
        def __init__(self, *args, **kwargs):
            pass

        def get_angles(self):
            return [0.0] * 6

        def close(self):
            pass

    glove_module.GloveExpert = _DummyGloveExpert
    dexhand_module = types.ModuleType("rlinf_dexhand")
    dexhand_module.glove = glove_module
    sys.modules["rlinf_dexhand"] = dexhand_module
    sys.modules["rlinf_dexhand.glove"] = glove_module

from rlinf.envs.realworld.common.camera.base_camera import CameraInfo
from rlinf.envs.realworld.franka.franka_env import FrankaEnv

apply_wrappers = importlib.import_module("rlinf.envs.realworld.common.wrappers.apply")


def _make_identity_wrapper(name: str, calls: list[tuple[str, dict]]):
    def _wrapper(env, *args, **kwargs):
        calls.append((name, kwargs))
        return env

    return _wrapper


def test_apply_single_arm_wrappers_uses_gripper_and_dexhand_paths(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        apply_wrappers,
        "GripperCloseEnv",
        _make_identity_wrapper("gripper_close", calls),
    )
    monkeypatch.setattr(
        apply_wrappers,
        "SpacemouseIntervention",
        _make_identity_wrapper("spacemouse", calls),
    )
    monkeypatch.setattr(
        apply_wrappers,
        "DexHandIntervention",
        _make_identity_wrapper("dexhand", calls),
    )
    monkeypatch.setattr(
        apply_wrappers,
        "Quat2EulerWrapper",
        _make_identity_wrapper("quat2euler", calls),
    )
    monkeypatch.setattr(
        apply_wrappers,
        "RelativeFrame",
        _make_identity_wrapper("relative_frame", calls),
    )

    gripper_env = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False, end_effector_type="franka_gripper")
    )
    apply_wrappers.apply_single_arm_wrappers(
        gripper_env,
        {
            "no_gripper": True,
            "use_spacemouse": True,
            "use_gello": False,
            "use_relative_frame": False,
        },
    )

    assert ("gripper_close", {}) in calls
    assert ("spacemouse", {"gripper_enabled": False}) in calls
    assert ("dexhand", {}) not in calls

    calls.clear()
    dexhand_env = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False, end_effector_type="ruiyan_hand")
    )
    apply_wrappers.apply_single_arm_wrappers(
        dexhand_env,
        {
            "no_gripper": True,
            "use_spacemouse": True,
            "use_gello": False,
            "use_relative_frame": False,
            "glove_config": {"left_port": "/dev/mock_glove", "frequency": 30},
        },
    )

    assert ("gripper_close", {}) not in calls
    assert ("spacemouse", {"gripper_enabled": False}) not in calls
    assert (
        "dexhand",
        {
            "left_port": "/dev/mock_glove",
            "right_port": None,
            "glove_frequency": 30,
            "glove_config_file": None,
        },
    ) in calls


def test_apply_single_arm_wrappers_rejects_gello_for_dexhand():
    env = SimpleNamespace(
        config=SimpleNamespace(is_dummy=False, end_effector_type="ruiyan_hand")
    )

    with pytest.raises(ValueError, match="use_gello=True is not supported"):
        apply_wrappers.apply_single_arm_wrappers(
            env,
            {
                "no_gripper": False,
                "use_spacemouse": False,
                "use_gello": True,
                "use_relative_frame": False,
            },
        )


def test_build_camera_infos_keeps_name_and_crop_only():
    camera_fields = {field.name for field in fields(CameraInfo)}
    assert "auto_exposure" not in camera_fields
    assert "exposure" not in camera_fields
    assert "gain" not in camera_fields

    env = object.__new__(FrankaEnv)
    env.config = SimpleNamespace(
        camera_serials=["218722271009", "105322251046"],
        camera_type="realsense",
        camera_crop_regions={},
        camera_configs={
            "camera_defaults": {
                "resolution": [800, 600],
                "fps": 30,
                "enable_depth": True,
                "auto_exposure": False,
                "exposure": 120,
                "gain": 16,
            },
            "overrides": {
                "105322251046": {
                    "name": "global",
                    "crop_region": [0.4, 0.3, 0.85, 0.7],
                }
            },
        },
    )

    infos = FrankaEnv._build_camera_infos(env)

    assert [info.serial_number for info in infos] == ["105322251046", "218722271009"]
    assert infos[0].name == "global"
    assert infos[0].crop_region == (0.4, 0.3, 0.85, 0.7)
    assert infos[0].resolution == (800, 600)
    assert infos[0].fps == 30
    assert infos[0].enable_depth is True
    assert infos[1].name == "wrist_1"
    assert infos[1].crop_region is None
