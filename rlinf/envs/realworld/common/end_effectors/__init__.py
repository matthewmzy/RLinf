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

"""Shared hand backends and end-effector abstractions."""

# Backward-compatible re-exports for older gripper import paths.
from rlinf.envs.realworld.common.gripper import BaseGripper as BaseGripper
from rlinf.envs.realworld.common.gripper import create_gripper as create_gripper

from .base import EndEffector, EndEffectorType, normalize_end_effector_type

__all__ = [
    "EndEffector",
    "EndEffectorType",
    "create_end_effector",
    "normalize_end_effector_type",
]


def create_end_effector(
    end_effector_type: str | EndEffectorType,
    **kwargs,
) -> EndEffector:
    """Factory function to create a non-gripper end-effector instance."""
    if isinstance(end_effector_type, str):
        end_effector_type = EndEffectorType(end_effector_type)

    if end_effector_type == EndEffectorType.RUIYAN_HAND:
        from .ruiyan_hand import RuiyanHand

        return RuiyanHand(**kwargs)

    raise ValueError(
        f"Unsupported end-effector type: {end_effector_type}. "
        "Supported types: ['ruiyan_hand']"
    )
