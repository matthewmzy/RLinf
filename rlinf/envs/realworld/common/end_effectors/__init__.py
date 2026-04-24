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

"""Shared real-world end-effector backends and abstractions."""

from typing import Optional

from .base import EndEffector, EndEffectorType, normalize_end_effector_type
from .base_gripper import BaseGripper

__all__ = [
    "BaseGripper",
    "EndEffector",
    "EndEffectorType",
    "create_end_effector",
    "create_gripper",
    "normalize_end_effector_type",
]


def create_gripper(
    gripper_type: str = "franka",
    ros=None,
    port: Optional[str] = None,
    **kwargs,
) -> BaseGripper:
    """Factory that instantiates the right gripper backend."""
    gt = gripper_type.lower()
    if gt == "robotiq":
        if port is None:
            raise ValueError(
                "gripper_connection (serial port) must be specified "
                "for Robotiq grippers."
            )
        from .robotiq_gripper import RobotiqGripper

        return RobotiqGripper(port=port, **kwargs)

    if gt == "franka":
        if ros is None:
            raise ValueError(
                "ROSController instance must be provided for Franka gripper."
            )
        from .franka_gripper import FrankaGripper

        return FrankaGripper(ros=ros, **kwargs)

    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        "Supported types: 'franka', 'robotiq'."
    )


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
