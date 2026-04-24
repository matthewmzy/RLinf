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

"""Abstract base class for robot end-effectors."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class EndEffectorType(str, Enum):
    """Supported end-effector types for the Franka robot arm."""

    FRANKA_GRIPPER = "franka_gripper"
    ROBOTIQ_GRIPPER = "robotiq_gripper"
    RUIYAN_HAND = "ruiyan_hand"

    @property
    def is_gripper(self) -> bool:
        return self in (
            type(self).FRANKA_GRIPPER,
            type(self).ROBOTIQ_GRIPPER,
        )

    @property
    def is_hand(self) -> bool:
        return self == type(self).RUIYAN_HAND

    @property
    def gripper_backend(self) -> str:
        if self == type(self).FRANKA_GRIPPER:
            return "franka"
        if self == type(self).ROBOTIQ_GRIPPER:
            return "robotiq"
        raise ValueError(f"{self.value!r} is not a gripper type")


def normalize_end_effector_type(
    end_effector_type: str | EndEffectorType,
    gripper_type: str | None = None,
) -> EndEffectorType:
    if isinstance(end_effector_type, str):
        end_effector_type = EndEffectorType(end_effector_type)

    if end_effector_type.is_hand or gripper_type is None:
        return end_effector_type
    if end_effector_type == EndEffectorType.ROBOTIQ_GRIPPER:
        return end_effector_type

    gt = gripper_type.lower()
    if gt == "franka":
        return EndEffectorType.FRANKA_GRIPPER
    if gt == "robotiq":
        return EndEffectorType.ROBOTIQ_GRIPPER
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        "Supported types: 'franka', 'robotiq'."
    )


class EndEffector(ABC):
    """Abstract interface for a robot end-effector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the end-effector action vector."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the end-effector state vector."""

    @property
    @abstractmethod
    def control_mode(self) -> str:
        """Control mode: ``"binary"`` (open/close) or ``"continuous"``."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Perform any hardware-level initialization (serial open, etc.)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully release hardware resources."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current end-effector state."""

    @property
    def finger_names(self) -> list[str]:
        """Human-readable names for each DOF."""
        return [f"dof_{i}" for i in range(self.state_dim)]

    def get_detailed_state(self) -> dict:
        """Return a detailed status dictionary for diagnostics."""
        state = self.get_state()
        return {
            "positions": state.tolist(),
            "finger_names": self.finger_names,
        }

    @abstractmethod
    def command(self, action: np.ndarray) -> bool:
        """Send a command to the end-effector."""

    @abstractmethod
    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a default or specified state."""
