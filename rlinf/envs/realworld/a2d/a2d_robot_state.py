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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class A2DRobotState:
    """Decoded A2D controller state returned by the gRPC robot service."""

    images: dict[str, np.ndarray] = field(default_factory=dict)
    states: dict[str, np.ndarray] = field(default_factory=dict)
    timestamps: dict[str, float] = field(default_factory=dict)
    control_mode: Optional[int] = None
    trajectory_label: Optional[int] = None
    is_switch_mode: Optional[bool] = None

    def copy(self) -> "A2DRobotState":
        """Return a deep-ish copy that is safe to hand to env code."""
        return A2DRobotState(
            images={key: value.copy() for key, value in self.images.items()},
            states={key: value.copy() for key, value in self.states.items()},
            timestamps=self.timestamps.copy(),
            control_mode=self.control_mode,
            trajectory_label=self.trajectory_label,
            is_switch_mode=self.is_switch_mode,
        )
