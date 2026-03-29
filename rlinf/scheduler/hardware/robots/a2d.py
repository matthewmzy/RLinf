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

from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class A2DHWInfo(HardwareInfo):
    """Hardware information for an A2D controller node."""

    config: "A2DConfig"


@Hardware.register()
class A2DRobot(Hardware):
    """Hardware policy for nodes running the official A2D controller docker."""

    HW_TYPE = "A2D"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["A2DConfig"]] = None
    ) -> Optional[HardwareResource]:
        assert configs is not None, "A2D hardware requires explicit configurations."
        robot_configs = [
            config
            for config in configs
            if isinstance(config, A2DConfig) and config.node_rank == node_rank
        ]
        if not robot_configs:
            return None
        return HardwareResource(
            type=cls.HW_TYPE,
            infos=[
                A2DHWInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
                for config in robot_configs
            ],
        )


@NodeHardwareConfig.register_hardware_config(A2DRobot.HW_TYPE)
@dataclass
class A2DConfig(HardwareConfig):
    """Configuration for a node exposing an A2D controller service."""

    controller_host: str = "127.0.0.1"
    grpc_port: int = 12321
    container_name: Optional[str] = None
    grpc_config_file: Optional[str] = None
    auto_start_server: bool = False
    server_command: Optional[str] = None

    def __post_init__(self):
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in A2D config must be an integer. But got {type(self.node_rank)}."
        )
        self.controller_host = str(self.controller_host)
        self.grpc_port = int(self.grpc_port)
