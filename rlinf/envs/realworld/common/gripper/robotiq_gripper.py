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

"""Robotiq 2F gripper via direct Modbus RTU over USB-RS485."""

import inspect
import time
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base_gripper import BaseGripper

_SLAVE_ID = 0x09
_OUTPUT_REG_ADDR = 0x03E8
_INPUT_REG_ADDR = 0x07D0
_NUM_REGS = 3
_rACT = 1 << 0
_rGTO = 1 << 3


def _create_modbus_client(port: str, baudrate: int = 115200):
    """Create a pymodbus serial client (compatible with v2 and v3+)."""
    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError:
        from pymodbus.client.sync import ModbusSerialClient

    return ModbusSerialClient(
        port=port,
        baudrate=baudrate,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1,
    )


class RobotiqGripper(BaseGripper):
    """Robotiq 2F-85 / 2F-140 controlled via Modbus RTU over USB-RS485."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        slave_id: int = _SLAVE_ID,
        max_width: float = 0.085,
    ):
        self._logger = get_logger()
        self._port = port
        self._slave_id = slave_id
        self._max_width = max_width

        self._client = _create_modbus_client(port, baudrate)
        self._client.connect()

        sig = inspect.signature(self._client.write_registers)
        if "device_id" in sig.parameters:
            self._slave_kwarg = "device_id"
        else:
            self._slave_kwarg = "slave"

        self._cached_position: int = 0
        self._is_open_flag: bool = True
        self._activated: bool = False

        self._activate()
        self._logger.info(
            f"Robotiq gripper activated on {port} "
            f"(slave=0x{slave_id:02X}, max_width={max_width}m)"
        )

    def open(self, speed: float = 0.3) -> None:
        self._goto(position=0, speed=speed, force=0.0)
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        norm_force = min(force / 255.0, 1.0) if force > 1.0 else force
        self._goto(position=255, speed=speed, force=norm_force)
        self._is_open_flag = False

    def move(self, position: float, speed: float = 0.3) -> None:
        self._goto(position=int(np.clip(position, 0, 255)), speed=speed, force=0.5)

    @property
    def position(self) -> float:
        status = self._read_status()
        if status is not None:
            self._cached_position = status["position"]
        return self._max_width * (1.0 - self._cached_position / 255.0)

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        return self._activated

    def cleanup(self) -> None:
        self._client.close()

    def _activate(self) -> None:
        self._write_output_regs(0x0000, 0x0000, 0x0000)
        time.sleep(0.5)
        self._write_output_regs(_rACT << 8, 0x0000, 0x0000)

        for _ in range(50):
            time.sleep(0.1)
            status = self._read_status()
            if status is not None and status["gSTA"] == 0x03:
                self._activated = True
                return

        raise RuntimeError(
            f"Robotiq gripper on {self._port} did not activate within 5 s"
        )

    def _goto(self, position: int, speed: float, force: float) -> None:
        pos = int(np.clip(position, 0, 255))
        spd = int(np.clip(speed * 255, 0, 255))
        frc = int(np.clip(force * 255, 0, 255))

        reg0 = (_rACT | _rGTO) << 8
        reg1 = pos
        reg2 = (spd << 8) | frc
        self._write_output_regs(reg0, reg1, reg2)

    def _write_output_regs(self, reg0: int, reg1: int, reg2: int) -> None:
        self._client.write_registers(
            address=_OUTPUT_REG_ADDR,
            values=[reg0, reg1, reg2],
            **{self._slave_kwarg: self._slave_id},
        )

    def _read_status(self) -> Optional[dict]:
        resp = self._client.read_holding_registers(
            address=_INPUT_REG_ADDR,
            count=_NUM_REGS,
            **{self._slave_kwarg: self._slave_id},
        )
        if resp.isError():
            self._logger.warning(f"Robotiq Modbus read error: {resp}")
            return None

        r0, r1, r2 = resp.registers
        status_byte = (r0 >> 8) & 0xFF
        return {
            "gACT": (status_byte >> 0) & 0x01,
            "gGTO": (status_byte >> 3) & 0x01,
            "gSTA": (status_byte >> 4) & 0x03,
            "gOBJ": (status_byte >> 6) & 0x03,
            "fault": (r1 >> 8) & 0xFF,
            "position_echo": r1 & 0xFF,
            "position": (r2 >> 8) & 0xFF,
            "current": r2 & 0xFF,
        }
