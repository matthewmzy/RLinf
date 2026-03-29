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

import subprocess
import time
from typing import Optional

import grpc
import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .a2d_robot_state import A2DRobotState
from .proto import robot_control_pb2


class RobotControlStub:
    """Minimal client stub for the vendored RobotControl gRPC service."""

    def __init__(self, channel: grpc.Channel):
        self.reset = channel.unary_unary(
            "/robot_control.RobotControl/reset",
            request_serializer=robot_control_pb2.ResetRequest.SerializeToString,
            response_deserializer=robot_control_pb2.ResetResponse.FromString,
        )
        self.get_obs = channel.unary_unary(
            "/robot_control.RobotControl/get_obs",
            request_serializer=robot_control_pb2.GetObsRequest.SerializeToString,
            response_deserializer=robot_control_pb2.GetObsResponse.FromString,
        )
        self.set_action = channel.unary_unary(
            "/robot_control.RobotControl/set_action",
            request_serializer=robot_control_pb2.SetActionRequest.SerializeToString,
            response_deserializer=robot_control_pb2.SetActionResponse.FromString,
        )
        self.health_check = channel.unary_unary(
            "/robot_control.RobotControl/health_check",
            request_serializer=robot_control_pb2.HealthRequest.SerializeToString,
            response_deserializer=robot_control_pb2.HealthResponse.FromString,
        )


class A2DController(Worker):
    """Client worker that treats a running A2D docker as the robot controller."""

    MAX_GRPC_MESSAGE_LENGTH = 100 * 1024 * 1024

    @staticmethod
    def launch_controller(
        controller_host: str,
        controller_port: int,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        grpc_timeout_s: float = 5.0,
        ready_timeout_s: float = 60.0,
        container_name: Optional[str] = None,
        grpc_config_file: Optional[str] = None,
        auto_start_server: bool = False,
        server_command: Optional[str] = None,
    ):
        """Launch an A2D controller client on the specified node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return A2DController.create_group(
            controller_host,
            controller_port,
            grpc_timeout_s,
            ready_timeout_s,
            container_name,
            grpc_config_file,
            auto_start_server,
            server_command,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"A2DController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        controller_host: str,
        controller_port: int = 12321,
        grpc_timeout_s: float = 5.0,
        ready_timeout_s: float = 60.0,
        container_name: Optional[str] = None,
        grpc_config_file: Optional[str] = None,
        auto_start_server: bool = False,
        server_command: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()
        self.controller_host = controller_host
        self.controller_port = int(controller_port)
        self.grpc_timeout_s = float(grpc_timeout_s)
        self.ready_timeout_s = float(ready_timeout_s)
        self.container_name = container_name
        self.grpc_config_file = grpc_config_file
        self.auto_start_server = auto_start_server
        self.server_command = server_command
        self._server_started_by_rlinf = False

        options = [
            ("grpc.max_send_message_length", self.MAX_GRPC_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", self.MAX_GRPC_MESSAGE_LENGTH),
        ]
        self._channel = grpc.insecure_channel(
            f"{self.controller_host}:{self.controller_port}",
            options=options,
        )
        self._stub = RobotControlStub(self._channel)

        if self.auto_start_server:
            self._start_server_in_container()
        self._wait_until_ready(self.ready_timeout_s)

    def _start_server_in_container(self) -> None:
        """Optionally start the official A2D gRPC server inside the running container."""
        if not self.container_name:
            raise ValueError(
                "container_name must be provided when auto_start_server is enabled."
            )
        if self.is_robot_up():
            return

        command = self.server_command or (
            "source /ros_entrypoint.sh && "
            "source /opt/psi/rt/a2d-tele/install/setup.bash && "
            "python3 -m model_inference.run_inference_server"
        )
        if self.grpc_config_file:
            command = (
                f"export MODEL_INFERENCE_CONFIG_FILE={self.grpc_config_file} && {command}"
            )

        subprocess.run(
            [
                "docker",
                "exec",
                "-d",
                self.container_name,
                "bash",
                "-lc",
                command,
            ],
            check=True,
        )
        self._server_started_by_rlinf = True
        self._logger.info(
            "Started A2D model_inference gRPC server inside container %s",
            self.container_name,
        )

    def _wait_until_ready(self, timeout_s: float) -> None:
        """Block until the controller service reports readiness."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_robot_up():
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"A2D controller at {self.controller_host}:{self.controller_port} "
            f"did not become ready within {timeout_s} seconds."
        )

    @staticmethod
    def _decode_image(image_data: robot_control_pb2.ImageData) -> np.ndarray:
        dtype = np.dtype(image_data.dtype)
        image = np.frombuffer(image_data.data, dtype=dtype)
        if image_data.channels == 1:
            return image.reshape(image_data.height, image_data.width).copy()
        return image.reshape(
            image_data.height, image_data.width, image_data.channels
        ).copy()

    @staticmethod
    def _decode_array(array_data: robot_control_pb2.ArrayData) -> np.ndarray:
        values = np.asarray(array_data.values, dtype=np.float32)
        if array_data.shape:
            return values.reshape(tuple(array_data.shape))
        return values

    @classmethod
    def _decode_observation(
        cls,
        observation: robot_control_pb2.Observation,
    ) -> A2DRobotState:
        return A2DRobotState(
            images={
                key: cls._decode_image(value)
                for key, value in observation.images.items()
            },
            states={
                key: cls._decode_array(value)
                for key, value in observation.states.items()
            },
            timestamps=dict(observation.timestamps),
            control_mode=observation.control_mode
            if observation.HasField("control_mode")
            else None,
            trajectory_label=observation.trajectory_label
            if observation.HasField("trajectory_label")
            else None,
            is_switch_mode=observation.is_switch_mode
            if observation.HasField("is_switch_mode")
            else None,
        )

    def is_robot_up(self) -> bool:
        """Check whether the remote controller service is ready."""
        try:
            response = self._stub.health_check(
                robot_control_pb2.HealthRequest(),
                timeout=self.grpc_timeout_s,
            )
            return bool(response.is_ready)
        except grpc.RpcError:
            return False

    def get_state(self) -> A2DRobotState:
        """Fetch the latest observation from the running A2D controller."""
        response = self._stub.get_obs(
            robot_control_pb2.GetObsRequest(),
            timeout=self.grpc_timeout_s,
        )
        if not response.success:
            raise RuntimeError(f"Failed to get A2D observation: {response.message}")
        return self._decode_observation(response.observation)

    def reset(self, seed: Optional[int] = None) -> A2DRobotState:
        """Call the controller reset API and return the initial observation."""
        request = robot_control_pb2.ResetRequest()
        if seed is not None:
            request.seed = int(seed)
        response = self._stub.reset(request, timeout=self.grpc_timeout_s)
        if not response.success:
            raise RuntimeError(f"Failed to reset A2D controller: {response.message}")
        return self._decode_observation(response.observation)

    def set_action(self, action: np.ndarray) -> bool:
        """Send one action vector to the controller service."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        request = robot_control_pb2.SetActionRequest(
            action=robot_control_pb2.Action(
                values=action.tolist(),
                dimension=int(action.size),
            )
        )
        response = self._stub.set_action(request, timeout=self.grpc_timeout_s)
        if not response.success:
            self._logger.warning("A2D controller rejected action: %s", response.message)
        return bool(response.success)
