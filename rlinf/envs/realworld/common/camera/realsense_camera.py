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

import queue
import time
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base_camera import BaseCamera, CameraInfo

_logger = get_logger()


class RealSenseCamera(BaseCamera):
    """Camera capture for Intel RealSense cameras.

    Adapted from SERL's RSCapture class.
    For RealSense usage, see
    https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/quick_start_live.ipynb.
    """

    def __init__(self, camera_info: CameraInfo):
        import pyrealsense2 as rs

        super().__init__(camera_info)
        self._rs = rs

        self._device_info = {}
        for device in rs.context().devices:
            serial = device.get_info(rs.camera_info.serial_number)
            self._device_info[serial] = device
        assert camera_info.serial_number in self._device_info, (
            f"Available RealSense devices: {tuple(self._device_info.keys())}"
        )

        self._serial_number = camera_info.serial_number
        self._device = self._device_info[self._serial_number]
        self._enable_depth = camera_info.enable_depth

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self._serial_number)
        self._config.enable_stream(
            rs.stream.color,
            camera_info.resolution[0],
            camera_info.resolution[1],
            rs.format.bgr8,
            camera_info.fps,
        )
        if self._enable_depth:
            self._config.enable_stream(
                rs.stream.depth,
                camera_info.resolution[0],
                camera_info.resolution[1],
                rs.format.z16,
                camera_info.fps,
            )
        self.profile = self._pipeline.start(self._config)
        self._configure_color_sensor_options()

        # Align depth frames to the color stream when depth is enabled.
        self._align = rs.align(rs.stream.color)

    def _configure_color_sensor_options(self) -> None:
        """Apply optional exposure settings for the RealSense color sensor."""
        rs = self._rs

        try:
            color_sensor = self.profile.get_device().first_color_sensor()
        except Exception as exc:
            _logger.warning(
                "Could not access RealSense color sensor for %s: %s",
                self._camera_info.name,
                exc,
            )
            return

        if color_sensor is None:
            return

        if color_sensor.supports(rs.option.enable_auto_exposure):
            auto_val = 1.0 if self._camera_info.auto_exposure else 0.0
            color_sensor.set_option(rs.option.enable_auto_exposure, auto_val)

        if self._camera_info.auto_exposure:
            return

        if (
            self._camera_info.exposure is not None
            and color_sensor.supports(rs.option.exposure)
        ):
            color_sensor.set_option(
                rs.option.exposure,
                float(self._camera_info.exposure),
            )

        if self._camera_info.gain is not None and color_sensor.supports(
            rs.option.gain
        ):
            color_sensor.set_option(
                rs.option.gain,
                float(self._camera_info.gain),
            )

    def _capture_frames(self):
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            try:
                has_frame, frame = self._read_frame()
            except RuntimeError:
                # RealSense may timeout on the first few frames; retry.
                continue
            if not has_frame:
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self._enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame.is_video_frame():
            return False, None

        frame = np.asarray(color_frame.get_data())
        if self._enable_depth and depth_frame.is_depth_frame():
            depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
            return True, np.concatenate((frame, depth), axis=-1)
        return True, frame

    def _close_device(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass
        self._config.disable_all_streams()

    @staticmethod
    def get_device_serial_numbers() -> set[str]:
        """Return serial numbers of all connected RealSense cameras."""
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras
        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras
