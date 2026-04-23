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

import os
import threading
from collections import deque


class KeyboardListener:
    """Keyboard listener with pynput and evdev backends."""

    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_F", "KEY_Q", "KEY_S")

    def __init__(self, allowed_keys: set[str] | None = None):
        self.state_lock = threading.Lock()
        self._pressed_keys = set()
        self._pending_keys = deque()
        self._allowed_keys = (
            {str(key).lower() for key in allowed_keys}
            if allowed_keys is not None
            else None
        )
        self.last_intervene = 0
        self._stop_event = threading.Event()
        self._thread = None
        self.device = None
        self.listener = None

        if allowed_keys is None:
            try:
                from evdev import InputDevice, ecodes, list_devices

                self._input_device_cls = InputDevice
                self._ecodes = ecodes
                self._list_devices = list_devices
                self.device = self._open_keyboard_device()
                self.listener = self
                self._thread = threading.Thread(
                    target=self._listen_loop,
                    name=f"KeyboardListener:{self.device.path}",
                    daemon=True,
                )
                self._thread.start()
                return
            except ImportError:
                pass

        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "KeyboardListener requires either 'evdev' or 'pynput'."
            ) from exc

        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
        )
        self.listener.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.listener is not None and hasattr(self.listener, "stop"):
            self.listener.stop()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.device is not None:
            self.device.close()

    def _normalize_key(self, key) -> str:
        if hasattr(key, "char") and key.char is not None:
            return str(key.char).lower()
        return str(key).lower()

    def _handle_key_press(self, normalized_key: str) -> None:
        if self._allowed_keys is not None and normalized_key not in self._allowed_keys:
            return
        with self.state_lock:
            if normalized_key in self._pressed_keys:
                return
            self._pressed_keys.add(normalized_key)
            self._pending_keys.append(normalized_key)

    def _handle_key_release(self, normalized_key: str) -> None:
        with self.state_lock:
            self._pressed_keys.discard(normalized_key)

    def on_key_press(self, key):
        self._handle_key_press(self._normalize_key(key))

    def on_key_release(self, key):
        self._handle_key_release(self._normalize_key(key))

    def _open_keyboard_device(self):
        override_path = os.environ.get("RLINF_KEYBOARD_DEVICE")
        if override_path:
            device = self._open_device(override_path, is_override=True)
            if not self._is_keyboard_device(device):
                device.close()
                raise RuntimeError(
                    "KeyboardListener device set by "
                    f"RLINF_KEYBOARD_DEVICE='{override_path}' does not look like a "
                    "keyboard device. Point it to the correct /dev/input/eventX path."
                )
            return device

        permission_denied_paths: list[str] = []
        for device_path in sorted(self._list_devices()):
            try:
                device = self._open_device(device_path)
            except PermissionError:
                permission_denied_paths.append(device_path)
                continue

            if self._is_keyboard_device(device):
                return device
            device.close()

        if permission_denied_paths:
            denied = ", ".join(permission_denied_paths)
            raise RuntimeError(
                "KeyboardListener could not open any readable keyboard device under "
                f"/dev/input/event*. Permission denied for: {denied}. Grant the runtime "
                "user read access via the input group or udev rules, or set "
                "RLINF_KEYBOARD_DEVICE to a readable keyboard event device."
            )

        raise RuntimeError(
            "KeyboardListener could not find a readable keyboard device under "
            "/dev/input/event*. Ensure a physical keyboard is connected, the runtime "
            "user has access to input devices, or set RLINF_KEYBOARD_DEVICE to the "
            "correct /dev/input/eventX path."
        )

    def _open_device(self, device_path: str, is_override: bool = False):
        try:
            return self._input_device_cls(device_path)
        except FileNotFoundError as exc:
            if is_override:
                raise RuntimeError(
                    f"KeyboardListener override path '{device_path}' does not exist."
                ) from exc
            raise
        except PermissionError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener cannot read the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}'. Grant the runtime user "
                    "read access via the input group or udev rules."
                ) from exc
            raise
        except OSError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener failed to open the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}': {exc}"
                ) from exc
            raise RuntimeError(
                f"KeyboardListener failed to open input device '{device_path}': {exc}"
            ) from exc

    def _is_keyboard_device(self, device) -> bool:
        required_codes = {
            getattr(self._ecodes, key_name) for key_name in self.REQUIRED_KEY_NAMES
        }
        capabilities = device.capabilities(verbose=False)
        supported_key_codes = set(capabilities.get(self._ecodes.EV_KEY, []))
        return required_codes.issubset(supported_key_codes)

    def _listen_loop(self) -> None:
        try:
            for event in self.device.read_loop():
                if self._stop_event.is_set():
                    break
                if event.type != self._ecodes.EV_KEY:
                    continue
                key = self._event_to_key(event.code)
                if key is None:
                    continue
                if event.value in (1, 2):
                    self._handle_key_press(key)
                elif event.value == 0:
                    self._handle_key_release(key)
        finally:
            self._handle_key_release("")
            if self.device is not None:
                self.device.close()

    def _event_to_key(self, key_code: int) -> str | None:
        key_name = self._ecodes.bytype[self._ecodes.EV_KEY].get(key_code)
        if isinstance(key_name, list):
            key_name = key_name[0]
        if not isinstance(key_name, str):
            return None

        if key_name.startswith("KEY_"):
            normalized_key = key_name.removeprefix("KEY_").lower()
            if len(normalized_key) == 1:
                return normalized_key
            return f"Key.{normalized_key}"
        return key_name.lower()

    def get_key(self) -> str | None:
        """Returns the next queued key press event."""
        with self.state_lock:
            if not self._pending_keys:
                return None
            return self._pending_keys.popleft()

    def clear_pending_keys(self) -> None:
        with self.state_lock:
            self._pending_keys.clear()
            self._pressed_keys.clear()
