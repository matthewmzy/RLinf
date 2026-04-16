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

import threading
from collections import deque


class KeyboardListener:
    def __init__(self, allowed_keys: set[str] | None = None):
        from pynput import keyboard

        self.state_lock = threading.Lock()
        self._pressed_keys = set()
        self._pending_keys = deque()
        self._allowed_keys = (
            {str(key).lower() for key in allowed_keys}
            if allowed_keys is not None
            else None
        )

        self.listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.listener.start()
        self.last_intervene = 0

    def _normalize_key(self, key) -> str:
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        return str(key)

    def on_key_press(self, key):
        normalized_key = self._normalize_key(key)
        if self._allowed_keys is not None and normalized_key not in self._allowed_keys:
            return
        with self.state_lock:
            if normalized_key in self._pressed_keys:
                return
            self._pressed_keys.add(normalized_key)
            self._pending_keys.append(normalized_key)

    def on_key_release(self, key):
        normalized_key = self._normalize_key(key)
        with self.state_lock:
            self._pressed_keys.discard(normalized_key)

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
