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

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RLINF_DASHBOARD_DIR_ENV = "RLINF_DASHBOARD_DIR"


def _now_ts() -> float:
    return time.time()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_ready(payload), ensure_ascii=False, separators=(",", ":"))
        )
        handle.write("\n")


@dataclass(frozen=True)
class DashboardPaths:
    root: Path

    @property
    def control(self) -> Path:
        return self.root / "control.json"

    @property
    def runner_state(self) -> Path:
        return self.root / "runner_state.json"

    @property
    def runner_metrics(self) -> Path:
        return self.root / "runner_metrics.jsonl"

    @property
    def runner_events(self) -> Path:
        return self.root / "runner_events.jsonl"

    @property
    def manager_session(self) -> Path:
        return self.root / "manager_session.json"

    @property
    def actor_updates_dir(self) -> Path:
        return self.root / "actor_updates"

    @property
    def live_buffers_dir(self) -> Path:
        return self.root / "live_buffers"

    def actor_updates(self, rank: int) -> Path:
        return self.actor_updates_dir / f"rank_{int(rank)}.jsonl"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.actor_updates_dir.mkdir(parents=True, exist_ok=True)
        self.live_buffers_dir.mkdir(parents=True, exist_ok=True)


def get_dashboard_paths_from_env() -> DashboardPaths | None:
    root = os.environ.get(RLINF_DASHBOARD_DIR_ENV)
    if not root:
        return None
    paths = DashboardPaths(Path(root).expanduser().resolve())
    paths.ensure()
    return paths


class DashboardTelemetry:
    def __init__(self, paths: DashboardPaths | None = None):
        self.paths = paths or get_dashboard_paths_from_env()

    @property
    def enabled(self) -> bool:
        return self.paths is not None

    def read_control(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        assert self.paths is not None
        if not self.paths.control.exists():
            return None
        try:
            with open(self.paths.control, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def write_runner_state(self, **payload: Any) -> None:
        if not self.enabled:
            return
        assert self.paths is not None
        payload.setdefault("updated_at", _now_ts())
        _atomic_write_json(self.paths.runner_state, payload)

    def append_runner_metrics(self, **payload: Any) -> None:
        if not self.enabled:
            return
        assert self.paths is not None
        payload.setdefault("timestamp", _now_ts())
        _append_jsonl(self.paths.runner_metrics, payload)

    def append_runner_event(self, kind: str, **payload: Any) -> None:
        if not self.enabled:
            return
        assert self.paths is not None
        event = {"kind": kind, "timestamp": _now_ts(), **payload}
        _append_jsonl(self.paths.runner_events, event)

    def append_actor_update(self, rank: int, **payload: Any) -> None:
        if not self.enabled:
            return
        assert self.paths is not None
        payload.setdefault("timestamp", _now_ts())
        payload.setdefault("rank", int(rank))
        _append_jsonl(self.paths.actor_updates(rank), payload)
