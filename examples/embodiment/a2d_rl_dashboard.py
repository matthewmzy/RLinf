#!/usr/bin/env python3

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

import argparse
import json
import mimetypes
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.utils.dashboard_telemetry import DashboardPaths  # noqa: E402


def _now_ts() -> float:
    return time.time()


def _ts_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = deque(maxlen=max(1, limit))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return list(rows)


def _read_tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    rows = deque(maxlen=max(1, limit))
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                rows.append(line.rstrip("\n"))
    except OSError:
        return []
    return list(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _pick_metric(metrics: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in metrics:
            return metrics[key]
    return None


def _series_from_metrics(
    entries: list[dict[str, Any]],
    metric_key: str,
    *,
    step_field: str = "step",
    metrics_field: str = "metrics",
) -> list[dict[str, float]]:
    points = []
    for entry in entries:
        metrics = entry.get(metrics_field, {})
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric_key, None)
        if value is None:
            continue
        step = entry.get(step_field, None)
        if step is None:
            continue
        try:
            points.append({"x": float(step), "y": float(value)})
        except (TypeError, ValueError):
            continue
    return points


@dataclass
class ManagedSession:
    session_id: str
    session_dir: Path
    dashboard_dir: Path
    training_dir: Path
    log_path: Path
    live_buffer_dir: Path
    config_name: str
    experiment_name: str
    extra_overrides: list[str]
    persist_buffers: bool
    command: list[str]
    process: subprocess.Popen | None
    log_handle: Any
    started_at: float
    stop_requested: bool = False
    save_buffers_on_stop: bool | None = None
    finalized: bool = False
    exit_code: int | None = None
    cleanup_result: str | None = None


class DashboardManager:
    def __init__(self, repo_root: Path, assets_dir: Path):
        self.repo_root = repo_root
        self.embodied_path = repo_root / "examples" / "embodiment"
        self.config_path = self.embodied_path / "config"
        self.assets_dir = assets_dir
        self.sessions_root = repo_root / "results" / "a2d_rl_dashboard"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._request_counter = 0
        self.current_session: ManagedSession | None = None

    def list_configs(self) -> list[str]:
        configs = [
            path.stem
            for path in sorted(self.config_path.glob("realworld_a2d*.yaml"))
            if path.is_file()
        ]
        return configs

    def _refresh_session(self) -> None:
        session = self.current_session
        if session is None or session.process is None:
            return
        if session.finalized:
            return

        exit_code = session.process.poll()
        if exit_code is None:
            return

        session.exit_code = int(exit_code)
        if session.log_handle is not None:
            session.log_handle.close()
            session.log_handle = None

        if session.persist_buffers and session.save_buffers_on_stop is False:
            shutil.rmtree(session.live_buffer_dir, ignore_errors=True)
            session.cleanup_result = "discarded_live_buffers"
        elif session.persist_buffers and session.live_buffer_dir.exists():
            session.cleanup_result = "kept_live_buffers"

        session.finalized = True

    def _manager_meta(self, session: ManagedSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "config_name": session.config_name,
            "experiment_name": session.experiment_name,
            "session_dir": str(session.session_dir),
            "dashboard_dir": str(session.dashboard_dir),
            "training_dir": str(session.training_dir),
            "log_path": str(session.log_path),
            "live_buffer_dir": str(session.live_buffer_dir),
            "persist_buffers": session.persist_buffers,
            "extra_overrides": session.extra_overrides,
            "command": session.command,
            "started_at": session.started_at,
        }

    def start_training(
        self,
        *,
        config_name: str,
        experiment_name: str | None,
        extra_overrides: str,
        persist_buffers: bool,
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_session()
            if (
                self.current_session is not None
                and self.current_session.process is not None
                and self.current_session.process.poll() is None
            ):
                raise RuntimeError("已有训练任务正在运行，请先停止当前任务。")

            if config_name not in self.list_configs():
                raise RuntimeError(f"找不到配置 {config_name}。")

            session_id = f"{_ts_slug()}-{config_name}"
            session_dir = self.sessions_root / session_id
            dashboard_dir = session_dir / "dashboard"
            training_dir = session_dir / "training"
            live_buffer_dir = dashboard_dir / "live_buffers"
            log_path = session_dir / "run.log"
            session_dir.mkdir(parents=True, exist_ok=True)
            dashboard_dir.mkdir(parents=True, exist_ok=True)
            training_dir.mkdir(parents=True, exist_ok=True)
            live_buffer_dir.mkdir(parents=True, exist_ok=True)

            parsed_overrides = shlex.split(extra_overrides) if extra_overrides else []
            experiment_name = experiment_name or config_name
            command = [
                sys.executable,
                "examples/embodiment/train_embodied_agent.py",
                "--config-name",
                config_name,
                f"runner.logger.log_path={training_dir}",
                f"runner.logger.experiment_name={experiment_name}",
            ]
            if persist_buffers:
                command.extend(
                    [
                        "algorithm.replay_buffer.auto_save=True",
                        f"algorithm.replay_buffer.auto_save_path={live_buffer_dir / 'replay_buffer'}",
                        "algorithm.demo_buffer.auto_save=True",
                        f"algorithm.demo_buffer.auto_save_path={live_buffer_dir / 'demo_buffer'}",
                    ]
                )
            command.extend(parsed_overrides)

            env = os.environ.copy()
            env.setdefault("EMBODIED_PATH", str(self.embodied_path))
            env["PYTHONPATH"] = (
                f"{self.repo_root}:{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else str(self.repo_root)
            )
            env.setdefault("HYDRA_FULL_ERROR", "1")
            env["PYTHONUNBUFFERED"] = "1"
            env["RLINF_DASHBOARD_DIR"] = str(dashboard_dir)

            log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            session = ManagedSession(
                session_id=session_id,
                session_dir=session_dir,
                dashboard_dir=dashboard_dir,
                training_dir=training_dir,
                log_path=log_path,
                live_buffer_dir=live_buffer_dir,
                config_name=config_name,
                experiment_name=experiment_name,
                extra_overrides=parsed_overrides,
                persist_buffers=persist_buffers,
                command=command,
                process=process,
                log_handle=log_handle,
                started_at=_now_ts(),
            )
            self.current_session = session
            _write_json(DashboardPaths(dashboard_dir).manager_session, self._manager_meta(session))
            return {
                "ok": True,
                "message": "训练已启动。",
                "session_id": session_id,
                "pid": process.pid,
            }

    def request_stop(self, *, save_buffers: bool) -> dict[str, Any]:
        with self._lock:
            self._refresh_session()
            session = self.current_session
            if (
                session is None
                or session.process is None
                or session.process.poll() is not None
            ):
                raise RuntimeError("当前没有正在运行的训练任务。")

            self._request_counter += 1
            control = {
                "request_id": self._request_counter,
                "stop_requested": True,
                "save_buffers": bool(save_buffers),
                "flush_buffers": True,
                "requested_at": _now_ts(),
            }
            _write_json(DashboardPaths(session.dashboard_dir).control, control)
            session.stop_requested = True
            session.save_buffers_on_stop = bool(save_buffers)
            return {
                "ok": True,
                "message": "已发送优雅停止请求，训练会在当前安全边界结束后停下。",
            }

    def force_stop(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_session()
            session = self.current_session
            if (
                session is None
                or session.process is None
                or session.process.poll() is not None
            ):
                raise RuntimeError("当前没有正在运行的训练任务。")

            os.killpg(session.process.pid, signal.SIGTERM)
            session.stop_requested = True
            session.save_buffers_on_stop = False
            return {
                "ok": True,
                "message": "已发送强制终止信号。若进程未及时退出，可再次点击强制终止。",
            }

    def build_state(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_session()
            session = self.current_session
            if session is None:
                return {
                    "session": None,
                    "summary": {},
                    "charts": {},
                    "updates": [],
                    "logs": [],
                    "available_configs": self.list_configs(),
                }

            paths = DashboardPaths(session.dashboard_dir)
            runner_state = _read_json(paths.runner_state) or {}
            manager_state = _read_json(paths.manager_session) or self._manager_meta(session)
            runner_metrics = _read_jsonl_tail(paths.runner_metrics, limit=240)
            actor_updates = _read_jsonl_tail(paths.actor_updates(0), limit=240)
            logs = _read_tail_lines(session.log_path, limit=180)

            status = runner_state.get("status")
            if not status:
                if session.process is not None and session.process.poll() is None:
                    status = "launching"
                elif session.exit_code == 0:
                    status = "completed"
                else:
                    status = "failed"
            elif session.finalized and session.exit_code is not None:
                terminal_statuses = {"completed", "stopped", "failed", "interrupted"}
                if status not in terminal_statuses:
                    status = "completed" if session.exit_code == 0 else "failed"

            latest_metrics = runner_state.get("latest_metrics", {})
            if not latest_metrics and runner_metrics:
                latest_metrics = runner_metrics[-1].get("metrics", {})
            latest_update = actor_updates[-1] if actor_updates else None
            latest_update_metrics = (
                latest_update.get("metrics", {}) if latest_update else {}
            )

            summary = {
                "status": status,
                "phase": runner_state.get("phase", "idle"),
                "global_step": runner_state.get("global_step", 0),
                "max_steps": runner_state.get("max_steps"),
                "actor_loss": _pick_metric(
                    latest_update_metrics,
                    ["sac/actor_loss", "dagger/actor_loss"],
                ),
                "critic_loss": _pick_metric(
                    latest_update_metrics,
                    ["sac/critic_loss"],
                ),
                "alpha": _pick_metric(latest_update_metrics, ["sac/alpha"]),
                "replay_buffer_size": _pick_metric(
                    latest_metrics, ["train/replay_buffer/size"]
                ),
                "replay_buffer_samples": _pick_metric(
                    latest_metrics, ["train/replay_buffer/total_samples"]
                ),
                "demo_buffer_size": _pick_metric(
                    latest_metrics, ["train/demo_buffer/size"]
                ),
                "demo_buffer_samples": _pick_metric(
                    latest_metrics, ["train/demo_buffer/total_samples"]
                ),
                "env_reward": _pick_metric(
                    latest_metrics,
                    [
                        "env/reward_mean",
                        "env/reward",
                        "env/episode_reward_mean",
                    ],
                ),
                "env_success": _pick_metric(
                    latest_metrics,
                    [
                        "env/success_rate",
                        "env/success",
                        "env/trajectory_success_rate",
                    ],
                ),
                "env_episode_length": _pick_metric(
                    latest_metrics,
                    [
                        "env/episode_length_mean",
                        "env/episode_length",
                        "env/length",
                    ],
                ),
                "actor_update_step": latest_update.get("update_step")
                if latest_update
                else None,
            }

            charts = {
                "actor_loss": _series_from_metrics(
                    actor_updates,
                    "sac/actor_loss",
                    step_field="update_step",
                ),
                "critic_loss": _series_from_metrics(
                    actor_updates,
                    "sac/critic_loss",
                    step_field="update_step",
                ),
                "alpha": _series_from_metrics(
                    actor_updates,
                    "sac/alpha",
                    step_field="update_step",
                ),
                "replay_buffer_size": _series_from_metrics(
                    runner_metrics,
                    "train/replay_buffer/size",
                ),
                "demo_buffer_size": _series_from_metrics(
                    runner_metrics,
                    "train/demo_buffer/size",
                ),
                "env_reward": _series_from_metrics(
                    runner_metrics,
                    "env/reward_mean",
                )
                or _series_from_metrics(runner_metrics, "env/reward")
                or _series_from_metrics(runner_metrics, "env/episode_reward_mean"),
                "env_success": _series_from_metrics(
                    runner_metrics,
                    "env/success_rate",
                )
                or _series_from_metrics(runner_metrics, "env/success")
                or _series_from_metrics(
                    runner_metrics, "env/trajectory_success_rate"
                ),
            }

            session_info = {
                **manager_state,
                "pid": session.process.pid if session.process is not None else None,
                "status": status,
                "phase": runner_state.get("phase", "idle"),
                "started_at": session.started_at,
                "stop_requested": session.stop_requested,
                "save_buffers_on_stop": session.save_buffers_on_stop,
                "exit_code": session.exit_code,
                "finalized": session.finalized,
                "cleanup_result": session.cleanup_result,
            }

            return {
                "session": session_info,
                "summary": summary,
                "charts": charts,
                "updates": actor_updates[-18:],
                "logs": logs,
                "available_configs": self.list_configs(),
            }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    manager: DashboardManager
    assets_dir: Path

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, content: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        payload = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            asset = self.assets_dir / "index.html"
            self._send_text(asset.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/assets/"):
            asset = self.assets_dir / parsed.path.removeprefix("/assets/")
            if not asset.exists() or not asset.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type, _ = mimetypes.guess_type(str(asset))
            self._send_text(
                asset.read_bytes(),
                content_type or "application/octet-stream",
            )
            return
        if parsed.path == "/api/state":
            self._send_json(self.manager.build_state())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/start":
                result = self.manager.start_training(
                    config_name=str(payload.get("config_name", "")).strip(),
                    experiment_name=str(payload.get("experiment_name", "")).strip()
                    or None,
                    extra_overrides=str(payload.get("extra_overrides", "")).strip(),
                    persist_buffers=bool(payload.get("persist_buffers", True)),
                )
                self._send_json(result)
                return
            if parsed.path == "/api/stop":
                result = self.manager.request_stop(
                    save_buffers=bool(payload.get("save_buffers", True))
                )
                self._send_json(result)
                return
            if parsed.path == "/api/force-stop":
                result = self.manager.force_stop()
                self._send_json(result)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=400)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A2D real-world RL dashboard for launching and monitoring training."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8787, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assets_dir = Path(__file__).resolve().parent / "dashboard_assets"
    manager = DashboardManager(repo_root=REPO_ROOT, assets_dir=assets_dir)

    class _Handler(DashboardRequestHandler):
        pass

    _Handler.manager = manager
    _Handler.assets_dir = assets_dir

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"A2D RL dashboard listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
