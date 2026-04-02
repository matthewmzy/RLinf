#!/usr/bin/env python3
"""Run a fake A2D actor with dataset pose interpolation and lightweight HIL buffers.

This script is meant for bring-up:
1. Start or verify the A2D gRPC rollout server.
2. Pick two poses from a zarr dataset.
3. Continuously interpolate between them over 3 seconds and send absolute joints.
4. Record lightweight replay/demo buffers plus keyboard sparse rewards.

Controls:
- `f`: fail, reward = -1, end current logical episode
- `s`: success, reward = +1, end current logical episode
- `x`: stop the script after finalizing current logical episode
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import zarr

try:
    from pynput import keyboard
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency 'pynput'. Install it in conda env 'a2d' first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = REPO_ROOT / "rlinf" / "envs" / "realworld" / "a2d" / "proto"
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

import robot_control_pb2  # noqa: E402


class RobotControlStub:
    """Minimal gRPC client stub for the A2D rollout server."""

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


@dataclass
class RobotObservation:
    states: dict[str, np.ndarray]
    timestamps: dict[str, float]
    control_mode: int | None
    trajectory_label: int | None
    is_switch_mode: bool | None


class GlobalKeyListener:
    """Global key listener with one-shot consumption semantics."""

    def __init__(self):
        self._latest_key: str | None = None
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        self._latest_key = key.char if hasattr(key, "char") else str(key)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        del key
        self._latest_key = None

    def consume(self) -> str | None:
        key = self._latest_key
        self._latest_key = None
        return key

    def stop(self) -> None:
        self._listener.stop()


def decode_array(array_data: robot_control_pb2.ArrayData) -> np.ndarray:
    values = np.asarray(array_data.values, dtype=np.float32)
    if array_data.shape:
        return values.reshape(tuple(array_data.shape))
    return values


def decode_observation(
    observation: robot_control_pb2.Observation,
) -> RobotObservation:
    return RobotObservation(
        states={
            key: decode_array(value) for key, value in observation.states.items()
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


def policy_action_from_obs(obs: RobotObservation) -> np.ndarray:
    arm = np.asarray(obs.states["arm_joint_states"], dtype=np.float32).reshape(-1)
    left_hand = np.asarray(obs.states["left_hand_states"], dtype=np.float32).reshape(-1)
    right_hand = np.asarray(
        obs.states["right_hand_states"], dtype=np.float32
    ).reshape(-1)
    return np.concatenate([arm, left_hand, right_hand], axis=0).astype(np.float32)


def waist_from_obs(obs: RobotObservation) -> np.ndarray:
    return np.asarray(obs.states["waist_joints_states"], dtype=np.float32).reshape(-1)


def load_dataset_actions(zarr_path: str) -> dict[str, Any]:
    root = zarr.open(zarr_path, mode="r")
    data = root["data"]
    left_arm = np.asarray(data["left_arm_states"], dtype=np.float32)
    right_arm = np.asarray(data["right_arm_states"], dtype=np.float32)
    left_hand = np.asarray(data["left_hand_states"], dtype=np.float32)
    right_hand = np.asarray(data["right_hand_states"], dtype=np.float32)
    waist = np.asarray(data["waist_joints_states"], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    actions26 = np.concatenate([left_arm, right_arm, left_hand, right_hand], axis=1)
    return {
        "actions26": actions26,
        "waist": waist,
        "episode_starts": episode_starts,
        "episode_ends": episode_ends,
    }


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def start_server_if_needed(args: argparse.Namespace) -> None:
    if wait_for_port(args.host, args.port, timeout_s=1.0):
        return
    if not args.auto_start_server:
        raise RuntimeError(
            f"A2D gRPC server is not listening on {args.host}:{args.port} and "
            "--auto-start-server is disabled."
        )

    command = (
        "source /ros_entrypoint.sh && "
        "source /opt/psi/rt/a2d-tele/install/setup.bash && "
        "python3 -m model_inference.run_inference_server"
    )
    if args.grpc_config_file:
        command = (
            f"export MODEL_INFERENCE_CONFIG_FILE={args.grpc_config_file} && {command}"
        )

    log_path = f"/tmp/a2d_fake_actor_server_{int(time.time())}.log"
    docker_command = (
        f"{command} > {log_path} 2>&1"
    )
    subprocess.run(
        [
            "docker",
            "exec",
            "-d",
            args.container_name,
            "bash",
            "-lc",
            docker_command,
        ],
        check=True,
    )
    if not wait_for_port(args.host, args.port, timeout_s=args.ready_timeout):
        raise TimeoutError(
            f"Started server in container but {args.host}:{args.port} was not ready "
            f"within {args.ready_timeout} seconds. Check {args.container_name}:{log_path}"
        )
    print(f"[server] A2D gRPC server is ready on {args.host}:{args.port}")


def create_client(args: argparse.Namespace) -> tuple[grpc.Channel, RobotControlStub]:
    options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(f"{args.host}:{args.port}", options=options)
    stub = RobotControlStub(channel)
    return channel, stub


def wait_until_ready(
    stub: RobotControlStub,
    timeout_s: float,
    request_timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    last_status = "unknown"
    while time.time() < deadline:
        try:
            response = stub.health_check(
                robot_control_pb2.HealthRequest(),
                timeout=request_timeout_s,
            )
            last_status = response.status or "ready"
            if response.is_ready:
                print(f"[server] health_check OK: {last_status}")
                return
            print(f"[server] waiting: {last_status}")
        except grpc.RpcError as exc:
            last_status = f"rpc_error: {exc.code().name}"
            print(f"[server] waiting: {last_status}")
        time.sleep(1.0)
    raise TimeoutError(
        f"A2D server did not become ready within {timeout_s} seconds. "
        f"Last status: {last_status}"
    )


def get_obs(stub: RobotControlStub, timeout_s: float) -> RobotObservation:
    response = stub.get_obs(robot_control_pb2.GetObsRequest(), timeout=timeout_s)
    if not response.success:
        raise RuntimeError(f"get_obs failed: {response.message}")
    return decode_observation(response.observation)


def set_action(stub: RobotControlStub, action28: np.ndarray, timeout_s: float) -> None:
    response = stub.set_action(
        robot_control_pb2.SetActionRequest(
            action=robot_control_pb2.Action(
                values=np.asarray(action28, dtype=np.float32).reshape(-1).tolist(),
                dimension=28,
            )
        ),
        timeout=timeout_s,
    )
    if not response.success:
        raise RuntimeError(f"set_action failed: {response.message}")


def choose_start_and_target(
    actions26: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    current_action26: np.ndarray,
    max_search_offset: int,
    start_candidate_pool: int,
    min_target_offset: int,
    max_pair_span: float,
) -> dict[str, Any]:
    del episode_starts
    current_dists = np.linalg.norm(actions26 - current_action26[None, :], axis=1)
    start_candidates = np.argsort(current_dists)[:start_candidate_pool]

    best_pair: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None

    for start_idx in start_candidates:
        start_idx = int(start_idx)
        current_ep = int(np.searchsorted(episode_ends, start_idx, side="right"))
        ep_end = int(episode_ends[current_ep])
        base = actions26[start_idx]
        start_to_current = np.abs(base - current_action26)
        search_end = min(ep_end, start_idx + max_search_offset + 1)

        for target_idx in range(start_idx + min_target_offset, search_end):
            diff = np.abs(actions26[target_idx] - base)
            if float(diff.max()) > max_pair_span:
                continue

            arm_diff = diff[:14]
            hand_diff = diff[14:]
            score = (
                float(arm_diff.mean()),
                float(arm_diff.max()),
                -float(diff.mean()),
                -float(diff.max()),
                -float(start_to_current.mean()),
                -float(start_to_current.max()),
                -float(hand_diff.mean()),
                float(target_idx - start_idx),
            )
            if best_score is not None and score <= best_score:
                continue

            best_score = score
            best_pair = {
                "episode": current_ep,
                "start_idx": start_idx,
                "start_action26": base.copy(),
                "target_idx": int(target_idx),
                "target_action26": actions26[target_idx].copy(),
                "target_offset": int(target_idx - start_idx),
                "start_to_current_max": float(start_to_current.max()),
                "start_to_current_mean": float(start_to_current.mean()),
                "start_to_target_max": float(diff.max()),
                "start_to_target_mean": float(diff.mean()),
                "arm_delta_max": float(arm_diff.max()),
                "arm_delta_mean": float(arm_diff.mean()),
                "hand_delta_max": float(hand_diff.max()),
                "hand_delta_mean": float(hand_diff.mean()),
            }

    if best_pair is None:
        raise RuntimeError(
            "Failed to find a suitable start/target pair under the current safety "
            "constraints. Try increasing --max-search-offset or --max-pair-span."
        )
    return best_pair


def build_motion_sequence(
    start_action26: np.ndarray,
    target_action26: np.ndarray,
    steps_per_half_cycle: int,
) -> np.ndarray:
    forward = np.linspace(
        start_action26,
        target_action26,
        num=steps_per_half_cycle,
        endpoint=True,
        dtype=np.float32,
    )
    backward = np.linspace(
        target_action26,
        start_action26,
        num=steps_per_half_cycle,
        endpoint=True,
        dtype=np.float32,
    )
    return np.concatenate([forward, backward], axis=0)


def save_episode_buffers(
    replay_dir: Path,
    demo_dir: Path,
    episode_id: int,
    episode_steps: list[dict[str, Any]],
) -> tuple[int, int]:
    if not episode_steps:
        return 0, 0

    valid_steps = [step for step in episode_steps if step["data_valid"]]
    demo_steps = [step for step in valid_steps if step["control_mode"] == 1]

    def stack_steps(steps: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not steps:
            return {}
        return {
            "curr_state28": np.stack([step["curr_state28"] for step in steps], axis=0),
            "next_state28": np.stack([step["next_state28"] for step in steps], axis=0),
            "action26": np.stack([step["action26"] for step in steps], axis=0),
            "intervene_action26": np.stack(
                [step["intervene_action26"] for step in steps], axis=0
            ),
            "reward": np.asarray([step["reward"] for step in steps], dtype=np.float32),
            "done": np.asarray([step["done"] for step in steps], dtype=np.bool_),
            "control_mode": np.asarray(
                [step["control_mode"] for step in steps], dtype=np.int32
            ),
            "data_valid": np.asarray(
                [step["data_valid"] for step in steps], dtype=np.bool_
            ),
        }

    replay_payload = stack_steps(valid_steps)
    if replay_payload:
        np.savez_compressed(replay_dir / f"episode_{episode_id:04d}.npz", **replay_payload)

    demo_payload = stack_steps(demo_steps)
    if demo_payload:
        np.savez_compressed(demo_dir / f"episode_{episode_id:04d}.npz", **demo_payload)

    return len(valid_steps), len(demo_steps)


def state28_from_obs(obs: RobotObservation) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs.states["arm_joint_states"], dtype=np.float32).reshape(-1),
            np.asarray(obs.states["left_hand_states"], dtype=np.float32).reshape(-1),
            np.asarray(obs.states["right_hand_states"], dtype=np.float32).reshape(-1),
            np.asarray(obs.states["waist_joints_states"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def logical_reward_from_key(key: str | None) -> tuple[float, bool, bool]:
    if key == "f":
        return -1.0, True, False
    if key == "s":
        return 1.0, True, False
    if key == "x":
        return 0.0, True, True
    return 0.0, False, False


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an A2D fake actor with zarr pose interpolation."
    )
    parser.add_argument(
        "--zarr-path",
        default=str(
            REPO_ROOT.parent / "psi-policy" / "data" / "bj01-dc09_20260325_107.zarr"
        ),
        help="Path to the zarr dataset.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="A2D gRPC host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12321,
        help="A2D gRPC port.",
    )
    parser.add_argument(
        "--container-name",
        default="a2d-tele-release-2-1-0rc3-latest",
        help="A2D docker container name.",
    )
    parser.add_argument(
        "--grpc-config-file",
        default=None,
        help="Optional MODEL_INFERENCE_CONFIG_FILE inside the container.",
    )
    parser.add_argument(
        "--auto-start-server",
        action="store_true",
        help="Start the gRPC server inside the container if needed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request gRPC timeout in seconds.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=60.0,
        help="Server readiness timeout in seconds.",
    )
    parser.add_argument(
        "--interp-seconds",
        type=float,
        default=3.0,
        help="Interpolation time from start pose to target pose.",
    )
    parser.add_argument(
        "--step-frequency",
        type=float,
        default=10.0,
        help="Command send frequency in Hz.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Warm up from current robot pose to selected dataset start pose.",
    )
    parser.add_argument(
        "--max-search-offset",
        type=int,
        default=180,
        help="How many later frames in the same episode to search for the target pose.",
    )
    parser.add_argument(
        "--start-candidate-pool",
        type=int,
        default=3000,
        help="How many nearest dataset frames to consider as possible start poses.",
    )
    parser.add_argument(
        "--min-target-offset",
        type=int,
        default=10,
        help="Minimum frame gap between start and target.",
    )
    parser.add_argument(
        "--max-pair-span",
        type=float,
        default=0.45,
        help="Maximum allowed per-joint absolute span between the chosen start/target pair.",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.0,
        help="Optional extra margin added to the pair min/max bounding box.",
    )
    parser.add_argument(
        "--record-root",
        default=str(REPO_ROOT / "results" / "a2d_fake_actor_hil"),
        help="Root directory to store replay/demo buffers and metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    start_server_if_needed(args)
    channel, stub = create_client(args)

    try:
        wait_until_ready(
            stub,
            timeout_s=args.ready_timeout,
            request_timeout_s=args.timeout,
        )
        current_obs = get_obs(stub, timeout_s=args.timeout)

        dataset = load_dataset_actions(args.zarr_path)
        current_action26 = policy_action_from_obs(current_obs)
        current_waist = waist_from_obs(current_obs)
        selection = choose_start_and_target(
            actions26=dataset["actions26"],
            episode_starts=dataset["episode_starts"],
            episode_ends=dataset["episode_ends"],
            current_action26=current_action26,
            max_search_offset=args.max_search_offset,
            start_candidate_pool=args.start_candidate_pool,
            min_target_offset=args.min_target_offset,
            max_pair_span=args.max_pair_span,
        )

        start_action26 = selection["start_action26"]
        target_action26 = selection["target_action26"]
        action_low26 = (
            np.minimum(start_action26, target_action26) - float(args.bbox_margin)
        )
        action_high26 = (
            np.maximum(start_action26, target_action26) + float(args.bbox_margin)
        )

        steps_per_half_cycle = max(2, int(round(args.interp_seconds * args.step_frequency)))
        steps_per_warmup = max(2, int(round(args.warmup_seconds * args.step_frequency)))

        motion_sequence = build_motion_sequence(
            start_action26=start_action26,
            target_action26=target_action26,
            steps_per_half_cycle=steps_per_half_cycle,
        )
        warmup_sequence = np.linspace(
            current_action26,
            start_action26,
            num=steps_per_warmup,
            endpoint=True,
            dtype=np.float32,
        )

        run_dir = Path(args.record_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_dir = run_dir / "replay_buffer"
        demo_dir = run_dir / "demo_buffer"
        replay_dir.mkdir(parents=True, exist_ok=True)
        demo_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "zarr_path": args.zarr_path,
            "host": args.host,
            "port": args.port,
            "container_name": args.container_name,
            "episode_index": selection["episode"],
            "start_idx": selection["start_idx"],
            "target_idx": selection["target_idx"],
            "target_offset": selection["target_offset"],
            "start_to_current_max": selection["start_to_current_max"],
            "start_to_current_mean": selection["start_to_current_mean"],
            "start_to_target_max": selection["start_to_target_max"],
            "start_to_target_mean": selection["start_to_target_mean"],
            "arm_delta_max": selection["arm_delta_max"],
            "arm_delta_mean": selection["arm_delta_mean"],
            "hand_delta_max": selection["hand_delta_max"],
            "hand_delta_mean": selection["hand_delta_mean"],
            "bbox_mode": "pair_minmax",
            "bbox_margin": args.bbox_margin,
            "max_pair_span": args.max_pair_span,
            "min_target_offset": args.min_target_offset,
            "action_low26": action_low26.tolist(),
            "action_high26": action_high26.tolist(),
            "waist_fixed": current_waist.tolist(),
            "step_frequency": args.step_frequency,
            "interp_seconds": args.interp_seconds,
            "warmup_seconds": args.warmup_seconds,
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("[selection]", json.dumps(metadata, indent=2, ensure_ascii=False))
        print("[controls] f=fail, s=success, x=stop")
        print(
            "[run] Warmup to selected dataset start pose, then loop start<->target "
            f"every {args.interp_seconds:.1f}s at {args.step_frequency:.1f}Hz."
        )
        print(f"[buffers] replay={replay_dir}")
        print(f"[buffers] demo={demo_dir}")

        listener = GlobalKeyListener()
        episode_steps: list[dict[str, Any]] = []
        logical_episode_id = 0
        total_replay_steps = 0
        total_demo_steps = 0
        should_stop = False
        previous_obs = current_obs
        phase = "warmup"
        phase_idx = 0

        def finalize_episode() -> None:
            nonlocal logical_episode_id, episode_steps, total_replay_steps, total_demo_steps
            replay_steps, demo_steps = save_episode_buffers(
                replay_dir=replay_dir,
                demo_dir=demo_dir,
                episode_id=logical_episode_id,
                episode_steps=episode_steps,
            )
            if episode_steps:
                logical_episode_id += 1
                total_replay_steps += replay_steps
                total_demo_steps += demo_steps
                print(
                    f"[episode] finalized id={logical_episode_id - 1} "
                    f"valid_steps={replay_steps} demo_steps={demo_steps} "
                    f"total_replay_steps={total_replay_steps} total_demo_steps={total_demo_steps}"
                )
            episode_steps = []

        try:
            while True:
                loop_start = time.time()
                key = listener.consume()
                reward, done, should_stop_from_key = logical_reward_from_key(key)
                should_stop = should_stop or should_stop_from_key

                if phase == "warmup":
                    action26 = warmup_sequence[min(phase_idx, len(warmup_sequence) - 1)]
                else:
                    action26 = motion_sequence[phase_idx % len(motion_sequence)]

                action26 = np.clip(action26, action_low26, action_high26)
                action28 = np.concatenate([current_waist, action26], axis=0).astype(
                    np.float32
                )
                set_action(stub, action28=action28, timeout_s=args.timeout)

                elapsed = time.time() - loop_start
                sleep_s = max(0.0, (1.0 / args.step_frequency) - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

                next_obs = get_obs(stub, timeout_s=args.timeout)
                control_mode = next_obs.control_mode if next_obs.control_mode is not None else -1
                data_valid = control_mode != 99
                intervene_action26 = (
                    policy_action_from_obs(next_obs)
                    if control_mode == 1
                    else np.zeros(26, dtype=np.float32)
                )

                episode_steps.append(
                    {
                        "curr_state28": state28_from_obs(previous_obs),
                        "next_state28": state28_from_obs(next_obs),
                        "action26": action26.copy(),
                        "intervene_action26": intervene_action26.copy(),
                        "reward": reward,
                        "done": done,
                        "control_mode": control_mode,
                        "data_valid": data_valid,
                    }
                )

                if phase == "warmup":
                    phase_idx += 1
                    if phase_idx >= len(warmup_sequence):
                        phase = "loop"
                        phase_idx = 0
                        print("[run] Warmup finished, now looping dataset start<->target poses.")
                else:
                    phase_idx += 1

                if key in {"f", "s", "x"}:
                    print(
                        f"[key] key={key} reward={reward} done={done} "
                        f"control_mode={control_mode} data_valid={data_valid}"
                    )

                previous_obs = next_obs

                if done:
                    finalize_episode()
                    if should_stop:
                        break
        except KeyboardInterrupt:
            print("[run] Ctrl+C received, finalizing current episode and stopping.")
        finally:
            finalize_episode()
            listener.stop()
            summary = {
                "logical_episodes": logical_episode_id,
                "total_replay_steps": total_replay_steps,
                "total_demo_steps": total_demo_steps,
                "run_dir": str(run_dir),
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("[summary]", json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        channel.close()


if __name__ == "__main__":
    main()
