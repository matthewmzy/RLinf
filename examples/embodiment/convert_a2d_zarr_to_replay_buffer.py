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

"""Convert A2D teleoperation zarr demos into an RLinf replay-buffer checkpoint."""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import torch
import zarr

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more A2D zarr datasets into an RLinf replay-buffer format."
        )
    )
    parser.add_argument(
        "--zarr-path",
        nargs="+",
        required=True,
        help=(
            "One or more input zarr dataset paths. Wildcards are supported, "
            "which is convenient for vis_and_cut_demo exports like '*_ep*.zarr'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for RLinf replay-buffer checkpoint files.",
    )
    parser.add_argument(
        "--model-weights-id",
        type=str,
        default="offline_success_demo",
        help="Stored trajectory tag for replay-buffer metadata.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to convert.",
    )
    parser.add_argument(
        "--success-reward",
        type=float,
        default=1.0,
        help="Terminal reward assigned to the final step of each converted episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Replay-buffer seed written into metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing.",
    )
    return parser.parse_args()


def _expand_zarr_paths(raw_paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in raw_paths:
        matches: list[str]
        if any(token in raw_path for token in "*?[]"):
            matches = sorted(glob.glob(raw_path))
        else:
            matches = [raw_path]
        for match in matches:
            path = Path(match).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Input zarr dataset not found: {path}")
            expanded.append(path)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError("No input zarr datasets matched the provided --zarr-path values.")
    return unique_paths


def _infer_total_frames(data_group) -> int:
    lengths: list[int] = []
    for key in data_group.keys():
        array = data_group[key]
        if hasattr(array, "shape") and len(array.shape) >= 1:
            lengths.append(int(array.shape[0]))
    if not lengths:
        raise ValueError("Dataset contains no frame-aligned arrays.")
    return min(lengths)


def _load_episode_layout(root) -> np.ndarray:
    data_group = root["data"] if "data" in root else root
    total_frames = _infer_total_frames(data_group)
    if "meta" in root and "episode_ends" in root["meta"]:
        episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64).reshape(-1)
        episode_ends = episode_ends[episode_ends > 0]
        episode_ends = np.unique(episode_ends)
        episode_ends = episode_ends[episode_ends <= total_frames]
        if episode_ends.size == 0 or episode_ends[-1] < total_frames:
            episode_ends = np.concatenate(
                [episode_ends, np.array([total_frames], dtype=np.int64)]
            )
        return episode_ends
    return np.array([total_frames], dtype=np.int64)


def _shift_episode_tensor(array: np.ndarray) -> np.ndarray:
    if array.shape[0] == 1:
        return array.copy()
    return np.concatenate([array[1:], array[-1:]], axis=0)


def _prev_from_next(array: np.ndarray) -> np.ndarray:
    if array.shape[0] == 1:
        return array.copy()
    return np.concatenate([array[:1], array[:-1]], axis=0)


def _slice_array(data, key: str, start: int, end: int, dtype=np.float32) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key][start:end], dtype=dtype)


def _state_field(
    data,
    curr_key: str,
    next_key: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    curr = _slice_array(data, curr_key, start, end)
    nxt = _slice_array(data, next_key, start, end)

    if curr is None and nxt is None:
        raise KeyError(f"Neither {curr_key} nor {next_key} exists in dataset.")
    if curr is None:
        curr = _prev_from_next(nxt)
    if nxt is None:
        nxt = _shift_episode_tensor(curr)
    return curr, nxt


def _states_from_episode(data, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    left_arm, next_left_arm = _state_field(
        data, "left_arm_states", "next_left_arm_states", start, end
    )
    right_arm, next_right_arm = _state_field(
        data, "right_arm_states", "next_right_arm_states", start, end
    )
    left_hand, next_left_hand = _state_field(
        data, "left_hand_states", "next_left_hand_states", start, end
    )
    right_hand, next_right_hand = _state_field(
        data, "right_hand_states", "next_right_hand_states", start, end
    )

    states = np.concatenate(
        [left_arm, right_arm, left_hand, right_hand],
        axis=-1,
    )
    next_states = np.concatenate(
        [next_left_arm, next_right_arm, next_left_hand, next_right_hand],
        axis=-1,
    )
    return states, next_states


def _camera_views_from_episode(
    data,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    main_images = np.asarray(data["rgb_head"][start:end], dtype=np.uint8)
    left_images = (
        np.asarray(data["rgb_left_hand"][start:end], dtype=np.uint8)
        if "rgb_left_hand" in data
        else main_images.copy()
    )
    right_images = (
        np.asarray(data["rgb_right_hand"][start:end], dtype=np.uint8)
        if "rgb_right_hand" in data
        else left_images.copy()
    )
    extra_view_images = np.stack([left_images, right_images], axis=1)
    return main_images, extra_view_images


def _build_trajectory(
    data,
    start: int,
    end: int,
    *,
    model_weights_id: str,
    success_reward: float,
) -> Trajectory:
    traj_len = end - start
    if traj_len <= 0:
        raise ValueError(f"Invalid empty episode span: start={start}, end={end}")

    actions = np.concatenate(
        [
            np.asarray(data["action_arm_joints"][start:end], dtype=np.float32),
            np.asarray(data["action_left_hand_joints"][start:end], dtype=np.float32),
            np.asarray(data["action_right_hand_joints"][start:end], dtype=np.float32),
        ],
        axis=-1,
    )
    if actions.shape[-1] != 26:
        raise ValueError(f"Expected 26-dim policy actions, got {actions.shape[-1]}")

    states, next_states = _states_from_episode(data, start, end)
    if states.shape[-1] != 26 or next_states.shape[-1] != 26:
        raise ValueError(
            f"Expected 26-dim arm+hand states, got curr={states.shape[-1]}, next={next_states.shape[-1]}"
        )

    main_images, extra_view_images = _camera_views_from_episode(data, start, end)

    next_main_images = _shift_episode_tensor(main_images)
    next_extra_view_images = _shift_episode_tensor(extra_view_images)

    rewards = np.zeros((traj_len, 1), dtype=np.float32)
    rewards[-1, 0] = success_reward
    terminations = np.zeros((traj_len, 1), dtype=bool)
    terminations[-1, 0] = True
    truncations = np.zeros((traj_len, 1), dtype=bool)
    dones = terminations.copy()
    transition_valids = np.ones((traj_len, 1), dtype=bool)

    return Trajectory(
        max_episode_length=traj_len,
        model_weights_id=model_weights_id,
        actions=torch.from_numpy(actions).unsqueeze(1),
        transition_valids=torch.from_numpy(transition_valids),
        rewards=torch.from_numpy(rewards),
        terminations=torch.from_numpy(terminations),
        truncations=torch.from_numpy(truncations),
        dones=torch.from_numpy(dones),
        curr_obs={
            "main_images": torch.from_numpy(main_images).unsqueeze(1),
            "extra_view_images": torch.from_numpy(extra_view_images).unsqueeze(1),
            "states": torch.from_numpy(states).unsqueeze(1),
        },
        next_obs={
            "main_images": torch.from_numpy(next_main_images).unsqueeze(1),
            "extra_view_images": torch.from_numpy(next_extra_view_images).unsqueeze(1),
            "states": torch.from_numpy(next_states).unsqueeze(1),
        },
    )


def main() -> None:
    args = parse_args()
    zarr_paths = _expand_zarr_paths(args.zarr_path)
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimated_sample_window = max(1, int(args.max_episodes or 1024))

    buffer = TrajectoryReplayBuffer(
        seed=args.seed,
        enable_cache=True,
        cache_size=8,
        sample_window_size=estimated_sample_window,
        auto_save=True,
        auto_save_path=str(output_dir),
        trajectory_format="pt",
    )

    total_steps = 0
    converted_episodes = 0
    total_input_datasets = len(zarr_paths)
    remaining_episode_budget = args.max_episodes

    for dataset_idx, zarr_path in enumerate(zarr_paths, start=1):
        root = zarr.open(str(zarr_path), mode="r")
        data = root["data"]
        episode_ends = _load_episode_layout(root)
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        num_episodes = len(episode_ends)

        if remaining_episode_budget is not None:
            num_episodes = min(num_episodes, int(remaining_episode_budget))
            episode_ends = episode_ends[:num_episodes]
            episode_starts = episode_starts[:num_episodes]
        if num_episodes == 0:
            continue

        print(f"[dataset {dataset_idx}/{total_input_datasets}] {zarr_path}")

        for local_episode_idx, (start, end) in enumerate(
            zip(episode_starts, episode_ends), start=1
        ):
            start_idx = int(start)
            end_idx = int(end)
            trajectory = _build_trajectory(
                data,
                start_idx,
                end_idx,
                model_weights_id=args.model_weights_id,
                success_reward=args.success_reward,
            )
            total_steps += trajectory.max_episode_length
            converted_episodes += 1
            buffer.add_trajectories([trajectory])
            if (
                converted_episodes == 1
                or local_episode_idx == num_episodes
                or local_episode_idx % 10 == 0
            ):
                print(
                    f"  [{local_episode_idx}/{num_episodes}] converted episode length={trajectory.max_episode_length}"
                )

        if remaining_episode_budget is not None:
            remaining_episode_budget -= num_episodes
            if remaining_episode_budget <= 0:
                break

    if converted_episodes == 0:
        raise ValueError("No episodes were converted from the provided zarr inputs.")

    buffer.flush()
    buffer.close(wait=True)

    print(
        "Saved replay buffer checkpoint to "
        f"{output_dir} with {converted_episodes} episodes and {total_steps} action-level transitions."
    )


if __name__ == "__main__":
    main()
