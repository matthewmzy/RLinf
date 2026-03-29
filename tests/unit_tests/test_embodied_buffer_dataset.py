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

import pytest

torch = pytest.importorskip("torch")

from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
)


class _FakeBuffer:
    def __init__(self, size: int, fill_value: float):
        self.size = size
        self.fill_value = fill_value
        self.sample_calls: list[int] = []

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def sample(self, num_chunks: int) -> dict[str, torch.Tensor]:
        self.sample_calls.append(num_chunks)
        return {
            "actions": torch.full(
                (num_chunks, 1), self.fill_value, dtype=torch.float32
            )
        }


def test_replay_buffer_dataset_falls_back_to_replay_only_until_demo_ready():
    replay_buffer = _FakeBuffer(size=4, fill_value=1.0)
    demo_buffer = _FakeBuffer(size=0, fill_value=2.0)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        batch_size=8,
        min_replay_buffer_size=2,
        min_demo_buffer_size=1,
        allow_replay_only_until_demo_ready=True,
    )

    assert dataset._get_sampling_mode() == (True, False)
    batch = next(iter(dataset))

    assert batch["actions"].shape == (8, 1)
    assert torch.all(batch["actions"] == 1.0)
    assert replay_buffer.sample_calls == [8]
    assert demo_buffer.sample_calls == []


def test_replay_buffer_dataset_mixes_demo_after_demo_ready():
    replay_buffer = _FakeBuffer(size=4, fill_value=1.0)
    demo_buffer = _FakeBuffer(size=1, fill_value=2.0)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        batch_size=8,
        min_replay_buffer_size=2,
        min_demo_buffer_size=1,
        allow_replay_only_until_demo_ready=True,
    )

    assert dataset._get_sampling_mode() == (True, True)
    batch = next(iter(dataset))

    assert batch["actions"].shape == (8, 1)
    assert torch.all(batch["actions"][:4] == 1.0)
    assert torch.all(batch["actions"][4:] == 2.0)
    assert replay_buffer.sample_calls == [4]
    assert demo_buffer.sample_calls == [4]


def test_preload_replay_buffer_dataset_close_is_safe_before_iteration():
    dataset = PreloadReplayBufferDataset(
        replay_buffer=_FakeBuffer(size=4, fill_value=1.0),
        demo_buffer=_FakeBuffer(size=0, fill_value=2.0),
        batch_size=8,
        min_replay_buffer_size=2,
        min_demo_buffer_size=1,
        allow_replay_only_until_demo_ready=True,
    )

    dataset.close()
