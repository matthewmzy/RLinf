# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _RecordingBuffer:
    def __init__(self):
        self.trajectories = []

    def add_trajectories(self, trajectories):
        self.trajectories.extend(trajectories)


def test_received_trajectory_helper_keeps_full_valid_intervened_demo():
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.replay_buffer = _RecordingBuffer()
    worker.demo_buffer = _RecordingBuffer()

    trajectory = Trajectory(
        max_episode_length=8,
        model_weights_id="weights-1",
        actions=torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]),
        intervene_flags=torch.tensor(
            [[[False, False, False, False, True, True]]], dtype=torch.bool
        ),
        transition_valids=torch.tensor([[[True, False, True]]], dtype=torch.bool),
        rewards=torch.tensor([[[1.0, 2.0, 3.0]]]),
    )

    EmbodiedSACFSDPPolicy._add_received_trajectories_to_buffers(worker, [trajectory])

    assert len(worker.replay_buffer.trajectories) == 1
    replay_traj = worker.replay_buffer.trajectories[0]
    assert torch.equal(
        replay_traj.actions[:, 0],
        torch.tensor([[1.0, 2.0], [5.0, 6.0]]),
    )

    # Demo should keep the full valid trajectory, not only the teleop chunk.
    assert len(worker.demo_buffer.trajectories) == 1
    demo_traj = worker.demo_buffer.trajectories[0]
    assert torch.equal(demo_traj.actions, replay_traj.actions)
    assert torch.equal(demo_traj.rewards, replay_traj.rewards)


def test_received_trajectory_helper_keeps_chunk_trajectory_for_chunk_rl():
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.replay_buffer = _RecordingBuffer()
    worker.demo_buffer = _RecordingBuffer()
    worker.use_chunk_rl = True

    trajectory = Trajectory(
        max_episode_length=8,
        model_weights_id="weights-chunk-1",
        actions=torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]),
        intervene_flags=torch.tensor(
            [[[False, False, False, False, True, True]]], dtype=torch.bool
        ),
        transition_valids=torch.tensor([[[True, True, False]]], dtype=torch.bool),
        rewards=torch.tensor([[[1.0, 2.0, 0.0]]]),
    )

    EmbodiedSACFSDPPolicy._add_received_trajectories_to_buffers(worker, [trajectory])

    assert len(worker.replay_buffer.trajectories) == 1
    replay_traj = worker.replay_buffer.trajectories[0]
    assert torch.equal(replay_traj.actions, trajectory.actions)
    assert torch.equal(replay_traj.transition_valids, trajectory.transition_valids)

    assert len(worker.demo_buffer.trajectories) == 1
    demo_traj = worker.demo_buffer.trajectories[0]
    assert torch.equal(demo_traj.actions, trajectory.actions)


def test_async_sac_run_training_respects_train_actor_steps(monkeypatch):
    worker = object.__new__(AsyncEmbodiedSACFSDPPolicy)
    worker._timer_metrics = {}
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "replay_buffer": {"min_buffer_size": 2},
                "train_actor_steps": 20,
                "update_epoch": 1,
            },
            "actor": {},
        }
    )
    worker.update_step = 0

    class _ReplayBuffer:
        def __init__(self):
            self.queries = []

        def is_ready(self, min_size):
            self.queries.append(int(min_size))
            return min_size <= 2

    worker.replay_buffer = _ReplayBuffer()

    async def _wait_for_replay_buffer_ready(min_buffer_size):
        assert min_buffer_size == 2

    worker._wait_for_replay_buffer_ready = _wait_for_replay_buffer_ready
    worker._prepare_training_loop = lambda: True

    observed_train_actor = []

    def _update_one_epoch(*, train_actor=True):
        observed_train_actor.append(bool(train_actor))
        return {"loss": 1.0}

    worker.update_one_epoch = _update_one_epoch
    worker.process_train_metrics = lambda metrics: metrics

    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    metrics = asyncio.run(worker.run_training())

    assert observed_train_actor == [False]
    assert worker.replay_buffer.queries == [20]
    assert metrics["loss"] == [1.0]


class _FakeHandle:
    def __init__(self, result=None):
        self._result = result

    def wait(self):
        return self._result

    def consume_durations(self, return_per_rank=False):
        if return_per_rank:
            return {}, []
        return {}

    def done(self):
        return True


class _FakeActor:
    worker_group_name = "ActorGroup"

    def __init__(self):
        self.global_steps = []

    def set_global_step(self, step):
        self.global_steps.append(int(step))
        return _FakeHandle()

    def sync_model_to_rollout(self):
        return _FakeHandle()

    def recv_rollout_trajectories(self, input_channel):
        del input_channel
        return _FakeHandle()

    def run_training(self):
        return _FakeHandle([{"loss": 1.0}])

    def stop(self):
        return _FakeHandle()


class _FakeRollout:
    worker_group_name = "RolloutGroup"

    def __init__(self):
        self.global_steps = []

    def set_global_step(self, step):
        self.global_steps.append(int(step))
        return _FakeHandle()

    def sync_model_from_actor(self):
        return _FakeHandle()

    def generate(self, input_channel, output_channel, metric_channel):
        del input_channel, output_channel, metric_channel
        return _FakeHandle()

    def stop(self):
        return _FakeHandle()


class _FakeEnv:
    worker_group_name = "EnvGroup"

    def interact(self, input_channel, output_channel, metric_channel, replay_channel):
        del input_channel, output_channel, metric_channel, replay_channel
        return _FakeHandle()

    def stop(self):
        return _FakeHandle()


def test_async_embodied_runner_updates_actor_and_rollout_global_steps(tmp_path):
    cfg = OmegaConf.create(
        {
            "runner": {
                "max_epochs": 1,
                "max_steps": 1,
                "only_eval": False,
                "val_check_interval": -1,
                "save_interval": -1,
                "weight_sync_interval": 1,
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "rlinf",
                    "experiment_name": "async-runner-test",
                    "logger_backends": [],
                },
            },
            "actor": {"sync_weight_no_wait": False},
        }
    )

    actor = _FakeActor()
    rollout = _FakeRollout()
    env = _FakeEnv()
    runner = AsyncEmbodiedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
    runner.print_metrics_table_async = lambda *args, **kwargs: None

    runner.run()

    assert actor.global_steps == [0, 1]
    assert rollout.global_steps == [0, 1]
