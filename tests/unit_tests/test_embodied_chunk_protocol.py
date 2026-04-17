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

import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rlinf.data.embodied_io_struct import EnvOutput, Trajectory
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_extract_valid_traj_splits_chunked_transition_mask():
    trajectory = Trajectory(
        max_episode_length=12,
        model_weights_id="weights-1",
        actions=torch.tensor(
            [
                [[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]],
                [[40.0, 41.0, 50.0, 51.0, 60.0, 61.0]],
            ]
        ),
        intervene_flags=torch.zeros((2, 1, 6), dtype=torch.bool),
        transition_valids=torch.tensor(
            [[[True, False, True]], [[False, True, False]]], dtype=torch.bool
        ),
        rewards=torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]),
        terminations=torch.tensor(
            [[[False, False, False]], [[False, False, True]]], dtype=torch.bool
        ),
        truncations=torch.zeros((2, 1, 3), dtype=torch.bool),
        dones=torch.tensor(
            [[[False, False, False]], [[False, False, True]]], dtype=torch.bool
        ),
        prev_logprobs=torch.tensor(
            [
                [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
                [[[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]],
            ]
        ),
        prev_values=torch.tensor([[[0.0]], [[1.0]]]),
        versions=torch.tensor([[[2.0]], [[3.0]]]),
        curr_obs={
            "states": torch.tensor([[[100.0, 101.0]], [[200.0, 201.0]]]),
        },
        next_obs={
            "states": torch.tensor([[[110.0, 111.0]], [[210.0, 211.0]]]),
        },
        forward_inputs={
            "action": torch.tensor(
                [
                    [[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]],
                    [[40.0, 41.0, 50.0, 51.0, 60.0, 61.0]],
                ]
            ),
            "states": torch.tensor([[[100.0, 101.0]], [[200.0, 201.0]]]),
        },
    )

    extracted = trajectory.extract_valid_traj()

    assert extracted is not None
    assert len(extracted) == 1
    valid_traj = extracted[0]
    assert torch.equal(
        valid_traj.actions[:, 0],
        torch.tensor([[10.0, 11.0], [30.0, 31.0], [50.0, 51.0]]),
    )
    assert torch.equal(valid_traj.rewards[:, 0], torch.tensor([1.0, 3.0, 5.0]))
    assert torch.equal(
        valid_traj.curr_obs["states"][:, 0],
        torch.tensor([[100.0, 101.0], [100.0, 101.0], [200.0, 201.0]]),
    )
    assert torch.equal(
        valid_traj.next_obs["states"][:, 0],
        torch.tensor([[110.0, 111.0], [110.0, 111.0], [210.0, 211.0]]),
    )
    assert torch.equal(
        valid_traj.transition_valids[:, 0],
        torch.tensor([True, True, True]),
    )


def test_extract_valid_chunk_traj_keeps_macro_actions_and_drops_non_prefix_masks():
    trajectory = Trajectory(
        max_episode_length=12,
        model_weights_id="weights-chunk-1",
        actions=torch.tensor(
            [
                [[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]],
                [[40.0, 41.0, 50.0, 51.0, 60.0, 61.0]],
                [[70.0, 71.0, 80.0, 81.0, 90.0, 91.0]],
            ]
        ),
        intervene_flags=torch.zeros((3, 1, 6), dtype=torch.bool),
        transition_valids=torch.tensor(
            [
                [[True, True, True]],
                [[True, True, False]],
                [[True, False, True]],
            ],
            dtype=torch.bool,
        ),
        rewards=torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 0.0]],
                [[6.0, 0.0, 7.0]],
            ]
        ),
        curr_obs={
            "states": torch.tensor([[[100.0, 101.0]], [[200.0, 201.0]], [[300.0, 301.0]]]),
        },
        next_obs={
            "states": torch.tensor([[[110.0, 111.0]], [[210.0, 211.0]], [[310.0, 311.0]]]),
        },
    )

    extracted = trajectory.extract_valid_chunk_traj()

    assert extracted is not None
    assert len(extracted) == 1
    valid_traj = extracted[0]
    assert torch.equal(
        valid_traj.actions[:, 0],
        torch.tensor(
            [
                [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                [40.0, 41.0, 50.0, 51.0, 60.0, 61.0],
            ]
        ),
    )
    assert torch.equal(
        valid_traj.transition_valids[:, 0],
        torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool),
    )
    assert torch.equal(
        valid_traj.curr_obs["states"][:, 0],
        torch.tensor([[100.0, 101.0], [200.0, 201.0]]),
    )


def test_extract_intervene_traj_splits_chunked_intervention_mask():
    trajectory = Trajectory(
        max_episode_length=8,
        model_weights_id="weights-2",
        actions=torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]),
        intervene_flags=torch.tensor(
            [[[True, True, False, False, True, False]]], dtype=torch.bool
        ),
        transition_valids=torch.tensor([[[True, True, True]]], dtype=torch.bool),
        rewards=torch.tensor([[[1.0, 2.0, 3.0]]]),
        curr_obs={"states": torch.tensor([[[7.0, 8.0]]])},
        next_obs={"states": torch.tensor([[[9.0, 10.0]]])},
    )

    extracted = trajectory.extract_intervene_traj(mode="any")

    assert extracted is not None
    assert len(extracted) == 1
    intervene_traj = extracted[0]
    assert torch.equal(
        intervene_traj.actions[:, 0],
        torch.tensor([[1.0, 2.0], [5.0, 6.0]]),
    )
    assert torch.equal(intervene_traj.rewards[:, 0], torch.tensor([1.0, 3.0]))


def test_env_worker_stops_rollout_on_episode_end_when_enabled():
    worker = object.__new__(EnvWorker)
    worker.n_train_chunk_steps = 18
    worker.stop_rollout_on_episode_end = True

    env_output = EnvOutput(
        obs={"states": torch.zeros((1, 2))},
        dones=torch.tensor([[False, True]], dtype=torch.bool),
    )

    assert worker._should_stop_rollout_after_step(env_output, 3) is True
    assert worker._should_stop_rollout_after_step(env_output, 17) is True

    worker.stop_rollout_on_episode_end = False
    assert worker._should_stop_rollout_after_step(env_output, 3) is False
    assert worker._should_stop_rollout_after_step(env_output, 17) is True


def test_chunk_step_preserves_chunked_data_valid_on_truncation():
    env = object.__new__(RealWorldEnv)
    env.num_envs = 1
    env.auto_reset = True
    env.ignore_terminations = False

    step_calls = []
    step_results = [
        (
            {"states": torch.tensor([[1.0]])},
            torch.tensor([0.1]),
            torch.tensor([False]),
            torch.tensor([False]),
            {"data_valid": torch.tensor([True])},
        ),
        (
            {"states": torch.tensor([[2.0]])},
            torch.tensor([0.2]),
            torch.tensor([False]),
            torch.tensor([True]),
            {
                "data_valid": torch.tensor([False]),
                "episode": {"timeout_fail": torch.tensor([True])},
            },
        ),
    ]

    def fake_step(actions, auto_reset=True):
        del auto_reset
        step_calls.append(actions.clone())
        return step_results[len(step_calls) - 1]

    def fake_handle_auto_reset(dones, final_obs, infos):
        return (
            {"states": torch.tensor([[99.0]])},
            {
                "final_observation": final_obs,
                "final_info": infos,
                "_final_info": dones,
                "_final_observation": dones,
                "_elapsed_steps": dones,
            },
        )

    env.step = fake_step
    env._handle_auto_reset = fake_handle_auto_reset

    _, _, terminations, truncations, infos_list = RealWorldEnv.chunk_step(
        env, torch.zeros(1, 4, 2)
    )

    assert torch.equal(terminations, torch.zeros(1, 4, dtype=torch.bool))
    assert torch.equal(truncations, torch.tensor([[False, False, False, True]]))
    assert torch.equal(
        infos_list[-1]["data_valid"],
        torch.tensor([[True, False, False, False]]),
    )
    assert torch.equal(
        infos_list[-1]["final_info"]["data_valid"],
        torch.tensor([[True, False, False, False]]),
    )


def test_generate_one_epoch_stops_after_rollout_stop_batch():
    worker = object.__new__(MultiStepRolloutWorker)
    worker.collect_prev_infos = True
    worker.num_pipeline_stages = 1
    worker.version = 3
    worker._timer_metrics = {}
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(model=SimpleNamespace(num_action_chunks=2))
    )
    worker.update_dagger_beta = lambda: None

    env_batches = iter(
        [
            {
                "obs": {"states": torch.zeros((1, 2))},
                "final_obs": None,
                "rollout_stop": False,
            },
            {
                "obs": {"states": torch.ones((1, 2))},
                "final_obs": {"states": torch.full((1, 2), 2.0)},
                "rollout_stop": True,
            },
        ]
    )

    async def fake_recv_env_output(input_channel):
        del input_channel
        return next(env_batches)

    def fake_predict(obs):
        batch_size = obs["states"].shape[0]
        actions = torch.ones((batch_size, 2, 2), dtype=torch.float32)
        result = {
            "prev_logprobs": torch.full((batch_size, 2), 0.5, dtype=torch.float32),
            "prev_values": torch.full((batch_size, 1), 0.25, dtype=torch.float32),
            "forward_inputs": {
                "action": torch.full((batch_size, 4), 2.0, dtype=torch.float32)
            },
        }
        return actions, result

    def fake_get_bootstrap_values(final_obs):
        if final_obs is None:
            return None
        return torch.full((final_obs["states"].shape[0], 1), 9.0)

    sent_results = []

    def fake_send_rollout_result(output_channel, rollout_result, mode="train"):
        del output_channel
        sent_results.append((mode, rollout_result))

    worker.recv_env_output = fake_recv_env_output
    worker.predict = fake_predict
    worker.get_bootstrap_values = fake_get_bootstrap_values
    worker.send_rollout_result = fake_send_rollout_result

    asyncio.run(MultiStepRolloutWorker.generate_one_epoch(worker, None, None))

    assert len(sent_results) == 2
    assert sent_results[0][0] == "train"
    assert sent_results[0][1].actions is not None
    assert torch.equal(sent_results[0][1].versions, torch.tensor([[3.0]]))
    assert sent_results[1][1].actions is None
    assert torch.equal(sent_results[1][1].bootstrap_values, torch.tensor([[9.0]]))
