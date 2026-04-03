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

import threading
from collections import deque
from types import SimpleNamespace

import pytest

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gymnasium import spaces

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
import rlinf.envs.realworld.common.wrappers.reward_done_wrapper as reward_done_wrapper
from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
    KeyboardRewardDoneWrapper,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv


class _DummyEnv(gym.Env):
    metadata = {}

    def __init__(self, *, terminated=False, truncated=False, info=None):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.terminated = terminated
        self.truncated = truncated
        self.info = dict(info or {})

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        del action
        return (
            np.zeros((1,), dtype=np.float32),
            0.25,
            self.terminated,
            self.truncated,
            dict(self.info),
        )


class _QueuedListener:
    def __init__(self, events=None):
        self.events = list(events or [])

    def get_key(self):
        if not self.events:
            return None
        return self.events.pop(0)


def _make_test_listener() -> KeyboardListener:
    listener = object.__new__(KeyboardListener)
    listener.state_lock = threading.Lock()
    listener._pressed_keys = set()
    listener._pending_keys = deque()
    listener.last_intervene = 0
    return listener


def _key(char: str):
    return SimpleNamespace(char=char)


def test_keyboard_listener_emits_one_event_per_press_until_release():
    listener = _make_test_listener()

    listener.on_key_press(_key("s"))
    listener.on_key_press(_key("s"))

    assert listener.get_key() == "s"
    assert listener.get_key() is None

    listener.on_key_release(_key("s"))
    listener.on_key_press(_key("s"))

    assert listener.get_key() == "s"
    assert listener.get_key() is None


def test_keyboard_reward_wrapper_consumes_success_once(monkeypatch):
    monkeypatch.setattr(
        reward_done_wrapper, "KeyboardListener", lambda: _QueuedListener(["s"])
    )
    env = KeyboardRewardDoneWrapper(_DummyEnv())

    _, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))
    assert reward == 1
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["fail"] is False

    _, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))
    assert reward == 0
    assert terminated is False
    assert truncated is False
    assert info["success"] is False
    assert info["fail"] is False


def test_keyboard_reward_wrapper_preserves_env_done_without_keypress(monkeypatch):
    monkeypatch.setattr(
        reward_done_wrapper, "KeyboardListener", lambda: _QueuedListener([])
    )
    env = KeyboardRewardDoneWrapper(_DummyEnv(terminated=True))

    _, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))

    assert reward == 0
    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    assert info["fail"] is False


def test_realworld_step_preserves_underlying_truncation():
    env = object.__new__(RealWorldEnv)
    env.cfg = SimpleNamespace(max_episode_steps=10)
    env.num_envs = 1
    env.auto_reset = False
    env.ignore_terminations = False
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env.env = SimpleNamespace(
        step=lambda actions: (
            {"unused": True},
            np.array([0.0], dtype=np.float32),
            np.array([False]),
            np.array([True]),
            {},
        )
    )
    env._wrap_obs = lambda raw_obs: {"raw_obs": raw_obs}
    env._calc_step_reward = lambda reward: reward.astype(np.float32)
    env._record_metrics = lambda step_reward, terminations, intervene_flag, infos: infos

    _, reward, terminations, truncations, infos = RealWorldEnv.step(
        env, np.zeros((1, 1), dtype=np.float32), auto_reset=False
    )

    assert reward.item() == 0.0
    assert terminations.item() is False
    assert truncations.item() is True
    assert infos["timeout_fail"].item() is False


def test_chunk_step_stops_after_terminal_event_and_pads_chunk_metadata():
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
            {
                "intervene_action": torch.tensor([[1.0, 2.0]]),
                "intervene_flag": torch.tensor([False]),
                "data_valid": torch.tensor([True]),
            },
        ),
        (
            {"states": torch.tensor([[2.0]])},
            torch.tensor([1.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            {
                "intervene_action": torch.tensor([[3.0, 4.0]]),
                "intervene_flag": torch.tensor([True]),
                "data_valid": torch.tensor([True]),
                "episode": {"success_once": torch.tensor([True])},
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

    obs_list, rewards, terminations, truncations, infos_list = RealWorldEnv.chunk_step(
        env, torch.zeros(1, 4, 2)
    )

    assert len(step_calls) == 2
    assert len(obs_list) == 2
    assert torch.equal(obs_list[-1]["states"], torch.tensor([[99.0]]))
    assert torch.allclose(rewards, torch.tensor([[0.1, 1.0, 0.0, 0.0]]))
    assert torch.equal(
        terminations, torch.tensor([[False, False, False, True]])
    )
    assert torch.equal(truncations, torch.zeros(1, 4, dtype=torch.bool))
    assert torch.equal(
        infos_list[-1]["data_valid"],
        torch.tensor([[True, True, False, False]]),
    )
    assert torch.equal(
        infos_list[-1]["final_info"]["data_valid"],
        torch.tensor([[True, True, False, False]]),
    )
    assert torch.equal(
        infos_list[-1]["intervene_action"],
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        infos_list[-1]["intervene_flag"],
        torch.tensor([[False, True, False, False]]),
    )
    assert torch.equal(
        infos_list[-1]["final_observation"]["states"],
        torch.tensor([[2.0]]),
    )
