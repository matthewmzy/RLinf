import numpy as np
import pytest

pytest.importorskip("omegaconf")

from rlinf.envs.realworld.franka.franka_env import FrankaEnv


class _AsyncResult:
    def __init__(self, value):
        self._value = value

    def wait(self):
        return [self._value]


class _ControllerStub:
    def __init__(self):
        self.last_command = None

    def command_end_effector(self, action):
        self.last_command = np.array(action, dtype=np.float64)
        return _AsyncResult(True)


def _make_dummy_hand_env() -> FrankaEnv:
    env = FrankaEnv(
        override_cfg={
            "camera_serials": ["1"],
            "end_effector_type": "ruiyan_hand",
            "is_dummy": True,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )
    env._controller = _ControllerStub()
    return env


def test_ruiyan_hand_action_space_stays_symmetric():
    env = _make_dummy_hand_env()

    np.testing.assert_allclose(env.action_space.low[6:], -1.0)
    np.testing.assert_allclose(env.action_space.high[6:], 1.0)


def test_ruiyan_hand_env_action_maps_to_normalized_command():
    env = _make_dummy_hand_env()

    hand_action = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.2], dtype=np.float64)
    effective = env._end_effector_action(hand_action)

    assert effective is True
    np.testing.assert_allclose(
        env._controller.last_command,
        np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.6], dtype=np.float64),
    )
