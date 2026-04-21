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


import os

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.total_cnt = 0
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        if self.cfg.env.eval.get("data_collection", None) and getattr(
            self.cfg.env.eval.data_collection, "enabled", False
        ):
            from rlinf.envs.wrappers import CollectEpisode

            self.env = CollectEpisode(
                self.env,
                save_dir=self.cfg.env.eval.data_collection.save_dir,
                export_format=getattr(
                    self.cfg.env.eval.data_collection, "export_format", "pickle"
                ),
                robot_type=getattr(
                    self.cfg.env.eval.data_collection, "robot_type", "panda"
                ),
                fps=getattr(self.cfg.env.eval.data_collection, "fps", 10),
                only_success=getattr(
                    self.cfg.env.eval.data_collection, "only_success", False
                ),
                finalize_interval=getattr(
                    self.cfg.env.eval.data_collection, "finalize_interval", 100
                ),
            )

        # Read from the wrapped action space so GripperCloseEnv / dual-arm all just work.
        self.action_dim = int(self.env.action_space.shape[-1])

        buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")

        self.buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

    def _process_obs(self, obs):
        """Reshape env obs into the dict EmbodiedRolloutResult expects."""
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            val = val.cpu()
            if key == "images":
                ret_obs["main_images"] = val.clone()
            else:
                ret_obs[key] = val.clone()
        return ret_obs

    def _check_classifier_success(self, info):
        """Check classifier-based success from info dict.

        Returns (is_success, classifier_reward) if classifier wrapper is
        active, otherwise falls back to reward-based check.
        """
        succeed = self._extract_scalar(info.get("succeed", None))
        clf_reward = self._extract_scalar(info.get("classifier_reward", None))
        if succeed is None:
            return None, clf_reward
        return bool(succeed), clf_reward

    @staticmethod
    def _extract_scalar(value):
        """Extract a Python scalar from tensor/ndarray/list-like values."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.reshape(-1)[0].item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return value[0]
        return value

    @staticmethod
    def _to_column_tensor(value, *, dtype):
        """Convert scalars or vectors to the `[B, 1]` layout used by trajectories."""
        tensor = torch.as_tensor(value, dtype=dtype).cpu()
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor.reshape(-1, 1)

    def run(self):
        max_steps = self.cfg.env.eval.max_episode_steps
        use_classifier = self.cfg.env.eval.get("classifier_reward_wrapper", None) is not None

        obs, _ = self.env.reset()
        success_cnt = 0
        episode_cnt = 0

        self.log_info(
            f"\n{'=' * 60}\n"
            f"  Data collection started\n"
            f"  Target successful demos: {self.num_data_episodes}\n"
            f"  Max steps per episode: {max_steps}\n"
            f"  Success criterion: {'visual classifier' if use_classifier else 'target pose (built-in)'}\n"
            f"{'=' * 60}"
        )

        current_rollout = EmbodiedRolloutResult(max_episode_length=max_steps)

        current_obs_processed = self._process_obs(obs)

        # Print first episode header
        episode_cnt += 1
        self.log_info(
            f"\n{'#' * 50}\n"
            f"  Episode {episode_cnt}  "
            f"success: {success_cnt}/{self.num_data_episodes}\n"
            f"  >>> Start teleoperation <<<\n"
            f"{'#' * 50}"
        )

        progress_bar = tqdm(
            total=self.num_data_episodes, desc="Collecting Data Episodes:"
        )

        while success_cnt < self.num_data_episodes:
            action = np.zeros((1, self.action_dim), dtype=np.float32)
            next_obs, reward, terminated, truncated, info = self.env.step(
                action, auto_reset=False
            )

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = self._process_obs(next_obs)

            terminated_tensor = self._to_column_tensor(terminated, dtype=torch.bool)
            truncated_tensor = self._to_column_tensor(truncated, dtype=torch.bool)
            done_tensor = terminated_tensor | truncated_tensor

            action_tensor = torch.as_tensor(action, dtype=torch.float32)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)
            reward_tensor = self._to_column_tensor(reward, dtype=torch.float32)

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor,
                terminations=terminated_tensor,
                truncations=truncated_tensor,
                forward_inputs={"action": action_tensor},
            )

            current_rollout.append_step_result(step_result)
            current_rollout.append_transitions(
                curr_obs=current_obs_processed, next_obs=next_obs_processed
            )

            episode_done = bool(done_tensor.any().item())
            if not episode_done:
                obs = next_obs
                current_obs_processed = next_obs_processed
                continue

            reward_value = self._extract_scalar(reward)
            reward_value = float(reward_value) if reward_value is not None else 0.0
            self.total_cnt += 1

            # Determine success: prefer classifier info, fallback to reward
            clf_success, clf_reward_val = self._check_classifier_success(info)
            if clf_success is not None:
                is_success = clf_success
                success_metric = (
                    float(clf_reward_val)
                    if clf_reward_val is not None
                    else reward_value
                )
            else:
                is_success = reward_value >= 0.5
                success_metric = reward_value

            if is_success:
                success_cnt += 1
                self.log_info(
                    f"Episode {episode_cnt} succeeded "
                    f"(metric={success_metric:.3f}). "
                    f"Total: {success_cnt}/{self.num_data_episodes}"
                )

                trajectory = current_rollout.to_trajectory()
                if trajectory.intervene_flags is not None:
                    trajectory.intervene_flags = torch.ones_like(
                        trajectory.intervene_flags
                    )
                self.buffer.add_trajectories([trajectory])
                progress_bar.update(1)
            else:
                self.log_info(
                    f"Episode {episode_cnt} ended without success "
                    f"(metric={success_metric:.3f}). "
                    f"Discarded. Total success: {success_cnt}/{self.num_data_episodes}"
                )

            if success_cnt >= self.num_data_episodes:
                break

            obs, _ = self.env.reset()
            current_obs_processed = self._process_obs(obs)
            current_rollout = EmbodiedRolloutResult(max_episode_length=max_steps)
            episode_cnt += 1
            self.log_info(
                f"\n{'#' * 50}\n"
                f"  Episode {episode_cnt}  "
                f"success: {success_cnt}/{self.num_data_episodes}\n"
                f"  >>> Start teleoperation <<<\n"
                f"{'#' * 50}"
            )

        progress_bar.close()
        self.buffer.close()
        self.log_info(
            f"Finished. Demos saved in: {os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
        )
        self.env.close()


def _launch_classifier_reward_server(cfg, cluster, component_placement):
    """Launch ClassifierRewardServer if configured in component_placement.

    Returns:
        WorkerGroup or None if not configured.
    """
    from rlinf.workers.reward import launch_classifier_reward_server

    reward_server_strategy = component_placement.get_strategy("reward_server", required=False)
    if reward_server_strategy is not None:
        reward_server_cfg = cfg.get("reward_server", None)
        if reward_server_cfg is None:
            raise ValueError(
                "component_placement has 'reward_server' but config missing 'reward_server' section"
            )
        return launch_classifier_reward_server(
            cfg=cfg,
            cluster=cluster,
            placement_strategy=reward_server_strategy,
        )

    return None


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")

    # Launch ClassifierRewardServer if configured
    server_handle = _launch_classifier_reward_server(cfg, cluster, component_placement)

    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()

    if server_handle is not None:
        server_handle.shutdown()


if __name__ == "__main__":
    main()
