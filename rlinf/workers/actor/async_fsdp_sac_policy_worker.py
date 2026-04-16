# Copyright 2025 The RLinf Authors.
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
import queue
import threading

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class AsyncEmbodiedSACFSDPPolicy(EmbodiedSACFSDPPolicy):
    should_stop = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.should_stop = False

    async def recv_rollout_trajectories(self, input_channel):
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        while not self.should_stop:
            for _ in range(split_num):
                trajectory = input_channel.get()
                self._recv_queue.put(trajectory)

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return

        self._add_received_trajectories_to_buffers(recv_list)

    async def _wait_for_replay_buffer_ready(self, min_buffer_size: int):
        while True:
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 256)
            )
            if await self.replay_buffer.is_ready_async(min_buffer_size):
                return
            await asyncio.sleep(1)

    @Worker.timer("run_training")
    async def run_training(self):
        """SAC training using replay buffer"""
        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        await self._wait_for_replay_buffer_ready(min_buffer_size)

        if not self._prepare_training_loop():
            return {}

        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        torch.distributed.barrier()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            await asyncio.sleep(0)
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    async def stop(self):
        self.should_stop = True
        recv_thread = getattr(self, "_recv_rollout_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
        self._drain_received_trajectories()
        if getattr(self, "buffer_dataset", None) is not None:
            self.buffer_dataset.close()
        if getattr(self, "replay_buffer", None) is not None:
            self.replay_buffer.close(wait=True)
        if getattr(self, "demo_buffer", None) is not None:
            self.demo_buffer.close(wait=True)
