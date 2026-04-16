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

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_sac_mlp_async"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
            AsyncEmbodiedSACFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
            AsyncEmbodiedDAGGERFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "decoupled_actor_critic":
        from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
        from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor

        runner_cls = AsyncPPOEmbodiedRunner
        actor_worker_cls = AsyncPPOEmbodiedFSDPActor
    else:
        raise ValueError(
            f"Unsupported loss type {cfg.algorithm.loss_type} for async embodied runner"
        )

    offline_critic_warmup_updates = int(
        cfg.runner.get("offline_critic_warmup_updates", 0)
    )
    if offline_critic_warmup_updates > 0:
        warmup_micro_batch_size = cfg.runner.get(
            "offline_critic_warmup_micro_batch_size", None
        )
        warmup_global_batch_size = cfg.runner.get(
            "offline_critic_warmup_global_batch_size", None
        )
        if warmup_micro_batch_size is not None:
            cfg.actor.micro_batch_size = int(warmup_micro_batch_size)
        if warmup_global_batch_size is not None:
            cfg.actor.global_batch_size = int(warmup_global_batch_size)
        actor_group = actor_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )
        actor_group.init_worker().wait()
        warmup_metrics = actor_group.run_critic_warmup(
            num_updates=offline_critic_warmup_updates
        ).wait()
        save_dir = cfg.runner.get("offline_critic_warmup_save_dir", None)
        if save_dir is None:
            save_dir = os.path.join(
                cfg.runner.logger.log_path,
                "offline_critic_warmup",
                f"updates_{offline_critic_warmup_updates}",
            )
        actor_save_dir = os.path.join(save_dir, "actor")
        actor_group.save_checkpoint(actor_save_dir, offline_critic_warmup_updates).wait()
        print(
            json.dumps(
                {
                    "offline_critic_warmup_updates": offline_critic_warmup_updates,
                    "offline_critic_warmup_save_dir": save_dir,
                    "offline_critic_warmup_actor_dir": actor_save_dir,
                    "offline_critic_warmup_micro_batch_size": cfg.actor.micro_batch_size,
                    "offline_critic_warmup_global_batch_size": cfg.actor.global_batch_size,
                    "offline_critic_warmup_metrics": warmup_metrics,
                },
                indent=2,
            )
        )
        return

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = AsyncEnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
