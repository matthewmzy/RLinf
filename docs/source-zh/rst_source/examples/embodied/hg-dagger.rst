真实 Franka 的 HG-DAgger 全流程
===============================

**HG-DAgger** （Human-Gated DAgger）是一种面向真实世界交互式模仿学习的算法
流程。该流程先采集带遥操作的真实数据，再基于收集到的 LeRobot 数据集执行
OpenPI SFT，最后在机器人上继续运行异步在线 HG-DAgger。

在 RLinf 配置中，HG-DAgger 主要通过
``algorithm.dagger.only_save_expert: True`` 启用。该选项表示仅保存专家实际执行
的 step，这也是现实世界干预式数据的默认用法。

环境
----

**真实 Franka Bin Relocation + Pi0**

- **环境**：运行在机器人节点上的 ``FrankaBinRelocationEnv-v1``
- **观测**：腕部 / 外部 RGB 图像与机器人状态
- **动作空间**：末端执行器 delta qpos 与夹爪动作
- **适用场景**：采集带人工引导的真实数据，进行 OpenPI SFT，然后继续异步 HG-DAgger

算法
----

**HG-DAgger 流程**

1. **人工引导数据采集**

   - 操作者通过 spacemouse 在真机上进行干预。
   - RLinf 将成功轨迹导出为 LeRobot 数据集，供后续 SFT 使用。

2. **监督预热**

   - 为采集到的数据集计算归一化统计量。
   - 先运行 OpenPI SFT，将人工引导数据训练成初始学生策略。

3. **在线 HG-DAgger**

   - 异步 rollout 在真机上继续执行，并使用 ``beta`` 调度专家引导。
   - 当 ``only_save_expert: True`` 时，只有专家实际执行的 step 会写入 replay buffer。

4. **Replay Buffer 更新**

   - actor 使用 ``embodied_dagger`` 损失在干预数据上继续训练。
   - SFT 阶段导出的 checkpoint 会作为在线 HG-DAgger 的初始化模型。

依赖安装
--------

真实世界流程的不同节点需要 **不同的软件环境**：

- **机器人 / env 节点**：使用 :doc:`franka` 中的 Franka 控制节点环境。
- **训练 / rollout 节点**：使用与模拟器 DAgger :doc:`dagger` 相同的环境。

机器人 / Env 节点
~~~~~~~~~~~~~~~~~

请先参考 :doc:`franka` 中的控制节点安装说明，完成固件检查、实时内核、ROS 与
Franka 控制依赖的准备。

**选项 1：Docker 镜像**

.. code:: bash

   docker run -it --rm \
      --privileged \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-franka
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-franka

随后切换到与你的 libfranka 版本兼容的环境：

.. code:: bash

   source switch_env franka-<libfranka_version>

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 参数。
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

在机器人节点执行 ``ray start`` 之前，请像 :doc:`franka` 中说明的那样，先
source 对应的 ROS / Franka controller 环境。

训练 / Rollout 节点
~~~~~~~~~~~~~~~~~~~

该节点使用与模拟器 Pi0 DAgger 相同的软件环境。

**选项 1：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后执行：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 参数。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

集群配置
--------

在启动采集或训练任务之前，请先完成 :doc:`franka` 中介绍的 Ray 集群配置。
通常训练 / rollout 节点作为 Ray head（``RLINF_NODE_RANK=0``），Franka 控制
节点作为 worker（``RLINF_NODE_RANK=1``）。

.. code-block:: bash

   # 在训练 / rollout 节点
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_node_ip>

   # 在机器人 / env 节点
   export RLINF_NODE_RANK=1
   ray start --address='<head_node_ip>:6379'

Ray 会在启动时记录当前 Python 解释器与环境变量，因此务必在 ``ray start``
之前完成对应环境的 source。

全流程
------

1. 采集带人工引导的真实数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~

从 ``examples/embodiment/config/realworld_collect_data.yaml`` 开始。对于抓放
任务，需要将环境从 peg insertion 切换为 bin relocation：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

然后填写机器人配置，并保持 LeRobot 导出开启：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 0

   env:
     eval:
       use_spacemouse: True
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         success_hold_steps: 1
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
      data_collection:
        enabled: True
        save_dir: ${runner.logger.log_path}/collected_data
        export_format: "lerobot"
        only_success: True
        robot_type: "panda"
        fps: 10

使用你复制后的配置启动采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

遥操作过程中，同一次运行会写出：

- replay-buffer 轨迹到 ``logs/{timestamp}/demos/``
- LeRobot 数据到 ``logs/{timestamp}/collected_data/``

关于采集格式，参见 :doc:`../../tutorials/components/data_collection`。

2. 计算归一化统计
~~~~~~~~~~~~~~~~~

在进行 SFT 或 HG-DAgger 之前，先为采集得到的 LeRobot 数据集计算 OpenPI
归一化统计：

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

这里使用的数据集根目录和数据集 id，需要与后续 SFT 保持一致。更多 OpenPI
数据集说明可参考 :doc:`sft_openpi`。

3. 运行 OpenPI SFT
~~~~~~~~~~~~~~~~~~

启动前，先修改 ``examples/sft/config/realworld_sft_openpi.yaml``：

.. code-block:: yaml

   data:
     train_data_paths: "/path/to/realworld-franka-bin-relocation-dataset"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_realworld"

然后执行：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_sft_openpi

SFT 导出的 checkpoint 会作为在线阶段的学生模型初始化。更多 OpenPI SFT 细节
可参考 :doc:`sft_openpi`。

4. 在真机上运行异步 HG-DAgger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml``，使其与你的
集群、相机、目标位姿与 checkpoint 一致：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     node_groups:
       - label: "train"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   runner:
     ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

   algorithm:
     dagger:
       init_beta: 1.0
       beta_schedule: "exponential"
       beta_decay: 0.99
       only_save_expert: True

   env:
     train:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
     eval:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

   rollout:
     model:
       model_path: "/path/to/pi0-model"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_realworld"

在 Ray head 节点上启动 HG-DAgger：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

5. A2D + psi-policy：先做链路验收
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果你现在的目标不是直接做最终 SAC 微调，而是先确认：

- A2D 真机观测能进入 RLinf rollout
- teleop 接管动作能被保存成 intervention 样本
- ``psi-policy`` actor 能基于这些样本真的发生更新

那更推荐先跑仓库里新增的 A2D ``psi-policy`` HG-DAgger 预设：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_a2d_dagger_psi

可选配置名：

- ``realworld_a2d_dagger_psi``：推荐入口，等价于 ``realworld_a2d_dagger_psi_rtc``
- ``realworld_a2d_dagger_psi_rtc``：显式 RTC 配置
- ``realworld_a2d_dagger_psi_direct``：不做 RTC，最适合查单步链路
- ``realworld_a2d_dagger_psi_window``：window overlap 平滑版本

这组配置默认做了下面这些约束，目的是先把链路单独跑通：

- ``algorithm.loss_type: embodied_dagger``
- ``algorithm.dagger.only_save_expert: True``
- ``runner.ckpt_path: null``，第一次启动不走 RLinf warmstart
- ``rollout.model.noise_std_rollout: 0.0``，避免额外 rollout 噪声干扰排障
- ``actor/rollout.model.model_path`` 指向原始 ``psi-policy`` `.ckpt`
- ``normalizer_path`` 指向与该 `.ckpt` 配对的 ``normalizer.pkl``
- 当前推荐 RTC 配置下，还固定使用：

  - ``num_action_chunks: 4``
  - ``action_dim: 26``
  - ``rollout.action_execution.rtc.search_window: 4``
  - ``rollout.action_execution.rtc.merge_weight_base: 0.8``
  - ``actor.micro_batch_size: 8``
  - ``actor.global_batch_size: 8``
  - ``actor.optim.lr: 3e-5``
  - ``algorithm.update_epoch: 1``
  - ``algorithm.replay_buffer.min_buffer_size: 1``
  - ``runner.weight_sync_interval: 1``

其中 ``action_horizon`` 要特别注意：

- ``examples/embodiment/config/model/psi_policy.yaml`` 的静态默认值是 ``16``
- 但 ``RLinf`` 真正加载 ``psi-policy`` checkpoint 后，会用 checkpoint 自己的 ``n_action_steps`` 覆盖它
- 你当前这套 A2D preset 指向的是 ``..._na32/...`` 这颗 checkpoint，所以 runtime 实际值应该按 ``32`` 走

这里最容易填错的是路径语义：

- ``actor.model.model_path`` / ``rollout.model.model_path``：原始 ``psi-policy`` checkpoint
- ``normalizer_path``：和上面 checkpoint 成对的 normalizer
- ``runner.ckpt_path``：RLinf 自己保存出来的 actor ``state_dict``；第一次跑 A2D HG-DAgger 时不要填原始 `.ckpt`

当前 ``psi-policy`` HG-DAgger 的 replay 语义也已经改成了更接近原始 IL 的版本：

- rollout / env 先记录最终真实执行动作流，标签源是 ``Trajectory.actions``
- 其中 RTC 裁剪 / 平滑后的动作，以及 teleop 接管后的最终动作，都会写回这里
- actor 收到 raw trajectory 后，不再直接用 action-level intervention 样本训练
- 对 ``psi-policy``，actor 会把 raw trajectory 重组成：

  - ``obs_t -> future executed action window``

- 当前窗口长度固定跟 runtime checkpoint 的 ``action_horizon`` 走；你现在这颗 ``na32`` ckpt 对应的是 ``32``
- 只有“当前 chunk 的 4 个子步都由 expert / teleop 执行”的位置会被保留为 anchor

也就是说，当前学的是“某个观测之后，系统最终真实执行出去的未来动作轨迹”，而不是 rollout 某一次原始预测的 raw action chunk。

如果你想按当前配置理解启动顺序，可以按下面这条链路看日志：

1. ``run_realworld_async.sh`` 调 ``train_async.py``
2. `Hydra` 组合出 ``realworld_a2d_dagger_psi_rtc + realworld_a2d_dagger_psi_base``
3. runner 选择 ``AsyncEmbodiedRunner`` + ``AsyncEmbodiedDAGGERFSDPPolicy`` + ``AsyncMultiStepRolloutWorker`` + ``AsyncEnvWorker``
4. actor / rollout 分别加载原始 ``psi-policy`` checkpoint 和 normalizer
5. rollout 用 RTC 模式把预测 horizon 处理成真实执行的 ``4`` 步动作
6. env 真正执行这 ``4`` 步动作，并记录 ``rewards`` / ``dones`` / ``transition_valids`` / intervention 信息
7. env 在 rollout epoch 结束后把 raw trajectory 发给 actor
8. actor 先把 raw trajectory 重组为 ``obs_t -> future 32-step executed action window``
9. 样本进入 replay buffer；达到 ``min_buffer_size=1`` 后 learner 开始训练
10. learner 每次用 ``global_batch_size=8``、``micro_batch_size=8``、``update_epoch=1``、``lr=3e-5`` 做一次 SFT 更新
11. runner 每 ``weight_sync_interval=1`` 步把新 actor 权重再同步回 rollout

验收时，建议人工切几次 teleop 接管，然后重点观察：

- ``train/replay_buffer/num_trajectories``
- ``train/replay_buffer/total_samples``
- ``train/dagger/actor_loss``

只要这些指标开始变化，就说明 A2D 的 intervention 数据链路和 ``psi-policy`` 的 HG-DAgger 更新入口已经接通。

可视化与监控
------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. 推荐关注的监控指标**

- ``train/dagger/actor_loss``：基于干预数据计算的 HG-DAgger 监督损失。
- ``train/replay_buffer/num_trajectories``：当前已保存轨迹数量。
- ``train/replay_buffer/total_samples``：当前可训练样本总数。
- ``train/actor/lr``：学习率。
- ``train/actor/grad_norm``：梯度范数。
