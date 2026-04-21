A2D 真机强化学习
=================

本文档说明如何把官方 A2D 遥操作 Docker 作为外部 controller 接入 RLinf。

环境
----

- **环境类型**: ``realworld``
- **Gym id**: ``A2DEnv-v1``
- **控制边界**: RLinf 通过正在运行的 A2D Docker 暴露的 ``model_inference`` gRPC 服务访问机器人
- **观测**:

  - ``rgb_head``、``rgb_left_hand``、``rgb_right_hand`` 三路 RGB 图像
  - ``arm_joint_states``、``left_hand_states``、``right_hand_states``、``waist_joints_states`` 等状态向量

- **动作空间**: 28 维连续动作

  - 16 维腰部 + 双臂关节
  - 6 维左手
  - 6 维右手

算法
----

示例配置默认使用 psi-policy 初始化的异步 SAC 微调：

- 三路 RGB 图像输入：``rgb_head``、``rgb_left_hand``、``rgb_right_hand``
- 策略动作维度默认是 ``26``，对应双臂 + 双手
- A2D env 在发送给 controller 前会补齐前 2 个 waist 维度，形成 28 维 controller 动作

安装
----

**Python 环境**

.. code:: bash

   bash requirements/install.sh embodied --env a2d
   source .venv/bin/activate

**Docker 镜像**

.. code:: bash

   export BUILD_TARGET=embodied-a2d
   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .

该镜像基于官方 ``a2d-tele`` 运行时镜像构建，并叠加 RLinf 与 A2D gRPC 客户端依赖。

快速开始
--------

1. 先使用现有部署流程启动官方 A2D Docker。
2. 在容器内启动官方 ``model_inference`` gRPC 服务：

.. code:: bash

   source /ros_entrypoint.sh
   source /opt/psi/rt/a2d-tele/install/setup.bash
   python3 -m model_inference.run_inference_server

3. 准备集群配置，使得：

   - ``env`` 运行在 A2D 工作站节点
   - ``rollout`` 和 ``actor`` 运行在 GPU 节点

4. 参考 ``examples/embodiment/config/realworld_a2d_sac_psi_async.yaml`` 作为起点。

5. 启动训练：

.. code:: bash

   python examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_async

配置说明
--------

A2D hardware 配置如下：

.. code:: yaml

   hardware:
     type: A2D
     configs:
       - node_rank: 1
         controller_host: 127.0.0.1
         grpc_port: 12321
         container_name: a2d-tele-release-2-1-0rc3-latest

环境侧可以通过 override config 修改：

- 图像键名和尺寸
- 状态键名和尺寸
- 动作空间上下界
- ``policy_action_dim``（当前 psi-policy 默认是 26）
- controller 暴露出来的奖励 / 成功状态键

Dummy 验证
----------

如果只想先检查整条训练链路，可使用 dummy 配置：

.. code:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-name ../tests/e2e_tests/embodied/realworld_a2d_dummy_sac_psi
