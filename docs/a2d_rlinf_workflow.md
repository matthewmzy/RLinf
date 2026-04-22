# A2D 接入 RLinf 工作流

本文档说明如何把官方 `a2d-tele` Docker 当作一个持续运行的 controller，让 RLinf 像支持 Franka 一样支持 A2D。

## 方案概览

当前实现采用下面的边界划分：

- `a2d-tele` 官方容器负责 ROS2、相机订阅、机器人控制与 `model_inference` gRPC 服务。
- RLinf 的 `EnvWorker` 通过 `A2DController` 访问该 gRPC 服务。
- RLinf 的 `RolloutWorker` 和 `ActorWorker` 保持原有职责，只处理策略推理与训练。

因此，A2D 在 RLinf 中走的是 `realworld env + external controller` 路线，而不是自定义 rollout worker。

## 1. 构建 RLinf A2D 镜像

在 RLinf 根目录执行：

```bash
export BUILD_TARGET=embodied-a2d
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

这个镜像会：

- 以官方 `docker-registry.psibot.net/synrobot/a2d-tele/x86/release/2.1.0rc3:latest` 为基础镜像
- 叠加 RLinf 代码与 A2D 所需 Python 依赖
- 默认激活 `/opt/venv/a2d`

## 2. 安装本地 Python 环境

如果不用 Docker，也可以直接在控制节点或开发机安装：

```bash
bash requirements/install.sh embodied --env a2d
source .venv/bin/activate
```

这个安装路径只会补充 RLinf 和 A2D gRPC 客户端依赖，不会尝试重新编译 `a2d-tele`。

## 3. 启动官方 A2D controller 容器

推荐继续使用已有的部署方案：

```bash
cd deploy-a2d-tele-hil
python3 deploy.py --config config/deploy-a2d-tele-2.0.0rc1.yaml
```

部署脚本会启动官方容器，并拉起 GUI。

如果要让 RLinf 直接连接该容器中的 `model_inference` gRPC 服务，需要确保容器里已经启动：

```bash
source /ros_entrypoint.sh
source /opt/psi/rt/a2d-tele/install/setup.bash
python3 -m model_inference.run_inference_server
```

默认监听端口是 `12321`。

如果你已经有外部配置文件，可以在容器内设置：

```bash
export MODEL_INFERENCE_CONFIG_FILE=/path/to/config.yaml
python3 -m model_inference.run_inference_server
```

## 4. 配置 RLinf 集群

推荐从新的显式 preset 开始，而不是继续直接改旧 alias：

- `realworld_a2d_sac_psi_direct_async`：推荐的真机 RL 起点
- `realworld_a2d_sac_psi_rtc_repro`：psi-policy RTC 复现
- `realworld_a2d_sac_psi_rtc_async`：RTC + 真机 RL 噪声 + safe joint box
- `realworld_a2d_sac_psi_direct_repro`：不带 RTC 的 direct chunk 复现
- `realworld_a2d_sac_psi_window_repro` / `realworld_a2d_sac_psi_window_async`：window blend 平滑实验

旧入口 `realworld_a2d_sac_psi_async`、`realworld_a2d_sac_psi_async_repro`、`realworld_a2d_sac_psi_safe_explore`、`realworld_a2d_sac_psi` 仍然保留为兼容 alias。

最关键的是 `node_groups` 里的 `A2D` hardware：

```yaml
cluster:
  num_nodes: 2
  component_placement:
    actor:
      node_group: "4090"
      placement: 0
    rollout:
      node_group: "4090"
      placement: 0
    env:
      node_group: a2d
      placement: 0
  node_groups:
    - label: "4090"
      node_ranks: 0
    - label: a2d
      node_ranks: 1
      hardware:
        type: A2D
        configs:
          - node_rank: 1
            controller_host: 127.0.0.1
            grpc_port: 12321
            container_name: a2d-tele-release-2-1-0rc3-latest
```

含义是：

- `actor` 和 `rollout` 在 GPU 训练节点
- `env` 在 A2D 工作站节点
- `env` 通过 `controller_host:grpc_port` 访问正在运行的 A2D controller 服务

## 5. 启动训练

在 Ray 集群就绪后，于 head 节点执行：

```bash
python examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_direct_async
```

如果只想验证配置链路，可先使用 dummy 配置：

```bash
python examples/embodiment/train_embodied_agent.py --config-name ../tests/e2e_tests/embodied/realworld_a2d_dummy_sac_psi
```

## 6. 当前观测与动作约定

默认观测：

- `rgb_head`
- `rgb_left_hand`
- `rgb_right_hand`
- `arm_joint_states`
- `left_hand_states`
- `right_hand_states`
- `waist_joints_states`

当前 psi-policy 配置默认输出 26 维动作：

- 0-13: 双臂关节
- 14-19: 左手
- 20-25: 右手

A2D env 会在发送给 controller 前补上前 2 个 waist 维度，形成 controller 需要的 28 维动作。

RLinf 侧现在只保留 absolute-action 语义：

- 策略输出直接是绝对关节目标
- `env.*.override_cfg.safe_box.low/high` 定义 policy-space 26 维 joint box
- `safe_box.enabled=true` 时自动启用 joint clipping
- 旧 `action_low/action_high/clip_policy_actions` 仍然兼容，但内部会统一折叠到 `safe_box`
- 不再支持 `[-1, 1] -> controller range` 的环境内二次缩放

## 6.1 rollout action execution 配置

psi-policy 的“完整 horizon 预测”和“真正执行哪几步动作”现在已经解耦。实验切换主要改：

```yaml
actor:
  model:
    num_action_chunks: 4   # RL macro step / env 每次 chunk_step 真正执行 4 步

rollout:
  action_execution:
    mode: direct | rtc | window_blend
    noise_stage: pre_smooth
    reset_on_episode_end: true
    rtc:
      search_window: 4
      merge_weight_base: 0.8
    window:
      window_size: 8
      overlap_steps: 4
```

## 7. 常见修改点

如果要接入你们自己的 `model_inference` 配置，优先修改这些字段：

- `env.train.override_cfg.image_keys`
- `env.train.override_cfg.state_keys`
- `env.train.override_cfg.image_shapes`
- `env.train.override_cfg.state_shapes`
- `env.train.override_cfg.reward_state_key`
- `env.train.override_cfg.success_state_key`
- `cluster.node_groups[].hardware.configs[].grpc_config_file`

## 8. 当前限制

- 当前实现默认通过 gRPC 访问官方容器，不直接 import 容器内 ROS 包。
- `reward` / `success` 需要由 `model_inference` 暴露相应状态键，或由你后续在 A2D 侧补充。
- `SpacemouseIntervention`、`RelativeFrame`、`Quat2EulerWrapper` 已对 A2D 禁用，因为 A2D 不是 Franka 风格的 6/7 维 TCP 控制接口。
