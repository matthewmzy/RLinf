# A2D 真机强化训练简明说明

这份文档只覆盖你当前这条链路：

- A2D 官方 docker 提供 gRPC rollout server
- RLinf `a2d-dev` 分支做真机 SAC
- `psi-policy` 作为 actor 初始化
- 稀疏奖励由键盘给信号
- `control_mode=1` 作为人工接管，`control_mode=99` 作为空闲帧过滤

## 1. 先准备什么

训练前至少要有下面这些东西：

1. 一套能正常工作的 A2D 真机环境
   - `deploy-a2d-tele-hil` 能把官方容器拉起
   - 机器人、相机、遥操作、踏板都正常
2. A2D gRPC server 已启动
   - 容器内能运行 `python3 -m model_inference.run_inference_server`
   - 默认端口是 `12321`
3. 一份可用的 `psi-policy` 预训练产物
   - checkpoint：`.ckpt`
   - normalizer：`normalizer.pkl`
4. Ray 集群已启动
   - 你当前场景是单机 4090，所以推荐先用单节点 Ray
   - 同一台机器同时承担 actor、rollout、env
5. RLinf 环境已安装
   - 在 `RLinf` 根目录能运行 `python examples/embodiment/train_async.py ...`

## 1.1 需不需要拉 RLinf docker

结论先说：

- 你的当前场景不需要强制再拉一层 RLinf docker
- 更推荐直接在宿主机 Python 环境里跑 RLinf

推荐原因：

1. 你已经有一个 A2D 官方 docker 在跑
2. 训练和推理都在同一台 4090 主机
3. RLinf 还要同时访问：
   - 本机 Ray
   - 本机文件系统里的 checkpoint / normalizer
   - 本机 `127.0.0.1:12321` 的 A2D gRPC
4. 这时候再套一层 RLinf docker，最容易额外引入：
   - host networking 问题
   - GPU 映射问题
   - Ray 进程可见性问题
   - 挂载路径不一致问题

所以默认建议：

- A2D 官方系统继续跑它自己的 docker
- RLinf 直接跑宿主机 `.venv`

宿主机安装方式：

```bash
bash requirements/install.sh embodied --env a2d
source .venv/bin/activate
```

只有在下面两种情况才建议额外使用 RLinf docker：

1. 宿主机依赖环境很乱，已经难以维护
2. 你们后面要把整套 RLinf 训练环境做成可复用镜像

如果你以后确实想用 RLinf docker，建议把它当“可选部署方式”，不要作为第一次真机联调的默认方案。

如果你还没有 `psi-policy` 产物，就先在 `psi-policy` 仓库按你们现有流程完成模仿学习训练。你当前对应的是：

- 训练脚本：`train.py`
- workspace 配置：`psi_policy_workspace.yaml`
- task 配置：`dex_image_state.yaml`

这一步的目标只是拿到 `ckpt + normalizer.pkl`，不是在 RLinf 里重复做 imitation training。

## 1.2 为什么建议先跑 HG-Dagger

如果你现在的首要目标是确认下面这条链路真的通：

1. A2D 真机观测进到了 RLinf rollout
2. 人类接管动作被正确标成 intervention 样本
3. actor 能基于这些样本发生参数更新

那先跑 HG-Dagger 比直接上 SAC 更稳。

原因是：

- HG-Dagger 只验证 `env -> rollout -> replay buffer -> actor` 这条监督更新链路
- 不会把 critic、entropy tuning、demo/replay 混采这些 SAC 变量一起带进来
- 你手动接管几次后，就应该能很快看到 `train/dagger/actor_loss` 和 replay buffer 指标变化

这次仓库里新增的入口是：

- `realworld_a2d_dagger_psi`
  - 推荐入口，等价于 `realworld_a2d_dagger_psi_rtc`
- `realworld_a2d_dagger_psi_rtc`
  - 显式 RTC 配置
- `realworld_a2d_dagger_psi_direct`
  - 不做 RTC 拼接，最适合查单步链路
- `realworld_a2d_dagger_psi_window`
  - 用 window overlap 平滑

其中 `realworld_a2d_dagger_psi` 默认对齐你当前 `policy_config.yaml` 的行为：

- `control.use_rtc: true`
- `policy.imle.execute_step: 4`

## 2. 关键配置改哪里

主配置文件：

- HG-Dagger 验链路：`examples/embodiment/config/realworld_a2d_dagger_psi.yaml`
- SAC 继续优化：`examples/embodiment/config/realworld_a2d_sac_psi.yaml`

模型配置：

- `examples/embodiment/config/model/psi_policy.yaml`

最需要按现场改的项目只有这些：

### 2.1 单机 4090 时，Ray 集群怎么配

如果训练、rollout、推理都在同一台 4090 上，最简单也最稳妥的配置就是单节点。当前仓库里的 A2D `psi-policy` 预设（包括 `realworld_a2d_dagger_psi.yaml` 和 `realworld_a2d_sac_psi.yaml`）都已经按这个思路配好了。

如果你想核对，`cluster` 应该长这样：

```yaml
cluster:
  num_nodes: 1
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
      node_ranks: 0
      hardware:
        type: A2D
        configs:
          - node_rank: 0
            controller_host: "127.0.0.1"
            grpc_port: 12321
            container_name: "a2d-tele-release-2-1-0rc3-latest"
            auto_start_server: false
```

这表示：

- `actor` 在本机
- `rollout` 在本机
- `env` 也在本机
- `env` 通过本机 `127.0.0.1:12321` 去连 A2D 官方 docker 中的 gRPC server

对你现在这个“单台 4090 做训练和推理”的场景，这是最直接的配置。

这里有一个容易写错的点：

- 不要把 `actor/rollout/env` 全都放进同一个带 `hardware: A2D` 的 node group
- 比较稳妥的写法是：
  - `actor/rollout` 放普通 GPU group，比如 `"4090"`
  - `env` 放带 `hardware: A2D` 的 group，比如 `a2d`
  - 这两个 group 可以都指向同一台物理机的 `node_rank: 0`

1. A2D 连接信息
   - `cluster.node_groups[].hardware.configs[].controller_host`
   - `cluster.node_groups[].hardware.configs[].grpc_port`
   - `cluster.node_groups[].hardware.configs[].container_name`

2. `psi-policy` 产物路径
   - `rollout.model.model_path`
   - `actor.model.model_path`
   - `actor.model.normalizer_path`
   - rollout 会复用 `actor.model.normalizer_path`
   - `runner.ckpt_path`

3. 日志实验名
   - `runner.logger.experiment_name`

4. 如果你的 server 输出键名或尺寸和默认不一致，再改：
   - `env.train.override_cfg.image_keys`
   - `env.train.override_cfg.state_keys`
   - `env.train.override_cfg.image_shapes`
   - `env.train.override_cfg.state_shapes`

当前这份 A2D psi 配置已经默认包含这些和你需求强相关的设定：

- `policy_action_dim: 26`
- `clip_policy_actions: False`
- `model_control_modes: [0]`
- `teleop_control_modes: [1]`
- `idle_control_modes: [99]`
- `keyboard_reward_wrapper: single_stage`
- `backup_entropy: False`
- `train_actor_steps: 20`
- `demo_buffer.allow_replay_only_until_ready: True`
- `entropy_tuning.target_entropy: auto`

含义是：

- `psi-policy` 输出的是绝对关节角，不再二次映射
- `0` 是模型控制帧
- `1` 是遥操作接管帧，会作为 intervention 样本进入数据流
- `99` 是空闲帧，不会进入 replay buffer
- 键盘奖励模式是：
  - `f`：失败，reward = -1，当前 episode 立刻 done
  - `s`：成功，reward = +1，当前 episode 立刻 done
  - 成功/失败是整条 episode 的标签，不是“按住这个键的那几帧才算成功/失败”
- 训练刚开始时先用 replay-only 训练 critic
- 至少积累约 20 条轨迹后 actor 才开始更新
- 一旦出现第一批人工接管轨迹，demo buffer 会自动开始参与混采
- 如果你打开 `actor.model.rl_action_mask`，`target_entropy` 会自动按实际参与 RL 的动作维度数调整

### 2.2 三个路径别填混

这里最容易出错的是把 RLinf checkpoint 和 `psi-policy` 原始 checkpoint 混掉。

这三个字段的语义分别是：

1. `actor.model.model_path` / `rollout.model.model_path`
   - 填 `psi-policy` 原始训练产物里的 `.ckpt`
   - RLinf 会直接从这里还原 `psi-policy` 网络和 `shape_meta`
2. `normalizer_path`
   - 填和上面 `.ckpt` 配对的 `normalizer.pkl`
   - 不要跨实验混用
3. `runner.ckpt_path`
   - 这是 RLinf 自己保存出来的 actor `state_dict`
   - 第一次跑 `realworld_a2d_dagger_psi*` 时应该保持 `null`
   - 只有当你要从之前一次 RLinf 训练结果继续接着训，才填这个字段

一句话记：

- 原始 `psi-policy` 初始化，用 `model_path + normalizer_path`
- RLinf 续训，用 `runner.ckpt_path`

### 2.3 动作维度 mask 怎么配

现在 `psi_policy.yaml` 里已经支持一个方便的 `rl_action_mask`：

```yaml
actor:
  model:
    rl_action_mask:
      enabled: False
      active_groups: ["all"]
      active_indices: []
```

它的语义是：

- `enabled: False`
  - 全 26 维动作都参与 RL 探索和学习
- `active_groups`
  - 可选：`all`、`arms`、`hands`、`left_arm`、`right_arm`、`left_hand`、`right_hand`
- `active_indices`
  - 额外补充的 0-based 动作索引

26 维动作顺序是：

- `left_arm[0:7]`
- `right_arm[7:14]`
- `left_hand[14:20]`
- `right_hand[20:26]`

最常见的几种写法：

1. 双臂 + 双手全学

```yaml
actor:
  model:
    rl_action_mask:
      enabled: False
```

2. 只学双臂，不让 RL 探索双手

```yaml
actor:
  model:
    rl_action_mask:
      enabled: True
      active_groups: ["arms"]
      active_indices: []
```

3. 只学几个手关节自由度

```yaml
actor:
  model:
    rl_action_mask:
      enabled: True
      active_groups: []
      active_indices: [14, 15, 20, 21]
```

4. 学双臂，再额外学几个手关节

```yaml
actor:
  model:
    rl_action_mask:
      enabled: True
      active_groups: ["arms"]
      active_indices: [14, 20]
```

当前这版 mask 的具体行为是：

- 被选中的维度：
  - 参与 rollout exploration noise
  - 参与 SAC actor/entropy 这部分训练
- 没被选中的维度：
  - 直接被覆盖为该 episode 的 reset pose
  - 不额外加 RL exploration noise
  - 不直接参与 SAC actor 的策略梯度
  - critic 学习时也会把 replay/demo 里的这些维度重新投影回 reset pose，避免学到 actor 永远发不出的动作分布

这很适合你这种真机场景，因为能明显减少探索自由度。

## 3. 建议的启动顺序

严格按这个顺序：

1. 启动本机 Ray head
2. 启动 A2D 官方容器
3. 启动 A2D gRPC server
4. 检查机器人观测 ready
5. 再启动 RLinf 真机训练

不要反过来。否则 RLinf env worker 会先连 gRPC，然后在 `ready_timeout` 内一直等观测。

### 3.1 单机 4090 的 Ray 启动方式

先清掉旧的 Ray：

```bash
ray stop --force
```

然后在这台 4090 主机上直接启动一个 head：

```bash
ray start --head
```

确认状态：

```bash
ray status
```

你在单机场景下不需要再启动 worker 节点，也不需要第二台机器加入。

## 4. 运行命令

### 4.0 先用 HG-Dagger 验链路

如果你现在是第一次把 `psi-policy` 接进 RLinf，建议先跑：

```bash
bash examples/embodiment/run_realworld_async.sh realworld_a2d_dagger_psi
```

如果你想切执行模式，对应配置名是：

- `realworld_a2d_dagger_psi`
  - 推荐，等价于 `realworld_a2d_dagger_psi_rtc`
- `realworld_a2d_dagger_psi_rtc`
  - 显式 RTC 配置
- `realworld_a2d_dagger_psi_direct`
  - 最容易查 rollout 原始动作
- `realworld_a2d_dagger_psi_window`
  - 更平滑，但排障时不如 direct 直观

这组配置默认已经做了几件适合“先验链路”的事情：

- `algorithm.loss_type: embodied_dagger`
- `algorithm.dagger.only_save_expert: True`
- `runner.ckpt_path: null`
- `rollout.model.noise_std_rollout: 0.0`
- `micro/global batch size = 8`

第一次联调时，建议你先人工接管几次，再看下面这些指标有没有开始变化：

- `train/replay_buffer/num_trajectories`
- `train/replay_buffer/total_samples`
- `train/dagger/actor_loss`

如果这三个指标都动了，说明：

- 人类接管样本进了 replay buffer
- `psi-policy` 的 HG-Dagger batch 组装是通的
- actor 确实在更新

### 4.1 这套 HG-Dagger 配置现在的实际值

如果你跑的是默认推荐入口 `realworld_a2d_dagger_psi`（等价于 `realworld_a2d_dagger_psi_rtc`），当前实际值是：

- `algorithm.loss_type: embodied_dagger`
- `algorithm.dagger.only_save_expert: True`
- `actor.model.num_action_chunks: 4`
- `actor.model.action_dim: 26`
- `rollout.action_execution.mode: rtc`
- `rollout.action_execution.rtc.search_window: 4`
- `rollout.action_execution.rtc.merge_weight_base: 0.8`
- `rollout.model.noise_std_rollout: 0.0`
- `actor.model.noise_std_train: 0.0`
- `actor.micro_batch_size: 8`
- `actor.global_batch_size: 8`
- `actor.optim.lr: 3e-5`
- `algorithm.update_epoch: 1`
- `algorithm.replay_buffer.min_buffer_size: 1`
- `runner.weight_sync_interval: 1`
- `env.train.total_num_envs: 1`
- `env.train.override_cfg.step_frequency: 10.0`
- `env.train.max_steps_per_rollout_epoch: 200`
- `env.train.max_episode_steps: 200`

这里有一个容易混淆的点：

- `examples/embodiment/config/model/psi_policy.yaml` 里的静态默认值是 `action_horizon: 16`
- 但 `RLinf` 在真正加载 `psi-policy` checkpoint 后，会用 checkpoint 自己的 `n_action_steps` 覆盖这个值
- 你当前这套 base preset 指向的 ckpt 路径是：
  - `.../2026.04.18/23.45_zjxc_v0.12_3rgb_joint_headmask_viewtokens_na32/...`
- 这颗 ckpt 对应的 runtime `action_horizon` 应该按 `na32` 走，也就是 `32`

所以对你现在这套 A2D `psi-policy` HG-Dagger 来说，更准确的说法是：

- 静态 yaml 默认值：`16`
- 当前这颗 `na32` checkpoint 的 runtime 实际值：`32`

配置里虽然还有：

- `algorithm.dagger.init_beta: 1.0`
- `algorithm.dagger.beta_schedule: exponential`
- `algorithm.dagger.beta_decay: 0.99`
- `algorithm.dagger.beta_min: 0.05`

但这套 A2D `psi-policy` HG-Dagger 没有配置 `rollout.expert_model`，所以这里不是“beta 混合 teacher policy”的经典 DAgger 路径，而是 **teleop 人类接管 + only_save_expert=True** 的 HG-Dagger 路径。

### 4.2 现在 `psi-policy` 的 HG-Dagger 样本语义

现在的 `psi-policy` HG-Dagger 不是“拿某一次 rollout sampled chunk 直接监督”。

当前实现分两步：

1. rollout / env 先记录 **最终真实执行动作流**
   - 标签源是 `Trajectory.actions`
   - 在 RTC 模式下，这里面已经是 RTC 裁剪 / 平滑之后真正执行的动作
   - 如果中间有人类接管，接管后的动作也会被写回这里

2. actor 收到 raw trajectory 之后，再重组训练样本
   - 只把“当前 chunk 的 4 个子步都由 expert/teleop 执行”的位置当成 anchor
   - 输入是该时刻的观测 `obs_t`
   - 标签不是 raw `model_action`
   - 标签是从 `Trajectory.actions` 往后拼出来的 future executed action window
   - 当前窗口长度固定跟运行时 checkpoint 的 `action_horizon` 走；你现在这颗 `na32` ckpt 对应的是 `32`

也就是说，现在学的是：

- `obs_t -> future 32-step executed action window`

这和 `psi-policy` 原本的 IL 训练语义是对齐的。

### 4.3 启动 HG-Dagger 之后，异步链路里先后发生什么

按默认配置 `realworld_a2d_dagger_psi`，细粒度顺序是：

1. `run_realworld_async.sh` 调 `train_async.py`
2. Hydra 组合出 `realworld_a2d_dagger_psi_rtc + realworld_a2d_dagger_psi_base`
3. 因为 `algorithm.loss_type=embodied_dagger`，runner 会选择：
   - `AsyncEmbodiedRunner`
   - `AsyncEmbodiedDAGGERFSDPPolicy`
   - `AsyncMultiStepRolloutWorker`
   - `AsyncEnvWorker`
4. actor 加载原始 `psi-policy` ckpt 和 normalizer，创建 replay buffer
5. rollout 加载自己的 `psi-policy` 副本，并创建 RTC action execution adapter
6. runner 先做一次 actor -> rollout 权重同步；之后每 `weight_sync_interval=1` 个训练 step 再同步一次
7. env 把当前观测发给 rollout
8. rollout 用 `psi-policy` 预测完整 horizon，但只取 RTC 处理后的 `num_action_chunks=4` 步真实执行动作发回 env
9. env 真正执行这 4 步动作，得到：
   - `rewards`
   - `dones`
   - `transition_valids`
   - 以及可能存在的 `intervene_action / intervene_flag`
10. 如果有人类接管，env 会把最后真实执行动作写回 `Trajectory.actions`
11. env 累积整段 raw trajectory，并在 rollout epoch 结束后发给 actor
12. 对 `psi-policy`，actor 不再直接做 action-level `extract_intervene_traj(mode="all")`
13. actor 会先把 raw trajectory 重组成：
   - `obs_t -> future 32-step executed action window`
14. 这些 windowized 样本进入 replay buffer
15. 只要 buffer 达到 `min_buffer_size=1`，actor 就开始训练
16. 每次训练用：
   - `global_batch_size=8`
   - `micro_batch_size=8`
   - `update_epoch=1`
   - `lr=3e-5`
17. learner 前向走 `ForwardType.SFT`，损失直接复用 `psi-policy` 原生 IMLE loss
18. actor 更新完权重后，runner 再把新权重同步给 rollout
19. rollout 用新权重继续在线跑，形成异步闭环

### 4.4 启动训练

在 `RLinf` 根目录：

```bash
bash examples/embodiment/run_realworld_async.sh realworld_a2d_sac_psi
```

如果你想直接看清楚底层入口，也可以不用包装脚本，直接运行：

```bash
python examples/embodiment/train_async.py \
  --config-path examples/embodiment/config \
  --config-name realworld_a2d_sac_psi
```

这条命令的含义：

- `run_realworld_async.sh`
  - A2D 真机异步训练启动脚本
  - 内部调用 `train_async.py`
- `train_async.py`
  - RLinf 具身异步训练主入口
- `--config-path examples/embodiment/config`
  - 使用仓库里的具身配置目录
- `--config-name realworld_a2d_sac_psi`
  - 使用 A2D 真机 + 异步 SAC + psi-policy 这份配置

如果你想自动生成带时间戳的日志目录，也可以直接用脚本：

```bash
bash examples/embodiment/run_realworld_async.sh realworld_a2d_sac_psi
```

这条脚本实际做的事情是：

- 调用同一个 `train_async.py`
- 自动把 `runner.logger.log_path` 改成 `logs/<时间>-realworld_a2d_sac_psi`
- 把终端输出同时保存到 `run_embodiment.log`

如果你是单机 4090 场景，推荐完整命令顺序就是：

```bash
cd /path/to/RLinf
source .venv/bin/activate
ray stop --force
ray start --head
bash examples/embodiment/run_realworld_async.sh realworld_a2d_sac_psi
```

### 4.2 只验证链路，不接真机

```bash
bash tests/e2e_tests/embodied/run_async.sh realworld_a2d_dummy_sac_psi
```

或者直接调异步主程序：

```bash
python examples/embodiment/train_async.py \
  --config-path tests/e2e_tests/embodied \
  --config-name realworld_a2d_dummy_sac_psi
```

这一步只验证 RLinf 训练链路，不验证 A2D 真机。

## 5. 实际操作流程

建议一轮 episode 这么做：

1. 启动训练后，先确认 env 正常收到了 A2D 观测
2. 让踏板/控制模式保持在 `0`
   - 这时是模型控制
3. 需要人工接管时，把模式切到 `1`
   - 当前机器人状态会被当作 `intervene_action`
   - 这部分样本会走 intervention 逻辑
4. 进入准备或切换阶段时，模式会到 `99`
   - 这部分帧不会进 replay
5. 一段任务结束后，用键盘给稀疏奖励
   - `s` 成功
   - `f` 失败
6. 自动 reset 后开始下一轮

你的目标轨迹在 RL 里会由两部分组成：

- 模型控制帧
- 遥操作接管帧

空闲帧已经被过滤掉，不会污染 replay buffer。

## 5.1 单机 4090 的实际部署形态

你当前最推荐的实际形态是：

1. 宿主机
   - 跑 RLinf
   - 跑 Ray head
   - 保存日志、checkpoint、tensorboard
2. A2D 官方 docker
   - 跑 ROS2
   - 跑相机和控制
   - 跑 `model_inference` gRPC server
3. RLinf 通过 `127.0.0.1:12321` 访问 A2D gRPC

也就是：

- 一个宿主机 Python 训练进程
- 一个官方 A2D docker
- 不额外再套 RLinf docker

## 6. 结果保存到哪里

如果你直接跑 `python examples/embodiment/train_async.py ...`，默认配置里的日志目录是：

- `../results`

如果你跑 `bash examples/embodiment/run_realworld_async.sh realworld_a2d_sac_psi`，日志目录会是：

- `logs/<时间>-realworld_a2d_sac_psi`

重点看这些位置：

1. 主日志
   - `logs/.../run_embodiment.log`

2. TensorBoard
   - `logs/.../tensorboard/`

3. 视频
   - `logs/.../video/train/`
   - 前提是配置里打开 `video_cfg.save_video`

4. replay buffer
  - 当前配置 `auto_save: False`
  - 默认不会自动落盘
  - 如果打开 `algorithm.replay_buffer.auto_save: True`，默认目录是：
    - `runner.logger.log_path/replay_buffer/rank_*`

5. demo buffer
   - 当前 A2D psi 配置已经启用了 `demo_buffer`
   - 但它只有在出现人工接管轨迹后才会真正写入内容
   - 默认目录是：
     - `runner.logger.log_path/demo_buffer/rank_*`

6. checkpoint
   - 当前默认还是 `save_interval: -1`
   - 也就是训练过程不定期自动存档，避免真机大图像 replay buffer 频繁落盘拖慢训练
   - 如果你确实需要周期性保存，可以手动把 `runner.save_interval` 改成一个较大的值，比如 `20` 或 `50`

## 7. 怎么做最小验证

最小验证建议按下面几步：

1. 连通性验证
   - RLinf 启动后不应该报 gRPC timeout
   - 第一帧 reset 应该成功

2. 观测验证
   - 模型开始 rollout 后不应出现缺少 `rgb_head`、`rgb_left_hand`、`rgb_right_hand`、`arm_joint_states` 之类的 key 错误

3. 模式验证
   - `control_mode=0` 时正常执行模型动作
   - `control_mode=1` 时人工接管
   - `control_mode=99` 时不应把空闲帧持续塞进 replay

4. 奖励验证
   - 按 `a/c` 后 episode 应结束
   - TensorBoard 里能看到 `env` 和 `replay_buffer` 相关指标变化

5. HIL 验证
   - 在一条轨迹中途切到遥操作，再切回模型
   - 训练不应中断
   - replay buffer 大小继续增长

## 8. 常见问题

### 8.1 一启动就卡住

通常是下面几种原因：

- A2D gRPC server 没启动
- `controller_host` / `grpc_port` 配错
- A2D server 还没 ready，图像或状态 topic 没齐
- 单机配置还保留着 `num_nodes: 2`
- `node_rank` 还在引用不存在的第二台机器

### 8.1.1 单机时最容易忘的一点

如果你只有一台 4090 主机，但配置里还是这种双节点结构：

- `cluster.num_nodes: 2`
- `actor/rollout` 放 `node_rank: 0`
- `env` 放 `node_rank: 1`

那训练一定起不来。

因为对 Ray 来说，你根本没有第二个节点可放 `env worker`。

所以单机场景一定要把 `cluster` 改成单节点版本。

### 8.2 动作看起来明显不对

优先检查：

- `model_path` 和 `normalizer_path` 是否配对
- 现在这份配置必须保持：
  - `policy_action_dim: 26`
  - `clip_policy_actions: False`

### 8.3 遥操作接管没有生效

优先检查：

- A2D 是否真的在往外发 `control_mode`
- 当前踏板配置是不是：
  - `0 -> 模型`
  - `1 -> 遥操作`
  - `99 -> IDLE`

### 8.4 想查权威观测/动作格式

直接看这两份文档：

- `A2D_HIL_IMPLEMENTATION_KB.md`
- `A2D_IMAGE_SERVER_KB.md`（如果你本地没带这份文档，去 A2D 侧知识库找同名文件）
