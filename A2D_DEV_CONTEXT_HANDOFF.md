# A2D / RLinf 开发接力上下文

最后更新：2026-03-30

这份文档用于下次新开对话时快速接住当前开发，不再重复做背景调研。

## 当前目标

- 把 A2D 官方 `a2d-tele` Docker 封装成 RLinf 里的真机设备
- 用 `psi-policy` 的模仿学习策略作为 SAC actor 初始化，在真机上微调
- 支持 inference-time human-in-the-loop
- 人工接管帧和模型推理帧一起组成连续轨迹进入 replay / demo 数据流
- 奖励使用键盘给的稀疏成功/失败信号

用户当前真实场景：

- 训练和推理在同一台 4090 上
- A2D 官方系统继续跑它自己的 Docker
- RLinf 更推荐直接在宿主机 Python 环境里运行，不强制再套一层 RLinf Docker

## 目录与仓库

工作目录：

- `/Users/matthew/Documents/a2d`

关键仓库/目录：

- `RLinf`
- `psi-policy`
- `a2d-tele`
- `src`
- `install`
- `deploy-a2d-tele-hil`

当前 git 状态：

- `RLinf`
  - 分支：`a2d-dev`
  - 最新提交：`15c1593`
  - 已推到远端 `origin/a2d-dev`
- `psi-policy`
  - 分支：`yuanpei_merge`
  - 当前工作区干净
  - 这轮开发没有需要提交的重要改动
- `a2d-tele`
  - 本地工作区非常脏
  - 本轮没有基于它做提交

## 已确认的权威事实

权威来源主要来自：

- `src/model_inference/model_inference/envs/grpc_server.py`
- `src/model_inference/model_inference/utils/obs_manager.py`
- `src/model_inference/model_inference/configs/data_source/base_data.yaml`
- `src/pedal_controller/config/*.yaml`
- `psi-policy/model_inference_client/envs/base_env.py`

### A2D gRPC 观测契约

gRPC 观测对象里有三类信息：

- `Observation.images`
- `Observation.states`
- 顶层字段
  - `control_mode`
  - `trajectory_label`
  - `is_switch_mode`

重要点：

- `control_mode` 是 protobuf 顶层字段，不是 `states` 里的一个键

### 默认 A2D 观测布局

图像键：

- `rgb_head`
- `rgb_left_hand`
- `rgb_right_hand`

状态键：

- `arm_joint_states`: 14D
- `left_hand_states`: 6D
- `right_hand_states`: 6D
- `waist_joints_states`: 2D

`ObsManager` 还会派生：

- `left_arm_states = arm_joint_states[:7]`
- `right_arm_states = arm_joint_states[7:14]`

所以权威双臂顺序是：

- `left_arm[0:7] + right_arm[7:14]`

### 默认动作布局

上游默认 28 维动作：

- `[0:2]` waist
- `[2:9]` left arm
- `[9:16]` right arm
- `[16:22]` left hand
- `[22:28]` right hand

当前 RLinf `psi-policy` 接入使用 26 维动作：

- `arm_joint_states[14] + left_hand_states[6] + right_hand_states[6]`
- waist 不进 policy action

### control_mode 语义

已确认并在 RLinf 中采用：

- `0 = model`
- `1 = teleop`
- `99 = idle`

当前语义：

- `0`
  - 模型控制帧
  - 正常进入 SAC rollout / replay
- `1`
  - 人工接管帧
  - 当前机器人 joint state 会作为 `intervene_action`
  - 可以进入 HIL / demo 数据流
- `99`
  - 准备/空闲帧
  - 通过 `data_valid=False` 被过滤，不进 replay

## 这轮已经在 RLinf 做完的工作

核心实现文件：

- `rlinf/envs/realworld/a2d/a2d_env.py`
- `rlinf/envs/realworld/realworld_env.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/data/embodied_io_struct.py`
- `rlinf/data/embodied_buffer_dataset.py`
- `rlinf/models/embodiment/psi_policy/psi_policy_for_rl.py`
- `rlinf/workers/actor/fsdp_sac_policy_worker.py`

已具备的行为：

- 从 `control_mode` 提取 intervention 语义
- `teleop` 帧生成 `info["intervene_action"]`
- `idle` 帧标记 `data_valid=False`
- transition validity 被带入轨迹
- replay buffer 只接收 valid transition
- demo buffer 在没有第一条人工接管轨迹前可以先 replay-only 训练
- 一旦 demo ready，自动混采 replay + demo

### psi-policy 接入 SAC

当前配置与实现已经确认：

- `policy_action_dim: 26`
- `normalize_actions: False`
- `clip_policy_actions: False`

原因：

- A2D 接收的是绝对关节角
- 不应再做 controller-space 的二次归一化或 clip

### RL 动作 mask

当前支持 `rl_action_mask`：

- `enabled`
- `active_groups`
  - `all`
  - `arms`
  - `hands`
  - `left_arm`
  - `right_arm`
  - `left_hand`
  - `right_hand`
- `active_indices`

动作顺序固定为：

- `left_arm[0:7]`
- `right_arm[7:14]`
- `left_hand[14:20]`
- `right_hand[20:26]`

重要语义：

- 未被选中的动作维度保持该 episode 的 `reset pose`
- rollout 时如此
- SAC actor 采样时如此
- critic 学 replay/demo 时也会把这些维度重新投影回 `reset pose`

这样做是为了避免 actor/critic 看到不一致的动作分布。

### 自动 target entropy

当前已支持：

- `target_entropy: auto`
- 实际使用 `-num_active_rl_action_dims`

### 键盘稀疏奖励

当前真机配置走键盘 sparse reward：

- `a = fail, reward -1, done=True`
- `b = neutral, reward 0`
- `c = success, reward +1, done=True`

## 文档与产物

已新增/整理：

- `A2D_HIL_IMPLEMENTATION_KB.md`
- `A2D_REALWORLD_RL_QUICKSTART.md`
- `A2D_REALWORLD_RL_FULL_FLOW.png`
- `A2D_REALWORLD_RL_FULL_FLOW.svg`

下次新对话优先看：

1. `A2D_REALWORLD_RL_QUICKSTART.md`
2. `A2D_HIL_IMPLEMENTATION_KB.md`
3. `rlinf/models/embodiment/psi_policy/psi_policy_for_rl.py`
4. `examples/embodiment/config/realworld_a2d_sac_psi.yaml`

## 当前推荐运行方式

### RLinf 是否需要 Docker

当前用户场景下，不推荐把 RLinf 再套一层 Docker。

原因：

- A2D 官方链路已经在官方 Docker 里
- 训练和推理都在同一台 4090
- RLinf 直接跑宿主机更方便访问：
  - 本机 Ray
  - 本机 checkpoint / normalizer
  - 本机 `127.0.0.1:12321` 的 A2D gRPC server

### 单机 4090 的 Ray 配置

当前推荐单节点：

- `actor` 在本机
- `rollout` 在本机
- `env` 也在本机
- `env` 通过 `127.0.0.1:12321` 连 A2D 官方 Docker 中的 gRPC server

对应主配置：

- `examples/embodiment/config/realworld_a2d_sac_psi.yaml`

### 环境安装

RLinf 正式环境不要用系统 `python3.13`。

原因：

- `pyproject.toml` 要求 Python `>=3.10, <=3.11.14`
- 当前系统 `python3` 虽然装了 `pytest`
- 但缺 `torch/numpy/omegaconf`

推荐命令：

```bash
cd /Users/matthew/Documents/a2d/RLinf
bash requirements/install.sh embodied --env a2d
source .venv/bin/activate
```

## 当前验证状态

已完成：

- 关键 Python 文件 `compileall` 通过
- `RLinf` 改动已经提交并推送
- `psi-policy` 当前没有需要提交的重要改动

未完成：

- 还没在 RLinf 正式 `.venv` 里跑完整单测
- 当前系统 `python3` 下相关测试会因缺依赖被跳过

当前新增/更新测试：

- `tests/unit_tests/test_a2d_integration.py`
- `tests/unit_tests/test_embodied_buffer_dataset.py`
- `tests/unit_tests/test_psi_policy_action_mask.py`

## 很重要的实现语义

### demo buffer “先 replay-only，再自动混采”

这不代表“人工干预前模型不能自己 rollout 学习”。

真实语义是：

- 人工干预前
  - replay buffer 已经可以积累模型自己 rollout 的数据
  - critic 可以先从 replay 学
  - 只是 demo buffer 还没数据，所以暂时不混 demo
- 一旦出现第一批人工接管轨迹
  - demo buffer ready
  - 自动切换成 replay + demo 混采

### mask 后冻结动作为什么选 reset pose

这是用户明确要求的语义：

- 被 mask 的动作维度应保持 `reset pose`

不是保持当前 policy 输出，也不是保持当前时刻观测 state。

当前实现：

- `RealWorldEnv` 把 `reset_states` 注入 obs
- `PsiPolicyForRL` 在 rollout / SAC actor 采样时，把冻结维度覆盖为 `reset pose`
- `SAC critic` 训练时，也会把 batch action 的冻结维度重新投影为 `reset pose`

## 下次新对话建议

建议按这个顺序接：

1. 先读这份文档
2. 再读 `A2D_REALWORLD_RL_QUICKSTART.md`
3. 再读 `A2D_HIL_IMPLEMENTATION_KB.md`
4. 再看 `a2d-dev` 上的关键实现文件
5. 如果要跑测试，先搭好 `RLinf/.venv`
6. 不要默认改 `psi-policy`
   - 除非用户明确提出要改网络本体

## 一句话总结

当前已经把 A2D 官方 Docker 的 gRPC 观测/动作链路、`control_mode` 的 HIL 语义、`psi-policy` 作为 SAC actor、稀疏键盘奖励、动作维度 mask、以及 quickstart 文档都接到了 `RLinf/a2d-dev`，而且这轮真正需要推送的只有 `RLinf`，不是 `psi-policy`。
