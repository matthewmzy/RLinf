# A2D `psi-policy` HG-Dagger 动作流字符图说明

这份文档只讲你当前这条链路：

- `bash examples/embodiment/run_realworld_async.sh realworld_a2d_dagger_psi`
- `realworld_a2d_dagger_psi -> realworld_a2d_dagger_psi_rtc`
- A2D 真机
- 异步 rollout / 异步 actor 学习器
- `psi-policy`
- HG-Dagger

除必要的配置键、代码路径、张量名和接口名外，本文尽量全部用中文描述。

本文重点回答两个问题：

1. 一次 rollout 中间切到遥操作，再切回模型，这些动作到底怎么一步步流动？
2. 一次 rollout 中间切到遥操作，不切回去，直到成功按 `s`，这些动作又怎么一步步流动？

同时，我只写**当前配置里能确定的实际值**。不确定的地方我会明确标出来。

---

## 0. 当前链路的实际超参数

下面这些值来自当前 `realworld_a2d_dagger_psi*` 配置和当前 preset 指向的 ckpt 路径。

### 0.1 RLinf 配置里的确定值

- `algorithm.loss_type = embodied_dagger`
- `algorithm.dagger.only_save_expert = True`
- `algorithm.rollout_epoch = 1`
- `algorithm.update_epoch = 1`
- `algorithm.replay_buffer.min_buffer_size = 1`
- `algorithm.replay_buffer.cache_size = 2000`
- `algorithm.replay_buffer.sample_window_size = 2000`

- `actor.micro_batch_size = 8`
- `actor.global_batch_size = 8`
- `actor.optim.lr = 3e-5`
- `actor.optim.clip_grad = 10.0`
- `actor.sync_weight_no_wait = true`
- `actor.recv_drain_max_trajectories = 32`

- `actor.model.action_dim = 26`
- `actor.model.num_action_chunks = 4`
- `actor.model.add_q_head = true`
- `actor.model.use_chunk_rl = true`
- `actor.model.noise_std_train = 0.0`
- `rollout.model.noise_std_rollout = 0.0`

- `rollout.action_execution.mode = rtc`
- `rollout.action_execution.noise_stage = pre_smooth`
- `rollout.action_execution.reset_on_episode_end = true`
- `rollout.action_execution.rtc.search_window = 4`
- `rollout.action_execution.rtc.merge_weight_base = 0.8`

- `runner.weight_sync_interval = 1`
- `runner.ckpt_path = null`

- `env.train.total_num_envs = 1`
- `env.train.max_steps_per_rollout_epoch = 200`
- `env.train.max_episode_steps = 200`
- `env.train.override_cfg.step_frequency = 10.0`
- `env.train.override_cfg.model_control_modes = [0]`
- `env.train.override_cfg.teleop_control_modes = [1]`
- `env.train.override_cfg.idle_control_modes = [99]`
- `env.train.keyboard_reward_wrapper = single_stage`

### 0.2 `action_horizon` 这次按什么算

这里最容易混：

- `examples/embodiment/config/model/psi_policy.yaml` 的静态默认值是 `action_horizon = 16`
- 但 `RLinf` 真正加载 `psi-policy` checkpoint 之后，会用 checkpoint 自己的 `n_action_steps` 覆盖它

你当前 preset 指向的 ckpt 路径是：

- `.../2026.04.18/23.45_zjxc_v0.12_3rgb_joint_headmask_viewtokens_na32/...`

所以本文按**当前这颗 ckpt 的 runtime 实际值**来讲：

- `action_horizon = 32`

如果你后面把 ckpt 换成 `na16 / na48 / na64`，下面所有 “32 步窗口” 都要跟着变。

### 0.3 把时间尺度先换算成人能直觉理解的量

因为：

- `step_frequency = 10.0 Hz`
- 所以 1 个物理 action step = `0.1 s`

又因为：

- `num_action_chunks = 4`
- 所以 1 个 RLinf chunk-step = `4` 个物理 step = `0.4 s`

又因为：

- `action_horizon = 32`
- 所以 1 个 HG-Dagger 监督窗口长度 = `32` 个物理 step = `8` 个 chunk-step = `3.2 s`

再因为：

- `max_steps_per_rollout_epoch = 200`
- 所以 1 个 rollout epoch 最多跑 `200 / 4 = 50` 个 chunk-step = `20 s`

---

## 1. 先定义本文的符号

### 1.1 张量维度

当前单环境训练，所以：

- `B = 1`
- `A = 26` (`action_dim`)
- `C = 4` (`num_action_chunks`, 也是 `execute_step`)
- `H = 32` (`action_horizon`)

常见张量形状：

- rollout 预测完整 horizon：
  - `predicted_actions`: `[B, H, A] = [1, 32, 26]`
- rollout 真正发给 env 的 chunk：
  - `executed_actions`: `[B, C, A] = [1, 4, 26]`
- env 按 chunk 返回的人类接管标记：
  - `intervene_flag`: `[B, C] = [1, 4]`
- env 按 chunk 返回的数据有效标记：
  - `data_valid`: `[B, C] = [1, 4]`
- actor 重组后的 DAgger 标签：
  - `future action window`: `[B, H, A] = [1, 32, 26]`

### 1.2 字符图里怎么记动作

- `M(k,j)`：第 `k` 个 chunk 的第 `j` 个子步，**模型 / RTC 最终给 env 的动作**
- `H(k,j)`：第 `k` 个 chunk 的第 `j` 个子步，**人类遥操作实际执行的动作**
- `X(k,j)`：第 `k` 个 chunk 的第 `j` 个子步，**无效 / 未实际执行 / padding**

其中：

- `k` 是 chunk-step 编号
- `j in {0,1,2,3}` 是 chunk 内 4 个物理 step

### 1.3 控制模式

当前 A2D 配置里：

- `0 = model`
- `1 = teleop`（遥操作）
- `99 = idle`

当前 reward wrapper：

- `s = success, reward = +1, terminated = True`
- `f = fail, reward = -1, terminated = True`

也就是说，**你现在确实是 `s / f`，不是 `c / a`。**

---

## 2. 不管你有没有切到遥操作，固定都会发生的链路

先把所有 case 都共享的主链路画出来。

```text
env(obs_t)
  |
  |  obs payload, B=1:
  |  - main_images      [1, 720, 1080, 3]
  |  - extra_view_images[1, 2, 720, 1080, 3]
  |  - states           [1, 28]
  |  - reset_states     [1, 28]
  v
rollout psi-policy
  |
  |  predict_action_horizon_batch(...)
  |  -> predicted_actions [1, 32, 26]
  |
  |  RTC adapter:
  |  - search_window    = 4
  |  - merge_weight_base= 0.8
  |  - execute_step     = 4
  v
executed_actions [1, 4, 26]
  |
  |  flatten into forward_inputs["action"] [1, 104]
  |  forward_inputs carries obs too
  v
env.chunk_step(...)
  |
  |  executes 4 physical steps at 10 Hz
  |  => 0.4 s per chunk-step
  |
  |  per-substep returns:
  |  - reward
  |  - terminated / truncated
  |  - control_mode (0/1/99)
  |  - `intervene_action`（如果当前是遥操作）
  |  - data_valid
  v
chunk result
  |
  |  rewards          [1, 4]
  |  intervene_action [1, 104]
  |  intervene_flag   [1, 4]
  |  data_valid       [1, 4]
  v
env worker rollout buffer（原始轨迹构建中）
  |
  |  上一个 chunk 会被回填成“真实执行动作”:
  |  - `intervene_flag=1` 的位置，用遥操作动作覆盖模型动作
  |  - 已存的 `model_action` 会被丢掉
  v
原始 Trajectory
  |
  |  关键字段:
  |  - actions           [T, 1, 104]   <- 最终真实执行的标准 chunk 动作流
  |  - intervene_flags   [T, 1, 104]
  |  - transition_valids [T, 1, 4]
  |  - rewards           [T, 1, 4]
  |  - forward_inputs(观测等)
  v
异步 actor worker
  |
  |  `psi-policy` 钩子:
  |  prepare_dagger_replay_trajectories(raw_trajectory)
  |
  |  只保留满足下面条件的锚点:
  |  这个 chunk 上 `intervene_flags` 在 4*26 维度上全为 True
  |
  |  标签来源:
  |  只用 `Trajectory.actions`
  |
  |  构造监督目标:
  |  `obs_t ->` 后续 32 个有效真实执行动作
  v
回放样本
  |
  |  forward_inputs["action"] [1, 32, 26]
  |  1 个有效锚点 = 1 条回放样本
  v
回放缓冲区
  |
  |  min_buffer_size = 1
  v
actor 学习器
  |
  |  训练批次:
  |  - global_batch_size = 8
  |  - micro_batch_size  = 8
  |  - update_epoch      = 1
  |  - lr                = 3e-5
  |
  |  forward_type = SFT
  |  loss         = psi-policy native IMLE loss
  v
new actor weights
  |
  |  runner.weight_sync_interval = 1
  v
rollout uses new weights in next loop
```

---

## 3. RTC 这一步到底怎么处理动作

当前是 `rtc` 模式，不是 direct。

所以 rollout **不是** 直接把 `predicted_actions[:, :4]` 原封不动送给 env，然后就结束。

当前 RTC 逻辑是：

```text
在第 `k` 个 chunk:

模型先预测:
  P_k[0:31]  # 32 actions, each dim=26

RTC 内部状态保存:
  old_queue      # 上一个未来队列，已经弹掉旧的 4 步
  last_action    # 上一个 chunk 里最后真实执行的动作

RTC 合并过程:
  1. 只在前 4 个新预测里找对齐点
     `search_window = 4`

  2. 选出离 `last_action` 最近的位置
     start_index = argmin ||P_k[i] - last_action||, i in {0,1,2,3}

  3. 保留 `P_k[start_index:]`

  4. 把 `old_queue` 和新的前缀做加权融合:
     weight(i) = 0.8 / (i^2 + 1)

     i=0 -> 0.80
     i=1 -> 0.40
     i=2 -> 0.16
     i=3 -> 0.08
     ...

  5. 从融合后的队列里弹出前 4 个动作
     => executed_actions = E_k[0:3]
```

所以你在 HG-Dagger 里最终学到的不是 raw `P_k`，而是：

- RTC 之后真实执行出去的 `E_k`
- 再叠加遥操作替换之后的最终动作

---

## 4. 情况 A：中途切到遥操作，再切回模型

这里我分两个子情况讲，因为它们在当前实现里结论完全不同。

---

## 4A. 子情况 A1：你是在一个 4 步 chunk 的中间切到遥操作，然后很快切回

这是最容易“你以为采到了纠错样本，但实际上一个都没进 replay”的情况。

### 4A.1 假设时序

假设：

- chunk `k=10`：纯模型
- chunk `k=11`：前两个子步还是模型，后两个子步切成遥操作
- chunk `k=12`：又切回模型

字符图：

```text
chunk k=10
  control_mode: [0, 0, 0, 0]
  source      : [M, M, M, M]

chunk k=11
  control_mode: [0, 0, 1, 1]
  source      : [M, M, H, H]

chunk k=12
  control_mode: [0, 0, 0, 0]
  source      : [M, M, M, M]
```

### 4A.2 这 3 个 chunk 在环境里会变成什么

在 `RealWorldEnv.chunk_step()` 之后：

```text
chunk k=10
  intervene_flag = [0, 0, 0, 0]
  data_valid     = [1, 1, 1, 1]

chunk k=11
  intervene_flag = [0, 0, 1, 1]
  data_valid     = [1, 1, 1, 1]   # 前提是中间没有 `idle(99)`

chunk k=12
  intervene_flag = [0, 0, 0, 0]
  data_valid     = [1, 1, 1, 1]
```

### 4A.3 这 3 个 chunk 在原始轨迹里怎么落盘

`env worker` 会先把 rollout 原始 chunk 存进去，然后在下一轮开始时调用 `update_last_actions(...)` 修正前一个 chunk。

所以：

```text
chunk k=11 初始存进去时:
  forward_inputs["action"]
    = [M(11,0), M(11,1), M(11,2), M(11,3)]

chunk k=11 被 env 回传 patch 之后:
  Trajectory.actions(k=11)
    = [M(11,0), M(11,1), H(11,2), H(11,3)]

  Trajectory.intervene_flags(k=11)
    = [0, 0, 1, 1]  # 实际上会扩展到 4*26 个 bool
```

### 4A.4 这会不会进入 HG-Dagger 回放缓冲区

**默认不会。**

原因是你现在的 `psi-policy` HG-Dagger 锚点规则是：

```text
当前 chunk 的 4 个子步都必须是 expert / 遥操作
```

而 chunk `k=11` 是：

```text
[0, 0, 1, 1]
```

不是全 `1`。

所以：

```text
anchor(k=11) = False
```

即使这个 chunk 里面后 2 个动作已经被人接管了，它也**不会**成为训练锚点。

### 4A.5 结论

```text
你在 chunk 中间才切到遥操作
-> 混合 chunk
-> 动作会被正确写回原始轨迹
-> 但这个 chunk 不满足“全 expert 锚点”
-> 默认不会直接产出 HG-Dagger 回放样本
```

所以，**“动作被正确记录”** 和 **“动作最终进入 HG-Dagger 回放缓冲区并被学习器学到”** 是两回事。

---

## 4B. 子情况 A2：你在 chunk 边界切到遥操作，完整接管 1 个 chunk，然后切回模型

这个情况会比 4A.1 更接近你直觉里的 “我接管了一下，模型应该学到点东西”。

### 4B.1 假设时序

假设：

- chunk `k=20`：模型
- chunk `k=21`：完整遥操作一整个 chunk
- chunk `k=22..28`：切回模型，继续正常跑
- 整个 episode 没结束，也没有 `data_valid=0` 的 idle 断点

字符图：

```text
chunk k=20
  control_mode: [0, 0, 0, 0]
  source      : [M, M, M, M]

chunk k=21
  control_mode: [1, 1, 1, 1]
  source      : [H, H, H, H]

chunk k=22
  control_mode: [0, 0, 0, 0]
  source      : [M, M, M, M]

chunk k=23
  control_mode: [0, 0, 0, 0]
  source      : [M, M, M, M]

...

chunk k=28
  control_mode: [0, 0, 0, 0]
  source      : [M, M, M, M]
```

### 4B.2 这个遥操作 chunk 会不会成为锚点

会。

因为：

```text
intervene_flag(k=21) = [1, 1, 1, 1]
```

所以：

```text
anchor(k=21) = True
```

### 4B.3 这个锚点什么时候能真正产出 1 条回放样本

还要满足第二个条件：

```text
从 `k=21` 开始，后面必须还能拿到 32 个连续有效的真实执行动作
```

因为：

- `action_horizon = 32`
- 每 chunk 只有 `4` 个 action
- 所以至少要 `8` 个连续 valid chunk-step

也就是：

```text
1 个有效锚点
需要从锚点起点开始至少再有 3.2 秒的有效动作流
```

在这个例子里：

- `k=21..28` 一共正好 `8` 个 chunk
- `8 * 4 = 32` 个 action

所以锚点 `k=21` 会产出 **1 条**回放样本。

### 4B.4 这 1 条样本的标签到底是什么

不是只学遥操作的 4 步。

而是：

```text
chunk `k=21` 起点处的观测
  ->
未来 32 步真实执行动作窗口
```

具体就是：

```text
[ H(21,0), H(21,1), H(21,2), H(21,3),
  M(22,0), M(22,1), M(22,2), M(22,3),
  M(23,0), M(23,1), M(23,2), M(23,3),
  M(24,0), M(24,1), M(24,2), M(24,3),
  M(25,0), M(25,1), M(25,2), M(25,3),
  M(26,0), M(26,1), M(26,2), M(26,3),
  M(27,0), M(27,1), M(27,2), M(27,3),
  M(28,0), M(28,1), M(28,2), M(28,3) ]
```

总长度：

- `32 x 26`

### 4B.5 这一条样本最后会怎样被学习器用掉

它会被写成：

```text
forward_inputs["action"] : [1, 32, 26]
```

然后在回放缓冲区里：

- 这 1 个锚点 = 1 条样本

接着 `actor` 学习器会开始训练：

- `min_buffer_size = 1`
- 所以只要这条样本进了缓冲区，下一次训练循环就可以开始

训练时：

```text
样本 -> prepare_dagger_sft_batch()
      -> action stays [B, 32, 26]
      -> forward_type = SFT
      -> psi-policy native IMLE loss
```

### 4B.6 结论

```text
完整遥操作 1 个 chunk，再切回模型
-> 可以成为锚点
-> 只要后面还跟着至少 7 个完整有效 chunk
-> 就能产出 1 条 `obs_t -> future 32-step executed window` 样本
```

---

## 5. 情况 B：切到遥操作之后不切回去，直到成功按 `s`

这是你更关心的第二种情况。

### 5.1 先说 `s` 键会做什么

当前 `single_stage` reward wrapper 下：

```text
press 's'
  -> reward = +1
  -> terminated = True
  -> info["success"] = True
```

也就是说：

- `s` 会结束 episode
- 并且这一步 reward 变成 `+1`

### 5.2 假设时序

我们举一个**能产生训练样本**的具体例子。

假设：

- 你在 chunk `k=30` 的开头切到遥操作
- 从 `k=30` 开始一直遥操作
- 在 chunk `k=38` 的第 4 个子步按下 `s`

字符图：

```text
chunk k=30  [H H H H]
chunk k=31  [H H H H]
chunk k=32  [H H H H]
chunk k=33  [H H H H]
chunk k=34  [H H H H]
chunk k=35  [H H H H]
chunk k=36  [H H H H]
chunk k=37  [H H H H]
chunk k=38  [H H H H + press s on last substep]
```

也就是：

- 从 `k=30` 到 `k=38` 一共 `9` 个完整遥操作 chunk
- 一共 `9 * 4 = 36` 个有效遥操作动作

### 5.3 原始轨迹里会是什么样

这 9 个 chunk 的规范动作流都会是遥操作动作：

```text
Trajectory.actions(k=30) = [H(30,0), H(30,1), H(30,2), H(30,3)]
Trajectory.actions(k=31) = [H(31,0), H(31,1), H(31,2), H(31,3)]
...
Trajectory.actions(k=38) = [H(38,0), H(38,1), H(38,2), H(38,3)]
```

对应标记：

```text
intervene_flag(k=30..38) = [1,1,1,1]
data_valid(k=30..38)     = [1,1,1,1]
```

### 5.4 哪些 chunk 会成为回放锚点

因为锚点规则是“当前 chunk 4 个子步都为遥操作 / expert”，

所以：

```text
anchor(k=30) = True
anchor(k=31) = True
anchor(k=32) = True
...
anchor(k=38) = True
```

但是：

**并不是所有锚点最后都能产出样本。**

因为还要满足：

```text
从该锚点开始必须能凑够 32 个有效动作
```

在这个例子里：

- 从 `k=30` 开始到 `k=37`，正好 `8` 个 chunk = `32` 个 action
- 从 `k=31` 开始到 `k=38`，也正好 `8` 个 chunk = `32` 个 action`
- 从 `k=32` 开始到 `k=38`，只有 `7` 个 chunk = `28` 个 action，不够

所以最后真正留下来的只有：

```text
valid replay anchors = { k=30, k=31 }
dropped anchors      = { k=32, k=33, ..., k=38 }
```

### 5.5 它们各自学到什么

#### 锚点 = `k=30`

标签窗口：

```text
[ H(30,0..3),
  H(31,0..3),
  H(32,0..3),
  H(33,0..3),
  H(34,0..3),
  H(35,0..3),
  H(36,0..3),
  H(37,0..3) ]
```

共 `32` 个 action。

#### 锚点 = `k=31`

标签窗口：

```text
[ H(31,0..3),
  H(32,0..3),
  H(33,0..3),
  H(34,0..3),
  H(35,0..3),
  H(36,0..3),
  H(37,0..3),
  H(38,0..3) ]
```

也是 `32` 个 action。

### 5.6 为什么后面的遥操作 chunk 会被扔掉

因为你一按 `s`，episode 就结束了。

episode 一结束：

- 后面没有更多 action 可用了
- 所以靠近 episode 尾部的锚点，拿不到足够长的 `32-step` 后缀

这就是为什么：

```text
“一直遥操作到成功”
!=
“所有遥操作 chunk 都会进入 HG-Dagger replay”
```

### 5.7 一个很重要的硬条件：至少要有 3.2 秒有效动作

你现在的窗口长度是：

- `32` action
- `10 Hz`

所以从某个锚点开始，**至少需要 3.2 秒连续有效动作**，它才可能变成一条训练样本。

所以：

```text
如果你切到遥操作以后，很快就按了 `s`
并且从第一个完整遥操作 chunk 开始到 episode 结束
总共还不到 32 个有效动作

=> 这一段虽然被正确记录了
=> 但 0 条样本会进入 HG-Dagger replay
```

这不是 bug，是当前 replay window 规则的直接结果。

---

## 6. 成功发生在 chunk 中间时，会怎样

再讲一个非常关键的边界情况。

假设：

- 你在 chunk `k=50` 的第 2 个子步按下 `s`

字符图：

```text
chunk k=50
  substep 0: H
  substep 1: H + press s
  substep 2: X
  substep 3: X
```

这里 `X` 的意思不是 “模型动作”，而是：

- env 在第 2 个子步已经终止
- 后两个子步不会真的被执行
- `chunk_step()` 会用 padding 把 chunk 填满到长度 4

所以最终：

```text
intervene_flag = [1, 1, 0, 0]
data_valid     = [1, 1, 0, 0]
```

注意：

- `Trajectory.actions` 这个 chunk 里后两个位置虽然仍然会占位
- 但 `transition_valids` 会是 `0`
- 这些无效位**不能**拿来填未来 `32-step` window

所以：

```text
如果某个锚点的后缀需要用到这两个 padding 位
=> 该锚点直接丢弃
```

---

## 7. idle(99) 手动切换窗口也很重要

当前配置里：

- `idle_control_modes = [99]`
- `filter_idle_transitions = true`

所以如果切换遥操作 / 模型的过程中 controller 给出 `99`：

```text
control_mode = 99
-> data_valid = False
```

一旦某个 future window 中间穿过这些 `data_valid=0` 的步：

```text
_build_executed_action_window(...)
-> 直接停止
-> 该锚点丢弃
```

所以在实际联调里，下面这两件事会极大影响样本产量：

1. 你是不是经常在 chunk 中间才切到遥操作
2. 切换时 controller 会不会吐很多 `idle(99)` 过渡步

---

## 8. 两种情况的最终对比

### 8.1 中途切到遥操作，再切回模型

```text
优点:
  - 轨迹里会保存 “真实执行动作”
  - 遥操作覆盖过的位置不会丢

缺点:
  - 如果遥操作只覆盖了半个 chunk
    -> 混合 chunk
    -> 不能当锚点

  - 即使完整遥操作了 1 个 chunk
    -> 后面还必须再有 7 个完整 valid chunk
    -> 才能凑满 32 步窗口
```

一句话总结：

```text
“切一下再切回去” 不等于一定会产生 HG-Dagger 样本
```

### 8.2 切到遥操作，不切回去，直到成功按 `s`

```text
优点:
  - 更容易产生完整 expert chunk
  - 更容易满足“锚点 = 全 expert”的条件

缺点:
  - 越靠近 episode 尾部的遥操作 chunk，越可能因为后缀不够 32 步被丢掉
  - 如果从第一个完整遥操作 chunk 到按 `s` 总共不到 3.2 秒
    -> 0 条样本
```

一句话总结：

```text
“一直遥操作到成功” 比 “短暂接管一下再放回去” 更容易产样本，
但也不是所有遥操作段都会被学习器学到。
```

---

## 9. 你现在跑 HG-Dagger 时，最该盯什么日志

当前配置：

- `min_buffer_size = 1`
- `global_batch_size = 8`
- `micro_batch_size = 8`

所以只要真的形成了第一条有效锚点样本，`actor` 很快就会开始更新。

建议重点看：

- `train/replay_buffer/num_trajectories`
- `train/replay_buffer/total_samples`
- `train/dagger/actor_loss`

如何解释：

```text
num_trajectories 增长
  -> env 确实把原始轨迹发给 actor 了

total_samples 增长
  -> 原始轨迹里真的抽出了有效的 32 步锚点样本

actor_loss 开始变化
  -> 学习器已经在用这些样本更新 `psi-policy`
```

如果出现下面这种情况：

```text
num_trajectories 在涨
但 total_samples 不涨
```

那通常意味着：

1. 你有遥操作，但没有完整 expert chunk
2. 有完整 expert chunk，但后面不足 32 个有效动作
3. 中间穿过了 idle(99) / data_valid=0

---

## 10. 最后一张总图

把 “动作被执行” 和 “动作被学到” 放在一张图里：

```text
                +-------------------------+
obs_t --------> | rollout psi-policy      |
                | predict 32x26           |
                | RTC(search=4, w0=0.8)   |
                +------------+------------+
                             |
                             v
                     真实执行 chunk 4x26
                             |
                             v
                +-------------------------+
                | A2D controller @ 10 Hz  |
                | 第0步 第1步 第2步 第3步 |
                +------------+------------+
                             |
                             | control_mode / 遥操作 / s/f
                             v
                +-------------------------+
                | RealWorldEnv.chunk_step  |
                | rewards[4]               |
                | intervene_flag[4]        |
                | data_valid[4]            |
                +------------+------------+
                             |
                             v
                +-------------------------+
                | 原始 Trajectory          |
                | actions[T,1,4,26] 最终值 |
                | flags[T,1,4,26]          |
                | valids[T,1,4]            |
                +------------+------------+
                             |
                             | `psi-policy` 钩子
                             | 只保留全 expert 锚点
                             | 需要 32 个有效未来动作
                             v
                +-------------------------+
                | 回放样本                |
                | obs_t -> [32,26]        |
                +------------+------------+
                             |
                             v
                +-------------------------+
                | actor 学习器            |
                | batch=8 lr=3e-5 IMLE     |
                +-------------------------+
```

---

## 11. 最短启动方式

```bash
bash examples/embodiment/run_realworld_async.sh realworld_a2d_dagger_psi
```

---

## 12. 你现在最该记住的 6 句话

1. 当前这条链路按 `na32` ckpt 算，`action_horizon = 32`。
2. 当前每个 chunk 真正执行 `4` 步，`10 Hz` 下就是 `0.4 s`。
3. HG-Dagger 现在学的是 `obs_t -> future 32-step executed action window`。
4. chunk 中间才切到遥操作，会形成混合 chunk，默认不能当锚点。
5. 一个锚点想进入 replay，后面必须还有 `32` 个有效动作，也就是 `3.2 s`。
6. `s` 会给 `+1` 并终止 episode，但它也会让靠近 episode 尾部的很多锚点因为后缀不够而被丢掉。
