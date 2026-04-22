可以。以后我就直接说“能跑什么、怎么跑”。

先进入项目目录，然后用同一条命令换不同配置名就行：

```bash
cd /home/psibot/workspace_zhiyuan/RLinf
export EMBODIED_PATH=$PWD/examples/embodiment
export PYTHONPATH=$PWD
```

如果你要用我这边测过的 Python，也可以直接用这个：

```bash
PYTHON_BIN=/home/psibot/workspace_zhiyuan/miniconda/envs/a2d/bin/python
```

然后 6 个实验分别是：

RTC 复现实验

```bash
$PYTHON_BIN examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_rtc_repro
```

RTC 真机强化实验

```bash
$PYTHON_BIN examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_rtc_async
```

不用 RTC，直接执行 action chunk 的复现实验

```bash
$PYTHON_BIN examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_direct_repro
```

不用 RTC，直接执行 action chunk 的真机强化实验

```bash
$PYTHON_BIN examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_direct_async
```

window 平滑的复现实验

```bash
$PYTHON_BIN examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_window_repro
```

window 平滑的真机强化实验

```bash
$PYTHON_BIN examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_window_async
```

你以前的老名字也还能继续用：

老的 `realworld_a2d_sac_psi_async`
现在等价于 `realworld_a2d_sac_psi_direct_async`
老的 `realworld_a2d_sac_psi_async_repro`
现在等价于 `realworld_a2d_sac_psi_rtc_repro`
老的 `realworld_a2d_sac_psi_safe_explore`
现在也还能跑，等价于 `realworld_a2d_sac_psi_direct_async`
老的 `realworld_a2d_sac_psi`
现在也还能跑，等价于 realworld_a2d_sac_psi_direct_async
