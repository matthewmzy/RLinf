# A2D 实验命令

```bash
cd /home/descfly/ceyao_dev/RLinf
export EMBODIED_PATH=$PWD/examples/embodiment
export PYTHONPATH=$PWD
export PYTHON_BIN=${PYTHON_BIN:-python}
```

复现实验（异步训练入口，无噪声）：

```bash
$PYTHON_BIN examples/embodiment/train_async.py --config-name realworld_a2d_sac_psi_rtc_repro
$PYTHON_BIN examples/embodiment/train_async.py --config-name realworld_a2d_sac_psi_direct_repro
$PYTHON_BIN examples/embodiment/train_async.py --config-name realworld_a2d_sac_psi_window_repro
```

真机 RL（异步训练入口，有噪声）：

```bash
$PYTHON_BIN examples/embodiment/train_async.py --config-name realworld_a2d_sac_psi_rtc_async
$PYTHON_BIN examples/embodiment/train_async.py --config-name realworld_a2d_sac_psi_direct_async
$PYTHON_BIN examples/embodiment/train_async.py --config-name realworld_a2d_sac_psi_window_async
```

旧名字：

```text
realworld_a2d_sac_psi_async        -> realworld_a2d_sac_psi_direct_async
realworld_a2d_sac_psi_async_repro  -> realworld_a2d_sac_psi_rtc_repro
realworld_a2d_sac_psi_safe_explore -> realworld_a2d_sac_psi_direct_async
realworld_a2d_sac_psi              -> realworld_a2d_sac_psi_direct_async
```
