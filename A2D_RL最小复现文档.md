最重要的配置文件
/home/psibot/workspace_zhiyuan/RLinf/examples/embodiment/config/realworld_a2d_sac_psi.yaml

在已经配置过的电脑上跑
cd /home/psibot/workspace_zhiyuan/deploy-a2d-tele-hil
python3 deploy.py

# 启动GUI上的所有服务

docker exec -it a2d-tele-release-2-1-0rc3-latest bash
ros2 run model_inference run_inference_server 

cd /home/psibot/workspace_zhiyuan/RLinf

PYTHONPATH=/home/psibot/workspace_zhiyuan/RLinf \
EMBODIED_PATH=/home/psibot/workspace_zhiyuan/RLinf/examples/embodiment \
/home/psibot/workspace_zhiyuan/miniconda/envs/a2d/bin/python \
examples/embodiment/train_async.py \
  --config-name realworld_a2d_sac_psi_async \
  runner.logger.log_path=/home/psibot/workspace_zhiyuan/results/a2d_async_run

第一次上新电脑部署
1. 下载模型和数据
在小黑上没有online的数据，要采的话：
原因：由于产出容器默认开启了端侧数据质检，但是在HIL模式（非数采模式）下，有些被监控的数据话题由于不是持续发布的，导致帧率不能满足质检要求，所以开启不了录制。因此需要关闭端侧数据质检监控。这样以后F3按下才能开启数据录制。
关闭方式：在docker中修改配置：
vim /opt/psi/rt/a2d-tele/install/data_collection/share/data_collection/config/topic_config.yaml
下到最后找到：
[Image]
Shift + i #进入vim的编辑模式
将true 改为 false：
[Image]
保存退出：
esc #退出编辑模式
Shift + : #进入vim的命令行模式
wq # 保存退出
2. 裁剪数据（Option）
cd /home/psibot/workspace_zhiyuan/psi-policy

python tools/vis_and_cut_demo.py \
  --input /home/psibot/workspace_zhiyuan/psi-policy/data/bj01-dc09_20260409_22.zarr \
  --output /home/psibot/workspace_zhiyuan/psi-policy/data/bj01-dc09_20260409_22_cut.zarr \
  --overwrite
3. 离线IQL训练
先把zarr demo转成buffer
cd /home/psibot/workspace_zhiyuan/RLinf

python examples/embodiment/convert_a2d_zarr_to_replay_buffer.py \
  --zarr-path /home/psibot/workspace_zhiyuan/psi-policy/data/bj01-dc09_20260409_22_cut.zarr \
  --output-dir /home/psibot/workspace_zhiyuan/RLinf/results/a2d_iql_20260415/offline_success_buffer_20260409_22_cut \
  --overwrite
再训练iql
cd /home/psibot/workspace_zhiyuan/RLinf

python examples/embodiment/pretrain_a2d_iql_critic.py \
  --config-name realworld_a2d_sac_psi \
  --replay-buffer-path /home/psibot/workspace_zhiyuan/RLinf/results/a2d_iql_20260415/offline_success_buffer_20260409_22_cut \
  --output-dir /home/psibot/workspace_zhiyuan/RLinf/results/a2d_iql_20260415/critic_warmstart_20260409_22_cut \
  --model-path /home/psibot/workspace_zhiyuan/psi-policy/data/outputs/2026.04.10/04.54_zjxc_v0.8_vit-small_rgb-state/checkpoints/latest.ckpt \
  --normalizer-path /home/psibot/workspace_zhiyuan/psi-policy/data/outputs/2026.04.10/04.54_zjxc_v0.8_vit-small_rgb-state/normalizer.pkl \
  --num-updates 1000 \
  --batch-size 16 \
  --log-interval 50 \
  --overwrite
4. 配置模型路径：
服务器上下载的psi-policy模型：
- actor.model.model_path
- actor.model.normalizer_path
- rollout.model.model_path
训练的offline critic：
- runner.ckpt_path
5. 启动容器GUI和server
cd /home/psibot/workspace_zhiyuan/deploy-a2d-tele-hil
python3 deploy.py

# 另开一个终端进容器
docker exec -it a2d-tele-release-2-1-0rc3-latest bash

# 标定
ros2 run vr_calibration calibrate_node

# 起server
ros2 run model_inference run_inference_server 

# 如果需要重启 GUI （Option）
ros2 launch scripts_pack start_gui.launch.py
6. 启动RL！
cd /home/psibot/workspace_zhiyuan/RLinf

PYTHONPATH=/home/psibot/workspace_zhiyuan/RLinf \
EMBODIED_PATH=/home/psibot/workspace_zhiyuan/RLinf/examples/embodiment \
/home/psibot/workspace_zhiyuan/miniconda/envs/a2d/bin/python \
examples/embodiment/train_async.py \
  --config-name realworld_a2d_sac_psi_async \
  runner.logger.log_path=/home/psibot/workspace_zhiyuan/results/a2d_async_run
训练开始后，一条 episode 的典型流程是：

1. 环境 reset，机器人回到配置里的 reset_controller_action
2. 机器人进入模型控制，策略开始执行
3. 人实时观察动作是否安全、是否接近成功
4. 如有必要，人切换到 idle -> teleop 接管
5. episode 成功时按 s
6. episode 失败时按 f
7. 这一条 episode 结束，环境 reset，开始下一条
