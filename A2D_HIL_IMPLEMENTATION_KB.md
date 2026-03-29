# A2D HIL Implementation KB

This note records only the source-backed facts that matter for RLinf A2D real-robot SAC + human-in-the-loop integration.

## Authoritative upstream sources

- A2D gRPC server:
  - `../src/model_inference/model_inference/envs/grpc_server.py`
- A2D observation assembly:
  - `../src/model_inference/model_inference/utils/obs_manager.py`
- A2D default obs/action config:
  - `../src/model_inference/model_inference/configs/data_source/base_data.yaml`
- A2D pedal / control-mode config:
  - `../src/pedal_controller/config/pedal_config_control.yaml`
  - `../src/pedal_controller/config/pedal_config_model_idle.yaml`
- psi-policy online client references:
  - `../psi-policy/model_inference_client/envs/base_env.py`
  - `../psi-policy/psi_policy/config/task/dex_image_rgb_state.yaml`

## gRPC observation contract

- `Observation.images` carries image tensors keyed by name.
- `Observation.states` carries numeric tensors keyed by name.
- `Observation.control_mode`, `Observation.trajectory_label`, and `Observation.is_switch_mode` are top-level protobuf fields, not entries in `states`.

## Default A2D observation layout

- Images:
  - `rgb_head`
  - `rgb_left_hand`
  - `rgb_right_hand`
- States:
  - `arm_joint_states`: 14D
  - `left_hand_states`: 6D
  - `right_hand_states`: 6D
  - `waist_joints_states`: 2D
- `ObsManager` additionally derives:
  - `left_arm_states = arm_joint_states[:7]`
  - `right_arm_states = arm_joint_states[7:14]`

Important consequence:

- The authoritative arm order is `left_arm` first, `right_arm` second in the 14D arm vector.

## Default A2D action layout

Source: `src/model_inference/model_inference/README.md`

- `[0:2]`: waist
- `[2:9]`: left arm
- `[9:16]`: right arm
- `[16:22]`: left hand
- `[22:28]`: right hand

For RLinf `psi-policy` integration we use `policy_action_dim=26`, so the policy action is:

- `arm_joint_states[14] + left_hand_states[6] + right_hand_states[6]`
- waist is excluded and may be held fixed by env config

## Control-mode semantics

Source-backed from pedal configs:

- `0`: model control
- `1`: teleoperation
- `99`: idle / preparation

These values are now treated in RLinf as:

- `0`: normal SAC rollout transition
- `1`: human intervention transition, with `intervene_action` filled from current robot joint state
- `99`: invalid for replay insertion (`data_valid=False`)

## RLinf implementation decisions on `a2d-dev`

- `rlinf/envs/realworld/a2d/a2d_env.py`
  - maps teleop observations into `info["intervene_action"]`
  - marks idle frames as `info["data_valid"] = False`
- `rlinf/envs/realworld/realworld_env.py`
  - propagates `data_valid` through chunked stepping
- `rlinf/workers/env/env_worker.py`
  - attaches per-transition validity flags to collected transitions
- `rlinf/workers/actor/fsdp_sac_policy_worker.py`
  - drops invalid transitions before adding trajectories to replay buffer
- `rlinf/models/embodiment/psi_policy/psi_policy_for_rl.py`
  - uses the corrected arm split:
    - `right_arm = [7:14]`
    - `left_arm = [0:7]`

## psi-policy integration caveats

- `psi-policy` checkpoint actions are treated as absolute joint targets, not extra normalized controller-space actions.
- For A2D SAC + psi-policy example config we therefore use:
  - `policy_action_dim: 26`
  - `normalize_actions: False`
  - `clip_policy_actions: False`
  - `keyboard_reward_wrapper: single_stage`
