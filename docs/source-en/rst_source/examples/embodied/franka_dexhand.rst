Real-World RL with Franka + Dexterous Hand
==========================================

This page summarizes the configuration differences when the Franka arm uses a Ruiyan dexterous hand.
For the end-to-end real-world workflow, see :doc:`franka` and :doc:`franka_reward_model`.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The dexterous-hand setup keeps the same real-world RL and reward-model workflow as Franka.
The main differences are in the end-effector, teleoperation, and action space:

- The action space is 12-D.
- The first 6 dimensions control arm pose deltas.
- The last 6 dimensions control the dexterous hand.
- ``RuiyanHand`` handles the hand hardware.
- ``DexHandIntervention`` combines SpaceMouse input and glove input into expert actions.

Teleoperation
-------------

Dexterous-hand teleoperation uses:

- SpaceMouse for 6-D arm motion
- a data glove for 6-D finger control
- the SpaceMouse left button to enable relative glove control

Reward Model
------------

The reward-model path is the same as the Franka real-world reward-model workflow described in :doc:`franka_reward_model`.

For the dexterous-hand pick-and-place environment:

- the default reward image follows ``env.main_image_key``
- ``main_image_key`` is ``global`` in ``examples/embodiment/config/env/realworld_dex_pnp.yaml``
- ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` uses the reward model through the ``reward`` section

Configurations
--------------

Use ``examples/embodiment/config/realworld_collect_dexhand_data.yaml`` for data collection.
This config includes:

- ``end_effector_type: "ruiyan_hand"``
- glove settings for teleoperation
- ``data_collection`` for raw episode export in ``pickle`` format

Use ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` for RL training.
Before running, fill in:

- ``robot_ip``
- ``target_ee_pose``
- policy ``model_path``
- reward ``model.model_path``
- dexterous-hand serial ports in ``end_effector_config`` and ``glove_config``

Workflow
--------

1. Follow :doc:`franka` to finish environment setup, dependency installation, and Ray cluster setup.
2. Collect expert demos with:

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data

3. Collect reward raw episodes with the same entrypoint. For this pass, increase ``env.eval.override_cfg.success_hold_steps`` and use a separate log directory.
4. Preprocess the raw reward episodes with ``examples/reward/preprocess_reward_dataset.py`` as described in :doc:`franka_reward_model`.
5. Train the reward model with ``examples/reward/run_reward_training.sh``.
6. Launch RL with:

   .. code-block:: bash

      bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async
