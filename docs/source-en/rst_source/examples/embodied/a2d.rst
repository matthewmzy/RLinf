Real-World RL with A2D
======================

This page describes how to use RLinf with the official A2D teleoperation docker as an external controller.

Environment
-----------

- **Environment type**: ``realworld``
- **Gym id**: ``A2DEnv-v1``
- **Controller boundary**: RLinf connects to the `model_inference` gRPC service exposed by the running A2D docker.
- **Observation**:

  - RGB images from ``rgb_head``, ``rgb_left_hand``, and ``rgb_right_hand``
  - Robot state arrays from ``arm_joint_states``, ``left_hand_states``, ``right_hand_states``, and ``waist_joints_states``

- **Action space**: 28-dimensional continuous vector

  - 16 dims for waist + dual-arm joints
  - 6 dims for left hand
  - 6 dims for right hand

Algorithm
---------

The provided example uses async SAC fine-tuning initialized from psi-policy:

- three RGB views: ``rgb_head``, ``rgb_left_hand``, and ``rgb_right_hand``
- a default policy action dimension of ``26`` for the two arms and two hands
- the A2D env prepends the 2 waist values before sending the final 28-dim action to the controller

Installation
------------

**Python environment**

.. code:: bash

   bash requirements/install.sh embodied --env a2d
   source .venv/bin/activate

**Docker image**

.. code:: bash

   export BUILD_TARGET=embodied-a2d
   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .

The image is built on top of the official ``a2d-tele`` runtime image and installs RLinf plus the A2D gRPC client dependencies.

Quick Start
-----------

1. Start the official A2D docker with your existing deployment flow.
2. Inside the container, start the official ``model_inference`` gRPC server:

.. code:: bash

   source /ros_entrypoint.sh
   source /opt/psi/rt/a2d-tele/install/setup.bash
   python3 -m model_inference.run_inference_server

3. Prepare a cluster config that places:

   - ``env`` on the A2D workstation node
   - ``rollout`` and ``actor`` on the GPU node

4. Use ``examples/embodiment/config/realworld_a2d_sac_psi_async.yaml`` as the starting point.

5. Launch training:

.. code:: bash

   python examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_psi_async

Configuration Notes
-------------------

The A2D hardware section looks like this:

.. code:: yaml

   hardware:
     type: A2D
     configs:
       - node_rank: 1
         controller_host: 127.0.0.1
         grpc_port: 12321
         container_name: a2d-tele-release-2-1-0rc3-latest

The environment override config can be used to change:

- image keys and shapes
- state keys and shapes
- action bounds
- ``policy_action_dim`` (the current psi-policy setup uses 26)
- reward/success keys exposed by the controller

Dummy Validation
----------------

For a lightweight pipeline check without a real robot:

.. code:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-name ../tests/e2e_tests/embodied/realworld_a2d_dummy_sac_psi
