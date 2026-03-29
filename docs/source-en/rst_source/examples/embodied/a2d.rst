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

The provided example uses SAC with a CNN policy:

- ``image_num: 3`` for the three RGB views
- ``state_dim: 28`` from the concatenated robot state vector
- ``action_dim: 28`` matching the A2D controller action contract

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

4. Use ``examples/embodiment/config/realworld_a2d_sac_cnn.yaml`` as the starting point.

5. Launch training:

.. code:: bash

   python examples/embodiment/train_embodied_agent.py --config-name realworld_a2d_sac_cnn

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
- action range mapping
- reward/success keys exposed by the controller

Dummy Validation
----------------

For a lightweight pipeline check without a real robot:

.. code:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-name ../tests/e2e_tests/embodied/realworld_a2d_dummy_sac_cnn
