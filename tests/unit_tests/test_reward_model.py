# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.data.datasets.reward_model import RewardBinaryDataset, RewardDatasetPayload
from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker


def _reward_model_cfg():
    return OmegaConf.create(
        {
            "model_type": "resnet",
            "model_path": None,
            "arch": "resnet18",
            "pretrained": False,
            "hidden_dim": None,
            "dropout": 0.0,
            "image_size": [3, 8, 8],
            "normalize": False,
            "precision": "fp32",
        }
    )


def test_reward_dataset_payload_save_load_and_collate(tmp_path):
    payload = RewardDatasetPayload(
        images=[
            torch.zeros(8, 8, 3, dtype=torch.uint8),
            torch.ones(8, 8, 3, dtype=torch.uint8),
        ],
        labels=[0, 1],
    )
    output_path = tmp_path / "reward.pt"
    payload.save(str(output_path))

    loaded = RewardDatasetPayload.load(str(output_path))
    assert loaded.metadata == {}

    dataset = RewardBinaryDataset(str(output_path))
    images, labels = next(iter(DataLoader(dataset, batch_size=2)))

    assert isinstance(images, torch.Tensor)
    assert images.shape == (2, 8, 8, 3)
    assert torch.equal(labels, torch.tensor([0.0, 1.0]))


def test_resnet_reward_model_forward_and_compute_reward(monkeypatch):
    model = ResNetRewardModel(_reward_model_cfg())

    class DummyBackbone(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.reshape(images.shape[0], -1).mean(dim=1, keepdim=True)

    monkeypatch.setattr(model, "backbone", DummyBackbone())

    images = torch.tensor([[[[0.0]]], [[[2.0]]]])
    labels = torch.tensor([0.0, 1.0])

    outputs = model.forward(images, labels)
    expected_logits = torch.tensor([0.0, 2.0])
    expected_probabilities = torch.sigmoid(expected_logits)

    assert torch.allclose(outputs["logits"], expected_logits)
    assert torch.allclose(outputs["probabilities"], expected_probabilities)
    assert torch.allclose(model.compute_reward(images), expected_probabilities)


def test_reward_worker_single_camera_helpers():
    batch_size = EmbodiedRewardWorker._infer_reward_batch_size(
        torch.zeros(2, 8, 8, 3, dtype=torch.uint8)
    )
    assert batch_size == 2

    merged = EmbodiedRewardWorker._merge_image_batches(
        [
            torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            np.ones((2, 8, 8, 3), dtype=np.uint8),
        ]
    )
    assert isinstance(merged, torch.Tensor)
    assert merged.shape == (3, 8, 8, 3)
