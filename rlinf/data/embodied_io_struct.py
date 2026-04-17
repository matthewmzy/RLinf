# Copyright 2025 The RLinf Authors.
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

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    pass

from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)


def get_model_weights_id(versions: torch.Tensor) -> str:
    """
    Get the model weights id from the tensor.

    Args:
        versions (torch.Tensor): The tensor to get the model weights id from.

    Returns:
        str: The model weights id.
    """

    name_bytes = versions.cpu().numpy().tobytes()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_bytes.hex()))


def _to_cpu_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.cpu().contiguous()


@dataclass(kw_only=True)
class EnvOutput:
    """Environment output for a single chunk step."""

    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]

    intervene_actions: Optional[torch.Tensor] = None  # [B]
    intervene_flags: Optional[torch.Tensor] = None  # [B]
    transition_valids: Optional[torch.Tensor] = None  # [B]
    episode_intervened: Optional[torch.Tensor] = None  # [B]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = _to_cpu_tensor(self.dones)
        self.terminations = _to_cpu_tensor(self.terminations)
        self.truncations = _to_cpu_tensor(self.truncations)
        self.rewards = _to_cpu_tensor(self.rewards)
        self.intervene_actions = _to_cpu_tensor(self.intervene_actions)
        self.intervene_flags = _to_cpu_tensor(self.intervene_flags)
        self.transition_valids = _to_cpu_tensor(self.transition_valids)
        self.episode_intervened = _to_cpu_tensor(self.episode_intervened)
        if self.transition_valids is not None and self.transition_valids.dim() == 1:
            self.transition_valids = self.transition_valids.unsqueeze(-1)

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    @staticmethod
    def merge_env_outputs(env_outputs: list[dict]) -> dict[str, Any]:
        """Merge multiple env output dicts into one batch-aligned env output.

        Merge strategy:

        - Tensor fields: concatenate on batch dimension.
        - List fields: flatten in source order.
        - ``None`` fields: keep ``None``.
        - ``final_obs`` supports partial ``None`` across shards. For shards
            without ``final_obs``, use the corresponding ``obs`` as fallback to
            keep batch alignment.

        Args:
            env_outputs: Per-source env output dicts that share the same schema.

        Returns:
            A merged env output dict produced via ``EnvOutput(...).to_dict()``.
        """

        def _get_batch_size(env_output: dict[str, Any]) -> int:
            dones = env_output.get("dones")
            if isinstance(dones, torch.Tensor):
                return dones.shape[0]

            obs = env_output["obs"]
            for key in ("states", "main_images", "task_descriptions"):
                value = obs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
                if isinstance(value, list):
                    return len(value)
            raise ValueError("Cannot infer batch size from env output.")

        def _merge_obs_dicts(obs_dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged_obs = {}
            for key in obs_dicts[0].keys():
                obs_elements = [obs_dict[key] for obs_dict in obs_dicts]
                first_non_none = next(
                    (element for element in obs_elements if element is not None), None
                )
                if first_non_none is None:
                    merged_obs[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged_obs[key] = torch.cat(obs_elements, dim=0)
                elif isinstance(first_non_none, list):
                    merged_obs[key] = [
                        item for sublist in obs_elements for item in sublist
                    ]
                else:
                    merged_obs[key] = obs_elements
            return merged_obs

        def _merge_optional_tensor_field(
            field_name: str,
            *,
            allow_partial_none: bool = False,
            fill_value: float | bool = 0,
        ) -> torch.Tensor | None:
            values = [env_output[field_name] for env_output in env_outputs]
            if all(value is None for value in values):
                return None

            if any(value is None for value in values):
                if not allow_partial_none:
                    raise ValueError(
                        f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                    )

                ref_tensor = next(value for value in values if value is not None)
                filled_values = []
                for env_output, value in zip(env_outputs, values):
                    if value is None:
                        batch_size = _get_batch_size(env_output)
                        fill_shape = (batch_size, *ref_tensor.shape[1:])
                        filled_values.append(
                            torch.full(
                                fill_shape,
                                fill_value=fill_value,
                                dtype=ref_tensor.dtype,
                            )
                        )
                    else:
                        filled_values.append(value)
                values = filled_values

            return torch.cat(values, dim=0)

        merged_obs = _merge_obs_dicts([env_output["obs"] for env_output in env_outputs])

        merged_final_obs = None
        final_obs_list = [env_output["final_obs"] for env_output in env_outputs]
        if any(final_obs is not None for final_obs in final_obs_list):
            # Some shards may not have done episodes in this step, so their final_obs
            # is None. Use obs as fallback to keep merged batch shape aligned.
            final_obs_or_obs = [
                final_obs if final_obs is not None else env_output["obs"]
                for env_output, final_obs in zip(env_outputs, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        merged_dones = _merge_optional_tensor_field("dones")
        merged_terminations = _merge_optional_tensor_field("terminations")
        merged_truncations = _merge_optional_tensor_field("truncations")
        merged_rewards = _merge_optional_tensor_field("rewards")
        merged_intervene_actions = _merge_optional_tensor_field(
            "intervene_actions",
            allow_partial_none=True,
            fill_value=0.0,
        )
        merged_intervene_flags = _merge_optional_tensor_field(
            "intervene_flags",
            allow_partial_none=True,
            fill_value=False,
        )
        merged_transition_valids = _merge_optional_tensor_field(
            "transition_valids",
            allow_partial_none=True,
            fill_value=True,
        )
        # turn to EnvOutput and turn to dict to call post init for tensor processing
        return EnvOutput(
            obs=merged_obs,
            final_obs=merged_final_obs,
            dones=merged_dones,
            terminations=merged_terminations,
            truncations=merged_truncations,
            rewards=merged_rewards,
            intervene_actions=merged_intervene_actions,
            intervene_flags=merged_intervene_flags,
            transition_valids=merged_transition_valids,
        ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["intervene_actions"] = self.intervene_actions
        env_output_dict["intervene_flags"] = self.intervene_flags
        env_output_dict["transition_valids"] = self.transition_valids
        env_output_dict["episode_intervened"] = self.episode_intervened

        return env_output_dict


@dataclass(kw_only=True)
class RolloutResult:
    """Rollout result for a single chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]

    bootstrap_values: torch.Tensor = None  # [B, 1]
    save_flags: torch.Tensor = None  # [B, num_action_chunks]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.save_flags is not None:
            self.save_flags = self.save_flags.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()

    @staticmethod
    def merge_rollout_results(
        rollout_results: list["RolloutResult"],
    ) -> "RolloutResult":
        def _merge_optional_tensor(field_name: str) -> torch.Tensor | None:
            values = [
                getattr(rollout_result, field_name)
                for rollout_result in rollout_results
            ]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                )
            return torch.cat(values, dim=0)

        merged_actions = _merge_optional_tensor("actions")
        merged_prev_logprobs = _merge_optional_tensor("prev_logprobs")
        merged_prev_values = _merge_optional_tensor("prev_values")
        merged_bootstrap_values = _merge_optional_tensor("bootstrap_values")
        merged_save_flags = _merge_optional_tensor("save_flags")
        merged_versions = _merge_optional_tensor("versions")

        forward_inputs_list = [
            rollout_result.forward_inputs for rollout_result in rollout_results
        ]
        if all(not forward_inputs for forward_inputs in forward_inputs_list):
            merged_forward_inputs = {}
        else:
            merged_forward_inputs = cat_list_of_dict_tensor(forward_inputs_list)
        return RolloutResult(
            actions=merged_actions,
            prev_logprobs=merged_prev_logprobs,
            prev_values=merged_prev_values,
            bootstrap_values=merged_bootstrap_values,
            save_flags=merged_save_flags,
            forward_inputs=merged_forward_inputs,
            versions=merged_versions,
        )


@dataclass(kw_only=True)
class ChunkStepResult:
    """Model outputs, env outputs (without observations), and training forward inputs for a chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.dones is not None:
            self.dones = self.dones.cpu().contiguous()
        if self.terminations is not None:
            self.terminations = self.terminations.cpu().contiguous()
        if self.truncations is not None:
            self.truncations = self.truncations.cpu().contiguous()
        if self.rewards is not None:
            self.rewards = self.rewards.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()


@dataclass
class Trajectory:
    """
    trajectory contains multiple episodes.
    """

    max_episode_length: int = 0  # max episode length
    model_weights_id: str = ""  # str(uuid(versions))
    actions: torch.Tensor = None
    intervene_flags: torch.Tensor = None
    transition_valids: torch.Tensor = None
    rewards: torch.Tensor = None
    terminations: torch.Tensor = None
    truncations: torch.Tensor = None
    dones: torch.Tensor = None
    prev_logprobs: torch.Tensor = None
    prev_values: torch.Tensor = None
    versions: torch.Tensor = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)

    curr_obs: dict[str, Any] = field(default_factory=dict)
    next_obs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _generate_field_mask(
        ref_tensor: torch.Tensor, mask: torch.Tensor, traj_len: int
    ) -> torch.Tensor:
        """
        Generate a mask for terminations/truncations/dones based on their original shape.
        """
        assert mask.dim() == 1, f"Expected 1D mask, got {mask.shape=}"
        if ref_tensor.shape[0] == traj_len:
            return mask
        elif ref_tensor.shape[0] > traj_len:
            extra = int(ref_tensor.shape[0] - traj_len)
            assert traj_len % extra == 0, (
                f"Trajectory length {traj_len} is not divisible by extra {extra} for terminations/truncations/dones"
            )
            epoch_len = traj_len // extra

            field_mask = torch.zeros(
                ref_tensor.shape[0], dtype=torch.bool, device=mask.device
            )
            original_indices = torch.arange(ref_tensor.shape[0], device=mask.device)
            epoch_idx = original_indices // (epoch_len + 1)
            step_idx = original_indices % (epoch_len + 1)

            # Keep the first position of each epoch (step_idx == 0)
            field_mask[step_idx == 0] = True

            # Map positions with step_idx >= 1 to mask
            valid_mask = step_idx >= 1
            mask_idx = epoch_idx[valid_mask] * epoch_len + (step_idx[valid_mask] - 1)
            valid_original_indices = original_indices[valid_mask]
            valid_mask_idx = mask_idx < len(mask)
            field_mask[valid_original_indices[valid_mask_idx]] = mask[
                mask_idx[valid_mask_idx]
            ].to(dtype=torch.bool)

            return field_mask
        else:
            raise ValueError(
                f"Reference tensor length {ref_tensor.shape[0]} < traj_len {traj_len}"
            )

    @staticmethod
    def _normalize_chunk_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        assert mask.dim() == 3, f"Expected 3D chunk mask, got {mask.shape=}"
        return mask

    @staticmethod
    def _align_field_with_traj_len(
        tensor: torch.Tensor | None, traj_len: int, field_name: str
    ) -> torch.Tensor | None:
        if tensor is None:
            return None

        if tensor.shape[0] == traj_len:
            return tensor

        if tensor.shape[0] > traj_len:
            extra = int(tensor.shape[0] - traj_len)
            assert traj_len % extra == 0, (
                f"Trajectory length {traj_len} is not divisible by extra {extra} "
                f"for field {field_name}"
            )
            epoch_len = traj_len // extra
            return tensor.reshape(
                extra, epoch_len + 1, *tensor.shape[1:]
            )[:, 1:].reshape(traj_len, *tensor.shape[1:])

        raise ValueError(
            f"Reference tensor length {tensor.shape[0]} < traj_len {traj_len} for field {field_name}"
        )

    @staticmethod
    def _infer_num_action_chunks(
        mask: torch.Tensor | None, fallback_tensor: torch.Tensor | None
    ) -> int:
        if mask is not None and mask.dim() == 3:
            return int(mask.shape[-1])
        if (
            fallback_tensor is not None
            and fallback_tensor.dim() >= 3
            and fallback_tensor.shape[-1] > 0
        ):
            return int(fallback_tensor.shape[-1])
        return 1

    @staticmethod
    def _reshape_action_like_tensor(
        tensor: torch.Tensor | None,
        *,
        traj_len: int,
        num_chunks: int,
        field_name: str,
    ) -> torch.Tensor | None:
        tensor = Trajectory._align_field_with_traj_len(tensor, traj_len, field_name)
        if tensor is None:
            return None
        assert tensor.dim() == 3, (
            f"Expected 3D tensor for field '{field_name}', got {tensor.shape=}"
        )
        assert tensor.shape[-1] % num_chunks == 0, (
            f"Field '{field_name}' last dim {tensor.shape[-1]} is not divisible by "
            f"{num_chunks=}"
        )
        return tensor.reshape(traj_len, tensor.shape[1], num_chunks, -1)

    @staticmethod
    def _reshape_chunk_scalar_tensor(
        tensor: torch.Tensor | None,
        *,
        traj_len: int,
        num_chunks: int,
        field_name: str,
    ) -> torch.Tensor | None:
        tensor = Trajectory._align_field_with_traj_len(tensor, traj_len, field_name)
        if tensor is None:
            return None
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        assert tensor.dim() == 3, (
            f"Expected 3D tensor for field '{field_name}', got {tensor.shape=}"
        )
        if tensor.shape[-1] == 1 and num_chunks == 1:
            return tensor
        assert tensor.shape[-1] == num_chunks, (
            f"Expected field '{field_name}' to have last dim {num_chunks}, got {tensor.shape=}"
        )
        return tensor

    @staticmethod
    def _reshape_repeated_chunk_tensor(
        tensor: torch.Tensor | None,
        *,
        traj_len: int,
        num_chunks: int,
        field_name: str,
    ) -> torch.Tensor | None:
        tensor = Trajectory._align_field_with_traj_len(tensor, traj_len, field_name)
        if tensor is None:
            return None
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        assert tensor.dim() == 3, (
            f"Expected 3D tensor for field '{field_name}', got {tensor.shape=}"
        )
        return tensor.unsqueeze(2).expand(-1, -1, num_chunks, -1)

    @staticmethod
    def _reshape_logprob_tensor(
        tensor: torch.Tensor | None,
        *,
        traj_len: int,
        num_chunks: int,
        field_name: str,
    ) -> torch.Tensor | None:
        tensor = Trajectory._align_field_with_traj_len(tensor, traj_len, field_name)
        if tensor is None:
            return None
        if tensor.dim() == 3:
            if num_chunks == 1:
                return tensor.unsqueeze(2)
            if tensor.shape[-1] == num_chunks:
                return tensor.unsqueeze(-1)
            assert tensor.shape[-1] % num_chunks == 0, (
                f"Field '{field_name}' last dim {tensor.shape[-1]} is not divisible by "
                f"{num_chunks=}"
            )
            return tensor.reshape(traj_len, tensor.shape[1], num_chunks, -1)
        assert tensor.dim() == 4 and tensor.shape[2] == num_chunks, (
            f"Expected field '{field_name}' to have shape [T, B, {num_chunks}, ...], "
            f"got {tensor.shape=}"
        )
        return tensor

    @staticmethod
    def _expand_obs_like_tensor(
        tensor: torch.Tensor | None,
        *,
        traj_len: int,
        num_chunks: int,
        field_name: str,
    ) -> torch.Tensor | None:
        tensor = Trajectory._align_field_with_traj_len(tensor, traj_len, field_name)
        if tensor is None:
            return None
        assert tensor.dim() >= 2, (
            f"Expected tensor field '{field_name}' to have at least 2 dims, got {tensor.shape=}"
        )
        return tensor.unsqueeze(2).expand(-1, -1, num_chunks, *tensor.shape[2:])

    def _extract_action_level_trajs(
        self,
        mask: torch.Tensor,
    ) -> list["Trajectory"] | None:
        mask = self._normalize_chunk_mask(mask)
        if (~mask).all():
            return None

        traj_len, batch_size, num_chunks = mask.shape

        actions = self._reshape_action_like_tensor(
            self.actions,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="actions",
        )
        intervene_flags = self._reshape_action_like_tensor(
            self.intervene_flags,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="intervene_flags",
        )
        transition_valids = self._reshape_chunk_scalar_tensor(
            self.transition_valids,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="transition_valids",
        )
        rewards = self._reshape_chunk_scalar_tensor(
            self.rewards,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="rewards",
        )
        prev_logprobs = self._reshape_logprob_tensor(
            self.prev_logprobs,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="prev_logprobs",
        )
        prev_values = self._reshape_repeated_chunk_tensor(
            self.prev_values,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="prev_values",
        )
        versions = self._reshape_repeated_chunk_tensor(
            self.versions,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="versions",
        )
        terminations = self._reshape_chunk_scalar_tensor(
            self.terminations,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="terminations",
        )
        truncations = self._reshape_chunk_scalar_tensor(
            self.truncations,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="truncations",
        )
        dones = self._reshape_chunk_scalar_tensor(
            self.dones,
            traj_len=traj_len,
            num_chunks=num_chunks,
            field_name="dones",
        )

        forward_inputs = (
            {
                key: (
                    self._reshape_action_like_tensor(
                        value,
                        traj_len=traj_len,
                        num_chunks=num_chunks,
                        field_name=f"forward_inputs.{key}",
                    )
                    if key in {"action", "model_action"}
                    else self._expand_obs_like_tensor(
                        value,
                        traj_len=traj_len,
                        num_chunks=num_chunks,
                        field_name=f"forward_inputs.{key}",
                    )
                )
                for key, value in self.forward_inputs.items()
            }
            if self.forward_inputs
            else {}
        )
        curr_obs = (
            {
                key: self._expand_obs_like_tensor(
                    value,
                    traj_len=traj_len,
                    num_chunks=num_chunks,
                    field_name=f"curr_obs.{key}",
                )
                for key, value in self.curr_obs.items()
            }
            if self.curr_obs
            else {}
        )
        next_obs = (
            {
                key: self._expand_obs_like_tensor(
                    value,
                    traj_len=traj_len,
                    num_chunks=num_chunks,
                    field_name=f"next_obs.{key}",
                )
                for key, value in self.next_obs.items()
            }
            if self.next_obs
            else {}
        )

        def select(tensor: torch.Tensor | None, batch_idx: int) -> torch.Tensor | None:
            if tensor is None:
                return None
            flat_tensor = tensor[:, batch_idx].reshape(
                traj_len * num_chunks, *tensor.shape[3:]
            )
            selected = flat_tensor[mask[:, batch_idx].reshape(-1)]
            return selected.unsqueeze(1)

        def select_dict(
            data: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            if not data:
                return {}
            return {
                key: select(value, batch_idx) for key, value in data.items() if value is not None
            }

        filtered_trajectories = []
        for batch_idx in range(batch_size):
            flat_mask = mask[:, batch_idx].reshape(-1)
            if not flat_mask.any():
                continue

            filtered_trajectories.append(
                Trajectory(
                    max_episode_length=self.max_episode_length,
                    model_weights_id=self.model_weights_id,
                    actions=select(actions, batch_idx),
                    intervene_flags=select(intervene_flags, batch_idx),
                    transition_valids=select(transition_valids, batch_idx),
                    rewards=select(rewards, batch_idx),
                    terminations=select(terminations, batch_idx),
                    truncations=select(truncations, batch_idx),
                    dones=select(dones, batch_idx),
                    prev_logprobs=select(prev_logprobs, batch_idx),
                    prev_values=select(prev_values, batch_idx),
                    versions=select(versions, batch_idx),
                    forward_inputs=select_dict(forward_inputs, batch_idx),
                    curr_obs=select_dict(curr_obs, batch_idx),
                    next_obs=select_dict(next_obs, batch_idx),
                )
            )

        return filtered_trajectories if filtered_trajectories else None

    def _extract_chunk_level_trajs(
        self,
        mask: torch.Tensor,
        *,
        require_valid_prefix: bool,
    ) -> list["Trajectory"] | None:
        mask = self._normalize_chunk_mask(mask)
        traj_len, batch_size, num_chunks = mask.shape

        chunk_keep = mask.any(dim=-1)
        if require_valid_prefix:
            valid_counts = mask.to(dtype=torch.long).sum(dim=-1)
            prefix_pattern = (
                torch.arange(num_chunks, device=mask.device).view(1, 1, -1)
                < valid_counts.unsqueeze(-1)
            )
            chunk_keep = chunk_keep & (mask == prefix_pattern).all(dim=-1)

        if (~chunk_keep).all():
            return None

        def align(
            tensor: torch.Tensor | None, field_name: str
        ) -> torch.Tensor | None:
            return self._align_field_with_traj_len(tensor, traj_len, field_name)

        actions = align(self.actions, "actions")
        intervene_flags = align(self.intervene_flags, "intervene_flags")
        transition_valids = align(self.transition_valids, "transition_valids")
        rewards = align(self.rewards, "rewards")
        prev_logprobs = align(self.prev_logprobs, "prev_logprobs")
        prev_values = align(self.prev_values, "prev_values")
        versions = align(self.versions, "versions")
        terminations = align(self.terminations, "terminations")
        truncations = align(self.truncations, "truncations")
        dones = align(self.dones, "dones")

        forward_inputs = (
            {
                key: align(value, f"forward_inputs.{key}")
                for key, value in self.forward_inputs.items()
            }
            if self.forward_inputs
            else {}
        )
        curr_obs = (
            {
                key: align(value, f"curr_obs.{key}")
                for key, value in self.curr_obs.items()
            }
            if self.curr_obs
            else {}
        )
        next_obs = (
            {
                key: align(value, f"next_obs.{key}")
                for key, value in self.next_obs.items()
            }
            if self.next_obs
            else {}
        )

        def select(tensor: torch.Tensor | None, batch_idx: int) -> torch.Tensor | None:
            if tensor is None:
                return None
            selected = tensor[:, batch_idx]
            selected = selected[chunk_keep[:, batch_idx]]
            return selected.unsqueeze(1)

        def select_dict(
            data: dict[str, torch.Tensor], batch_idx: int
        ) -> dict[str, torch.Tensor]:
            if not data:
                return {}
            return {
                key: select(value, batch_idx)
                for key, value in data.items()
                if value is not None
            }

        filtered_trajectories = []
        for batch_idx in range(batch_size):
            if not chunk_keep[:, batch_idx].any():
                continue

            filtered_trajectories.append(
                Trajectory(
                    max_episode_length=self.max_episode_length,
                    model_weights_id=self.model_weights_id,
                    actions=select(actions, batch_idx),
                    intervene_flags=select(intervene_flags, batch_idx),
                    transition_valids=select(transition_valids, batch_idx),
                    rewards=select(rewards, batch_idx),
                    terminations=select(terminations, batch_idx),
                    truncations=select(truncations, batch_idx),
                    dones=select(dones, batch_idx),
                    prev_logprobs=select(prev_logprobs, batch_idx),
                    prev_values=select(prev_values, batch_idx),
                    versions=select(versions, batch_idx),
                    forward_inputs=select_dict(forward_inputs, batch_idx),
                    curr_obs=select_dict(curr_obs, batch_idx),
                    next_obs=select_dict(next_obs, batch_idx),
                )
            )

        return filtered_trajectories if filtered_trajectories else None

    def extract_intervene_traj(self, mode="any"):
        if self.intervene_flags is None or (~self.intervene_flags).all():
            return None

        num_chunks = self._infer_num_action_chunks(
            self.transition_valids, self.rewards
        )
        flags = self._align_field_with_traj_len(
            self.intervene_flags,
            int(self.intervene_flags.shape[0]),
            "intervene_flags",
        )
        assert flags is not None

        if flags.dim() == 2:
            flags = flags.unsqueeze(-1)
        if flags.dim() == 3:
            assert flags.shape[-1] % num_chunks == 0, (
                f"intervene_flags last dim {flags.shape[-1]} is not divisible by "
                f"{num_chunks=}"
            )
            flags = flags.reshape(flags.shape[0], flags.shape[1], num_chunks, -1)
        else:
            raise AssertionError(
                f"Expected intervene_flags to be 2D or 3D, got {flags.shape=}"
            )

        if mode == "any":
            mask = flags.any(dim=-1)
        elif mode == "all":
            mask = flags.all(dim=-1)
        else:
            raise NotImplementedError(
                f"Unsupported extract_intervene_traj mode: {mode}"
            )

        mask = self._normalize_chunk_mask(mask)
        if mask.shape[-1] == 1 and mask.all():
            return [self]
        return self._extract_action_level_trajs(mask)

    def extract_valid_traj(self):
        if self.transition_valids is None:
            return [self]

        mask = self._normalize_chunk_mask(self.transition_valids)
        if (~mask).all():
            return None
        if mask.shape[-1] == 1 and mask.all():
            return [self]
        return self._extract_action_level_trajs(mask)

    def extract_valid_chunk_traj(
        self, require_valid_prefix: bool = True
    ) -> list["Trajectory"] | None:
        if self.transition_valids is None:
            return [self]

        mask = self._normalize_chunk_mask(self.transition_valids)
        if (~mask).all():
            return None
        if mask.shape[-1] == 1 and mask.all():
            return [self]
        return self._extract_chunk_level_trajs(
            mask,
            require_valid_prefix=require_valid_prefix,
        )


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    """
    Collect chunk-step results and transitions during rollout,
    and convert them into trajectory tensors.
    """

    max_episode_length: int = 0

    actions: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    intervene_flags: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length
    rewards: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    terminations: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    truncations: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    dones: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    prev_logprobs: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    prev_values: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    versions: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    forward_inputs: list[dict[str, Any]] = field(
        default_factory=list
    )  # trajectory_length

    curr_obs: list[dict[str, Any]] = field(default_factory=list)  # trajectory_length
    next_obs: list[dict[str, Any]] = field(default_factory=list)  # trajectory_length
    transition_valids: list[torch.Tensor] = field(default_factory=list)  # trajectory_length

    def append_step_result(self, result: ChunkStepResult):
        if result.actions is not None:
            self.actions.append(result.actions)
            self.intervene_flags.append(
                torch.zeros_like(result.actions, dtype=torch.bool)
            )
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.versions is not None:
            self.versions.append(result.versions)
        if result.forward_inputs:
            self.forward_inputs.append(result.forward_inputs)

    def mark_last_step_with_flags(self, save_flags: torch.Tensor):
        if not self.intervene_flags:
            return

        if save_flags.dim() == 1:
            save_flags = save_flags[:, None]
        assert save_flags.dim() == 2, f"Expected 2D tensor, got {save_flags.shape=}"

        last_action = self.actions[-1]
        bsz, num_action_chunks = save_flags.shape
        expanded_flags = save_flags.reshape(bsz, num_action_chunks, 1).expand_as(
            last_action.reshape(bsz, num_action_chunks, -1)
        )
        self.intervene_flags[-1] = expanded_flags.reshape(bsz, -1).to(torch.bool)

    def update_last_actions(
        self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor
    ):
        # action: [bsz, num-chunk-size x action-dim]
        # intervene_actions: [bsz, num-chunk-size x action-dim]
        # intervene_flags: [bsz, num-chunk-size]

        if self.actions and len(self.actions) > 0:
            last_action = self.actions[-1]
            assert last_action.dim() == 2, (
                f"Expected 2D tensor, got {last_action.shape=}"
            )
            assert intervene_actions.dim() == 2, (
                f"Expected 2D tensor, got {intervene_actions.shape=}"
            )

            # Normalize intervene_flags dimensions
            if intervene_flags.dim() == 1:
                intervene_flags = intervene_flags[:, None]
            assert intervene_flags.dim() == 2, (
                f"Expected 2D tensor, got {intervene_flags.shape=}"
            )

            bsz, num_action_chunks = intervene_flags.shape[:2]
            flags = intervene_flags.reshape(-1, num_action_chunks, 1)

            # Combine intervene_actions and last_action based on flags
            last_full_action = intervene_actions.reshape(
                bsz, num_action_chunks, -1
            ) * flags + last_action.reshape(bsz, num_action_chunks, -1) * (~flags)
            self.actions[-1] = last_full_action.reshape(bsz, -1)

            full_flags = flags.expand_as(last_full_action).reshape(bsz, -1)
            self.intervene_flags[-1] = full_flags

            if self.forward_inputs:
                last_fi = self.forward_inputs[-1]
                if "action" in last_fi:
                    last_fi["action"] = (
                        last_full_action.reshape(bsz, -1).cpu().contiguous()
                    )
                last_fi.pop("model_action", None)

    def append_transitions(self, curr_obs=None, next_obs=None, transition_valids=None):
        assert curr_obs is not None and next_obs is not None
        if "task_descriptions" in curr_obs:
            curr_obs.pop("task_descriptions")
        if "task_descriptions" in next_obs:
            next_obs.pop("task_descriptions")
        self.curr_obs.append(curr_obs)
        self.next_obs.append(next_obs)
        if transition_valids is not None:
            self.transition_valids.append(transition_valids.cpu().contiguous())

    def to_trajectory(self) -> Trajectory:
        # return [trajectory_length, B, ...]
        trajectory = Trajectory(
            max_episode_length=self.max_episode_length,
        )
        if len(self.actions) > 0:
            trajectory.actions = torch.stack(self.actions, dim=0).cpu().contiguous()
        if len(self.intervene_flags) > 0:
            trajectory.intervene_flags = (
                torch.stack(self.intervene_flags, dim=0).cpu().contiguous()
            )
        if len(self.transition_valids) > 0:
            trajectory.transition_valids = torch.stack(
                self.transition_valids, dim=0
            ).cpu().contiguous()
            if (
                trajectory.transition_valids.dim() == 3
                and trajectory.transition_valids.shape[-1] == 1
            ):
                trajectory.transition_valids = trajectory.transition_valids.squeeze(-1)
        if len(self.rewards) > 0:
            trajectory.rewards = torch.stack(self.rewards, dim=0).cpu().contiguous()
        if len(self.terminations) > 0:
            trajectory.terminations = (
                torch.stack(self.terminations, dim=0).cpu().contiguous()
            )
        if len(self.truncations) > 0:
            trajectory.truncations = (
                torch.stack(self.truncations, dim=0).cpu().contiguous()
            )
        if len(self.dones) > 0:
            trajectory.dones = torch.stack(self.dones, dim=0).cpu().contiguous()
        if len(self.prev_logprobs) > 0:
            trajectory.prev_logprobs = (
                torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
            )
        if len(self.prev_values) > 0:
            trajectory.prev_values = (
                torch.stack(self.prev_values, dim=0).cpu().contiguous()
            )
        if len(self.versions) > 0:
            trajectory.versions = torch.stack(self.versions, dim=0).cpu().contiguous()
        if len(self.forward_inputs) > 0:
            trajectory.forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
            for key in trajectory.forward_inputs.keys():
                trajectory.forward_inputs[key] = (
                    trajectory.forward_inputs[key].cpu().contiguous()
                )

        if len(self.curr_obs) > 0:
            trajectory.curr_obs = stack_list_of_dict_tensor(self.curr_obs)
            for key in trajectory.curr_obs.keys():
                trajectory.curr_obs[key] = trajectory.curr_obs[key].cpu().contiguous()
        if len(self.next_obs) > 0:
            trajectory.next_obs = stack_list_of_dict_tensor(self.next_obs)
            for key in trajectory.next_obs.keys():
                trajectory.next_obs[key] = trajectory.next_obs[key].cpu().contiguous()

        trajectory.model_weights_id = get_model_weights_id(
            trajectory.versions
            if trajectory.versions is not None
            else torch.zeros(1, dtype=torch.float32)
        )

        return trajectory

    def to_splited_trajectories(self, split_size: int) -> list[Trajectory]:
        all_trajectory: Trajectory = self.to_trajectory()
        splited_trajectories: list[Trajectory] = [
            Trajectory() for _ in range(split_size)
        ]

        if len(all_trajectory.curr_obs) > 0:
            splited_obs = split_dict_to_chunk(
                all_trajectory.curr_obs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].curr_obs = splited_obs[i]
        if len(all_trajectory.next_obs) > 0:
            splited_obs = split_dict_to_chunk(
                all_trajectory.next_obs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].next_obs = splited_obs[i]

        if (
            all_trajectory.forward_inputs is not None
            and len(all_trajectory.forward_inputs) > 0
        ):
            splited_forward_inputs = split_dict_to_chunk(
                all_trajectory.forward_inputs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].forward_inputs = splited_forward_inputs[i]

        for field_name in all_trajectory.__dataclass_fields__.keys():
            value = getattr(all_trajectory, field_name)

            if value is None or isinstance(value, dict):
                continue

            if isinstance(value, int) or isinstance(value, str):
                for i in range(split_size):
                    setattr(splited_trajectories[i], field_name, value)
                continue
            elif isinstance(value, torch.Tensor):
                chunks = torch.chunk(value, split_size, dim=1)
                for i in range(split_size):
                    setattr(splited_trajectories[i], field_name, chunks[i])
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        return splited_trajectories


def convert_trajectories_to_batch(
    trajectories: list[Trajectory],
) -> dict[str, torch.Tensor]:
    """
    convert a list of trajectories to a batch dict, the shape of the batch is [T, B, ...].
    """
    if not trajectories:
        return {}

    batch: dict[str, torch.Tensor] = {}

    # -------- obs / forward_inputs: dict[str, Tensor] --------
    if trajectories[0].curr_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.curr_obs.keys())
        batch["curr_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.curr_obs[key] for traj in trajectories if key in traj.curr_obs
            ]
            if tensors:
                batch["curr_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].next_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.next_obs.keys())
        batch["next_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.next_obs[key] for traj in trajectories if key in traj.next_obs
            ]
            if tensors:
                batch["next_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].forward_inputs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.forward_inputs.keys())
        batch["forward_inputs"] = {}
        for key in all_keys:
            tensors = [
                traj.forward_inputs[key]
                for traj in trajectories
                if key in traj.forward_inputs
            ]
            if tensors:
                batch["forward_inputs"][key] = torch.cat(tensors, dim=1)

    # -------- tensor fields --------
    reference_trajectory = trajectories[0]
    for field_name in reference_trajectory.__dataclass_fields__.keys():
        if not isinstance(getattr(reference_trajectory, field_name), torch.Tensor):
            continue
        field_list = [
            getattr(traj, field_name)
            for traj in trajectories
            if getattr(traj, field_name) is not None
        ]
        if field_list:
            batch[field_name] = torch.cat(field_list, dim=1)

    return batch
