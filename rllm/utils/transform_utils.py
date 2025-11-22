"""Transform utilities for converting Episodes to verl DataProto format."""

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm.parser import ChatTemplateParser
    from transformers import PreTrainedTokenizer
    from verl import DataProto

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for trajectory transformations."""
    max_prompt_length: int = 4096
    max_response_length: int = 8192
    stepwise_advantage_enable: bool = False

    @classmethod
    def from_verl_config(cls, config) -> "TransformConfig":
        return cls(
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
            stepwise_advantage_enable=config.rllm.stepwise_advantage.enable,
        )


def _pad_left(sequences: list[torch.Tensor], pad_value: int, max_length: int) -> torch.Tensor:
    """Left-pad sequences to max_length."""
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.flip(seq, dims=[0]) for seq in sequences],
        batch_first=True,
        padding_value=pad_value,
    ).flip(dims=[1])

    if padded.shape[1] < max_length:
        padding = torch.full((padded.shape[0], max_length - padded.shape[1]), pad_value, dtype=padded.dtype)
        padded = torch.cat([padding, padded], dim=1)
    elif padded.shape[1] > max_length:
        padded = padded[:, -max_length:]
    return padded


def _pad_right(sequences: list[torch.Tensor], pad_value: int, max_length: int) -> torch.Tensor:
    """Right-pad sequences to max_length."""
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_value)

    if padded.shape[1] < max_length:
        padding = torch.full((padded.shape[0], max_length - padded.shape[1]), pad_value, dtype=padded.dtype)
        padded = torch.cat([padded, padding], dim=1)
    elif padded.shape[1] > max_length:
        padded = padded[:, :max_length]
    return padded


def _tokenize_step(step: Step, chat_parser: "ChatTemplateParser") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a single step using available token IDs or chat completions."""
    from rllm.engine.rollout import ModelOutput

    # Use pre-computed token IDs if available
    if step.prompt_ids and step.response_ids:
        return (
            torch.tensor(step.prompt_ids, dtype=torch.long),
            torch.tensor(step.response_ids, dtype=torch.long),
            torch.ones(len(step.response_ids), dtype=torch.long),
        )
    if isinstance(step.model_output, ModelOutput):
        return (
            torch.tensor(step.model_output.prompt_ids, dtype=torch.long),
            torch.tensor(step.model_output.completion_ids, dtype=torch.long),
            torch.ones(len(step.model_output.completion_ids), dtype=torch.long),
        )
    # Fall back to tokenizing from chat completions
    return chat_parser.tokenize_and_mask(step.chat_completions)


def _tokenize_trajectory(
    trajectory: Trajectory,
    chat_parser: "ChatTemplateParser",
    stepwise: bool,
) -> list[dict]:
    """Convert a Trajectory to list of tensor dicts."""
    if len(trajectory.steps) == 0:
        return []

    results = []
    if stepwise:
        for step_idx, step in enumerate(trajectory.steps):
            prompt, response, mask = _tokenize_step(step, chat_parser)
            results.append({
                "prompt_ids": prompt,
                "response_ids": response,
                "response_mask": mask,
                "reward": trajectory.reward,
                "is_last_step": step_idx == len(trajectory.steps) - 1,
                "n_steps": len(trajectory.steps),
            })
    else:
        # For multi-step, use cumulative tokenization from final step
        if len(trajectory.steps) > 1:
            if not trajectory.is_cumulative():
                logger.warning(f"Multi-step trajectory {trajectory.name} is not cumulative but stepwise is disabled.")
            prompt, response, mask = chat_parser.tokenize_and_mask_cumulative(trajectory.steps[-1].chat_completions)
        else:
            prompt, response, mask = _tokenize_step(trajectory.steps[-1], chat_parser)

        results.append({
            "prompt_ids": prompt,
            "response_ids": response,
            "response_mask": mask,
            "reward": trajectory.reward,
            "is_last_step": True,
            "n_steps": 1,
        })
    return results


def episodes_to_dataproto(
    episodes: list[Episode],
    task_ids: list[str],
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
    config: TransformConfig,
) -> "DataProto":
    """Convert Episodes to verl DataProto format."""
    from verl import DataProto

    prompts, responses, traj_mask = [], [], []
    rewards, episode_ids, is_last_step, step_nums = [], [], [], []
    repeat_counts = []

    for i, episode in enumerate(episodes):
        if episode is None or all(len(t.steps) == 0 for t in episode.trajectories):
            repeat_counts.append(0)
            continue

        count = 0
        for traj in episode.trajectories:
            for step_data in _tokenize_trajectory(traj, chat_parser, config.stepwise_advantage_enable):
                prompts.append(step_data["prompt_ids"])
                responses.append(step_data["response_ids"])
                traj_mask.append(step_data["response_mask"])
                rewards.append(step_data["reward"])
                is_last_step.append(step_data["is_last_step"])
                step_nums.append(step_data["n_steps"])
                episode_ids.append(episode.id)
                count += 1
        repeat_counts.append(count)

    if not prompts:
        return DataProto.from_dict(tensors={}, non_tensors={}, meta_info={"repeat_counts": repeat_counts})

    # Pad sequences
    max_prompt = config.max_prompt_length
    max_response = config.max_response_length
    prompts_batch = _pad_left(prompts, tokenizer.pad_token_id, max_prompt)
    response_batch = _pad_right(responses, tokenizer.pad_token_id, max_response)
    traj_mask_batch = _pad_right(traj_mask, 0, max_response)

    # Compute lengths and masks
    prompt_lengths = torch.as_tensor([len(p) for p in prompts]).clamp_(min=0, max=max_prompt)
    response_lengths = torch.as_tensor([len(r) for r in responses]).clamp_(min=0, max=max_response)

    prompt_pos = torch.arange(max_prompt).unsqueeze(0)
    prompt_mask = prompt_pos >= (max_prompt - prompt_lengths.unsqueeze(1))
    resp_pos = torch.arange(max_response).unsqueeze(0)
    response_mask = resp_pos < response_lengths.unsqueeze(1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    # Place rewards at last token
    score_batch = torch.zeros(len(rewards), max_response, dtype=torch.float32)
    for i, reward in enumerate(rewards):
        resp_len = response_lengths[i].item()
        if 0 < resp_len <= max_response:
            score_batch[i, resp_len - 1] = reward

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.cat([prompts_batch, response_batch], dim=1),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_batch,
            "responses": response_batch,
            "response_mask": traj_mask_batch,
            "token_level_scores": score_batch,
        },
        non_tensors={
            "episode_ids": np.array(episode_ids),
            "is_last_step": np.array(is_last_step),
            "is_pad_step": np.array([False] * len(episode_ids)),
            "step_nums": np.array(step_nums),
        },
        meta_info={"repeat_counts": repeat_counts},
    )
