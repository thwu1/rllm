"""
Transform utilities for converting Episodes/Trajectories to training-ready formats.

This module provides shared utilities for transforming agent outputs (Episodes, Trajectories)
into formats required by trainers (verl DataProto or tinker Datum).

Data Flow Architecture
======================

All engines produce Episodes or Trajectories, which can then be transformed:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                              ENGINES                                    │
    │  AgentWorkflowEngine, AgentExecutionEngine, TinkerEngine                │
    │  Returns: list[Episode] with trajectories containing Steps              │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        TrajectoryGroup                                  │
    │  Intermediate grouping for advantage computation                        │
    │  Groups trajectories by task_id, trajectory_name, or step_idx           │
    └────────┬──────────────────────────────────────────────────┬─────────────┘
             │                                                  │
             ▼                                                  ▼
    ┌────────────────────────┐                    ┌────────────────────────────┐
    │   VERL TRAINERS        │                    │   TINKER TRAINERS          │
    │   episodes_to_dataproto│                    │   process_trajectory_groups│
    │   → DataProto          │                    │   → tinker.Datum           │
    └────────────────────────┘                    └────────────────────────────┘

Key Classes:
- Episode: Contains trajectories for a single task rollout
- Trajectory: Contains Steps for a single agent's execution
- Step: Single model interaction with prompt_ids, response_ids, logprobs
- TrajectoryGroup: Groups trajectories for advantage computation (shared with tinker)

For verl: Use episodes_to_dataproto() to convert Episodes → DataProto
For tinker: Use rllm.trainer.tinker.tinker_data_processor.process_episodes()
"""

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm.engine.rollout import ModelOutput
    from rllm.parser import ChatTemplateParser
    from transformers import PreTrainedTokenizer
    from verl import DataProto


# Re-export TrajectoryGroup from tinker_data_processor for consistency
# This ensures a single source of truth for the grouping abstraction
try:
    from rllm.trainer.tinker.tinker_data_processor import TrajectoryGroup
except ImportError:
    # Fallback definition if tinker is not available

    @dataclass
    class TrajectoryGroup:
        """A group of trajectories for advantage computation.

        Unlike Episode (which represents raw rollout data), TrajectoryGroup is specifically
        structured for advantage computation. All trajectories in a group will have their
        rewards compared to compute advantages (e.g., via GRPO).

        Note: This is a fallback. Prefer importing from tinker_data_processor when available.

        Attributes:
            trajectories: List of trajectories to compare for advantage computation
            group_id: Optional identifier for the group (e.g., "task1:agent_0")
        """

        trajectories: list[Trajectory] = field(default_factory=list)
        group_id: str = None


def episodes_to_trajectory_groups(
    episodes: list[Episode],
    grouping_level: str = "episode",
) -> list[TrajectoryGroup]:
    """Convert Episodes to TrajectoryGroups for advantage computation.

    This function supports different grouping strategies:
    - "episode": All trajectories in an episode form one group (default)
    - "trajectory": Group by (task_id, trajectory_name) across episodes
    - "step": Group individual steps at same position (for step-level advantage)

    Args:
        episodes: List of Episode objects from engine execution
        grouping_level: How to group trajectories ("episode", "trajectory", or "step")

    Returns:
        List of TrajectoryGroup objects for advantage computation
    """
    from collections import defaultdict

    trajectory_groups_dict = defaultdict(list)

    def get_task_id(episode: Episode) -> str:
        """Extract task_id from episode.id (format: task_id:rollout_idx)"""
        return ":".join(episode.id.split(":")[:-1]) if ":" in episode.id else episode.id

    if grouping_level == "trajectory":
        # Group by (task_id, trajectory_name) - for multi-agent workflows like solver-judge
        for episode in episodes:
            task_id = get_task_id(episode)
            for trajectory in episode.trajectories:
                group_key = (task_id, trajectory.name)
                trajectory_groups_dict[group_key].append(trajectory)

    elif grouping_level == "step":
        # Group by (task_id, trajectory_name, step_idx) - for step-level advantages
        for episode in episodes:
            task_id = get_task_id(episode)
            for trajectory in episode.trajectories:
                for step_idx, step in enumerate(trajectory.steps):
                    group_key = (task_id, trajectory.name, step_idx)
                    # Create single-step trajectory
                    single_step_traj = Trajectory(steps=[step], reward=step.reward, name=trajectory.name)
                    trajectory_groups_dict[group_key].append(single_step_traj)

    else:  # "episode" or default
        # Simple grouping: all trajectories in an episode form one group
        for episode in episodes:
            group_key = episode.id
            trajectory_groups_dict[group_key].extend(episode.trajectories)

    # Convert dict to list of TrajectoryGroup objects
    return [
        TrajectoryGroup(trajectories=trajs, group_id=str(key))
        for key, trajs in trajectory_groups_dict.items()
    ]


@dataclass
class TransformConfig:
    """Configuration for trajectory transformations.

    Attributes:
        max_prompt_length: Maximum number of tokens for prompts (left-padded)
        max_response_length: Maximum number of tokens for responses (right-padded)
        stepwise_advantage_enable: Whether to enable step-level advantage
        stepwise_advantage_mode: Mode for step-level advantage ("per_step" or "broadcast")
        compact_filtering: Config for filtering trajectories based on termination reason
    """
    max_prompt_length: int = 4096
    max_response_length: int = 8192
    stepwise_advantage_enable: bool = False
    stepwise_advantage_mode: str = "broadcast"  # "per_step" or "broadcast"
    compact_filtering: dict = field(default_factory=dict)

    @classmethod
    def from_verl_config(cls, config) -> "TransformConfig":
        """Create TransformConfig from verl OmegaConf config."""
        cf = config.rllm.get("compact_filtering", {})
        return cls(
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
            stepwise_advantage_enable=config.rllm.stepwise_advantage.enable,
            stepwise_advantage_mode=config.rllm.stepwise_advantage.mode,
            compact_filtering={
                "enable": cf.get("enable", False),
                "mask_max_prompt_length_exceeded": cf.get("mask_max_prompt_length_exceeded", False),
                "mask_max_response_length_exceeded": cf.get("mask_max_response_length_exceeded", False),
                "mask_env_done": cf.get("mask_env_done", False),
                "mask_max_turns_exceeded": cf.get("mask_max_turns_exceeded", False),
                "mask_timeout": cf.get("mask_timeout", False),
                "mask_unknown": cf.get("mask_unknown", False),
                "mask_error": cf.get("mask_error", False),
            }
        )


def pad_sequence_left(sequences: list[torch.Tensor], pad_value: int, max_length: int) -> torch.Tensor:
    """Left-pad sequences to max_length.

    Args:
        sequences: List of 1D tensors to pad
        pad_value: Value to use for padding
        max_length: Target length after padding

    Returns:
        Batched tensor of shape (batch_size, max_length)
    """
    # Flip, pad, flip back for left padding
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.flip(seq, dims=[0]) for seq in sequences],
        batch_first=True,
        padding_value=pad_value,
    ).flip(dims=[1])

    # Ensure we have exactly max_length
    if padded.shape[1] < max_length:
        padding = torch.full((padded.shape[0], max_length - padded.shape[1]), pad_value, dtype=padded.dtype)
        padded = torch.cat([padding, padded], dim=1)
    elif padded.shape[1] > max_length:
        padded = padded[:, -max_length:]

    return padded


def pad_sequence_right(sequences: list[torch.Tensor], pad_value: int, max_length: int) -> torch.Tensor:
    """Right-pad sequences to max_length.

    Args:
        sequences: List of 1D tensors to pad
        pad_value: Value to use for padding
        max_length: Target length after padding

    Returns:
        Batched tensor of shape (batch_size, max_length)
    """
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=pad_value,
    )

    # Ensure we have exactly max_length
    if padded.shape[1] < max_length:
        padding = torch.full((padded.shape[0], max_length - padded.shape[1]), pad_value, dtype=padded.dtype)
        padded = torch.cat([padded, padding], dim=1)
    elif padded.shape[1] > max_length:
        padded = padded[:, :max_length]

    return padded


def compute_attention_mask(
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    max_prompt_length: int,
    max_response_length: int,
) -> torch.Tensor:
    """Compute attention mask for prompt + response sequences.

    Args:
        prompt_lengths: Tensor of original prompt lengths
        response_lengths: Tensor of original response lengths
        max_prompt_length: Maximum prompt length after padding
        max_response_length: Maximum response length after padding

    Returns:
        Attention mask tensor of shape (batch_size, max_prompt_length + max_response_length)
    """
    # Prompt mask (left-padded, so mask = pos >= (max_length - actual_length))
    prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
    prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

    # Response mask (right-padded, so mask = pos < actual_length)
    resp_pos = torch.arange(max_response_length).unsqueeze(0)
    response_mask = resp_pos < response_lengths.unsqueeze(1)

    return torch.cat([prompt_mask, response_mask], dim=1).long()


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute position IDs from attention mask.

    Args:
        attention_mask: Attention mask tensor

    Returns:
        Position IDs tensor (cumsum of attention mask, masked)
    """
    return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask


def place_rewards_at_last_token(
    rewards: list[float],
    response_lengths: torch.Tensor,
    max_response_length: int,
) -> torch.Tensor:
    """Place rewards at the last valid token of each response.

    Args:
        rewards: List of reward values
        response_lengths: Tensor of response lengths
        max_response_length: Maximum response length

    Returns:
        Reward tensor of shape (batch_size, max_response_length)
    """
    batch_size = len(rewards)
    reward_tensor = torch.zeros(batch_size, max_response_length, dtype=torch.float32)

    for i, reward in enumerate(rewards):
        resp_len = response_lengths[i].item()
        if resp_len > 0 and resp_len <= max_response_length:
            reward_tensor[i, resp_len - 1] = reward

    return reward_tensor


def tokenize_step(
    step: Step,
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float] | None]:
    """Tokenize a single step into prompt_ids, response_ids, response_mask, and optionally logprobs.

    Args:
        step: The Step object to tokenize
        tokenizer: The tokenizer to use
        chat_parser: The chat template parser

    Returns:
        Tuple of (prompt_ids, response_ids, response_mask, logprobs) tensors.
        logprobs may be None if not available.
    """
    from rllm.engine.rollout import ModelOutput

    logprobs = None

    # Priority 1: Use Step's direct prompt_ids/response_ids if available (otel branch)
    if step.prompt_ids and step.response_ids:
        prompt_ids = torch.tensor(step.prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(step.response_ids, dtype=torch.long)
        response_mask = torch.ones_like(response_ids, dtype=torch.long)
        logprobs = step.logprobs if step.logprobs else None
    # Priority 2: Use model_output if available
    elif isinstance(step.model_output, ModelOutput):
        prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
        response_mask = torch.ones_like(response_ids, dtype=torch.long)
        logprobs = step.model_output.logprobs if step.model_output.logprobs else None
    # Priority 3: Tokenize from chat completions
    else:
        chat_completions = step.chat_completions
        prompt_ids, response_ids, response_mask = chat_parser.tokenize_and_mask(chat_completions)

    return prompt_ids, response_ids, response_mask, logprobs


def tokenize_trajectory_cumulative(
    trajectory: Trajectory,
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a cumulative trajectory (where steps build on each other).

    For cumulative trajectories, we use the final step's chat_completions
    which contains the full conversation.

    Args:
        trajectory: The Trajectory object to tokenize
        tokenizer: The tokenizer to use
        chat_parser: The chat template parser

    Returns:
        Tuple of (prompt_ids, response_ids, response_mask) tensors
    """
    if len(trajectory.steps) == 0:
        raise ValueError("Cannot tokenize empty trajectory")

    final_step = trajectory.steps[-1]
    chat_completions = final_step.chat_completions
    prompt, response, mask = chat_parser.tokenize_and_mask_cumulative(chat_completions)
    return prompt, response, mask


def trajectory_to_tensors(
    trajectory: Trajectory,
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
    stepwise: bool = False,
    trajectory_id: str = None,
) -> list[dict]:
    """Convert a Trajectory to tensor representations.

    This function maintains compatibility with the original transform_results_for_verl logic:
    - For non-stepwise multi-step trajectories: always use cumulative tokenization
    - For non-stepwise single-step: use step's token IDs or tokenize from chat completions
    - For stepwise: process each step independently

    Args:
        trajectory: The Trajectory to convert
        tokenizer: The tokenizer to use
        chat_parser: The chat template parser
        stepwise: If True, return one dict per step; otherwise return one dict for the trajectory
        trajectory_id: Optional trajectory ID for logging warnings

    Returns:
        List of dicts containing prompt_ids, response_ids, response_mask, reward, and optionally logprobs
    """
    import logging

    logger = logging.getLogger(__name__)
    results = []

    if not stepwise:
        # Single trajectory mode - matches original transform_results_for_verl behavior
        if len(trajectory.steps) == 0:
            return results

        logprobs = None
        if len(trajectory.steps) > 1:
            # For multi-step trajectories, always use cumulative tokenization
            # (this matches the original behavior)
            if not trajectory.is_cumulative():
                logger.warning(
                    f"Warning: Multi-step trajectory {trajectory_id or trajectory.name} is not cumulative, "
                    "but stepwise mode is not enabled. There could be a token mismatch during trajectory generation."
                )
            prompt, response, mask = tokenize_trajectory_cumulative(trajectory, tokenizer, chat_parser)
        else:
            # Single step - use tokenize_step which handles model_output and chat_completions
            prompt, response, mask, logprobs = tokenize_step(trajectory.steps[-1], tokenizer, chat_parser)

        results.append({
            "prompt_ids": prompt,
            "response_ids": response,
            "response_mask": mask,
            "logprobs": logprobs,
            "reward": trajectory.reward,
            "step_reward": trajectory.reward,
            "trajectory_name": trajectory.name,
            "trajectory_uid": trajectory.uid,
            "is_last_step": True,
            "step_idx": 0,
            "n_steps": 1,
        })
    else:
        # Stepwise mode - return one dict per step
        for step_idx, step in enumerate(trajectory.steps):
            prompt, response, mask, logprobs = tokenize_step(step, tokenizer, chat_parser)
            results.append({
                "prompt_ids": prompt,
                "response_ids": response,
                "response_mask": mask,
                "logprobs": logprobs,
                "reward": trajectory.reward,  # Trajectory reward assigned to all steps
                "step_reward": step.reward,
                "trajectory_name": trajectory.name,
                "trajectory_uid": trajectory.uid,
                "is_last_step": step_idx == len(trajectory.steps) - 1,
                "step_idx": step_idx,
                "n_steps": len(trajectory.steps),
            })

    return results


def episodes_to_dataproto(
    episodes: list[Episode],
    task_ids: list[str],
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
    config: TransformConfig,
) -> "DataProto":
    """Convert a list of Episodes to verl DataProto format.

    This is the main transformation function that handles:
    - Tokenization of prompts and responses
    - Padding to max lengths
    - Computing attention masks and position IDs
    - Placing rewards at the correct positions
    - Handling stepwise vs trajectory-level processing
    - Compact filtering based on termination reasons

    Args:
        episodes: List of Episode objects from engine
        task_ids: List of task IDs corresponding to episodes
        tokenizer: The tokenizer to use
        chat_parser: The chat template parser
        config: Transform configuration

    Returns:
        DataProto ready for trainer
    """
    from verl import DataProto

    # Collect all data from episodes
    prompts = []
    responses = []
    traj_rewards = []
    step_rewards = []
    episode_ids = []
    trajectory_ids = []
    step_ids = []
    step_nums = []
    repeat_counts = []
    is_last_step = []
    is_correct = []
    traj_mask = []
    termination_reasons = []
    metrics = []

    stepwise = config.stepwise_advantage_enable

    for i, episode in enumerate(episodes):
        total_steps = 0

        if episode is None:
            print(f"Episode {i} is None (failed task), dropping it from the batch")
            repeat_counts.append(0)
            continue

        if all(len(traj.steps) == 0 for traj in episode.trajectories):
            print(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
            repeat_counts.append(0)
            continue

        for trajectory in episode.trajectories:
            trajectory_id = f"{task_ids[i]}_{trajectory.name}"

            if len(trajectory.steps) == 0:
                continue

            step_tensors = trajectory_to_tensors(
                trajectory,
                tokenizer,
                chat_parser,
                stepwise=stepwise,
                trajectory_id=trajectory_id,
            )

            for step_data in step_tensors:
                prompts.append(step_data["prompt_ids"])
                responses.append(step_data["response_ids"])
                traj_mask.append(step_data["response_mask"])
                traj_rewards.append(trajectory.reward)
                step_rewards.append(step_data["step_reward"])

                step_id = f"{trajectory_id}_step{step_data['step_idx']}" if stepwise else trajectory_id
                step_ids.append(step_id)
                trajectory_ids.append(trajectory_id)
                step_nums.append(step_data["n_steps"])
                is_last_step.append(step_data["is_last_step"])

                total_steps += 1

        episode_ids.extend([episode.id] * total_steps)
        is_correct.extend([episode.is_correct] * total_steps)
        termination_reasons.extend([
            episode.termination_reason if episode.termination_reason is not None
            else TerminationReason.UNKNOWN
        ] * total_steps)
        metrics.extend([episode.metrics] * total_steps)
        repeat_counts.append(total_steps)

    if len(prompts) == 0:
        # Return empty DataProto
        return DataProto.from_dict(
            tensors={},
            non_tensors={},
            meta_info={"repeat_counts": repeat_counts}
        )

    # Pad prompts (left-pad) and responses (right-pad)
    prompts_batch = pad_sequence_left(prompts, tokenizer.pad_token_id, config.max_prompt_length)
    response_batch = pad_sequence_right(responses, tokenizer.pad_token_id, config.max_response_length)

    # Compute input_ids, attention_mask, position_ids
    input_ids = torch.cat([prompts_batch, response_batch], dim=1)

    prompt_lengths = torch.as_tensor([len(p) for p in prompts]).clamp_(min=0, max=config.max_prompt_length)
    response_lengths = torch.as_tensor([len(r) for r in responses]).clamp_(min=0, max=config.max_response_length)

    attention_mask = compute_attention_mask(
        prompt_lengths, response_lengths,
        config.max_prompt_length, config.max_response_length
    )
    position_ids = compute_position_ids(attention_mask)

    # Pad response masks
    traj_mask_batch = pad_sequence_right(traj_mask, 0, config.max_response_length)

    # Place rewards at last token
    traj_rewards_batch = place_rewards_at_last_token(
        traj_rewards, response_lengths, config.max_response_length
    )
    step_rewards_batch = place_rewards_at_last_token(
        step_rewards, response_lengths, config.max_response_length
    )

    # Compact filtering
    is_valid = [True] * len(episode_ids)
    cf = config.compact_filtering
    if cf.get("enable", False):
        for i in range(len(episode_ids)):
            reason = termination_reasons[i]
            if ((cf.get("mask_max_prompt_length_exceeded") and reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED) or
                (cf.get("mask_max_response_length_exceeded") and reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) or
                (cf.get("mask_env_done") and reason == TerminationReason.ENV_DONE) or
                (cf.get("mask_max_turns_exceeded") and reason == TerminationReason.MAX_TURNS_EXCEEDED) or
                (cf.get("mask_timeout") and reason == TerminationReason.TIMEOUT) or
                (cf.get("mask_unknown") and reason == TerminationReason.UNKNOWN) or
                (cf.get("mask_error") and reason == TerminationReason.ERROR)):
                is_valid[i] = False

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_batch,
            "responses": response_batch,
            "response_mask": traj_mask_batch,
            "traj_rewards": traj_rewards_batch,
            "step_rewards": step_rewards_batch,
        },
        non_tensors={
            "episode_ids": np.array(episode_ids),
            "trajectory_ids": np.array(trajectory_ids),
            "step_ids": np.array(step_ids),
            "batch_ids": np.array([str(uuid.uuid4())] * len(episode_ids)),
            "step_nums": np.array(step_nums),
            "is_correct": np.array(is_correct),
            "termination_reasons": np.array([x.value for x in termination_reasons]),
            "metrics": np.array(metrics),
            "is_valid": np.array(is_valid),
            "is_last_step": np.array(is_last_step),
            "is_pad_step": np.array([False] * len(episode_ids)),
        },
        meta_info={
            "repeat_counts": repeat_counts,
        },
    )


def trajectory_groups_to_dataproto(
    groups: list[TrajectoryGroup],
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
    config: TransformConfig,
) -> "DataProto":
    """Convert a list of TrajectoryGroups to verl DataProto format.

    This is a convenience wrapper around episodes_to_dataproto that first
    converts TrajectoryGroups to Episodes.

    Args:
        groups: List of TrajectoryGroup objects
        tokenizer: The tokenizer to use
        chat_parser: The chat template parser
        config: Transform configuration

    Returns:
        DataProto ready for trainer
    """
    episodes = [group.to_episode() for group in groups]
    task_ids = [group.task_id for group in groups]
    return episodes_to_dataproto(episodes, task_ids, tokenizer, chat_parser, config)


def trajectories_to_dataproto(
    trajectories: list[Trajectory],
    task_ids: list[str],
    tokenizer: "PreTrainedTokenizer",
    chat_parser: "ChatTemplateParser",
    config: TransformConfig,
    is_correct: list[bool] = None,
    termination_reasons: list[TerminationReason] = None,
) -> "DataProto":
    """Convert a list of Trajectories to verl DataProto format.

    This wraps each Trajectory in an Episode and uses episodes_to_dataproto.

    Args:
        trajectories: List of Trajectory objects
        task_ids: List of task IDs
        tokenizer: The tokenizer to use
        chat_parser: The chat template parser
        config: Transform configuration
        is_correct: Optional list of correctness flags
        termination_reasons: Optional list of termination reasons

    Returns:
        DataProto ready for trainer
    """
    episodes = []
    for i, traj in enumerate(trajectories):
        episode = Episode(
            id=task_ids[i] if task_ids else str(uuid.uuid4()),
            trajectories=[traj],
            is_correct=is_correct[i] if is_correct else traj.reward > 0,
            termination_reason=termination_reasons[i] if termination_reasons else TerminationReason.UNKNOWN,
        )
        episodes.append(episode)

    return episodes_to_dataproto(episodes, task_ids, tokenizer, chat_parser, config)
