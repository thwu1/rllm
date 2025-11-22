"""Utilities for the rllm package."""

from rllm.utils.compute_pass_at_k import compute_pass_at_k
from rllm.utils.episode_logger import EpisodeLogger
from rllm.utils.transform_utils import (
    TrajectoryGroup,
    TransformConfig,
    compute_attention_mask,
    compute_position_ids,
    episodes_to_dataproto,
    episodes_to_trajectory_groups,
    pad_sequence_left,
    pad_sequence_right,
    place_rewards_at_last_token,
    tokenize_step,
    tokenize_trajectory_cumulative,
    trajectories_to_dataproto,
    trajectory_groups_to_dataproto,
    trajectory_to_tensors,
)

__all__ = [
    "EpisodeLogger",
    "compute_pass_at_k",
    # Transform utilities
    "TrajectoryGroup",
    "TransformConfig",
    "episodes_to_dataproto",
    "episodes_to_trajectory_groups",
    "trajectories_to_dataproto",
    "trajectory_groups_to_dataproto",
    "trajectory_to_tensors",
    "tokenize_step",
    "tokenize_trajectory_cumulative",
    "pad_sequence_left",
    "pad_sequence_right",
    "compute_attention_mask",
    "compute_position_ids",
    "place_rewards_at_last_token",
]
