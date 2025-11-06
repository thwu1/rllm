"""Patch for VERL vLLM servers to enable token IDs instrumentation.

This module provides a patched vLLMHttpServer that automatically instruments
vLLM to return token IDs, even for vLLM < 0.10.2.

Usage:
    from rllm.engine.verl_vllm_patch import get_patched_vllm_server_class

    # Get the patched server class
    PatchedServer = get_patched_vllm_server_class()

    # Use it in VERL replica
    replica.server_class = PatchedServer
"""

import logging
from typing import Any

import ray

from rllm.engine.vllm_instrumentation import get_vllm_token_ids_support, instrument_vllm

logger = logging.getLogger(__name__)


def get_patched_vllm_server_class():
    """Get a patched vLLMHttpServer class that instruments vLLM for token IDs.

    This function dynamically creates a patched version of VERL's vLLMHttpServer
    that automatically calls instrument_vllm() before launching the server.

    Returns:
        A Ray remote class that extends vLLMHttpServerBase with instrumentation.

    Example:
        >>> PatchedServer = get_patched_vllm_server_class()
        >>> # Use in VERL replica
        >>> replica.server_class = PatchedServer
    """
    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase
    except ImportError as e:
        raise ImportError("VERL is not installed. Please install VERL to use this patch.") from e

    class InstrumentedvLLMHttpServer(vLLMHttpServerBase):
        """vLLM HTTP server with automatic token IDs instrumentation.

        This class extends VERL's vLLMHttpServerBase and automatically instruments
        vLLM to return token IDs before launching the server.
        """

        def __init__(self, *args, **kwargs):
            """Initialize the server and instrument vLLM."""
            super().__init__(*args, **kwargs)

            # Instrument vLLM if needed
            support = get_vllm_token_ids_support()
            if support == "none":
                logger.info(f"[Replica {self.replica_rank}, Node {self.node_rank}] Instrumenting vLLM to return token IDs (vLLM < 0.10.2 detected)")
                success = instrument_vllm()
                if success:
                    logger.info(f"[Replica {self.replica_rank}, Node {self.node_rank}] vLLM instrumented successfully")
                else:
                    logger.warning(f"[Replica {self.replica_rank}, Node {self.node_rank}] Failed to instrument vLLM")
            elif support == "native":
                logger.info(f"[Replica {self.replica_rank}, Node {self.node_rank}] vLLM >= 0.10.2 detected, using native token IDs support")
            elif support == "instrumented":
                logger.info(f"[Replica {self.replica_rank}, Node {self.node_rank}] vLLM already instrumented")
            else:
                logger.warning(f"[Replica {self.replica_rank}, Node {self.node_rank}] vLLM not available, cannot instrument")

    # Create Ray remote class
    return ray.remote(num_cpus=1)(InstrumentedvLLMHttpServer)


def patch_verl_replica(replica: Any, auto_instrument: bool = True) -> None:
    """Patch a VERL replica to use instrumented vLLM servers.

    This function modifies a VERL replica instance to use the patched vLLM
    server class that automatically instruments vLLM for token IDs.

    Args:
        replica: A VERL vLLMReplica instance.
        auto_instrument: Whether to automatically instrument vLLM. Default: True.

    Example:
        >>> from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
        >>> replica = vLLMReplica(...)
        >>> patch_verl_replica(replica)
        >>> # Now replica will use instrumented vLLM servers
    """
    if not auto_instrument:
        logger.info("auto_instrument=False, skipping vLLM instrumentation")
        return

    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
    except ImportError as e:
        raise ImportError("VERL is not installed. Please install VERL to use this patch.") from e

    if not isinstance(replica, vLLMReplica):
        logger.warning(f"replica is not a vLLMReplica instance (got {type(replica).__name__}), skipping patch")
        return

    # Replace server class with patched version
    PatchedServer = get_patched_vllm_server_class()
    replica.server_class = PatchedServer
    logger.info(f"Patched replica {replica.replica_rank} to use instrumented vLLM servers")


def patch_verl_rollout_manager(rollout_manager: Any, auto_instrument: bool = True) -> None:
    """Patch a VERL AgentLoopManager to use instrumented vLLM servers.

    This function patches all replicas in a VERL AgentLoopManager to use
    instrumented vLLM servers.

    Args:
        rollout_manager: A VERL AgentLoopManager instance.
        auto_instrument: Whether to automatically instrument vLLM. Default: True.

    Example:
        >>> from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        >>> manager = AgentLoopManager(...)
        >>> patch_verl_rollout_manager(manager)
        >>> # Now all replicas will use instrumented vLLM servers
    """
    if not auto_instrument:
        logger.info("auto_instrument=False, skipping vLLM instrumentation")
        return

    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
    except ImportError as e:
        raise ImportError("VERL is not installed. Please install VERL to use this patch.") from e

    if not isinstance(rollout_manager, AgentLoopManager):
        logger.warning(f"rollout_manager is not an AgentLoopManager instance (got {type(rollout_manager).__name__}), skipping patch")
        return

    # Patch all replicas
    for replica in rollout_manager.rollout_replicas:
        patch_verl_replica(replica, auto_instrument=auto_instrument)

    logger.info(f"Patched {len(rollout_manager.rollout_replicas)} replicas in AgentLoopManager to use instrumented vLLM servers")


__all__ = [
    "get_patched_vllm_server_class",
    "patch_verl_replica",
    "patch_verl_rollout_manager",
]
