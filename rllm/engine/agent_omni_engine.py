import asyncio
import functools
import inspect
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from tqdm import tqdm

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.proxy_manager import VerlProxyManager
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.misc import colorful_print
from rllm.sdk.data_process import group_steps, trace_to_step
from rllm.sdk.protocol import TrajectoryProto
from rllm.sdk.shortcuts import _session_with_id
from rllm.sdk.store.sqlite_store import SqliteTraceStore
from rllm.workflows.workflow import TerminationReason

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from rllm.sdk.tracers import EpisodicTracer
    from verl import DataProto

logger = logging.getLogger(__name__)


"""
Trajectory ID Structure Documentation
======================================

This module uses a hierarchical ID system to track tasks, episodes, trajectories, and steps:

ID Hierarchy:
-------------
1. task_id: Unique identifier for each task
   - Format: UUID string or custom identifier
   - Example: "abc123-def456-ghi789" or "task_001"
   - Scope: Identifies a unique task across all rollouts

2. episode.id (session_id): Unique identifier for each rollout attempt
   - Format: "{task_id}:{rollout_idx}:{retry_attempt}"
   - Example: "abc123:0:1" (task abc123, first rollout, first retry)
   - Scope: Identifies a specific execution attempt of a task
   - Note: In agent_workflow_engine, format is "{task_id}:{rollout_idx}" (no retry_attempt)

3. trajectory_id: Identifier for trajectories within an episode
   - Format: "{task_id}_{trajectory_name}"
   - Example: "abc123_solver", "abc123_judge"
   - Scope: Identifies a trajectory type for a task
   - Important: Does NOT contain rollout_idx or retry_attempt
   - Multiple trajectories can share the same trajectory_id across different rollouts
   - Multiple trajectories within the same episode can have the same trajectory_id
     (e.g., in solver-judge workflow, multiple solver trajectories all have "task_id_solver")

4. step_id: Identifier for individual steps within a trajectory
   - Format: "{trajectory_id}_step{step_idx}"
   - Example: "abc123_solver_step0", "abc123_judge_step1"
   - Scope: Identifies a specific step within a trajectory

Example for Solver-Judge Workflow:
-----------------------------------
Given task_id="abc123" with 2 solver trajectories and 1 judge trajectory:

Episode 1 (rollout_idx=0, retry_attempt=1):
  - episode.id = "abc123:0:1"
  - Trajectories:
    * trajectory_id = "abc123_solver" (first solver, name="solver")
    * trajectory_id = "abc123_solver" (second solver, name="solver") <- SAME ID!
    * trajectory_id = "abc123_judge" (judge, name="judge")
  - Steps (if stepwise mode enabled):
    * "abc123_solver_step0", "abc123_solver_step1", ...
    * "abc123_judge_step0", "abc123_judge_step1", ...

Episode 2 (rollout_idx=1, retry_attempt=1):
  - episode.id = "abc123:1:1"
  - Trajectories:
    * trajectory_id = "abc123_solver" <- SAME as Episode 1!
    * trajectory_id = "abc123_solver" <- SAME as Episode 1!
    * trajectory_id = "abc123_judge" <- SAME as Episode 1!

Key Observations:
-----------------
- trajectory_id shares a common prefix (task_id) across all trajectories for a task
- trajectory_id does NOT contain rollout information (rollout_idx or retry_attempt)
- Multiple rollouts of the same task will produce trajectories with identical trajectory_ids
- This design allows grouping trajectories by task and type across different rollouts
- episode.id uniquely identifies each rollout attempt and contains rollout information
"""


class AgentOmniEngine:
    def __init__(self, agent_run_func: Callable, rollout_engine: RolloutEngine, config=None, n_parallel_tasks: int = 128, retry_limit: int = 3, raise_on_error: bool = True, proxy_config: dict | None = None, tracer: Optional["EpisodicTracer"] = None, **kwargs):
        """Initialize the AgentOmniEngine.

        Args:
            agent_run_func: The agent function to run for each task.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            proxy_config: Optional dict with proxy configuration:
                - model_name: Model name to expose (required for VERL)
                - proxy_host: Host to bind proxy (default: "127.0.0.1")
                - proxy_port: Port to bind proxy (default: 4000)
                - auto_start: Whether to auto-start proxy (default: False)
                - proxy_access_log: Emit LiteLLM proxy access logs (default: False)
            tracer: Optional EpisodicTracer for logging.
            **kwargs: Additional keyword arguments.
        """
        self.rollout_engine = rollout_engine
        self.agent_run_func = agent_run_func
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)
        self.agent_queue = None

        # Initialize proxy manager for VERL engines
        self.proxy_manager: VerlProxyManager | None = None
        self.rollout_engine_endpoint: str | None = None

        proxy_config = proxy_config or {}

        if isinstance(rollout_engine, VerlEngine):
            self._setup_verl_proxy(proxy_config, tracer)
        else:
            raise NotImplementedError(f"Rollout engine type {type(rollout_engine)} not supported")

        self.wrapped_agent_run_func = self._prepare_run_func_with_tracing(self.agent_run_func)
        self.groupby_key = self.config.rllm.omni.processing.groupby_key
        self.traj_name_key = self.config.rllm.omni.processing.traj_name_key
        self.store = SqliteTraceStore(db_path=self.config.rllm.omni.store.path)

    def _prepare_run_func_with_tracing(self, func):
        """Wrap agent function with session context for tracing.

        Uses _session_with_id to set explicit session_id for internal tracking.
        """
        if inspect.iscoroutinefunction(func):

            async def wrapped_func_async(metadata, *args, **kwargs):
                session_id = metadata.pop("session_id", None)
                with _session_with_id(session_id=session_id, **metadata) as session:
                    output = await func(*args, **kwargs)
                return output, session._uid

            return wrapped_func_async
        else:

            def wrapped_func_sync(metadata, *args, **kwargs):
                session_id = metadata.pop("session_id", None)
                with _session_with_id(session_id=session_id, **metadata) as session:
                    output = func(*args, **kwargs)
                return output, session._uid

        return wrapped_func_sync

    def _setup_verl_proxy(self, proxy_config: dict, tracer: Optional["EpisodicTracer"]) -> None:
        """Setup LiteLLM proxy for VERL rollout engine.

        Args:
            proxy_config: Proxy configuration dict
            tracer: Optional EpisodicTracer instance
        """
        model_name = proxy_config.get("model_name")
        if not model_name:
            logger.warning("No model_name provided in proxy_config. Proxy manager will not be initialized. Provide proxy_config={'model_name': 'your-model'} to enable proxy.")
            return

        proxy_host = proxy_config.get("proxy_host", "127.0.0.1")
        proxy_port = proxy_config.get("proxy_port", 4000)
        auto_start = proxy_config.get("auto_start", False)
        admin_token = proxy_config.get("admin_token", "my-shared-secret")

        self.proxy_manager = VerlProxyManager(
            rollout_engine=self.rollout_engine,
            model_name=model_name,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            tracer=tracer,
            admin_token=admin_token,
            proxy_access_log=False,
        )

        self.rollout_engine_endpoint = self.proxy_manager.get_proxy_url()

        print(f"Initialized VerlProxyManager with {len(self.proxy_manager.get_server_addresses())} vLLM replicas. Proxy endpoint: {self.rollout_engine_endpoint}")

        if auto_start:
            self.proxy_manager.start_proxy_server()
            logger.info(f"Auto-started LiteLLM proxy at {self.rollout_engine_endpoint}")
        else:
            self.proxy_manager.reload_external_proxy(
                inline_payload=True,
            )

    def get_server_addresses(self) -> list[str] | None:
        """Get all vLLM server addresses (for VERL engines).

        Returns:
            List of server addresses if using VERL, None otherwise.
        """
        if self.proxy_manager:
            return self.proxy_manager.get_server_addresses()
        return None

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        if self.agent_queue is not None:
            return
        self.agent_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for _ in range(self.n_parallel_tasks):
            self.agent_queue.put_nowait(self.wrapped_agent_run_func)

    async def _execute_with_exception_handling(self, func, task, task_id, rollout_idx, attempt_idx, **kwargs):
        # Construct session_id (which becomes episode.id)
        # Format: "{task_id}:{rollout_idx}:{attempt_idx}"
        # Example: "abc123:0:1" (task abc123, first rollout, first retry attempt)
        # This uniquely identifies each rollout attempt
        metadata = {"session_id": f"{task_id}:{rollout_idx}:{attempt_idx}", "task": task}
        try:
            if inspect.iscoroutinefunction(self.wrapped_agent_run_func):
                output, session_uid = await func(metadata, **task, **kwargs)
                return True, output, session_uid
            else:
                loop = asyncio.get_event_loop()
                bound_func = functools.partial(func, metadata, **task, **kwargs)
                output, session_uid = await loop.run_in_executor(self.executor, bound_func)
                return True, output, session_uid
        except Exception as e:
            return False, e, None

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, **kwargs) -> tuple[str, int, int, float, str]:
        """Process a single task rollout with retry logic on exceptions.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            **kwargs: Additional arguments passed to the agent function.

        Returns:
            tuple[str, int, int, float]: Task ID, rollout index, retry attempt, and reward.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        func = await self.agent_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}:{retry_attempt}"
                success, output, session_uid = await self._execute_with_exception_handling(func, task=task, task_id=task_id, rollout_idx=rollout_idx, attempt_idx=retry_attempt, **kwargs)
                if success and isinstance(output, float | int | bool):
                    colorful_print(f"[{uid}] Rollout completed with reward: {float(output)}", fg="green" if float(output) > 0 else "yellow")
                    return task_id, rollout_idx, retry_attempt, float(output), session_uid
                elif success and isinstance(output, list):
                    assert all(isinstance(t, TrajectoryProto) for t in output), "Must be a list of TrajectoryProto"
                    return task_id, rollout_idx, retry_attempt, output, session_uid
                if retry_attempt < self.retry_limit:
                    print(f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts: {output}")
        finally:
            await self.agent_queue.put(func)

    async def _execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        """Run asynchronous workflow execution with retry logic for multiple tasks.

        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers. If None, UUIDs are generated.
            **kwargs: Additional arguments passed to individual task processing.

        Returns:
            list[Episode]: List of completed episodes from all tasks.
        """
        if self.agent_queue is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_states = defaultdict(lambda: {"idx": None, "task": None, "episodes": [], "completed": 0, "total_rollouts": 0, "is_complete": False})

        # Capture rollout start time BEFORE launching tasks to ensure all traces are included
        rollout_start_time = time.time()

        futures = []
        idx_counter = 0
        for task, task_id in zip(tasks, task_ids, strict=True):
            state = task_states[task_id]
            if state["idx"] is None:  # First time seeing this task_id
                state["idx"] = idx_counter
                state["task"] = task
                idx_counter += 1
            rollout_idx = state["total_rollouts"]
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, **kwargs))
            state["total_rollouts"] += 1

        rollout_ids = set()
        session_uids = set()
        outputs = dict()
        start_time = time.time()
        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, retry_attempt, output, session_uid = await future
                session_uids.add(session_uid)
                rollout_ids.add(f"{task_id}:{rollout_idx}:{retry_attempt}")
                outputs[f"{task_id}:{rollout_idx}:{retry_attempt}"] = output
                state = task_states[task_id]
                state["completed"] += 1
                pbar.update(1)

        print(f"Total time for generating trajectories: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        flush_success = await self.flush_traces(timeout=60.0)
        flush_time = time.time() - start_time

        all_traces = []
        collect_trajectory_start = time.time()
        for session_uid in session_uids:
            traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)
            all_traces.extend(traces)
        collect_sqlite_time = time.time() - collect_trajectory_start

        traces_by_session_id = {}
        for uid in rollout_ids:
            traces_by_session_id[uid] = []

        for trace in all_traces:
            session_id = trace.data.get("session_id", None)
            if not session_id or session_id not in rollout_ids:
                continue
            traces_by_session_id[session_id].append((trace.id, trace.data))

        num_traces_collected = sum(len(traces) for traces in traces_by_session_id.values())

        for session_id, traces in traces_by_session_id.items():
            steps = [trace_to_step(trace[1]) for trace in traces]
            step_id_to_step = {trace[0]: step for trace, step in zip(traces, steps, strict=False)}

            task_id = session_id.split(":")[0]
            retry_attempt = int(session_id.split(":")[2])

            output = outputs[session_id]
            if isinstance(output, float):
                trajectories = group_steps(steps, by=self.groupby_key, name_key=self.traj_name_key)
                # fill reward for each trajectory using the final reward
                for trajectory in trajectories:
                    trajectory.reward = output
                is_correct = output >= 1.0
            else:
                # assemble and assign rewards based on user provide traj_proto
                trajectories = []
                for traj_proto in output:
                    steps_no_rw = [step_id_to_step.get(step.id, None) for step in traj_proto.steps]
                    for step_proto, step in zip(traj_proto.steps, steps_no_rw, strict=True):
                        if step is None:
                            print(f"Step {step_proto.id} not found in step_id_to_step, persistant storage failed?")
                            continue
                        step.reward = step_proto.reward
                    trajectories.append(
                        Trajectory(
                            name=traj_proto.name,
                            steps=steps_no_rw,
                            reward=traj_proto.reward,
                        )
                    )
                is_correct = trajectories[-1].reward >= 1.0 if len(trajectories) > 0 else False

            # episode.id is the full session_id including retry_attempt
            episode = Episode(id=session_id, is_correct=is_correct, trajectories=trajectories, metrics={"retry_attempt": retry_attempt, "empty": int(len(steps) == 0), "flush_success": int(flush_success), "num_trajectories": len(trajectories), "traces_collected": num_traces_collected, "collect_sqlite_time": collect_sqlite_time, "flush_time": flush_time})
            task_states[task_id]["episodes"].append(episode)

        results = []
        sorted_tasks = sorted(task_states.keys(), key=lambda task_id: task_states[task_id]["idx"])
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])
        return results

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        for _ in range(3):
            results = [None] * len(tasks)
            try:
                results = await self._execute_tasks(tasks, task_ids, **kwargs)
            except Exception as e:
                print(f"Error in execute_tasks: {e}, retrying...")

            error_count = 0
            for episode in results:
                if episode is None:
                    error_count += 1
                    continue
                if episode.trajectories is None or len(episode.trajectories) == 0:
                    error_count += 1
            if error_count / len(results) > 0.01:
                print(f"Too many errors in execute_tasks: {error_count} / {len(results)} > 0.01, sleeping for 120s before retrying...")
                await asyncio.sleep(120.0)
            else:
                return results
        raise Exception("Failed to execute tasks after 3 retries")

    async def flush_traces(self, timeout: float = 30.0) -> bool:
        """Flush all traces to ensure they are persisted to storage.

        This method sends a signal to the LiteLLM proxy, which then flushes
        the tracer queue. All queued traces will be persisted before this
        method returns.

        This is useful for synchronization to ensure all traces are available
        in storage before collecting them from the database.

        Args:
            timeout: Maximum time to wait for flush operation (default: 30.0 seconds)

        Returns:
            True if flush succeeds, False otherwise

        Example:
            ```python
            # After generating trajectories, flush traces before collecting
            success = await engine.flush_traces(timeout=60.0)
            if success:
                # All traces are now persisted to storage
                traces = await collect_traces_from_database()
            ```
        """
        if not self.proxy_manager:
            logger.warning("No proxy manager available, cannot flush traces")
            return False

        logger.info("Flushing traces via proxy manager (timeout=%s)", timeout)
        success = await self.proxy_manager.flush_tracer(timeout=timeout)

        if success:
            logger.info("Successfully flushed all traces")
        else:
            logger.warning("Failed to flush traces")

        return success

    async def execute_tasks_verl(self, batch: "DataProto", **kwargs) -> "DataProto":
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
        free_cache_engine = self.config.actor_rollout_ref.rollout.free_cache_engine if self.config else False
        if free_cache_engine:
            # TODO: later probably should make the `wake_up` and `sleep` methods in base class to be async
            if isinstance(self.rollout_engine, VerlEngine):
                await self.rollout_engine.wake_up()
            else:
                self.rollout_engine.wake_up()

        if batch.meta_info.get("validate", False):
            self.rollout_engine.validate = True
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
        self.rollout_engine.validate = False

        if free_cache_engine:
            if isinstance(self.rollout_engine, VerlEngine):
                await self.rollout_engine.sleep()
            else:
                self.rollout_engine.sleep()
        return self.transform_results_for_verl(episodes, task_ids)

    def transform_results_for_verl(self, episodes: list[Episode], task_ids: np.ndarray) -> "DataProto":
        """Transform episode results into Verl-compatible DataProto format.

        Args:
            episodes: List of completed episodes from workflow execution.
            task_ids: Array of task identifiers corresponding to episodes.

        Returns:
            DataProto: Formatted data ready for Verl training pipeline.
        """
        # Local import to keep verl optional
        from verl import DataProto
        from verl.utils.torch_functional import pad_sequence_to_length

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

        for i, episode in enumerate(episodes):
            total_steps = 0

            if episode is None:
                print(f"Episode {i} is None (failed task), dropping it from the batch")
                repeat_counts.append(0)
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                # termination hits before an agent finishes it's first step
                # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
                # we delete the episode from the batch by setting repeat_counts to 0
                print(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
                repeat_counts.append(0)
                continue

            for trajectory in episode.trajectories:
                name = trajectory.name
                # Construct trajectory_id from task_id and trajectory name
                # Format: "{task_id}_{trajectory_name}"
                # Example: "abc123_solver", "abc123_judge"
                #
                # IMPORTANT: trajectory_id does NOT contain rollout_idx or retry_attempt!
                # This means:
                # - Multiple rollouts of the same task will have identical trajectory_ids
                # - Multiple trajectories in the same episode can have the same trajectory_id
                #   (e.g., in solver-judge workflow with 2 solvers, both have "task_id_solver")
                # - trajectory_id shares a common prefix (task_id) for all trajectories of a task
                #
                # The rollout information is only in episode.id (format: "task_id:rollout_idx:retry_attempt")
                trajectory_id = f"{task_ids[i]}_{name}"  # e.g., "1234567890_solver"

                if len(trajectory.steps) == 0:
                    print(f"Trajectory {trajectory_id} has no steps, skipping")
                    continue

                # if not self.config.rllm.stepwise_advantage.enable:
                #     if len(trajectory.steps) > 1:
                #         if not trajectory.is_cumulative():
                #             logger.warning(f"Warning: Multi-step trajectory {trajectory_id} is not cumulative, but stepwise mode is not enabled. There could be a token mismatch during trajectory generation.")

                #         chat_completions = trajectory.steps[-1].chat_completions
                #         prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask_cumulative(chat_completions)
                #         prompts.append(prompt)
                #         responses.append(response)
                #         traj_mask.append(mask)

                #     elif isinstance(trajectory.steps[0].model_output, ModelOutput):
                #         step = trajectory.steps[0]

                #         prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
                #         prompts.append(prompt_ids)

                #         response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                #         responses.append(response_ids)

                #         mask = torch.ones_like(response_ids, dtype=torch.long)
                #         traj_mask.append(mask)

                #     else:
                #         chat_completions = trajectory.steps[0].chat_completions
                #         prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions)
                #         prompts.append(prompt)
                #         responses.append(response)
                #         traj_mask.append(mask)

                #     step_rewards.append(trajectory.reward)
                #     step_ids.append(trajectory_id)
                #     n_steps = 1

                # else:
                # TODO: auto merge the steps if they share some prefix
                for step_idx, step in enumerate(trajectory.steps):
                    if isinstance(step.model_output, ModelOutput):
                        prompt_ids = torch.tensor(step.model_output.prompt_ids, dtype=torch.long)
                        prompts.append(prompt_ids)

                        response_ids = torch.tensor(step.model_output.completion_ids, dtype=torch.long)
                        responses.append(response_ids)

                        mask = torch.ones_like(response_ids, dtype=torch.long)
                        traj_mask.append(mask)

                        # else:
                        #     chat_completions = step.chat_completions
                        #     prompt, response, mask = self.rollout_engine.chat_parser.tokenize_and_mask(chat_completions)
                        #     prompts.append(prompt)
                        #     responses.append(response)
                        #     traj_mask.append(mask)

                        step_rewards.append(step.reward)
                        # Construct step_id from trajectory_id and step index
                        # Format: "{trajectory_id}_step{step_idx}"
                        # Example: "abc123_solver_step0", "abc123_judge_step1"
                        # Since trajectory_id doesn't contain rollout info, step_id doesn't either
                        step_ids.append(f"{trajectory_id}_step{step_idx}")  # e.g., "1234567890_solver_step0"

                    n_steps = len(trajectory.steps)

                trajectory_ids.extend([trajectory_id] * n_steps)
                step_nums.extend([n_steps] * n_steps)
                traj_rewards.extend([trajectory.reward] * n_steps)
                is_last_step.extend([False] * n_steps)
                is_last_step[-1] = True
                total_steps += n_steps

            episode_ids.extend([episode.id] * total_steps)
            is_correct.extend([episode.is_correct] * total_steps)
            termination_reasons.extend([episode.termination_reason if episode.termination_reason is not None else TerminationReason.UNKNOWN] * total_steps)
            metrics.extend([episode.metrics] * total_steps)
            repeat_counts.append(total_steps)

        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in prompts],
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        ).flip(dims=[1])
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.rollout_engine.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]  # truncate if necessary

        response_batch = torch.nn.utils.rnn.pad_sequence(
            responses,
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        )
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.rollout_engine.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]  # truncate if necessary

        input_ids = torch.concat([prompts_batch, response_batch], dim=1)

        prompt_lengths = torch.as_tensor([len(t) for t in prompts]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in responses]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        traj_mask = torch.nn.utils.rnn.pad_sequence(traj_mask, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]  # truncate if necessary

        # Place all rewards to last response token of the last_step response
        traj_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        step_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, (traj_reward, step_reward) in enumerate(zip(traj_rewards, step_rewards, strict=False)):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= traj_rewards_batch.shape[1]:
                traj_rewards_batch[i, resp_len - 1] = traj_reward
                step_rewards_batch[i, resp_len - 1] = step_reward

        # compact filtering
        cf = self.config.rllm.compact_filtering
        is_valid = [True] * len(episode_ids)
        if cf.enable:
            for i in range(len(episode_ids)):
                termination_reason = termination_reasons[i]
                if (cf.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED) or (cf.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) or (cf.mask_env_done and termination_reason == TerminationReason.ENV_DONE) or (cf.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED) or (cf.mask_timeout and termination_reason == TerminationReason.TIMEOUT) or (cf.mask_unknown and termination_reason == TerminationReason.UNKNOWN) or (cf.mask_error and termination_reason == TerminationReason.ERROR):
                    is_valid[i] = False  # set flag to filter out the episode later (after advantages are computed)

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_batch,
                "responses": response_batch,
                "response_mask": traj_mask,
                "traj_rewards": traj_rewards_batch,
                "step_rewards": step_rewards_batch,
            },
            non_tensors={
                # episode_ids: Format "task_id:rollout_idx:retry_attempt" (e.g., "abc123:0:1")
                # Uniquely identifies each rollout attempt
                "episode_ids": np.array(episode_ids),
                # trajectory_ids: Format "task_id_trajectory_name" (e.g., "abc123_solver")
                # Does NOT contain rollout_idx - shared across rollouts of the same task
                # Multiple trajectories can have the same trajectory_id
                "trajectory_ids": np.array(trajectory_ids),
                # step_ids: Format "task_id_trajectory_name_step{idx}" (e.g., "abc123_solver_step0")
                # Does NOT contain rollout_idx - shared across rollouts of the same task
                "step_ids": np.array(step_ids),
                # batch_ids: Unique identifier for each training batch
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

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
