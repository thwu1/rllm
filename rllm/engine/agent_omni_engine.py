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

from episodic import ContextSubscriber
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.proxy_manager import VerlProxyManager
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.misc import colorful_print
from rllm.sdk import RLLMClient
from rllm.workflows.workflow import TerminationReason

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from rllm.sdk.tracing import LLMTracer
    from verl import DataProto

logger = logging.getLogger(__name__)


class AgentOmniEngine:
    def __init__(self, agent_run_func: Callable, rollout_engine: RolloutEngine, config=None, n_parallel_tasks: int = 128, retry_limit: int = 3, raise_on_error: bool = True, proxy_config: dict | None = None, tracer: Optional["LLMTracer"] = None, **kwargs):
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
            tracer: Optional LLMTracer for logging.
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

        self.rllm_client = RLLMClient(
            base_url=self.proxy_manager.get_proxy_url(),
            api_key="EMPTY",
            project="rllm-agent-omni-engine",
            cs_endpoint="http://localhost:8000",
            cs_api_key="your-api-key-here",
        )

        self.context_store = self.rllm_client.get_context_store()
        self.context_subscriber = ContextSubscriber(
            context_store=self.context_store,
        )
        self.trace_idle_timeout = kwargs.get("trace_idle_timeout", 1.0)
        self.trace_wait_timeout = kwargs.get("trace_wait_timeout", 30.0)
        self.wrapped_agent_run_func = self.prepare_agent_run_func_with_tracing(self.agent_run_func)

    def prepare_agent_run_func_with_tracing(self, func):
        if inspect.iscoroutinefunction(func):

            async def wrapped_func_async(metadata, *args, **kwargs):
                with self.rllm_client.session(**metadata):
                    reward = await func(*args, **kwargs)
                return reward

            return wrapped_func_async
        else:

            def wrapped_func_sync(metadata, *args, **kwargs):
                with self.rllm_client.session(**metadata):
                    reward = func(*args, **kwargs)
                return reward

        return wrapped_func_sync

    def _setup_verl_proxy(self, proxy_config: dict, tracer: Optional["LLMTracer"]) -> None:
        """Setup LiteLLM proxy for VERL rollout engine.

        Args:
            proxy_config: Proxy configuration dict
            tracer: Optional LLMTracer instance
        """
        model_name = proxy_config.get("model_name")
        if not model_name:
            logger.warning("No model_name provided in proxy_config. Proxy manager will not be initialized. Provide proxy_config={'model_name': 'your-model'} to enable proxy.")
            return

        proxy_host = proxy_config.get("proxy_host", "127.0.0.1")
        proxy_port = proxy_config.get("proxy_port", 4000)
        auto_start = proxy_config.get("auto_start", False)
        proxy_access_log = proxy_config.get("proxy_access_log", False)
        admin_token = proxy_config.get("admin_token", "my-shared-secret")

        self.proxy_manager = VerlProxyManager(
            rollout_engine=self.rollout_engine,
            model_name=model_name,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            tracer=tracer,
            proxy_access_log=proxy_access_log,
            admin_token=admin_token,
        )

        self.rollout_engine_endpoint = self.proxy_manager.get_proxy_url()

        logger.info(f"Initialized VerlProxyManager with {len(self.proxy_manager.get_server_addresses())} vLLM replicas. Proxy endpoint: {self.rollout_engine_endpoint}")

        if auto_start:
            self.proxy_manager.start_proxy_server()
            logger.info(f"Auto-started LiteLLM proxy at {self.rollout_engine_endpoint}")
        else:
            self.proxy_manager.reload_external_proxy(
                inline_payload=True,
            )

    def get_openai_endpoint(self) -> str | None:
        """Get the OpenAI-compatible endpoint URL.

        Returns:
            URL string if proxy is configured, None otherwise.
        """
        return self.rollout_engine_endpoint

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

    async def _execute_agent_func_with_exception_handling(self, func, task, task_id, rollout_idx, attempt_idx, **kwargs):
        metadata = {"session_id": f"{task_id}:{rollout_idx}:{attempt_idx}", "task": task}
        try:
            if inspect.iscoroutinefunction(self.wrapped_agent_run_func):
                output = await func(metadata, **task, **kwargs)
                return True, output
            else:
                loop = asyncio.get_event_loop()
                bound_func = functools.partial(func, metadata, **task, **kwargs)
                output = await loop.run_in_executor(self.executor, bound_func)
                return True, output
        except Exception as e:
            return False, e

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, **kwargs) -> tuple[str, int, int, float]:
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
        agent_run_func = await self.agent_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}:{retry_attempt}"
                success, reward_or_error = await self._execute_agent_func_with_exception_handling(agent_run_func, task=task, task_id=task_id, rollout_idx=rollout_idx, attempt_idx=retry_attempt, **kwargs)
                if success:
                    colorful_print(f"[{uid}] Rollout completed with reward: {reward_or_error}", fg="green" if reward_or_error > 0 else "yellow")
                    return task_id, rollout_idx, retry_attempt, reward_or_error
                if retry_attempt < self.retry_limit:
                    print(f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts: {reward_or_error}")
        finally:
            await self.agent_queue.put(agent_run_func)

    async def start_collect_traces(self, batch_end_token: str):
        traces_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._batch_end_future: asyncio.Future | None = loop.create_future()

        @self.context_subscriber.on_context_update(namespaces=["rllm-agent-omni-engine"])
        async def add_to_queue(update):
            # Resolve future when the matching batch-end marker arrives
            if update.context.type == "trace_batch_end":
                if update.context.id == batch_end_token and self._batch_end_future:
                    print(f"Received batch end signal for token {batch_end_token}")
                    self._batch_end_future.set_result(True)

                return

            event = {
                "id": update.context.id,
                "type": update.context.type,
                "data": update.context.data,
            }
            traces_queue.put_nowait(event)

        await self.context_subscriber.start()

        return traces_queue, self._batch_end_future

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
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

        batch_end_token = f"trace-batch-end-{uuid.uuid4().hex}"
        traces_queue, end_future = await self.start_collect_traces(batch_end_token)

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

        uids_to_collect = set()
        rewards = dict()
        start_time = time.time()
        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, retry_attempt, reward = await future
                uids_to_collect.add(f"{task_id}:{rollout_idx}:{retry_attempt}")
                rewards[f"{task_id}:{rollout_idx}:{retry_attempt}"] = reward
                state = task_states[task_id]
                state["completed"] += 1
                pbar.update(1)

        print(f"Total time for generating trajectories: {time.time() - start_time:.2f} seconds")

        # Emit batch-end marker via proxy and await it
        self._current_batch_end_token = batch_end_token
        start_time = time.time()
        await self.emit_batch_end_signal(batch_end_token)
        print(f"Batch end signal emitted in {time.time() - start_time:.2f} seconds")
        start_time = time.time()

        received_batch_end = False
        try:
            await asyncio.wait_for(end_future, timeout=60.0)
            received_batch_end = True
            print(f"Batch end future resolved in {time.time() - start_time:.2f} seconds")
        except asyncio.TimeoutError:
            print("⚠️ WARNING: Batch end signal timeout after 60s, collecting available traces")

        # Drain remaining trace events without exceptions
        traces_by_session_id = {}
        for uid in uids_to_collect:
            traces_by_session_id[uid] = []

        while not traces_queue.empty():
            event = traces_queue.get_nowait()
            trace = event.get("data", {})
            session_id = trace.get("session_id", None)
            if not session_id or session_id not in uids_to_collect:
                continue
            traces_by_session_id[session_id].append(trace)

        await self.context_subscriber.stop()

        for session_id, traces in traces_by_session_id.items():
            steps = [self.convert_trace_to_step(trace) for trace in traces]
            task_id = session_id.split(":")[0]
            rollout_idx = int(session_id.split(":")[1])
            retry_attempt = int(session_id.split(":")[2])

            trajectory = Trajectory(
                uid=f"{task_id}:{rollout_idx}",
                steps=steps,
                reward=rewards[session_id],
            )
            episode = Episode(id=session_id, trajectories=[trajectory], metrics={"retry_attempt": retry_attempt, "empty": int(len(steps) == 0), "received_batch_end": int(received_batch_end)})
            task_states[task_id]["episodes"].append(episode)

        results = []
        sorted_tasks = sorted(task_states.keys(), key=lambda task_id: task_states[task_id]["idx"])
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])
        return results

    def convert_trace_to_model_output(self, trace: dict) -> ModelOutput:
        output = trace.get("output", {})
        prompt_ids = output.get("prompt_token_ids", [])
        choices = output.get("choices", [])
        content = choices[0].get("message", {}).get("content", "")
        reasoning = choices[0].get("message", {}).get("reasoning", "")
        provider_specific_fields = choices[0].get("provider_specific_fields", {})
        completion_ids = provider_specific_fields.get("token_ids", [])
        assert output, trace
        assert len(choices) == 1, "Only one choice is supported for now"
        assert prompt_ids, "Prompt IDs are required"
        assert completion_ids, "Completion IDs are required"
        return ModelOutput(
            text="",
            content=content,
            reasoning=reasoning,
            tool_calls=[],  # need fix
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=[],  # need fix
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            finish_reason=choices[0].get("finish_reason", "stop"),
        )

    def convert_trace_to_step(self, trace: dict) -> Step:
        model_output = self.convert_trace_to_model_output(trace)
        messages = trace.get("input", {}).get("messages", [])
        response_message = trace.get("output", {}).get("choices", [])[0].get("message", {})

        assert response_message
        return Step(
            chat_completions=messages + [response_message],
            model_output=model_output,
        )

    async def emit_batch_end_signal(self, token: str) -> bool:
        return await self.proxy_manager.emit_batch_end_signal(token)

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
        results = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
        self.rollout_engine.validate = False

        # ADD PRINT 4: Show what's passed to transform_results_for_verl
        print(f"Number of episodes: {len(results)}")
        print(f"Number of task_ids: {len(task_ids)}")
        # for i in range(min(10, len(results))):
        #     episode = results[i]
        #     session_id = episode.id
        #     task_id_from_episode = session_id.split(":")[0]
        #     task_id_from_array = task_ids[i] if i < len(task_ids) else "N/A"
        #     # print(f"  [{i}] episode.id={session_id}, task_ids[{i}]={task_id_from_array}, match={task_id_from_episode == task_id_from_array}")
        # print()

        if free_cache_engine:
            if isinstance(self.rollout_engine, VerlEngine):
                await self.rollout_engine.sleep()
            else:
                self.rollout_engine.sleep()
        return self.transform_results_for_verl(results, task_ids)

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
            # ADD PRINT 5: Show mapping in transform
            # if i < 10:  # Only print first 10
            #     session_id = episode.id if episode else "None"
            #     task_id_from_episode = session_id.split(":")[0] if episode and ":" in session_id else "N/A"
            #     task_id_from_array = task_ids[i] if i < len(task_ids) else "N/A"
            #     print(f"  transform[{i}]: episode.id={session_id}, task_ids[{i}]={task_id_from_array}, match={task_id_from_episode == task_id_from_array}")

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
                trajectory_id = f"{task_ids[i]}_{name}"  # unique trajectory identifier e.g., 1234567890_solver

                if len(trajectory.steps) == 0:
                    logger.info(f"Trajectory {trajectory_id} has no steps, skipping")
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
                        step_ids.append(f"{trajectory_id}_step{step_idx}")  # unique step identifier e.g., 1234567890_solver_step0

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
                "episode_ids": np.array(episode_ids),  # unique identifier for each rollout
                "trajectory_ids": np.array(trajectory_ids),  # unique identifier for each trajectory (shares prefix with task_id) and shared across rollouts
                "step_ids": np.array(step_ids),  # unique identifier for each step (shares prefix with task_id) and shared across rollouts
                "batch_ids": np.array([str(uuid.uuid4())] * len(episode_ids)),  # unique identifier for each batch
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
