set -x

# Fix Ray permission issues by cleaning up stale sessions
# Remove old Ray session directories that may have permission issues
if [ -d "/tmp/ray" ]; then
    # Remove stale Ray session directories (older than 1 hour)
    find /tmp/ray -maxdepth 1 -type d -name "session_*" -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    # Also remove the specific stale session if it exists
    rm -rf /tmp/ray/session_2025-11-23_01-48-20_255610_1739350 2>/dev/null || true
    # Fix permissions on /tmp/ray directory
    chmod -R 755 /tmp/ray 2>/dev/null || true
fi

# Force Ray to start a local instance instead of auto-connecting to remote clusters
# Explicitly set RAY_ADDRESS to empty string to prevent auto-discovery of remote clusters
export RAY_ADDRESS=""

# Stop any local Ray instances that might be interfering
ray stop --force 2>/dev/null || true

# Kill any remaining Ray processes (more aggressive cleanup)
pkill -9 -f "ray::" 2>/dev/null || true
sleep 1

# Use a custom Ray temp directory to avoid conflicts with other users' Ray instances
# This ensures your Ray instance is completely isolated
export RAY_TMPDIR="${HOME}/.ray_tmp"
mkdir -p "${RAY_TMPDIR}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

# CUDA compatibility setup for vLLM PTX compatibility (no sudo required)
export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH
echo "Using conda-installed cuda-compat from $CONDA_PREFIX/cuda-compat"


python3 -m examples.sdk.geo3k.train_geo3k \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='geo3k' \
    trainer.experiment_name='qwen3-vl-2b-instruct' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    rllm.workflow.use_workflow=True \
    trainer.total_epochs=15 \
    rllm.sdk.proxy.host=127.0.0.1 \
    rllm.sdk.proxy.port=4000 \
    rllm.sdk.proxy.mode=subprocess \
    rllm.sdk.store.path="${HOME}/rllm-traces-geo3k.db"