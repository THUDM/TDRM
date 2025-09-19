TORCH_COMPILE=1
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
mkdir -p logs_grpo

LR=${1:-1e-6}
batch_size=${2:-8}
gradient_accumulation_steps=${3:-1}
num_epochs=${4:-1}
run_name=${5:-"qwen25-7b-level3-ours-without-rule-"}

MODEL_PATH=/path/to/your/base/model
REWARD_MODEL_PATH=/path/to/your/reward/model

EXPNAME=qwen2.5-7b-baseline-prm-level3-lr$LR-bs$batch_size-accum$gradient_accumulation_steps-epoch$num_epochs-$TIMESTAMP

ARGS="train_grpo_zero.py \
    --model_name $MODEL_PATH \
    --reward_model_name $REWARD_MODEL_PATH \
    --train_path ./data/grpo_data_level3.json \
    --num_generations 7 \
    --learning_rate $LR \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --max_completion_length 2048 \
    --optim paged_adamw_32bit \
    --lr_scheduler_type cosine \
    --bf16 \
    --warmup_ratio 0.03 \
    --num_train_epochs $num_epochs \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 40 \
    --save_only_model \
    --report_to none \
    --output_dir runs_grpo/ours-without-rule/$EXPNAME \
    --use_vllm \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.6 \
    --log_completions True \
    --deepspeed /path/to/TDRM/trl/configs/ds_config_zero3.json \
    --wandb_run_name $run_name"

run_cmd="torchrun --nnodes=1 --nproc_per_node=7 --node_rank=0 $ARGS"
echo $run_cmd
LOG_PATH=logs_grpo/${EXPNAME}
mkdir -p $LOG_PATH
eval ${run_cmd} 2>&1 | tee ${LOG_PATH}/output_${MLP_ROLE_INDEX}.log