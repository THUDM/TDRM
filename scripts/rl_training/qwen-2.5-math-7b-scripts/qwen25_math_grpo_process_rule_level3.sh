TORCH_COMPILE=1
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
mkdir -p logs_grpo

LR=${1:-1e-6}
batch_size=${2:-8}
gradient_accumulation_steps=${3:-1}
num_epochs=${4:-1}

method=td2
level=level3+4
run_name=${5:-"qwen2.5-7b"}
data_path=./data/grpo_data_level3.json
echo "Data path: $data_path"

MODEL_PATH=/path/to/your/base/model
REWARD_MODEL_PATH=/path/to/your/reward/model


if [[ ! "$run_name" =~ "$method" ]]; then
    echo "Error: '$method' is not in run_name!"
    exit 1
fi

if [[ ! "$REWARD_MODEL_PATH" =~ "$method" ]]; then
    echo "Error: '$method' is not in REWARD_MODEL_PATH!"
    exit 1
fi

if [[ ! "$data_path" =~ "$level" ]]; then
    echo "Error: '$level' is not in run_name!"
    exit 1
fi

if [[ ! "$run_name" =~ "$level" ]]; then
    echo "Error: '$level' is not in run_name!"
    exit 1
fi


EXPNAME=$run_name-lr$LR-bs$batch_size-accum$gradient_accumulation_steps-epoch$num_epochs-$TIMESTAMP

ARGS="train_grpo_process_rule.py \
    --model_name $MODEL_PATH \
    --reward_model_name $REWARD_MODEL_PATH \
    --train_path $data_path \
    --num_generations 7 \
    --learning_rate $LR \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --max_completion_length 2048 \
    --num_iterations 1 \
    --optim paged_adamw_32bit \
    --lr_scheduler_type cosine \
    --bf16 \
    --warmup_ratio 0.03 \
    --num_train_epochs $num_epochs \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 80 \
    --save_only_model \
    --report_to wandb \
    --output_dir runs_grpo/qwen25-7b/$EXPNAME \
    --use_vllm \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.5 \
    --log_completions True \
    --deepspeed /path/to/TDRM/trl/configs/ds_config_zero3.json \
    --wandb_run_name $run_name \
    --no_std False \
    --shuffle False \
    --prm_weight 0.2 \
    --rule_weight 0.8"

run_cmd="torchrun --master_port=29501 --nnodes=1 --nproc_per_node=7 --node_rank=0 $ARGS"
echo $run_cmd
LOG_PATH=logs_grpo/${EXPNAME}
mkdir -p $LOG_PATH
eval ${run_cmd} 2>&1 | tee ${LOG_PATH}/output_${MLP_ROLE_INDEX}.log
