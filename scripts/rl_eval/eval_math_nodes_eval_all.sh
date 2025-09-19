#!/bin/bash

# example usage: bash eval_math.sh --run_name verl-grpo-fix-math-eval-large-reward_temp1.0_ppomicro4_Qwen2.5-14B_simplelr_math_35 --init_model Qwen2.5-14B --template qwen25-math-cot  --tp_size 1

cd examples/simplelr_math_eval
pip uninstall latex2sympy2 -y
cd latex2sympy
pip install -e . --use-pep517
pip install Pebble
pip install sympy==1.12
pip install antlr4-python3-runtime==4.11.1
pip install timeout-decorator
pip install jieba
cd ..


export NCCL_DEBUG=warn
# 定义评估脚本路径
set -x

export WANDB_OFFICIAL=1
export WANDB_API_KEY=2de3defdedb87a44ee32a7e5d02a764d56e9d765
TOTAL_NODES=${ARNOLD_WORKER_NUM:-1}  # Default to 1 if not set
CURRENT_NODE=${ARNOLD_ID:-0}  # Default to 0 if not set

add_step_0=false
temperature=0.0
max_tokens=16000
top_p=1
benchmarks="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
output_dir="eval_results"
overwrite=false
n_sampling=1
specific_steps=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --init_model)
            INIT_MODEL_PATH="$2"
            shift 2
            ;;
        --template)
            template="$2"
            shift 2
            ;;
        --tp_size)
            tp_size="$2"
            shift 2
            ;;
        --temperature)
            temperature="$2"
            shift 2
            ;;
        --top_p)
            top_p="$2"
            shift 2
            ;;
        --max_tokens)
            max_tokens="$2"
            shift 2
            ;;
        --add_step_0)
            add_step_0="$2"
            shift 2
            ;;
        --benchmarks)
            benchmarks="$2"
            shift 2
            ;;
        --just_wandb)
            just_wandb="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --overwrite)
            overwrite="$2"
            shift 2
            ;;
        --n_sampling)
            n_sampling="$2"
            shift 2
            ;;
        --specific_steps)
            specific_steps="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$RUN_NAME" ] || [ -z "$INIT_MODEL_PATH" ] || [ -z "$template" ] || [ -z "$tp_size" ]; then
    echo "Missing required parameters. Usage:"
    echo "--run_name <run_name> --init_model <init_model> --template <template> --tp_size <tp_size>"
    exit 1
fi


eval_script_path="sh/eval.sh"

HDFS_HOME=/path/to/the/project/dir/

base_checkpoint_path="${HDFS_HOME}/checkpoints/${RUN_NAME}"


init_model_path="${HDFS_HOME}/${INIT_MODEL_PATH}"
chmod +x sh/convert_and_evaluate_gpu_nodes.sh


get_all_checkpoints() {
    local base_path="$1"
    local specific_steps="$2"
    local checkpoints=()
    
    # If specific steps are provided, only collect those checkpoints
    if [ -n "$specific_steps" ]; then
        IFS=',' read -r -a step_array <<< "$specific_steps"
        for step in "${step_array[@]}"; do
            step_dir="$base_path/global_step_$step"
            if [ -d "$step_dir" ]; then
                checkpoints+=("global_step_$step")
            else
                echo "Warning: Requested step $step does not exist at $step_dir"
            fi
        done
    else
        # Otherwise, collect all checkpoints
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")
                checkpoints+=("$step_tag")
            fi
        done
    fi
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo ""
    else
        # Sort the checkpoints to ensure consistent ordering across nodes
        printf "%s\n" "${checkpoints[@]}" | sort -V
    fi
}

# Only evaluate the initial model
sub_dir=your_subdir
target_dir="${HDFS_HOME}/${sub_dir}"

step_tag="init_model_eval"

# Loop over each folder in target_dir
for folder in "$target_dir"/*/; do
    echo "Checking folder $folder"

    for checkpoint in "$folder"checkpoint-*; do
        if [ -d "$checkpoint" ]; then
            init_model_path="$checkpoint"
            single_run_name="$checkpoint"
            base_checkpoint_path="${HDFS_HOME}/checkpoints/${folder}"

            echo "Evaluating model at $init_model_path"

            output_path="$base_checkpoint_path/$output_dir/$step_tag/$(basename "$folder")/$(basename "$checkpoint")"
            if [ -d "$output_path" ]; then
                echo "Output path $output_path already exists. Skipping..."
            else
                mkdir -p "$output_path"
                # you may change the device id as you need
                CUDA_VISIBLE_DEVICES=7 bash "$eval_script_path" "${template}" "$init_model_path" "$output_path" "$temperature" "$max_tokens" "$top_p" "$benchmarks" "$overwrite" "$n_sampling"
            fi
        fi
    done
done


exit 0


bash eval_math_nodes_eval_all.sh \
    --run_name TD2  \
    --init_model last_checkpoint \
    --template qwen-boxed  \
    --tp_size 8 \
    --add_step_0 true  \
    --temperature 0 \
    --top_p 1 \
    --max_tokens 16000 \
    --benchmarks aime24,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 
