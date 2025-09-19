# Scripts for PRM Training & Eval

You can launch the PRM training by running:
```bash
accelerate launch -m tdrm.tdrm_1_step_train --deepspeed ./configs/zero3.json
```

You can train a baseline ScalarPRM using:
```bash
accelerate launch -m tdrm.scalar_prm_train --deepspeed ./configs/zero3.json \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --train_set_path=/path/to/TDRM/outputs/datasets/rlhflow_scalar_prm_1413k \
    --run_name=qwen-distill-mistral-scalar-prm \
    --model_name=/path/to/TDRM/checkpoint/Qwen2.5-Math-7B \
    --output_path=../../rest-online-checkpoints/qwen_scalar_prm_baseline \
    --save_every_steps=2000
```

And you can evaluate your PRM using:
```bash
cd ../..
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" \
    --master_port=12345 prm_evaluation/src/rewarding/get_reward_math_td_lambda_prm.py \
    --data_path evaluation/outputs/math-500/mistral_rlhflow_bo128/ \
    --save_path /path/to/scored_data/ \
    --prm_path /path/to/checkpoint/ \
    --rescore
```
1. This will load the data from `--data_path`, score each data point using the model at `--prm_path`, and then save the data with scores to `--save_path`. Use `--rescore` when you want to re-run the evaluation.
2. You will need to run a script to calculate the Best-of-N metric:
```bash
python get_results_math --completion_path /path/to/scored_data/output.json
```