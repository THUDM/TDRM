This directory contains utilities for PRM evalution.

You can evaluate your PRM using:
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