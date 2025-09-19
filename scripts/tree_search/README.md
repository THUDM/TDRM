# Inference-time Tree Search with RMs

This directory contains code for tree search using RMs. We use `hydra` to manage our configs in this experiment. You can run the following command with the `prm` environment:
```bash
python -m scripts.tree_search.beam_search \
    dataset=math-500 \
    method=greedy-search \
    model=qwen-math-7b \
    +num_branch=2,4,8,16 \
    +prefix_name="new-line-ds-baseline-prm-" \
    +value_model_path="/path/to/your/reward_model" \
    +output_path="greedy_search/qwen-baseline-prm/" \
    +epoch=3 \
    +step_max_tokens=1024 \
    +max_length=1024 \
    +stop="\n\n" \
    +stop_name="new_line" \
    +temperature=0.4 \
    --multirun
```
This will gives you results of greedy search using PRM at `value_model_path` and the model backbone of `qwen-math-7b`. This will run 3 times for each branch count in {2, 4, 8, 16}.