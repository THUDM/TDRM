For policy evaluation, we use the evaluation protocal in simpleRL-reason. `eval_math_nodes_eval_all.sh` is a modified script that can evaluate all the checkpoints under a directory one by one. You can first copy the `eval_math_nodes_eval_all.sh` to simpleRL-reason after installing the submodule, and then launch the script using the following command:

```bash
bash eval_math_nodes_eval_all.sh \
    --run_name TD2-level4-single-loss  \
    --template qwen-boxed  \
    --tp_size 8 \
    --add_step_0 true  \
    --temperature 0 \
    --top_p 1 \
    --max_tokens 16000 \
    --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math \
    --n_sampling 1 
```

P.S. You may need to add templates at `simpleRL-reason/examples/simplelr_math_eval/utils.py` to the `PROMPT_TEMPLATES` when needed. For example,
```python
"glm-boxed": (
    "[gMASK]<sop><|user|>\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|assistant|>\n",
    "{output}",
    "\n\n",
)
```