from dataclasses import dataclass, field
import torch.distributed
from typing import Any, Dict, List, Optional, Union

import os
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModel
from trl import GRPOConfig, GRPOTrainer
import wandb

import json
import torch.distributed as dist
from chat_template import CHAT_TEMPLATE_LLAMA31
from func_rewards.math import math_compute_score
from utils import pass_at_1

def setup_wandb(args, training_args):
    """Initialize wandb only on rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0  # Get rank, default to 0 if not distributed

def convert_to_conversational_format(examples):
    # NOTE: simply passing this is correct
    SYS_PROMPT="""Please reason step by step, and put your final answer within \\boxed{}"""
    examples["prompt"] = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": examples["prompt"]}]
    # examples["prompt"] = [{"role": "user", "content": examples["prompt"]}]
    # print(examples['prompt'])
    return examples

@dataclass
class Args:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-Coder-32B-Instruct")
    train_path: Union[str, List[str]] = field(default="")
    eval_path: Union[str, List[str]] = field(default="")
    wandb_run_name: str=field(default="grpo")
    num_datapoints: int=field(default=1000)
    constant_normalizer: float=field(default=None)

if __name__ == '__main__':
    # Initialize torch distributed
    # dist.init_process_group(backend="nccl")
    parser = HfArgumentParser((GRPOConfig, Args))
    training_args, args = parser.parse_args_into_dataclasses()
    # setup_wandb(args, training_args)
    if training_args.local_rank == 0:
        print(training_args)
        print(args)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    print(f"Train path is {args.train_path}")
    with open(args.train_path[0], "r") as f:
        dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(dataset)
    dataset = dataset.rename_columns({"question": "prompt"})
    # add chat template
    dataset = dataset.map(convert_to_conversational_format)
    eval_dataset = dataset.select(range(20))

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=math_compute_score,
        args=training_args,
        # train_dataset=data['train'],
        # eval_dataset=data['eval'],
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        constant_normalizer=args.constant_normalizer,
        # compute_metrics=lambda eval_pred: pass_at_1(eval_pred, tokenizer)
    )
    trainer.train()

    print("Saving last checkpoint of the model")
    save_path = os.path.join(training_args.output_dir, "last_checkpoint")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)