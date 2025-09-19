from dataclasses import dataclass, field
import torch.distributed
from typing import Any, Dict, List, Optional, Union

import os
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModel
from trl import GRPOConfig, GRPOProcessTrainer
# import wandb

import json
import torch.distributed as dist
from chat_template import CHAT_TEMPLATE_LLAMA31

from mathruler.grader import extract_boxed_content, grade_answer
from func_rewards.math import math_compute_score

# def setup_wandb(args, training_args):
#     """Initialize wandb only on rank 0."""
#     rank = dist.get_rank() if dist.is_initialized() else 0  # Get rank, default to 0 if not distributed

#     if rank == 0:  # Only initialize wandb on rank 0
#         os.environ["WANDB_API_KEY"] = ""

#         # Now, log in using the set API key
#         wandb.login()
#         wandb.init(project="TDRM", name=f"{args.wandb_run_name}-grpo-with-template-rule-lr={training_args.learning_rate}-mbs={training_args.per_device_train_batch_size}-epochs={training_args.num_train_epochs}-num_gens{training_args.num_generations}", entity="miniac-workspace")
    
    # dist.barrier()  # Sync all processes to avoid errors

def convert_to_conversational_format(examples):
    # NOTE: simply passing this is correct
    SYS_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
    examples["prompt"] = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": examples["prompt"]}]
    # examples["prompt"] = [{"role": "user", "content": examples["prompt"]}]
    return examples


@dataclass
class Args:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-Coder-32B-Instruct")
    reward_model_name: Optional[str] = field(default="")
    train_path: Union[str, List[str]] = field(default="")
    eval_path: Union[str, List[str]] = field(default="")
    wandb_run_name: str=field(default="grpo")
    no_std: bool=field(default=False)
    shuffle: bool=field(default=False)
    prm_weight: float=field(default=0.5)
    rule_weight: float=field(default=0.5)

if __name__ == '__main__':
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
    with open("./data/grpo_eval_manual.json", "r") as f:
        eval_dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "Llama-3.1" in args.model_name:
        tokenizer.chat_template = CHAT_TEMPLATE_LLAMA31

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)

    dataset = Dataset.from_list(dataset)
    dataset = dataset.rename_columns({"question": "prompt"})
    # add chat template
    dataset = dataset.map(convert_to_conversational_format)

    eval_dataset = Dataset.from_list(eval_dataset)
    eval_dataset = eval_dataset.rename_columns({"question": "prompt"})
    # add chat template
    eval_dataset = eval_dataset.map(convert_to_conversational_format)

    training_args.reward_weights = [args.prm_weight, args.rule_weight]
    print(f"Weights of rewards: {training_args.reward_weights}")
    trainer = GRPOProcessTrainer(
        model=args.model_name,
        reward_funcs=[args.reward_model_name, math_compute_score],
        args=training_args,
        # train_dataset=data['train'],
        # eval_dataset=data['eval'],
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_processing_classes=[reward_tokenizer, tokenizer],
        no_std=args.no_std,
        shuffle=args.shuffle,
    )
    trainer.train()

    print("Saving last checkpoint of the model")
    save_path = os.path.join(training_args.output_dir, "last_checkpoint")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)