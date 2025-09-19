import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams, RequestOutput
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

from search.tasks import SearchTask
from search.beam_search import BeamSearch
from transformers import AutoTokenizer
from .utils import flatten_args, get_file_line_num
import random
import numpy as np
from pathlib import Path

PROJ_ROOT = Path.cwd()

def set_random_seed(seed: int):
    """
    Sets random seeds for reproducibility across random, numpy, and PyTorch.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)                        # Python's built-in random module
    np.random.seed(seed)                     # Numpy random seed
    torch.manual_seed(seed)                  # PyTorch random seed
    torch.cuda.manual_seed(seed)             # PyTorch random seed for CUDA
    torch.cuda.manual_seed_all(seed)         # PyTorch random seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disables autotuning for reproducibility



@hydra.main(config_path=f"{PROJ_ROOT}/configs/exp_configs", config_name="beam-search")
def main(args: DictConfig):
    args = flatten_args(args)
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # search
    task = SearchTask(
        "",
        tokenizer,
        temperature=args.temperature,
        min_p=args.min_p,
        max_new_tokens=args.step_max_tokens,
        stop=args.stop,
        inference_model_path=args.model_name_or_path,
        value_model_path=args.value_model_path,
        system_prompt=args.sys_prompt,
    )

    # load data
    dataset = args.dataset_name
    data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
    dataset = dataset.split("@")[0].strip()
    dataset_basename = os.path.basename(dataset)

    ext = os.path.splitext(dataset)[-1]
    # local python script
    if ext == ".py" or (
        os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
    ):
        seed_data = load_dataset(dataset, trust_remote_code=True)
        print(f"loaded {dataset} with python script")
    # local text file
    elif ext in [".json", ".jsonl", ".csv"]:
        ext = ext.lower().strip(".")
        if ext == "jsonl":
            ext = "json"
        seed_data = load_dataset(ext, data_files=dataset)
        print(f"loaded {dataset} with data_files={dataset}")
    # local dataset saved with `datasets.Dataset.save_to_disk`
    elif os.path.isdir(dataset):
        seed_data = load_from_disk(dataset)
        print(f"loaded {dataset} from disk")
    # remote/local folder or common file
    else:
        seed_data = load_dataset(dataset, data_dir=data_dir)
        print(f"loaded {dataset} from files")

    seed_data = seed_data[args.split][args.prompt_key]
    if args.num_samples is not None:
        num_samples = min(args.num_samples, len(seed_data))
        seed_data = seed_data[args.start_from:args.start_from+num_samples]

    def collate_fn(batch):
        return [[{"role": "user", "content": item}] for item in batch]
    assert args.batch_size == 1, "batch size of beam search should be 1 for now"
    dataloader = DataLoader(seed_data, collate_fn=collate_fn, batch_size=args.batch_size)

    print("="*100)
    print(f"Data: {args.dataset_name}; Split: {args.split}")
    if args.prefix is not None:
        print(f"Generating with Prefix: {args.prefix}")
    else:
        print("No prefixes pre-pended")
    if args.sys_prompt is not None:
        print(f"Generating with System Prompt: {args.sys_prompt}")
    else:
        print("No system prompts pre-pended")
    print("="*100)

    if args.dataset_output is not None:
        output_dir = os.path.join(PROJ_ROOT, "evaluation", "outputs", args.dataset_output, args.output_path)
    else:
        output_dir = os.path.join(PROJ_ROOT, "evaluation", "outputs", args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(args.epoch):
        set_random_seed(epoch)
        output_path = f"{output_dir}/{args.prefix_name}-{args.model_name_or_path.split('/')[-1]}-model-temp{args.temperature}_{args.num_branch}branch_minp{args.min_p}_stop{args.stop_name}_initdepth{args.initial_depth}top{args.top_k}_run{epoch}.json"
        output_tree_info_path = f"{output_dir}/tree_info_temp{args.temperature}_minp{args.min_p}_stop{args.stop_name}_initdepth{args.initial_depth}top{args.top_k}_run{epoch}.json"
        print(f"The data will be written to file: {output_path}")
        print(f"The search tree info will be written to file: {output_tree_info_path}")
        print("="*100)
        file_line_num = get_file_line_num(output_path)
        if file_line_num != 0:
            print(f"Resume from {file_line_num}")
        if file_line_num == len(seed_data):
            print("This epoch has finished")
            continue
        dataloader = DataLoader(seed_data[file_line_num:], collate_fn=collate_fn, batch_size=args.batch_size)
        for batched_data in tqdm(dataloader):

            task.set_question(list(batched_data)[0][0]["content"])
            search = BeamSearch(
                task,
                depth=args.depth,
                top_k=args.top_k,
                num_branch=args.num_branch,
                max_length=args.max_length,
                initial_depth=args.initial_depth,
                initial_branch=args.initial_branch,
                branch_decay=args.branch_decay
            )

            user = list(batched_data)[0]

            write_item: dict[str, list[dict]] = {
                "user": "",
                "assistant": [] # BoN
            }
            write_item['user'] = user[0]["content"]
            write_item['assistant'] = [
                {
                    "content": tokenizer.decode(search[0]),
                    "num_tokens": task.num_tokens,
                    "num_states": task.node_count,
                    # "reward": float(search[2].V.cpu().float().numpy())
                    "reward": float(search[2].V)
                }
            ]

            nodes = search[1].traverse(node_list=[], id=0, pre_id=-1)

            with open(output_path, "a") as f:
                f.write(json.dumps(write_item))
                f.write("\n")

            write_item = {
                "user": "",
                "assistant": [] # BoN
            }


if __name__ == "__main__":
    main()