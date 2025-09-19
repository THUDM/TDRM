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
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from .utils import flatten_args, get_file_line_num
from datasets import load_dataset, Dataset

SYS_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

@hydra.main(config_path=f"/path/to/TDRM/configs/exp_configs", config_name="rollouts")
def main(args: DictConfig):
    args = flatten_args(args)
    print(args)
    llm = LLM(model=args.model_name_or_path, gpu_memory_utilization=0.8)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load data
    with open(args.dataset_name, "r") as f:
        dataset = json.load(f)

    def convert_to_conversational_format(examples):
        # NOTE: simply passing this is correct
        SYS_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
        examples["prompt"] = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": examples["prompt"]}]
        examples["prompt"] = tokenizer.apply_chat_template(examples["prompt"], tokenize=False, add_generation_prompt=True)
        # examples["prompt"] = [{"role": "user", "content": examples["prompt"]}]
        return examples

    dataset = Dataset.from_list(dataset)
    dataset = dataset.rename_columns({"question": "prompt"})
    # add chat template
    dataset = dataset.map(convert_to_conversational_format)

    seed_data = dataset["prompt"]

    def collate_fn(batch):
        return [item for item in batch]

    # generate
    generation_args = SamplingParams(
        n=args.n,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_p=args.min_p
    )

    print("="*100)
    print(f"Data: {args.dataset_name}")

    print(f"Sample with the following generation config:\n{generation_args}")
    print("="*100)

    output_dir = os.path.join("/path/to/TDRM/evaluation", "outputs", args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # try:
    for epoch in range(args.epoch):
        final_write_item = []
        output_path = f"{output_dir}/{args.prefix_name}_temp{args.temperature}_minp{args.min_p}_run{epoch}.json"
        print(f"The data will be written to file: {output_path}")
        print("="*100)
        file_line_num = get_file_line_num(output_path)
        if file_line_num != 0:
            print(f"Resume from {file_line_num}")
        dataloader = DataLoader(seed_data[file_line_num:], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)
        for batched_data in tqdm(dataloader):
            print(f"Length: {len(batched_data)}")
            print(batched_data)
            gens_output = llm.generate(
                batched_data,
                sampling_params=generation_args,
            )
            print(f"Output length: {len(gens_output)}")
            # print(gens_output)
            for user, assistant_outputs in zip(list(batched_data), gens_output):
                write_item: dict[str, list[dict]] = {
                    "user": "",
                    "assistant": [] # BoN
                }
                write_item['user'] = user
                for assistant in assistant_outputs.outputs:
                    write_item['assistant'].append({
                        "content": assistant.text,   # for the convinience of later reward assignment
                        "num_tokens": len(assistant.token_ids)
                    })

                # with open(output_path, "a") as f:
                #     f.write(json.dumps(write_item))
                #     f.write("\n")
                final_write_item.append(write_item)

                write_item = {
                    "user": "",
                    "assistant": [] # BoN
                }

            if args.debug:
                for item in final_write_item:
                    with open(output_path, "w") as f:
                        f.write(json.dumps(item))
                        f.write("\n")
                break

        for item in final_write_item:
            with open(output_path, "w") as f:
                f.write(json.dumps(item))
                f.write("\n")
        final_write_item = []
    # except:
    #     # if file_line_num
    #     for item in final_write_item:
    #         with open(output_path, "w") as f:
    #             f.write(json.dumps(item))
    #             f.write("\n")


if __name__ == "__main__":
    main()