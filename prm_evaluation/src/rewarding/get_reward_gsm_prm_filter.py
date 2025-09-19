import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from copy import deepcopy
import fcntl


def get_score(model, tokenizer, prompts, responses):
    # Prepare messages for each prompt-response pair
    messages = []
    for prompt, response in zip(prompts, responses):
        messages.append([{"role": "user", "content": prompt[0]},
                         {"role": "assistant", "content": response[0]}])

    # print(messages)
    # Apply chat template to each message pair and tokenize the input
    batch_input = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
    enc = tokenizer(batch_input, return_tensors='pt', padding=True, truncation=True)

    # Move all tensors to the model's device
    enc['input_ids'] = enc['input_ids'].to(model.module.device)
    enc['attention_mask'] = enc['attention_mask'].to(model.module.device)

    # Perform batch inference with no gradient computation
    with torch.no_grad():
        outputs = model(**enc)

    # Apply sigmoid to logits (assuming binary classification)
    rewards = torch.sigmoid(outputs.logits)  # Shape: [batch_size, sequence_length]
    
    # Assuming rewards are for each prompt-response pair, return them
    return rewards.squeeze(1)  # Return shape: [batch_size]


class RewardDataset(Dataset):
    def __init__(self, data_list, rescore=False):
        self.rescore = rescore
        self.data_list = data_list
        print(f"Number of items: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        prompts, responses, indices = [], [], []
        
        # Add original index for tracking purposes
        original_index = idx  # Store the original index

        for item_idx, item in enumerate(data['assistant']):
            # If the reward already exists and rescore is not requested, skip this item
            if "reward" in item and not self.rescore:
                continue

            # Append the user prompt, assistant response, and the index
            prompts.append(deepcopy(data['user']))
            responses.append(str(deepcopy(item['content'])))
            indices.append(item_idx)  # Use item_idx for specific assistant response

        # Return the data with the original index included
        return prompts, responses, indices, data, original_index


def write_to_file(data, temp_file_path):
    """ Helper function to write data to file (only for rank 0) """
    dist.barrier()
    with open(temp_file_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(data))
        f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file after writing

    dist.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/path/to/TDRM/outputs/bon_naive/')
    parser.add_argument('--save_path', type=str, default='/path/to/TDRM/RLHF-Reward-Modeling/math-rm/outputs/bon_naive/')
    parser.add_argument('--prm_path', type=str, default='/path/to/TDRM/RLHF-Reward-Modeling/math-rm/models/llama3_orm/checkpoint-1200/')
    parser.add_argument('--clean', type=str, default='/path/to/TDRM/RLHF-Reward-Modeling/math-rm/models/llama3_orm/checkpoint-1200/')
    parser.add_argument('--rescore', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1)  # for DDP
    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = args.local_rank  # Default value for non-DDP (if not set)

    def init_distributed_mode():
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    # Set up model and tokenizer
    def setup_model_and_tokenizer():
        path = args.prm_path
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=1, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        tokenizer.padding_side = 'right'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        return model, tokenizer

    # Setup for Distributed Data Parallel (DDP)
    # dist.init_process_group(backend='nccl')  # This is necessary for DDP to work with multiple GPUs
    # torch.cuda.set_device(local_rank)  # Set the device based on LOCAL_RANK
    print(f"Local rank: {local_rank}")

    init_distributed_mode()

    model, tokenizer = setup_model_and_tokenizer()

    # Move model to the correct device
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Process data in batches
    temp_file_path = os.path.join(args.save_path, "temp_file.json")
    if dist.get_rank() == 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        with open(temp_file_path, 'w') as f:
            f.write('')

    # Prepare dataset and dataloader
    data_list = []
    for file in os.listdir(args.data_path):
        if file.endswith('.json') and not file == 'temp_file.json':
            with open(os.path.join(args.data_path, file), 'r', encoding='utf-8') as f:
                for line in f:
                    data_list.append(json.loads(line))

    print(f"Number of data points: {len(data_list)}")

    dataset = RewardDataset(data_list, args.rescore)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=None)


    for batch_idx, (batch_prompts, batch_responses, batch_indices, data, original_index) in enumerate(tqdm(dataloader, desc='Processing batches')):
        # print(batch_prompts)
        
        # Get rewards for the current batch
        rewards = get_score(model, tokenizer, batch_prompts, batch_responses)

        # Gather rewards across all GPUs
        # gathered_rewards = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_rewards, rewards)

        # Combine the rewards from all GPUs
        # gathered_rewards = torch.cat(gathered_rewards, dim=0)

        # Store the modified data with rewards and original indices
        for idx, reward in zip(batch_indices, rewards):
            data['user'] = batch_prompts[idx][0]  # Ensure this is the original prompt (string)
            data['assistant'][idx]['content'] = data['assistant'][idx]['content'][0]  # Ensure 'content' is string
            data['assistant'][idx]['reward'] = float(reward.float().cpu().numpy())
            data['assistant'][idx]['num_tokens'] = int(data['assistant'][idx]['num_tokens'].numpy())
            data['original_index'] = int(original_index.numpy()[0])

        # Include the original index in the data for sorting later
        # all_data.append((data, original_index))
        write_to_file(data, temp_file_path)

    # Sort data by original indices before writing
    if dist.get_rank() == 0:
        old_data = []
        with open(temp_file_path, 'r') as f:
            for line in f:
                old_data.append(json.loads(line))
            old_data.append(data)
            old_data = sorted(old_data, key=lambda x: x['original_index'])
            # filter redundant data
            old_data = [old_data[i] for i in range(len(old_data)) if i == 0 or old_data[i]['original_index'] != old_data[i-1]['original_index']]

        with open(temp_file_path, 'w') as f:
            for data in old_data:
                f.write(json.dumps(data) + '\n')
        # Move temp file to the final destination
        os.replace(temp_file_path, os.path.join(args.save_path, "output.json"))
        with open(temp_file_path, "w") as f:
            f.write("")  # Clear the file if something went wrong

    # Clean up and finalize DDP
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
