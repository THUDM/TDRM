import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_score(model, prompt, response):
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    
    prompt = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False).replace(tokenizer.bos_token, "")
    enc = tokenizer(prompt, return_tensors='pt')
    enc['input_ids'] = enc['input_ids'].to(model.device)
    enc['attention_mask'] = enc['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(**enc)
        print(torch.sigmoid(outputs.logits[0]))
        return outputs.logits[0]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/path/to/TDRM/outputs/bon_naive/')
parser.add_argument('--prm_path', type=str, default='/path/to/TDRM/RLHF-Reward-Modeling/math-rm/models/llama3_orm/checkpoint-1200/')
parser.add_argument('--rescore', action='store_true', default=False)
args = parser.parse_args()

path = args.prm_path
model = AutoModelForSequenceClassification.from_pretrained(path, device_map="auto", num_labels=1,
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

temp_file_path = os.path.join(args.data_path, "temp_file.json")
for file in os.listdir(args.data_path):
    if file.endswith('.json'):
        data_list = []
        with open(os.path.join(args.data_path, file), 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
    else:
        continue
    if len(data_list) == 0:
        print(f"File {file} is empty")
        continue
    if "reward" in data_list[-1]['assistant'][0]:
        print(f"File {file} has already been rewarded")
                    

    for data in tqdm(data_list, desc=f'rewarding {file}'):
        for idx, item in enumerate(data['assistant']):
            if "reward" in data['assistant'][idx] and not args.rescore:
                print("already rewarded")
                continue
            # print(data['user'], item['content'])
            reward = get_score(model, data['user'], str(item['content']))
            data['assistant'][idx]['reward'] = float(reward.float().cpu().numpy())
        with open(temp_file_path, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")
    os.replace(temp_file_path, os.path.join(args.data_path, file))
    # clear the temp file
    with open(temp_file_path, "w") as f:
        f.write("")