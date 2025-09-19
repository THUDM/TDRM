import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "/path/to/TDRM/gen_prm_ckpts/baseline_orm_sign/checkpoint-1067"
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)


def get_score(model, tokenizer, prompt, response):
    conversation = []
    query_response = "Question:\n" + prompt + "Answer:\n" + response
    conversation.append({"content": query_response, "role": "user"}) # in their case, the query_response is the user's response
    conversation.append({"content": "+", "role": "assistant"})

    input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt").to(model.device)

    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id,minus_tag_id]

    logits = model(input_ids).logits[:,-3,candidate_tokens] #simple version for llama3.1-instruct, the +/- is predicted by the '-3' position
    scores = logits.softmax(dim=-1)[:,0]

    return scores[0].detach().cpu().float()


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/path/to/TDRM/outputs/bon_naive/')
parser.add_argument('--save_path', type=str, default='/path/to/TDRM/RLHF-Reward-Modeling/math-rm/outputs/bon_naive/')
parser.add_argument('--rescore', action='store_true', default=False)
args = parser.parse_args()

temp_file_path = os.path.join(args.save_path, "temp_file.json")
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
for file in os.listdir(args.data_path):
    if file.endswith('.json'):
        data_list = []
        with open(os.path.join(args.data_path, file), 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
    if len(data_list) == 0:
        print(f"File {file} is empty")
        continue
    # if "reward" in data_list[-1]['assistant'][0]:
    #     print(f"File {file} has already been rewarded")
                    

    for data in tqdm(data_list, desc=f'rewarding {file}'):
        for idx, item in enumerate(data['assistant']):
            if "reward" in data['assistant'][idx] and not args.rescore:
                print("already rewarded")
                continue
            # print(data['user'], item['content'])
            reward = get_score(model, tokenizer, data['user'], str(item['content']))
            data['assistant'][idx]['reward'] = float(reward.cpu().numpy())
        with open(temp_file_path, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")
    os.replace(temp_file_path, os.path.join(args.save_path, file))
    # clear the temp file
    with open(temp_file_path, "w") as f:
        f.write("")