import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# get value model
def get_value_model(base_model_dir):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if "gen_prm_ckpts" in base_model_dir:
        value_base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, trust_remote_code=True).bfloat16().cuda()
    else:
        value_base_model = AutoModelForSequenceClassification.from_pretrained(base_model_dir, trust_remote_code=True).bfloat16().cuda()

    return value_tokenizer, value_base_model

def get_local_value(model, tokenizer, prompt: str, response: str):
    # question = prompt
    # state = response
    # messages = [{"role": "user", "content": "Question:\n" + question + "Answer:\n" + state}, {"role": "assistant", "content": "Is the answer correct (Yes/No)? Yes"}]
    # input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    # plus_tag_id = tokenizer.encode(' Yes')[-1]
    # minus_tag_id = tokenizer.encode(' No')[-1]
    # candidate_tokens = [plus_tag_id, minus_tag_id]
    # logits = model(input_ids).logits[:, -3, candidate_tokens]  # simple version for llama3.1-instruct, the +/- is predicted by the '-3' position
    # scores = logits.softmax(dim=-1)[:, 0]
    
    # return scores.detach().cpu().float()
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    attention_mask = torch.ones(input_ids.shape).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        print("Shape of logits: ", outputs.logits.shape)
        if outputs.logits.size(1) == 1:
            rewards = torch.sigmoid(outputs.logits).squeeze(1)  # Shape: [batch_size, sequence_length]
        else:
            rewards = torch.softmax(outputs.logits, dim=-1)  # Shape: [batch_size, num_category]
            bin_values = torch.linspace(0, 1, 5).to(rewards.device)
            rewards = torch.sum(rewards * bin_values, dim=-1)
    
    # Assuming rewards are for each prompt-response pair, return them
    return rewards  # Return shape: [batch_size]