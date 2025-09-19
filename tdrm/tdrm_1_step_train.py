from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

import accelerate
import wandb

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=2)
    learning_rate: Optional[float] = field(default=1e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="/workspace/ckpt/Meta-Llama-3.1-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="zd21/TDRM-1-step-TD",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    output_path: Optional[str] = field(
        default="./output_path/",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=20,
        metadata={"help": "Eval the model every x steps"},
    )
    terminal_weight: Optional[float] = field(
        default=1,
        metadata={"help": "The weight of the terminal reward"},
    )
    gamma: Optional[float] = field(
        default=0.8,
        metadata={"help": "The discount factor"},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "wandb run name"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "right"
print(tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length


# Get the dataset
train_path = script_args.train_set_path
# eval_path = script_args.eval_set_path
output_name = script_args.output_path


def build_dataset(tokenizer, train_path, eval_path):

    def wrap_state(question, ans):
        if ans == '':
            message = [
                {"role":"user", "content":question},
            ]
        else:
            message = [
                {"role":"user", "content":question},
                {"role":"assistant", "content":ans}
            ]
        return message

    def tokenize(sample):
        question = sample['conversations']['question']
        state1 = sample['conversations']['state1']
        state2 = sample['conversations']['state2']
        # message = [
        #     {"role":"user", "content":question},
        #     {"role":"assistant", "content":ans}
        # ]
        message1 = wrap_state(question, state1)
        sample['state1'] = tokenizer.apply_chat_template(
            message1, tokenize=False, add_generation_prompt=False)
        tokenized_pos1 = tokenizer(sample['state1'], truncation=True)
        sample["input_ids_1"] = tokenized_pos1["input_ids"]
        sample["attention_mask_1"] = tokenized_pos1["attention_mask"]

        message2 = wrap_state(question, state2)
        sample['state2'] = tokenizer.apply_chat_template(
            message2, tokenize=False, add_generation_prompt=False)
        tokenized_pos2 = tokenizer(sample['state2'], truncation=True)
        sample["input_ids_2"] = tokenized_pos2["input_ids"]
        sample["attention_mask_2"] = tokenized_pos2["attention_mask"]

        sample['label'] = sample['conversations']['label']
        sample['reward'] = sample['conversations']['reward']
        sample['done'] = sample['conversations']['done']
        return sample

    ds = load_dataset(train_path)['train'].shuffle(seed=42)

    ds = ds.map(tokenize, num_proc=24)

    # print 10 data point
    print(ds[:10])

    eval_dataset = None

    train_dataset = ds
    #eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
    eval_dataset = ds.select(range(500))
    return train_dataset, eval_dataset


train_dataset, eval_dataset = build_dataset(tokenizer, train_path, None)
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

# Define the trainer


# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=1,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb',
    save_only_model = True
)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True,
)

model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        label_list = []
        done_list = []
        reward_list = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_1"],
                    "attention_mask": feature["attention_mask_1"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_2"],
                    "attention_mask": feature["attention_mask_2"],
                }
            )

            label_list.append([feature['label']])
            done_list.append([feature['done']])
            reward_list.append([feature['reward']])
    
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "label": label_list,
            "return_loss": True,
            "done": done_list,
            "reward": reward_list
        }
        return batch


class RewardTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to include generation during evaluation.
        """
        # Perform standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        with torch.no_grad():
            test_prompts = [
                [{"role": "user", "content": "What's 1+1?"}, {"role": "assistant", "content": "1+1=2"}],
                [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}, {"role": "assistant", "content": "Step 1:\nNatalia sold 48 clips to her friends in April.\n\nStep 2:\nShe sold half as many clips in May, which means she sold 48/2 = 24 clips in May.\n\nStep 3:\nTo find the total number of clips she sold in April and May, we add the number of clips she sold in each month:\n\n48 (April) + 24 (May) = 72\n\nAnswer:\n\\boxed{72}"}],
                [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}, {"role": "assistant", "content": "3+3=6"}],
            ]
            test_features = tokenizer(tokenizer.apply_chat_template(test_prompts, tokenize=False, add_generation_prompt=False), padding=True, return_tensors='pt')
            test_features['input_ids'] = test_features['input_ids'].to(self.model.device)
            test_features['attention_mask'] = test_features['attention_mask'].to(self.model.device)
            values = model(**test_features)
            print("Test values:\nOracle: 1, 1, 0")
            print(values)
            print(torch.sigmoid(values.logits))
        output['test_values'] = values
    

    def compute_loss(self, model, inputs, return_outputs=False):
        # outputs = model(
        #     input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        # )

        state1_outputs = model(
            input_ids=inputs["input_ids"][0::2, :], attention_mask=inputs["attention_mask"][0::2, :]
        )
        # with torch.no_grad():
        state2_outputs = model(
            input_ids=inputs["input_ids"][1::2, :], attention_mask=inputs["attention_mask"][1::2, :]
        )
        label = inputs['label']
        label = torch.tensor(label,dtype=torch.bfloat16).to(state1_outputs.logits.device)
        rewards = inputs['reward']
        rewards = torch.tensor(rewards,dtype=torch.bfloat16).to(state1_outputs.logits.device)

        state1_q = state1_outputs.logits
        state2_q = state2_outputs.logits.detach()
        state1_probs = torch.sigmoid(state1_q)
        td_state2_probs = torch.clamp(rewards + script_args.gamma * torch.sigmoid(state2_q), min=0, max=1)
        # td_state2_probs = 0.8 * torch.sigmoid(state2_q)
        print(f"TD Prob 2: {td_state2_probs}")
        # assert state1_probs.shape == state2_probs.shape

        # torch version
        done = inputs['done']
        done = torch.tensor(done).long().to(label.device)

        loss = done * script_args.terminal_weight * ((label * torch.log(state1_probs+1e-10) + (1 - label) * torch.log(1 - state1_probs+1e-10))) + \
            (1 - done) * (td_state2_probs * torch.log(state1_probs+1e-10) + (1 - td_state2_probs) * torch.log(1 - state1_probs+1e-10))

        final_loss = -torch.mean(loss)

        if return_outputs:
            return final_loss, {"loss": final_loss}
        return final_loss

# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)

accelerator = accelerate.Accelerator()
if accelerator.is_main_process:
    print("is main process")
    wandb.init(project="PRM-Training", name=f"{script_args.run_name}_terminal_weight={script_args.terminal_weight}_{script_args.num_train_epochs}epoch_{script_args.learning_rate}lr", entity="miniac-workspace")

print(script_args)

trainer.train()

# if accelerator.is_main_process:
print("Saving last checkpoint of the model")
#model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")

wandb.finish()
