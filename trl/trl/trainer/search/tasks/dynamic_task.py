import re
import os
from .prompts import *
from ..models import get_proposal, get_value, get_value_model, get_inference_model_vllm


# data: question: str
# mode: 'cot', 'tot', 'mcts'
# method: 'glm', 'gpt', 'local'
class DynamicSearchTask(object):
    def __init__(self, data: str, tokenizer,
                 propose_method='zephyr',
                 value_method='rlhflow',
                 temperature=0.7,
                 max_new_tokens=1024,
                 min_p=0.05,
                 stop=None,
                 system_prompt=None,
                #  value_tokenizer=None,
                 ):
        super().__init__()
        self.question = data
        self.system_prompt = system_prompt
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}
        self.tokenizer = tokenizer
        self.stop_token_id = tokenizer.eos_token_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.min_p = min_p
        self.node_count = 0
        self.stop = stop
        self.num_tokens = 0

    def set_models(self, inference_model, inference_tokenizer, value_model, value_tokenizer):
        self.inference_model = inference_model
        self.inference_tokenizer = inference_tokenizer
        self.value_model = value_model
        self.value_tokenizer = value_tokenizer


    def set_question(self, data):
        self.question = data
        self.node_count = 0
        self.num_tokens = 0
        self.clear_cache()

    def set_stop(self, stop):
        self.stop = stop

    def clear_cache(self):
        self.value_cache = {}

    def update_count(self):
        self.node_count += 1

    def update_budget(self, num_tokens):
        self.num_tokens += num_tokens

    def get_next_step(self, y, node_depth, num_branch, stop=None, stop_token_ids=None, max_length=1024, **kwargs) -> list[tuple[str, bool, int]]:
        # stop_token_ids = [self.stop_token_id]
        print(f"Current step y is {self.tokenizer.decode(y)}")
        if self.system_prompt is not None:
            prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.question}, {"role": "assistant", "content": self.tokenizer.decode(y)}], add_generation_prompt=False)[:-2]
            question_len = len(self.tokenizer.apply_chat_template([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.question}], add_generation_prompt=False))
        else:
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": self.question}, {"role": "assistant", "content": self.tokenizer.decode(y)}], add_generation_prompt=False)[:-2]
            question_len = len(self.tokenizer.apply_chat_template([{"role": "user", "content": self.question}], add_generation_prompt=False))
        # prompt = self.tokenizer.encode(y, add_special_tokens=False)
        print("Prompt: ", self.tokenizer.decode(prompt))

        # handle two cases: stop or stop_token_ids
        if stop_token_ids is not None:
            responses = get_proposal(
                question_len,
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_length=max_length,
                min_p=self.min_p,
                num_branch=num_branch,
                stop=None,
                stop_token_ids=stop_token_ids,
                num_stop_times=1,
                inference_model=self.inference_model,
                inference_tokenizer=self.inference_tokenizer
            )
        else:
            responses = get_proposal(
                question_len,
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_length=max_length,
                min_p=self.min_p,
                num_branch=num_branch,
                stop=self.stop if stop is None else None, # TODO: whenever stop is not None, we should use stop, which means we should not stop
                num_stop_times=1,
                inference_model=self.inference_model,
                inference_tokenizer=self.inference_tokenizer
            )
        # print(f"Response is {responses}")
        return responses
    
    def get_step_value(self, y) -> float:
        if str(y) in self.value_cache:
            return self.value_cache[str(y)]
        else:
            print(f"Output string is {self.tokenizer.decode(y)}")
            value = get_value(
                self.question,
                self.tokenizer.decode(y),
                value_model=self.value_model,
                value_tokenizer=self.value_tokenizer
            )
            self.value_cache[str(y)] = value
            return value
