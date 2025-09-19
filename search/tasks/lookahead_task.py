import re
import os
from .prompts import *
from ..models import get_proposal, get_value
import warnings


# data: question: str
# mode: 'cot', 'tot', 'mcts'
# method: 'glm', 'gpt', 'local'
class LookaheadSearchTask(object):
    def __init__(self, data: str, tokenizer,
                 propose_method='zephyr',
                 value_method='rlhflow',
                 temperature=0.7,
                 max_new_tokens=1024,
                 min_p=0.05,
                 stop=None,
                 ):
        super().__init__()
        self.question = data
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

    def clear_cache(self):
        self.value_cache = {}

    def update_count(self):
        self.node_count += 1

    def update_budget(self, num_tokens):
        self.num_tokens += num_tokens

    def get_next_step(self, y, node_depth, num_branch, stop_token_ids=None, max_length=1024, **kwargs) -> list[tuple[str, bool, int]]:
        # stop_token_ids = [self.stop_token_id]
        prompt = self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "user", "content": self.question}, {"role": "assistant", "content": y}], add_generation_prompt=False), skip_special_tokens=True)
        # print("Prompt: ", prompt)
        initial_branch = kwargs.get("initial_branch", num_branch)
        initial_depth = kwargs.get("initial_depth", 1)
        if node_depth == 0:
            num_stop_times = initial_depth
        else:
            num_stop_times = 1
        if y == '':
            print(f"Initial Branch: {initial_branch}")
            branch = initial_branch
        else:
            branch = num_branch
        question_len = len(self.tokenizer.apply_chat_template([{"role": "user", "content": self.question}]))

        # handle two cases: stop or stop_token_ids
        if stop_token_ids is not None:
            warnings.warn("Only supports stop as string for now")
            # responses = get_proposal(
            #     question_len,
            #     prompt,
            #     temperature=self.temperature,
            #     max_new_tokens=self.max_new_tokens,
            #     max_length=max_length,
            #     min_p=self.min_p,
            #     num_branch=branch,
            #     stop=None,
            #     stop_token_ids=stop_token_ids,
            #     num_stop_times=num_stop_times,
            # )
        else:
            responses = get_proposal(
                question_len,
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_length=max_length,
                min_p=self.min_p,
                num_branch=branch,
                stop=None,
                num_stop_times=num_stop_times,
            )

        return responses
    
    def get_step_value(self, y) -> float:
        if y in self.value_cache:
            return self.value_cache[y]
        else:
            value = get_value(self.question, y)
            self.value_cache[y] = value
            return value
