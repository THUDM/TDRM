import os
import json
import argparse
from .tasks import SearchTask
from .beam_search import BeamSearch
from transformers import AutoTokenizer

def run():
    INFERENCE_MODEL_DIR = "/workspace/ckpt/zephyr-7b-sft-full/"
    tokenizer = AutoTokenizer.from_pretrained(INFERENCE_MODEL_DIR, trust_remote_code=True)
    task = SearchTask("Write a blog post about getting good grades at school", tokenizer)
    search = BeamSearch(task)
    print(search)


if __name__ == '__main__':

    run()