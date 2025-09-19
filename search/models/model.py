# initialize models
import os
import openai
import requests
import json
from .inference_models import get_local_response, get_inference_model_vllm, get_local_response_llama, get_inference_model_mistral, get_local_response_mistral, get_vllm_response
from .value_models import get_value_model, get_local_value
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput

# openai api settings
API_KEY = 'sk-**'
API_BASE = 'base'
BASE_MODEL_GPT = "gpt-3.5-turbo"

# GLM api settings
URL = "https://api.chatglm.cn/v1/chat/completions"
ID = "**"
AUTH = '**'
CONTENT_TYPE = 'application/json; charset=utf-8'
BASE_MODEL_GLM = 'GLM4'

# local model settings
# if you want to use local models, set these two directories
# INFERENCE_MODEL_DIR = "/workspace/ckpt/Meta-Llama-3-8B-Instruct"
INFERENCE_MODEL_DIR = "/path/to/TDRM/checkpoint/Meta-Llama-3.1-8B-Instruct/"
LOCAL_INFERENCE_TYPES = ['glm', 'llama', 'mistral']
LOCAL_INFERENCE_IDX = 0

# VALUE_BASE_MODEL_DIR = "/workspace/ckpt/MetaMath-Mistral-7B"
VALUE_BASE_MODEL_DIR = "/path/to/TDRM/last_checkpoint"
# VALUE_MODEL_STATE_DICT = "/Path/to/PRM/records/Mistral/VM_best_checkpoint.pt"
VALUE_MODEL_STATE_DICT = None
LOCAL_VALUE_TYPES = ['glm', 'mistral']
LOCAL_VALUE_IDX = 0
USE_PRM = False

INFERENCE_LOCAL = True
VALUE_LOCAL = True

# # implement the inference model
# if INFERENCE_MODEL_DIR is not None:
#     inference_type = LOCAL_INFERENCE_TYPES[LOCAL_INFERENCE_IDX]
#     inference_tokenizer, inference_model = get_inference_model_vllm(INFERENCE_MODEL_DIR)

# # implement the value model (reward model)
# if VALUE_BASE_MODEL_DIR is not None:
#     value_tokenizer, value_model = get_value_model(VALUE_BASE_MODEL_DIR)

completion_tokens = prompt_tokens = 0
api_key = API_KEY
if api_key != "":
    openai.api_key = api_key
    print(f'api_key:{api_key}\n')
else:
    print("Warning: OPENAI_API_KEY is not set")

api_base = API_BASE
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

def vllm_inference_model(query, inference_model, inference_tokenizer, sampling_args: SamplingParams) -> RequestOutput:
    return get_vllm_response(query=query,
                             model=inference_model,
                             tokenizer=inference_tokenizer,
                             sampling_args=sampling_args
                             )

def local_value_model(prompt: str, response: str, value_model, value_tokenizer):
    return get_local_value(
        model=value_model,
        tokenizer=value_tokenizer,
        prompt=prompt,
        response=response
    )