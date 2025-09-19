# get different responses based on the model type
import torch
from .model import *
from vllm import LLM, SamplingParams, RequestOutput

# given prompt, generate proposal under instruction, unwrap is required
def get_proposal(question_len: int, prompt: str, temperature=0.7, max_new_tokens=50, min_p=0, max_length=2048, num_branch: int=3, stop: str=". ", stop_token_ids=None, num_stop_times=1, verbose=False, inference_model=None, inference_tokenizer=None) -> list[tuple[str, bool, int]]:
    """
    Get the proposal from the current node. The proposal is a list of completions/generations, with the length equal to the beam size. For each layer, we generate up to `num_branch` (branches of a single node) * `top_k` generations.

    Args:
        max_length: The maximum length of the response, which is `y` of the current node, regardless of the input question.
    """
    print(f"Stop is {stop}")
    current_length = len(prompt)
    sampling_args = SamplingParams(
        temperature=temperature,
        max_tokens=max_length - current_length,
        min_p=min_p,
        stop=stop,
        stop_token_ids=stop_token_ids,
        n=num_branch,
    )
    final_returns = []
    activated_generations = []
    for _ in range(num_stop_times):
        responses = vllm_inference_model(prompt, inference_model, inference_tokenizer, sampling_args)
        return_responses = []
        for output in responses:
            for idx in range(min(num_branch, len(output.outputs))):
                is_terminal = False
                # TODO: check if we can do so
                if inference_tokenizer.eos_token_id in output.outputs[idx].token_ids:
                    # NOTE: truncate things after eos token
                    output.outputs[idx].token_ids = output.outputs[idx].token_ids[:output.outputs[idx].token_ids.index(inference_tokenizer.eos_token_id)+1]
                    # print(output.outputs[idx].token_ids)
                    print("Terminal state by eos!")
                    if verbose:
                        print("Terminal state by eos!")
                    is_terminal = True
                total_len = len(output.outputs[idx].token_ids) + current_length # TODO: should subtract the input prompt length
                if total_len >= max_length:
                    if verbose:
                        print("Terminal state by length!")
                    is_terminal = True

                if sampling_args.stop != []:
                    if not is_terminal:
                        # print(f"Token ids: {output.outputs[idx].token_ids}")
                        # print(f"Stop token ids: {inference_tokenizer.encode(sampling_args.stop[0], add_special_tokens=False)}")
                        return_responses.append((output.outputs[idx].token_ids, is_terminal, len(output.outputs[idx].token_ids)))
                    else:
                        return_responses.append((output.outputs[idx].token_ids, is_terminal, len(output.outputs[idx].token_ids)))
                else:
                    return_responses.append((output.outputs[idx].token_ids, is_terminal, len(output.outputs[idx].token_ids)))
        
        activated_generations = []
        full_activated_generations = []
        for return_response in return_responses:
            if return_response[1] == False:
                activated_generations.append(return_response[0])
                full_activated_generations.append(return_response)
            else:
                final_returns.append(return_response)
        current_length += len(activated_generations)
        sampling_args = SamplingParams(
            temperature=temperature,
            max_tokens=max_length - current_length,
            min_p=min_p,
            stop=stop,
            stop_token_ids=stop_token_ids,
            n=1,
        )
        prompt = activated_generations
    if len(full_activated_generations) > 0:
        final_returns += full_activated_generations
    assert num_branch == len(final_returns), f"num_branch {num_branch} should be equal to the length of final_returns {len(final_returns)}"
    return return_responses

def get_value(prompt: str, response: str, value_model, value_tokenizer):
    value = local_value_model(prompt, response, value_model, value_tokenizer)

    return value