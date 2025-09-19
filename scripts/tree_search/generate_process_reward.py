import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM


from transformers import AutoTokenizer
from prm.labeler import TDLabeler, StepParser
from .utils import flatten_args, load_data, parse_answer, parse_answer_mapping, get_file_line_num

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra.main(config_path=f"{PROJ_ROOT}/experiments/configs", config_name="default")
def main(args: DictConfig):
    args = flatten_args(args)
    print(args)
    llm = LLM(model=args.model_name_or_path, gpu_memory_utilization=0.8)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    raw_data = load_data(args.dataset_name, args.ds_config)[args.split]

    step_parser = StepParser()
    labeler = TDLabeler(llm, step_parser, tokenizer)

    outputs = []

    # parse answers from the solutions
    local_scope = {}
    # try:
    #     parse_answer = eval(args.answer_parse_func)
    # except:
    exec(args.answer_parse_func, globals(), local_scope)
    with open(args.source_data_path, "r") as f:
        for line in f:
            outputs.append(eval(line))

    parsed_data = raw_data.map(
        lambda example: {**example, "parsed_answer": parse_answer(example[args.solution_key])}
    )
    output_dir = f"{PROJ_ROOT}/outputs/process_reward/{args.dataset_output}/"
    output_path = f"{output_dir}{args.split}.json"
    if not os.path.exists(output_dir):
        os.makedirs(os.path.dirname(output_dir))

    file_line_num = get_file_line_num(output_path)
    if file_line_num != 0:
        print(f"Resume from {file_line_num}")

    for question, output, target in zip(parsed_data[args.prompt_key][file_line_num:], outputs[file_line_num:], parsed_data["parsed_answer"][file_line_num:]):
        write_item = {
            "question": question,
            "outputs": [], # (steps, process_reward, task_reward, target_reward, correctness, spliter)
        }
        for item in output["assistant"]:
            process_rewards, correctness, task_reward, target_reward = labeler.label(question, item['content'], target)
            if correctness == 1:
                correct = "correct"
            else:
                correct = "incorrect"
            write_item['outputs'].append((step_parser(item['content']), process_rewards, task_reward, target_reward, correct, step_parser.spliter))

            if args.debug:
                print(f"process_rewards: {process_rewards}")
                break
        with open(output_path, "a") as f:
            f.write(json.dumps(write_item))
            f.write('\n')
        if args.debug:
            break

if __name__ == "__main__":
    print(PROJ_ROOT)
    main()