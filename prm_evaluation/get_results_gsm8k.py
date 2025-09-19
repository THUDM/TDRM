import json
from tqdm import tqdm
from datasets import load_dataset

gsm8k = load_dataset("openai/gsm8k", "main")['test']

gts = [x.split("#### ")[-1] for x in gsm8k['answer']]

import re

def apply_filters(completion, filter_list, regexes_to_ignore):
    """
    Extract pure number answers from completions using specified filters.

    Args:
        completion (str): The completion text to process.
        filter_list (list): List of filters to apply.
        regexes_to_ignore (list): List of regex patterns to remove from the extracted results.

    Returns:
        str or None: The extracted number, or None if no number matches.
    """

    def clean_result(result, regexes_to_ignore):
        """Remove patterns specified in regexes_to_ignore from the result."""
        for pattern in regexes_to_ignore:
            result = re.sub(pattern, "", result)
        return result.strip()

    for filter_item in filter_list:
        for filter_step in filter_item["filter"]:
            if filter_step["function"] == "regex":
                regex_pattern = filter_step["regex_pattern"]
                matches = re.findall(regex_pattern, completion)

                if matches:
                    # If group_select is specified, select the appropriate group
                    if "group_select" in filter_step:
                        group_select = filter_step["group_select"]
                        if group_select == -1:  # Select the last group
                            matches = [match[-1] if isinstance(match, tuple) else match for match in matches]
                    
                    # Take the first match if required
                    if "take_first" in [step["function"] for step in filter_item["filter"]]:
                        result = matches[-1] if matches else None
                        if result:
                            return clean_result(result, regexes_to_ignore)
    return None

# Example usage
filter_list = [
    {
        "name": "strict-match",
        "filter": [
            {"function": "regex", "regex_pattern": "#### (\\-?[0-9\\.\\,]+)"},
            {"function": "take_first"}
        ]
    },
    {
        "name": "flexible-extract",
        "filter": [
            {"function": "regex", "group_select": -1, "regex_pattern": "(-?[0-9]{1,}[\\.,]?[0-9]*)"},
            {"function": "take_first"}
        ]
    }
]

regexes_to_ignore = [
    ",",
    "\\$",
    "(?s).*#### ",
    "\\.$"
]

if __name__ == "__main__":
    completion_path = "/path/to/TDRM/prm_evaluation/outputs/gsm8k/qwen25_scalar_orm_baseline/output.json"


    completions = []
    with open(completion_path, "r") as f:
        for line in f:
            completions.append(json.loads(line))

    # sort the answers by reward for each question
    for i in tqdm(range(len(completions))):
        completions[i]["assistant"] = sorted(completions[i]["assistant"], key=lambda x: x["reward"])

    # extract answer for each item in completions
    for i in tqdm(range(len(completions))):
        for j in range(len(completions[i]["assistant"])):
            completions[i]["assistant"][j]["answer"] = apply_filters(completions[i]["assistant"][j]["content"], filter_list, regexes_to_ignore)

    best_of_k_acc = []

    def cal_acc(gts, preds):
        acc = 0
        for i in range(len(gts)):
            if gts[i] == preds[i]:
                acc += 1
        return acc / len(gts)

    for i in tqdm(range(len(completions[0]["assistant"]))):
        best_of_k_acc.append(cal_acc(gts, [x["answer"] for x in [completions[j]["assistant"][i] for j in range(len(completions))]]))

    print(best_of_k_acc[-1])

    def get_pass_at_k(data_path, num=-1):
        completions = []
        with open(data_path, "r") as f:
            for line in f:
                completions.append(json.loads(line))

        completions = completions

        # extract answer for each item in completions
        for i in tqdm(range(len(completions))):
            for j in range(len(completions[i]["assistant"])):
                completions[i]["assistant"][j]["answer"] = apply_filters(completions[i]["assistant"][j]["content"], filter_list, regexes_to_ignore)
        
        def get_pass_list(gts, preds_list):
            pass_list = []
            for i in range(len(gts)):
                if gts[i] in preds_list[i]:
                    pass_list.append(1)
                else:
                    pass_list.append(0)
            return pass_list

        # pass@k
        def cal_pass(gts, preds_list):
            gts = gts
            pass_k = 0
            for i in range(len(gts)):
                if gts[i] in preds_list[i]:
                    pass_k += 1
            return pass_k / len(gts)

        pass_at_k = []
        completion_list = [[] for j in range(len(completions))]
        for i in range(len(completions[0]["assistant"])):
            for j in range(len(completions)):
                completion_list[j].append(completions[j]["assistant"][i]["answer"])

            pass_at_k.append(cal_pass(gts, completion_list))
        pass_list = get_pass_list(gts, completion_list)

        best_of_k_acc = []

        def argmax_answers(preds):
            return max(preds, key=lambda x: x["reward"])["answer"]

        def cal_acc(gts, preds):
            gts = gts
            acc = 0
            for i in range(len(gts)):
                if gts[i] == preds[i]:
                    acc += 1
            return acc / len(gts)

        for i in tqdm(range(len(completions[0]["assistant"]))):
            best_of_k_acc.append(cal_acc(gts, [argmax_answers(x["assistant"][:i+1]) for x in completions]))

        return pass_at_k, best_of_k_acc, pass_list

    import matplotlib.pyplot as plt

    # Sample data for two lists of accuracy

    _, best_of_k_acc, _ = get_pass_at_k(completion_path)

    epochs = range(1, len(best_of_k_acc) + 1)

    print(best_of_k_acc[-1])
    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, best_of_k_acc, marker='o', linestyle='-', label="Best-of-N", linewidth=2)

    # Adding chart details
    # The line `plt.title("Bo128 on GSM8K TD1 int_cos_clamp 1413k(ds) mistral-v2", fontsize=16,
    # fontweight='bold')` in the code snippet is setting the title of the plot that will be generated.
    plt.title("Bo128 on GSM8K ORM", fontsize=16, fontweight='bold')
    plt.xlabel("# of Samples (N)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(title="Methods", fontsize=10, title_fontsize=12)
    plt.tight_layout()


    plt.savefig("/path/to/TDRM/prm_evaluation/bon_128_gsm8k/qwen_scalar_orm.png")
    # Display the chart
    plt.show()
