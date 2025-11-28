import re
from .math_grader import boxed_reward_fn
from .oat_math_grader import boxed_reward_fn as oat_evaluate
from math_verify import parse, verify

def think_accuracy_reward(completions, **kwargs):
    # Regular expression to capture content inside \boxed{}
    completion_contents = [completion[0]["content"] for completion in completions] 
    matches = [re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL) for completion in completion_contents]
    contents = [match.group(1) if match else "" for match in matches]
    #import pdb; pdb.set_trace()
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, kwargs["answer"])]

# def box_accuracy_reward(completions, **kwargs):
#     completion_contents = [completion[0]["content"] for completion in completions] 
#     #matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completion_contents]
#     #contents = [match.group(1).strip() if match else "" for match in matches]
#     result = []
#     for c, gt in zip(completion_contents, kwargs["answer"]):
#         info, r = boxed_reward_fn(c, gt, fast=False)
#         result.append(r)
#     # print(result)
#     return result

def box_accuracy_reward(completions, **kwargs):
    gold_answers = kwargs.get('answer', [])
    # 先用math_verify验证
    try:
        gold_answers_parsed = [parse("$" + answer + "$") for answer in gold_answers]
        models_responses = [completion[0]['content'] for completion in completions]
        models_answers_parsed = list(map(parse, models_responses))
        labels = list(map(verify, gold_answers_parsed, models_answers_parsed))
    except Exception as e:
        print(f"Error parsing answers: {e}")
        labels = [False] * len(completions)
    # 如果labels有False，则用oat_evaluate补充验证
    for i, label in enumerate(labels):
        if not label:
            if oat_evaluate(models_responses[i], gold_answers[i], fast=False)[1]==1.0:
                labels[i] = True
    # print("Model Responses:", models_responses)
    result = [1.0 if label else 0.0 for label in labels]
    # print("Reward:", result)
    return result