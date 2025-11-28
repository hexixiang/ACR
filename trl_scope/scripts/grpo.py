# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import weave
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from trl_scope import (
    DatasetMixtureConfig,
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)
from peft import get_peft_model
from trl_scope.rewards import get_soft_overlong_punishment, think_format_reward, think_format_reward, box_format_reward, think_accuracy_reward, box_accuracy_reward


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
# WandB configuration - set via environment variables or command line args
os.environ.setdefault("WANDB_PROJECT", "math_grpo")
# Note: Set WANDB_API_KEY in your environment or through `wandb login`

reward_funcs_registry = {
    "think_format_reward": think_format_reward,
    "get_soft_overlong_punishment": get_soft_overlong_punishment(max_completion_len=1280, soft_punish_cache=256),
    "box_format_reward": box_format_reward,
    "think_accuracy_reward": think_accuracy_reward,
    "box_accuracy_reward": box_accuracy_reward,
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        reward_funcs (`list[str]` or `None`, *optional*, defaults to `None`):
            Reward functions to use. Supported values are:

                - `"think_format_reward"`
                - `"get_soft_overlong_punishment"` (used value are `max_completion_len=1280`, `soft_punish_cache=256`)
                - any dotted import path " (e.g., `'my_lib.rewards.custom_reward'`).
    """

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: `think_format_reward`, "
            "`get_soft_overlong_punishment` (used value are `max_completion_len=1280`, `soft_punish_cache=256`), or "
            "any dotted import path (e.g., `'my_lib.rewards.custom_reward'`)."
        },
    )


def main(script_args, training_args, model_args, dataset_args):
    level = script_args.dataset_name.split("/")[-1].split("-")[-1]
    os.environ["WANDB_NAME"] = f"{training_args.run_name}"  # 设置运行时名称
    # Load a model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=config.torch_dtype, trust_remote_code=model_args.trust_remote_code
    )
    print("model:",model.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )

    # # Load the dataset
    # if dataset_args.datasets and script_args.dataset_name:
    #     logger.warning(
    #         "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
    #         "dataset and `dataset_name` will be ignored."
    #     )
    # elif dataset_args.datasets and not script_args.dataset_name:
    #     dataset = get_dataset(dataset_args)
    # elif not dataset_args.datasets and script_args.dataset_name:
    #     dataset = load_dataset(
    #         script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
    #     )
    # else:
    #     raise ValueError("Either `datasets` or `dataset_name` must be provided.")
    

    # Load the dataset
    if "." in script_args.dataset_name: 
        dataset = load_dataset('json', data_files=script_args.dataset_name)
    else: 
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Load evaluation dataset
    eval_dataset_path = os.path.join(os.path.dirname(__file__), "../../data/eval/math500.jsonl")
    math_eval_dataset = load_dataset('json', data_files=eval_dataset_path)['train']
    math_eval_dataset = math_eval_dataset.shuffle(seed=42).select(range(100))
    # Filter_level and select 500 items
    # train_dataset = dataset.filter(lambda x: x['level'] == 'Level 3')['train'].select(range(500))
    train_dataset = dataset['train']
    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
    def make_conversation(example):
        if 'problem' in example:
            question = example['problem']
        elif 'question' in example:
            question = example['question']
        msg ={
            "prompt": 
               [{'role': 'system', 'content': SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": question
                }]
            }
        
        return msg
    
    def make_llama_conversation(example):
        question = example['problem']
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n\n"
                }]
            }
        
        return msg

    train_dataset = train_dataset.map(make_conversation)
    math_eval_dataset = math_eval_dataset.map(make_conversation)
    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=math_eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args,_ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    # print(script_args,training_args,model_args, dataset_args)
    main(script_args, training_args, model_args, dataset_args)
    # main(script_args, training_args, model_args)

