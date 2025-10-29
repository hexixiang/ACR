# Adaptive Curriculum Reinforcement Learning for Mathematical Reasoning

[![Paper](https://img.shields.io/badge/Paper-WWW%202026-blue)](WWW_2026_paper_4458.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Abstract

This repository contains the official implementation of our WWW 2026 paper on **Adaptive Curriculum Reinforcement Learning** for mathematical reasoning tasks. We propose a novel approach that extends the Group Relative Policy Optimization (GRPO) algorithm with adaptive curriculum learning strategies and custom reward shaping mechanisms to enhance large language models' mathematical problem-solving capabilities.

Our method demonstrates significant improvements over baseline approaches by:
- **Adaptive Curriculum Design**: Progressively training on problems of increasing difficulty levels
- **Multi-faceted Reward Functions**: Combining accuracy, format compliance, and reasoning quality metrics
- **Efficient Training**: Leveraging vLLM for accelerated inference during reinforcement learning

## Overview

### Research Motivation

Mathematical reasoning remains a challenging task for large language models. Traditional reinforcement learning approaches often struggle with:
1. High variance in reward signals
2. Inefficient exploration of solution spaces  
3. Difficulty in learning from sparse feedback

This work addresses these challenges through curriculum-based training with carefully designed reward functions that guide the model toward producing well-formatted, step-by-step mathematical reasoning.

### Key Contributions

1. **Enhanced GRPO Trainer**: Extended implementation with support for adaptive difficulty progression
2. **Custom Reward Functions**: Novel reward shaping including:
   - **Accuracy Rewards**: Verifying correctness of mathematical solutions
   - **Format Rewards**: Encouraging proper reasoning structure  
   - **Cross-Entropy Rewards**: Fine-grained probability-based feedback
   - **Length Rewards**: Optimizing response verbosity
3. **Curriculum Learning Pipeline**: Systematic progression through difficulty levels
4. **Comprehensive Evaluation**: Benchmarking on MATH dataset with 500 test problems

## Repository Structure

```
.
├── trl_scope/                      # Core implementation
│   ├── trainer/
│   │   ├── grpo_trainer.py        # Modified GRPO trainer with adaptive features
│   │   ├── grpo_config.py         # Training configuration
│   │   └── ...                    # Other trainer components
│   ├── rewards/                    # Custom reward functions
│   │   ├── accuracy_rewards.py    # Correctness verification (box_accuracy_reward, think_accuracy_reward)
│   │   ├── reward_new.py          # Novel reward functions (cross_entropy, format, length)
│   │   ├── format_rewards.py      # Format compliance rewards
│   │   ├── other_rewards.py       # Additional utilities (soft overlong punishment)
│   │   └── math_grader.py         # Mathematical answer evaluation
│   ├── scripts/
│   │   └── grpo.py                # Main training script
│   └── accelerate_configs/        # Distributed training configurations
│       └── zero3.yaml             # DeepSpeed ZeRO-3 configuration
├── data/
│   └── eval/
│       └── math500.jsonl          # Evaluation dataset (500 MATH problems)
├── train_grpo.sh                   # Training launcher script
├── requirements.txt                # Python dependencies
└── WWW_2026_paper_4458.pdf        # Full paper
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 12.x (for GPU support)
- 4+ GPUs recommended (for distributed training)

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ACR-github.git
cd ACR-github
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `transformers>=4.56.2`: Hugging Face model library
- `torch>=2.8.0`: PyTorch framework
- `accelerate>=1.10.1`: Distributed training
- `vllm>=0.10.2`: Fast inference engine
- `deepspeed>=0.17.6`: Training optimization
- `math-verify>=0.8.0`: Mathematical answer verification
- `wandb>=0.22.0`: Experiment tracking

## Usage

### Training

The main training script supports adaptive curriculum learning across difficulty levels. 

**Basic Training Command**:
```bash
bash train_grpo.sh
```

**Key Training Parameters** (configured in `train_grpo.sh`):
- `--model_name_or_path`: Base model (e.g., `Qwen/Qwen2.5-Math-1.5B`)
- `--dataset_name`: Training dataset path (with difficulty level indicator)
- `--reward_funcs`: Reward function selection (e.g., `box_accuracy_reward`)
- `--num_generations`: Number of samples per prompt (default: 4)
- `--beta`: KL divergence penalty coefficient (default: 0.0)
- `--temperature`: Sampling temperature (default: 0.6)
- `--learning_rate`: Optimizer learning rate (default: 1e-6)
- `--use_vllm`: Enable vLLM acceleration (default: True)
- `--vllm_mode`: vLLM execution mode (`colocate` or `server`)

**Custom Training Script**:
```python
from trl_scope import GRPOTrainer, GRPOConfig
from trl_scope.rewards import box_accuracy_reward
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

# Load dataset
dataset = load_dataset('json', data_files='path/to/your/data.json')

# Configure training
config = GRPOConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    num_generations=4,
    max_completion_length=1024,
    temperature=0.6,
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=box_accuracy_reward,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# Train
trainer.train()
```

### Evaluation

Evaluation is integrated into the training loop. The model is periodically evaluated on `data/eval/math500.jsonl`.

**Standalone Evaluation**:
```python
from datasets import load_dataset
from trl_scope.rewards import box_accuracy_reward

# Load evaluation dataset
eval_dataset = load_dataset('json', data_files='data/eval/math500.jsonl')['train']

# Generate and evaluate
# (Use your trained model to generate solutions, then compute accuracy)
```

### Reward Functions

The framework supports multiple reward functions that can be combined:

1. **`box_accuracy_reward`**: Verifies mathematical correctness using `math-verify` and custom graders
2. **`cross_entropy_reward`**: Probability-based fine-grained feedback (0-7.5 scale)
3. **`format_reward`**: Ensures proper `<reason>` and `<json>` tag structure (0-5.75 scale)
4. **`length_reward`**: Penalizes overly short or long responses (0-1.0 scale)

**Custom Reward Function Example**:
```python
def custom_reward(completions, **kwargs):
    """
    Custom reward function template.
    
    Args:
        completions: List of model-generated completions
        **kwargs: Additional dataset columns (e.g., 'answer', 'problem')
    
    Returns:
        List of float rewards
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # Your custom logic here
        reward = compute_your_reward(content, kwargs.get('answer'))
        rewards.append(reward)
    return rewards
```

## Dataset

### Training Data Format

Training data should be in JSON format with the following structure:
```json
{
  "problem": "Solve for x: 2x + 5 = 13",
  "answer": "4",
  "level": 0,
  "subject": "Algebra"
}
```

### Evaluation Data

The evaluation set (`data/eval/math500.jsonl`) contains 500 problems sampled from the MATH dataset, covering:
- **Subjects**: Algebra, Precalculus, Geometry, Number Theory, etc.
- **Difficulty Levels**: Level 1-5 (increasing complexity)

Each problem includes:
- `problem`: Problem statement (may include LaTeX)
- `solution`: Step-by-step solution
- `answer`: Final answer in boxed format
- `subject`: Mathematical subject area
- `level`: Difficulty level (1-5)

## Results

Our approach demonstrates significant improvements over baseline methods:

| Method | Accuracy (%) | Avg. Reward | Reasoning Quality |
|--------|-------------|-------------|-------------------|
| Baseline GRPO | - | - | - |
| **Ours (ACR)** | **-** | **-** | **-** |

*Note: Please refer to the [paper](WWW_2026_paper_4458.pdf) for detailed experimental results.*

### Key Findings

1. **Curriculum Learning Benefits**: Progressive difficulty training improves final performance by X%
2. **Reward Shaping Impact**: Multi-faceted rewards lead to better-formatted, more accurate solutions
3. **Efficiency Gains**: vLLM integration reduces training time by X%

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{anonymous2026acr,
  title={Adaptive Curriculum Reinforcement Learning for Mathematical Reasoning},
  author={Anonymous},
  booktitle={Proceedings of The Web Conference 2026},
  year={2026},
  organization={ACM}
}
```

## Acknowledgments

This work builds upon:
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) - Base training framework
- [DeepSeekMath](https://github.com/deepseek-ai/DeepSeek-Math) - GRPO algorithm inspiration
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference engine
- [MATH Dataset](https://github.com/hendrycks/math) - Evaluation benchmark

## License

This project is released under the Apache 2.0 License. See the paper for full details.

## Contact

For questions or issues, please:
- Open an issue in this repository
- Contact: [your-email@institution.edu]

---

**Note**: This is a research codebase under active development. Some features may be experimental.
