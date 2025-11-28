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

## Results

Our approach demonstrates significant improvements over baseline methods across multiple model scales and benchmarks.

### Main Results

Performance comparison on mathematical reasoning benchmarks. All methods are trained for one epoch (500 steps) on Level3-500 dataset and evaluated using greedy decoding (pass@1 accuracy %).

#### Qwen2.5-3B

| Method | ACR↓ | MATH | GSM8K | Minerva | Olympiad | AMC | AIME | Avg. |
|--------|------|------|-------|---------|----------|-----|------|------|
| Base | - | 28.0 | 8.0 | 7.4 | 7.5 | 9.0 | 0.3 | 10.0 |
| INTUITOR | - | 32.4 | 45.8 | 12.6 | 18.7 | 21.3 | 4.1 | 22.5 |
| RENT | - | 29.7 | 41.5 | 11.1 | 16.9 | 19.2 | 3.3 | 20.3 |
| Vanilla GRPO | 0.37 | 36.8 | 52.6 | 15.3 | 22.4 | 24.8 | 5.9 | 26.3 |
| **AVSPO (Ours)** | **0.14** | **42.7** | **61.3** | **18.9** | **26.5** | **29.5** | **7.8** | **31.1** |
| Δ vs. GRPO | -62% | +5.9 | +8.7 | +3.6 | +4.1 | +4.7 | +1.9 | **+4.8** |

#### Qwen2.5-3B-Instruct

| Method | ACR↓ | MATH | GSM8K | Minerva | Olympiad | AMC | AIME | Avg. |
|--------|------|------|-------|---------|----------|-----|------|------|
| Base | - | 63.0 | 43.7 | 25.7 | 24.4 | 27.4 | 5.0 | 31.5 |
| INTUITOR | - | 65.3 | 61.4 | 26.9 | 27.6 | 30.8 | 7.6 | 36.6 |
| RENT | - | 64.1 | 56.2 | 26.1 | 25.8 | 28.9 | 6.4 | 34.6 |
| Vanilla GRPO | 0.35 | 68.9 | 68.7 | 28.4 | 31.2 | 33.7 | 9.8 | 40.1 |
| **AVSPO (Ours)** | **0.13** | **73.6** | **75.8** | **31.2** | **35.1** | **37.4** | **12.3** | **44.2** |
| Δ vs. GRPO | -63% | +4.7 | +7.1 | +2.8 | +3.9 | +3.7 | +2.5 | **+4.1** |

#### Qwen2.5-Math-1.5B

| Method | ACR↓ | MATH | GSM8K | Minerva | Olympiad | AMC | AIME | Avg. |
|--------|------|------|-------|---------|----------|-----|------|------|
| Base | - | 31.8 | 15.1 | 11.4 | 22.2 | 27.0 | 3.2 | 18.4 |
| INTUITOR | - | 52.4 | 43.6 | 19.7 | 31.4 | 35.2 | 8.5 | 31.8 |
| RENT | - | 47.8 | 38.4 | 17.2 | 28.5 | 32.8 | 6.9 | 28.6 |
| Vanilla GRPO | 0.40 | 58.6 | 49.8 | 19.2 | 31.7 | 37.6 | 10.6 | 34.6 |
| **AVSPO (Ours)** | **0.15** | **67.2** | **59.3** | **28.9** | **37.8** | **41.6** | **14.2** | **41.5** |
| Δ vs. GRPO | -63% | +8.6 | +9.5 | +9.7 | +6.1 | +4.0 | +3.6 | **+6.9** |

#### Qwen2.5-Math-7B

| Method | ACR↓ | MATH | GSM8K | Minerva | Olympiad | AMC | AIME | Avg. |
|--------|------|------|-------|---------|----------|-----|------|------|
| Base | - | 60.8 | 51.2 | 20.2 | 30.4 | 35.0 | 13.3 | 35.2 |
| INTUITOR | - | 68.9 | 62.5 | 25.3 | 37.1 | 39.2 | 18.4 | 41.9 |
| RENT | - | 65.4 | 57.8 | 23.1 | 34.6 | 37.5 | 16.1 | 39.1 |
| Vanilla GRPO | 0.33 | 65.0 | 65.3 | 25.7 | 36.2 | 43.8 | 20.6 | 42.8 |
| **AVSPO (Ours)** | **0.14** | **74.1** | **69.7** | **29.4** | **43.6** | 40.9 | **23.2** | **46.8** |
| Δ vs. GRPO | -58% | +9.1 | +4.4 | +3.7 | +7.4 | -2.9 | +2.6 | **+4.0** |

### Key Findings

| Metric | Improvement |
|--------|-------------|
| **ACR Reduction** | 58-63% (from 0.33-0.40 → 0.13-0.15) |
| **Sample Efficiency** | 60% relative improvement |
| **Average Accuracy Gain** | +4.0 to +6.9 percentage points |
| **Collapse Rate Reduction** | From 33-40% → 14% |

### ACR Predictive Power

Early-stage ACR (first 100 steps) strongly predicts final model performance:

- **Pearson correlation**: r = -0.785 (p < 10⁻⁸)
- **Regression model**: `Final Accuracy = 51.4 - 29.6 × ACR₁₀₀`
- **Coefficient of determination**: R² = 0.617

This means every 0.1 increase in early ACR corresponds to ~3% decrease in final accuracy.

### Cross-Model Generalization

AVSPO generalizes beyond Qwen2.5 to other model families:

| Model | Vanilla GRPO | AVSPO | Δ Accuracy |
|-------|--------------|-------|------------|
| LLaMA-3-8B | 52.4% (ACR: 0.36) | 57.8% (ACR: 0.16) | **+5.4%** |
| Mistral-7B | 49.1% (ACR: 0.38) | 54.3% (ACR: 0.17) | **+5.2%** |

### Ablation Studies

#### Virtual Sample Generation Strategies

| Strategy | MATH-500 Acc. | Avg. ACR |
|----------|---------------|----------|
| No augmentation (GRPO) | 58.6% | 0.40 |
| Random uniform | 62.1% | 0.32 |
| Fixed partial credit | 63.5% | 0.29 |
| **Stratified (Ours)** | **67.2%** | **0.15** |

#### Sensitivity Parameter α

| α | MATH-500 Acc. | Avg. ACR |
|---|---------------|----------|
| 0.3 | 64.8% | 0.22 |
| **0.5** | **67.2%** | **0.15** |
| 0.7 | 65.9% | 0.18 |
| 1.0 | 63.2% | 0.24 |

#### Adaptive vs. Fixed Thresholding

| Threshold Type | MATH-500 | Avg. ACR | Steps to 65% |
|----------------|----------|----------|--------------|
| Fixed τ = 0.3 | 63.9% | 0.27 | 420 |
| Fixed τ = 0.5 | 65.4% | 0.21 | 380 |
| Fixed τ = 0.7 | 62.1% | 0.31 | 450+ |
| **Adaptive (Ours)** | **67.2%** | **0.15** | **310** |

The adaptive mechanism converges **22% faster** than the best fixed threshold.

*Note: Please refer to the [paper](WWW_2026_paper_4458.pdf) for detailed experimental results.*


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
- Contact: xxx

---

**Note**: This is a research codebase under active development. Some features may be experimental.
