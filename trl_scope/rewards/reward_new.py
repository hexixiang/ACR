import re
import numpy as np
import json
import os
import datetime

# 正则表达式
think_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
reason_regex = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)
json_regex = re.compile(r"\s*<json>(.*?)</json>\s*", re.DOTALL)

def parse_predictions(completions):
    """从模型输出中解析概率数组"""
    predictions = []
    for completion in completions:
        # 处理嵌套列表格式
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        else:
            content = str(completion)
        if not isinstance(content, str):
            predictions.append(None)
            continue
        
        # 新增：忽略 Markdown 代码块，移除 ```...``` 包装
        content = re.sub(r'```[\w]*\n?(.*?)```', r'\1', content, flags=re.DOTALL)
        
        match = json_regex.search(content)
        if not match:
            predictions.append(None)
            continue
        try:
            probs = json.loads(match.group(1))
            if (isinstance(probs, list) and 
                len(probs) == 1 and 
                all(isinstance(p, (int, float)) for p in probs) and
                0.0 <= probs[0] <= 1.0):
                predictions.append(np.array(probs))
            else:
                predictions.append(None)
        except:
            predictions.append(None)
    return predictions

def cross_entropy_reward(prompts, completions, labels, step=0, **kwargs):
    """使用交叉熵计算奖励"""
    rewards = []
    predictions = parse_predictions(completions)
    
    for pred, true_label in zip(predictions, labels):
        if pred is None:
            rewards.append(0.0)
            continue
        try:
            if isinstance(true_label, list) and len(true_label) > 0:
                true_prob = float(true_label[0])
            elif isinstance(true_label, (int, float)):
                true_prob = float(true_label)
            else:
                rewards.append(0.0)
                continue
            pred_prob = np.clip(pred[0], 1e-10, 1.0 - 1e-10)
            ce_loss = - (true_prob * np.log(pred_prob) + (1 - true_prob) * np.log(1 - pred_prob))
            score = max(0.0, 100.0 - (ce_loss * 50.0))
            rewards.append(score * 0.075)  # 缩放到 0-7.5
        except:
            rewards.append(0.0)

    # 记录日志
    if step % 50 == 0 and any(r > 0 for r in rewards):
        log_completions = []
        for c in completions:
            if isinstance(c, list) and len(c) > 0:
                content = c[0].get("content", "") if isinstance(c[0], dict) else str(c[0])
            else:
                content = str(c)
            log_completions.append(content)  # 移除长度限制，直接记录完整内容
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "ce_r_mean": float(np.mean(rewards)) if rewards else 0.0,
            "completions": log_completions,
            "labels": [str(l) for l in labels],
            "predictions": [str(p) if p is not None else "None" for p in predictions]  # 新增：记录提取的 pred
        }
        os.makedirs("completion_samples", exist_ok=True)
        with open("completion_samples/ce_rewards.json", "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    
    return rewards

def format_reward(prompts, completions, step=0, **kwargs):
    """计算格式奖励"""
    rewards = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        else:
            content = str(completion)
        if not isinstance(content, str):
            rewards.append(0.0)
            continue
        
        # 新增：忽略 Markdown 代码块，移除 ```...``` 包装（与 parse_predictions 一致）
        content = re.sub(r'```[\w]*\n?(.*?)```', r'\1', content, flags=re.DOTALL)
        
        score = 100.0
        reason_valid = bool(reason_regex.search(content))
        json_valid = bool(json_regex.search(content))
        if not reason_valid:
            score -= 30.0
        if not json_valid:
            score -= 50.0
        tags = re.findall(r'(<reason>|<json>)', content)
        if tags and (tags[0] != '<reason>' or len(tags) != 2):
            score -= 20.0
        if re.search(r'(<reason>.*<json>)|(<json>.*<reason>)', content, re.DOTALL):
            score -= 15.0
        match = json_regex.search(content)
        if match:
            try:
                probs = json.loads(match.group(1))
                if (isinstance(probs, list) and len(probs) == 1 and
                    all(isinstance(p, (int, float)) for p in probs) and
                    0.0 <= probs[0] <= 1.0):
                    score += 15.0
            except:
                score -= 10.0
        rewards.append(max(0.0, score * 0.05))  # 缩放到 0-5.75

    # 记录日志
    if step % 50 == 0 and any(r > 0 for r in rewards):
        log_completions = []
        for c in completions:
            if isinstance(c, list) and len(c) > 0:
                content = c[0].get("content", "") if isinstance(c[0], dict) else str(c[0])
            else:
                content = str(c)
            log_completions.append(content)  # 移除长度限制，直接记录完整内容
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "format_r_mean": float(np.mean(rewards)) if rewards else 0.0,
            "completions": log_completions
        }
        os.makedirs("completion_samples", exist_ok=True)
        with open("completion_samples/format_rewards.json", "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    
    return rewards

def length_reward(prompts, completions, step=0, **kwargs):
    """计算长度奖励"""
    rewards = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        else:
            content = str(completion)
        if not isinstance(content, str):
            rewards.append(0.0)
            continue
        length = len(content)
        ideal_min = 600
        ideal_max = 1000
        if length < ideal_min:
            penalty = (ideal_min - length) // 100 * 10
            score = 100.0 - min(100.0, penalty)
        elif length > ideal_max:
            penalty = (length - ideal_max) // 100 * 5
            score = 100.0 - min(100.0, penalty)
        else:
            score = 100.0
        rewards.append(score * 0.01)  # 缩放到 0-1.0

    # 记录日志
    if step % 50 == 0 and any(r > 0 for r in rewards):
        log_completions = []
        for c in completions:
            if isinstance(c, list) and len(c) > 0:
                content = c[0].get("content", "") if isinstance(c[0], dict) else str(c[0])
            else:
                content = str(c)
            log_completions.append(content)  # 移除长度限制，直接记录完整内容
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "len_r_mean": float(np.mean(rewards)) if rewards else 0.0,
            "completions": log_completions
        }
        os.makedirs("completion_samples", exist_ok=True)
        with open("completion_samples/length_rewards.json", "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    
    return rewards