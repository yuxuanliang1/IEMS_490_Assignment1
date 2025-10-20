# src/iems490/dataset.py
from typing import List, Dict
import random
from .config import RANDOM_SEED, N_SAMPLES

def load_gsm8k_subset(split: str = "test") -> List[Dict]:
    """
    加载 Hugging Face 的 openai/gsm8k 并从指定 split 随机抽取 N_SAMPLES 条。
    返回：[{id, question, answer}]；answer 的最后一行形如 '#### <number>'
    说明：严格按照你的要求使用以下两行来加载数据集：
          from datasets import load_dataset
          ds = load_dataset("openai/gsm8k", "main")
    """
    # ★ 按你的要求写在函数体内
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")

    data_split = ds[split]
    total = len(data_split)
    if N_SAMPLES > total:
        raise ValueError(f"N_SAMPLES={N_SAMPLES} > {split} size={total}")

    idxs = list(range(total))
    random.Random(RANDOM_SEED).shuffle(idxs)
    idxs = idxs[:N_SAMPLES]
    subset = data_split.select(idxs)

    return [
        {"id": i, "question": r["question"], "answer": r["answer"]}
        for i, r in zip(idxs, subset)
    ]
