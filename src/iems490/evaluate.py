# src/iems490/evaluate.py
import pandas as pd
from tqdm import tqdm
from typing import Optional

from .dataset import load_gsm8k_subset
from .solve import solve_one
from .score import extract_final_number, extract_gold_number, numerically_equal

# 默认导入“基线”模板（不传参数时用它）
from .prompts_base import BASE_SYSTEM as DEFAULT_SYSTEM
from .prompts_base import BASE_USER_TEMPLATE as DEFAULT_USER

def run_eval(
    split: str = "test",
    csv_out: str = "results_base.csv",
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> float:
    """
    评测入口：可显式传入 (system, user_template)，否则使用基线模板。
    返回准确率，并写出 CSV。
    """
    system_prompt = system_prompt or DEFAULT_SYSTEM
    user_template = user_template or DEFAULT_USER

    rows = load_gsm8k_subset(split)
    recs = []
    for r in tqdm(rows, desc="Evaluating"):
        pred_text = solve_one(r["question"], system_prompt, user_template)
        gold = extract_gold_number(r["answer"])
        pred = extract_final_number(pred_text)
        correct = int(pred is not None and gold is not None and numerically_equal(pred, gold))
        recs.append(dict(
            id=r["id"],
            question=r["question"],
            gold=gold,
            pred_text=pred_text,
            pred=pred,
            correct=correct,
        ))
    df = pd.DataFrame(recs)
    df.to_csv(csv_out, index=False)
    acc = df["correct"].mean()
    print(f"Base prompt accuracy on {len(df)} samples: {acc:.3%}")
    return acc

if __name__ == "__main__":
    # 允许直接 python -m iems490.evaluate 跑基线
    run_eval(split="test", csv_out="results_base.csv")
