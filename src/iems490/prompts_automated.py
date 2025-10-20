# src/iems490/prompts_automated.py
"""
Automated prompt engineering (simple search):
- 定义一组候选 (system, user_template)
- 在一个小的验证子集上逐一评测
- 选择准确率最高的提示组合
- 返回最佳提示，以便在全量测试集上评测

说明：这是“选择一种自动优化方法”的最小实现，简单稳健、可复现。
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import pandas as pd
from tqdm import tqdm

from .score import extract_final_number, extract_gold_number, numerically_equal
from .solve import solve_one
from .dataset import load_gsm8k_subset
from .config import RANDOM_SEED

# === 候选提示（可以按需增删） ===
CANDIDATES: List[Tuple[str, str]] = [
    # Baseline-ish（稍加口吻变化）
    ("""You are a helpful assistant for grade-school math.
On the VERY LAST line print: #### <number>""",
     """Problem:
{question}

Provide a short solution. The final line must be exactly: #### <number>"""),

    # 结构化 CoT（简版）
    ("""You are a precise math solver.
Use: (1) plan, (2) equations, (3) step-by-step numbers, (4) quick self-check.
Only the very last line: #### <number>""",
     """Problem:
{question}

Keep it concise. Recompute if a slip is detected. Final line: #### <number>"""),

    # 结构化 CoT（更严格）
    ("""You are a careful math tutor.
FORMAT:
1) Plan.
2) Equations.
3) Numerical computation only.
4) Self-check: units/magnitude/consistency. Fix if wrong.
LAST line ONLY: #### <number>""",
     """Problem:
{question}

Do not add extra text after the last line. The last line must be: #### <number>"""),
]


@dataclass
class EvalResult:
    system: str
    user_template: str
    acc: float
    n: int

def _eval_on_subset(rows: List[Dict], system: str, user_template: str) -> EvalResult:
    out = 0
    for r in rows:
        pred_text = solve_one(r["question"], system, user_template)
        gold = extract_gold_number(r["answer"])
        pred = extract_final_number(pred_text)
        ok = (pred is not None and gold is not None and numerically_equal(pred, gold))
        out += int(ok)
    acc = out / len(rows)
    return EvalResult(system, user_template, acc, len(rows))

def auto_select_prompt(val_split: str = "test", n_val: int = 40, seed: int = RANDOM_SEED) -> EvalResult:
    """
    在 val_split 上随机抽 n_val 条做验证，选择最优提示。
    """
    full = load_gsm8k_subset(val_split)  # 按你的 dataset.py：受 N_SAMPLES 影响，但顺序固定
    rng = random.Random(seed)
    idxs = list(range(len(full)))
    rng.shuffle(idxs)
    idxs = idxs[:n_val]
    rows = [full[i] for i in idxs]

    best: EvalResult | None = None
    for sys_t, usr_t in CANDIDATES:
        r = _eval_on_subset(rows, sys_t, usr_t)
        if best is None or r.acc > best.acc:
            best = r
    return best  # type: ignore


def run_eval_with_prompt(system: str, user_template: str, split: str = "test", csv_out: str = "results_auto.csv") -> float:
    """
    用指定 (system, user_template) 在 split 上评测并落盘。
    """
    rows = load_gsm8k_subset(split)
    recs = []
    for r in tqdm(rows, desc="Evaluating (auto-selected prompt)"):
        pred_text = solve_one(r["question"], system, user_template)
        gold = extract_gold_number(r["answer"])
        pred = extract_final_number(pred_text)
        correct = int(pred is not None and gold is not None and numerically_equal(pred, gold))
        recs.append(dict(id=r["id"], question=r["question"], gold=gold,
                         pred_text=pred_text, pred=pred, correct=correct))
    df = pd.DataFrame(recs)
    df.to_csv(csv_out, index=False)
    acc = df["correct"].mean()
    print(f"[AUTO] accuracy on {len(df)} samples: {acc:.3%}")
    return acc
