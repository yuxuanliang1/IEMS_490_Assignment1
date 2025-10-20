# src/iems490/evaluate.py
import argparse
import os
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .dataset import load_gsm8k_subset
from .solve import solve_one
from .score import extract_final_number, extract_gold_number, numerically_equal

# Standard Baseline Template
from .prompts_base import BASE_SYSTEM as BASE_SYSTEM_BASE
from .prompts_base import BASE_USER_TEMPLATE as BASE_USER_BASE

# Manual Optimization Template (Structured CoT + Self-Check)
from .prompts_manual import MANUAL_SYSTEM, MANUAL_USER_TEMPLATE


from .prompts_automated import auto_select_prompt, run_eval_with_prompt


def run_eval(
    split: str = "test",
    csv_out: str = "results_base.csv",
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> float:

    system_prompt = system_prompt or BASE_SYSTEM_BASE
    user_template = user_template or BASE_USER_BASE

    rows = load_gsm8k_subset(split)
    recs = []
    for r in tqdm(rows, desc="Evaluating"):
        pred_text = solve_one(r["question"], system_prompt, user_template)
        gold = extract_gold_number(r["answer"])
        pred = extract_final_number(pred_text)
        correct = int(pred is not None and gold is not None and numerically_equal(pred, gold))
        recs.append(
            dict(
                id=r["id"],
                question=r["question"],
                gold=gold,
                pred_text=pred_text,
                pred=pred,
                correct=correct,
            )
        )
    df = pd.DataFrame(recs)
    df.to_csv(csv_out, index=False)
    acc = df["correct"].mean()
    print(f"Accuracy on {len(df)} samples: {acc:.3%}")
    return acc


def _auto_out_name(kind: str, split: str, default: str) -> str:
    n = os.getenv("N_SAMPLES", "")
    tag = f"{split}_{kind}"
    if n:
        tag += f"_n{n}"
    return f"results_{tag}.csv" if not default else default


def _parse_args():
    ap = argparse.ArgumentParser(description="Evaluate GSM8K with different prompt modes.")
    ap.add_argument("--prompt", choices=["base", "manual", "auto"], default="base",
                    help="Prompt mode: base=baseline, manual=structured CoT, auto=automated search")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    ap.add_argument("--out", default="", help="Output CSV path (default auto-named)")
    ap.add_argument("--n-val", type=int, default=40,
                    help="Auto mode: validation subset size for prompt search (default: 40)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.prompt == "base":
        out = _auto_out_name("base", args.split, args.out)
        print(f"[RUN] BASELINE -> {out}")
        run_eval(split=args.split, csv_out=out,
                 system_prompt=BASE_SYSTEM_BASE, user_template=BASE_USER_BASE)

    elif args.prompt == "manual":
        out = _auto_out_name("manual", args.split, args.out)
        print(f"[RUN] MANUAL (Structured CoT + self-check) -> {out}")
        run_eval(split=args.split, csv_out=out,
                 system_prompt=MANUAL_SYSTEM, user_template=MANUAL_USER_TEMPLATE)

    else:  # args.prompt == "auto"
        out = _auto_out_name("auto", args.split, args.out)
        print(f"[RUN] AUTO-SEARCH (n_val={args.n_val}) -> {out}")
        best = auto_select_prompt(val_split=args.split, n_val=args.n_val)
        print(f"[AUTO] selected acc={best.acc:.2%} on n={best.n}")
        run_eval_with_prompt(best.system, best.user_template,
                             split=args.split, csv_out=out)
