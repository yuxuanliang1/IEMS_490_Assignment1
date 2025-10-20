import pandas as pd
from tqdm import tqdm

from .dataset import load_gsm8k_subset
from .prompt import BASE_SYSTEM, BASE_USER_TEMPLATE
from .solve import solve_one
from .score import extract_final_number, extract_gold_number, numerically_equal



def run_eval(split="test", csv_out="results_base.csv"):
    rows = load_gsm8k_subset(split=split)
    records = []
    correct = 0
    for r in tqdm(rows, desc="Evaluating"):
        q, gold_raw = r["question"], r["answer"]
        pred_text = solve_one(q, BASE_SYSTEM, BASE_USER_TEMPLATE)

        pred_num = extract_final_number(pred_text)
        gold_num = extract_gold_number(gold_raw)
        is_ok = (pred_num is not None and gold_num is not None
                 and numerically_equal(pred_num, gold_num))

        correct += int(is_ok)
        records.append({
            "id": r["id"],
            "question": q,
            "gold": gold_num,
            "pred_text": pred_text,
            "pred": pred_num,
            "correct": int(is_ok)
        })

    df = pd.DataFrame(records)
    acc = correct / len(records) if records else 0.0
    df.to_csv(csv_out, index=False)
    print(f"Base prompt accuracy on {len(records)} samples: {acc:.3%}")
    return acc


if __name__ == "__main__":
    try:
        from .config import N_SAMPLES, PROVIDER
        print(f"[Evaluate] PROVIDER={PROVIDER}, N_SAMPLES={N_SAMPLES}")
    except Exception:
        pass
    run_eval(split="test", csv_out="results_base.csv")
