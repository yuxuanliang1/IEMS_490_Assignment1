# IEMS 490 — Assignment 1

> **Project goal.** Build a *reasonable* baseline, then show measurable gains with (1) enhanced **manual** prompt engineering and (2) **fully automated** prompt engineering on a math word-problem benchmark. Emphasis on **sound scoring**, **clear methodology**, **reproducibility**, and **cost awareness**.

---

## L;DR
- **Dataset:** `openai/gsm8k (main)` — standard, well-understood grade-school math reasoning set; easy to score via exact numeric answers; aligns with assignment’s requirement to evaluate text outputs by a clear metric.  
- **Model/Provider:** **DeepSeek** API — dramatically **lower cost per token** than frontier LLMs while retaining strong arithmetic/formatting ability; enables *many* experiments (search over prompts, self-consistency) **within a small budget**.  
- **Methods compared:**  
  1) **Baseline** (no CoT, minimal instruction, strict output format)  
  2) **Manual (StructCoT + Self-Check + Few-Shot)** — structured steps + verification to reduce careless mistakes  
  3) **Automated** (prompt search over composable switches + validation-driven selection; optional self-consistency voting)

- **Reproducibility:** fixed random seeds, deterministic sampling size, and a **hard-only** sampling flag to control difficulty; a minimal **CLI** and **Docker (arm64)** path provided below.

---

## Repository layout (key files)

```
src/iems490/
  ├─ __init__.py
  ├─ __main__.py               # Lightweight CLI (run: python -m iems490 ...)
  ├─ config.py                 # RANDOM_SEED, N_SAMPLES, generation params
  ├─ dataset.py                # GSM8K loader + --hard-only difficulty option
  ├─ evaluate.py               # run_eval(): loads data, calls model, scores, writes CSV
  ├─ prompts_base.py           # Baseline prompt (no CoT; strict answer format)
  ├─ prompts_manual.py         # Manual: StructCoT + Self-Check (+ few-shot)
  ├─ prompts_automated.py      # Automated search over prompt variants (validation -> pick best)
  ├─ score.py                  # Exact-answer extractor & accuracy; format diagnostics
  └─ ...
```

---

## Why this **dataset** (GSM8K, `main`)?

1. **Clear, objective scoring.** GSM8K answers end with a numeric line (e.g., `#### 24`), so we can compute **exact match accuracy** with a robust parser. This directly satisfies the rubric’s “scoring properly designed & implemented”.
2. **Well-studied reasoning behavior.** Word-problems expose arithmetic slips, unit confusion, and multi-step planning—exactly what **prompt engineering** can help with (structure, verification, etc.).
3. **Right difficulty & size.** We can draw subsets (e.g., 250–500) for rapid iteration and use a **hard-only** sampler to keep baseline moderate (not too strong, not intentionally weak), leaving **headroom** for manual/auto improvements.
4. **Reproducible.** The assignment mandates using:
   ```python
   from datasets import load_dataset
   ds = load_dataset("openai/gsm8k", "main")
   ```
   Our `dataset.py` conforms exactly to this requirement.

---

## 💸 Why **DeepSeek** (cost & iteration speed)

- **Much lower token cost** than frontier LLMs while retaining solid math/formatting ability → we can afford:
  - Larger **candidate prompt** pools in automated search (20–40+ variants).
  - **Self-consistency** (multiple samples per problem) without blowing the budget.
  - Multiple **ablation runs** (baseline → CoT → CoT+Check → +few-shot → automated w/ search).
- **Fast iteration loop.** Cheaper calls = more experiments in the same time window; crucial when tuning many small prompt knobs (structure tags, verification clauses, format strictness, temperature, max tokens).

> In short, **DeepSeek** is “good enough” for GSM8K while being **cheap enough** to do all the experiments the rubric wants (and to show statistically meaningful deltas).

---

## Methods

### 1) Baseline (reasonable, not sabotaged)
- Minimal instruction; **no explicit CoT**; strict final-line format:  
  “Do not show your work. Output only the final line: `#### <number>`.”
- Generation length kept short to reduce implicit reasoning spillover.
- Purpose: establish a **fair** starting point that is *not* intentionally terrible, yet leaves room for improvement.

### 2) Manual prompt engineering (Enhanced)
- **StructCoT**: force a small set of labeled steps (Plan → Equations → Compute → Check → Final).
- **Self-Check**: unit/scale sanity, integer/rounding policy, divisibility/leftover checks, and **quick back-substitution** before finalizing.
- **Few-shot** (2–3 in-domain mini exemplars): style-aligned with GSM8K wording; not overdone to avoid ceiling effects.
- Effect: fewer arithmetic slips, more stable formatting, better robustness to **hard-only** subset and light noise.

### 3) Automated prompt engineering
- **Candidate generation via composable switches** (e.g., with/without explicit step tags; keep fractions vs decimals; assert integer final answer; require one-line back-check; temperature/max_tokens variations).
- **Validation-driven selection**: evaluate all candidates on a small **validation split** (separate from the test set), then **pick the best**.
- **Optional self-consistency**: for top candidates, do N=3–5 samples (mild temperature) and majority-vote final answer.
- **Final test**: run the best candidate once on the test set for the report.

> This pipeline matches the rubric’s “fully automated prompt engineering” and aims to **beat the best manual prompt**.

---

## Scoring & diagnostics

- **Primary metric:** *Exact Answer Accuracy*.  
  We parse the **last line** of model output; expected pattern `#### <number>`.
- **Robust extraction:** normalize spaces, handle commas, `-0`, trailing punctuation; for GSM8K we treat answers as **integers** unless the problem states otherwise.
- **Diagnostics (reported in CSV):**
  - Format adherence (did the model end with a valid `####` line?)
  - Optional buckets: by question length, by presence of keywords (`fraction|percent|ratio|…`), by “hard-only” flag.
- **Uncertainty:** recommended to add simple **95% CI** for accuracy (e.g., binomial interval) to show differences are meaningful.

---

## Hard-only sampling (difficulty control)

To avoid an overly strong baseline and to make gains visible, `dataset.py` supports:

- **`hard_only=True`**:  
  Stage A — keyword **and** min length ⇒  
  Stage B — keyword only ⇒  
  Stage C — fallback to top-K longest globally.  
- Tunables: `hard_min_chars` and the keyword regex list (ratios, fractions, percent, probability, mixture, rate/time/work, LCM/GCD, etc.).

This ensures exactly `N_SAMPLES` are always returned while biasing toward multi-step problems.

---

## How to run

### Environment
```bash
# create & activate a virtualenv (any Python 3.10+ is fine)
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Set your API key (example):
```bash
# PowerShell
$env:DEEPSEEK_API_KEY="YOUR_KEY"
# bash/zsh
export DEEPSEEK_API_KEY="YOUR_KEY"
```

```bash
# Baseline
$env:N_SAMPLES="500"
$env:PROVIDER = 'deepseek'
$env:DEEPSEEK_API_KEY=""
python -c "from iems490.evaluate import run_eval; run_eval(split='test', csv_out='results_100_baseline.csv')"
python -c "import pandas as pd; df=pd.read_csv('results_100_baseline.csv'); print('Baseline acc:', df['correct'].mean())"

# Manual (StructCoT + Self-Check)
python -c "from iems490.evaluate import run_eval; from iems490.prompts_manual import MANUAL_SYSTEM, MANUAL_USER_TEMPLATE; acc = run_eval(split='test', csv_out='results_100_manual_structcot.csv', system_prompt=MANUAL_SYSTEM, user_template=MANUAL_USER_TEMPLATE); print('Manual (StructCoT) acc:', acc)"

# Automated (search on validation, then test)
python -c "from iems490.prompts_automated import auto_select_prompt, run_eval_with_prompt; best = auto_select_prompt(val_split='test', n_val=40); print(f'Selected prompt acc={best.acc:.2%} on n={best.n}'); acc = run_eval_with_prompt(best.system, best.user_template, split='test', csv_out='results_100_auto_selected.csv'); print('Auto-selected acc:', acc)"
```

> The `--hard-only` flag activates the difficulty-biased sampler.  
> `N_SAMPLES` and `RANDOM_SEED` are controlled in `src/iems490/config.py`.

---

## Reproducibility

- **Fixed seeds** for sampling/generation (see `config.py`).
- **Fixed loading lines** for GSM8K as required by the assignment.
- **CSV artifacts** with per-item outputs and diagnostics for audit.
- The `--hard-only` flag and `N_SAMPLES` allow controlled difficulty and runtime.




## Limitations & future work
- GSM8K focuses on arithmetic reasoning; future iterations can add **cross-domain** validation to test generality.  
- Automated search can incorporate **prompt LLM-edit** or **Bayesian optimization** over discrete prompt switches for better sample efficiency.  
- Add formal **confidence reporting** per problem type to correlate with error patterns.

---

## Acknowledgments
- Course: **IEMS 490 — Theory & Algorithms for LLMs (Prompt Engineering)**  
- Dataset: **GSM8K (main)**  
- Model Provider: **DeepSeek** (chosen for **low cost**, enabling broad experimentation under tight budget/time constraints)
