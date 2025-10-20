#!/bin/bash
# Filename: run.sh

# Exit immediately if any command fails
set -e

# 1. Check if the API key is present
# The Dockerfile will get this variable from the environment
if [ -z "$DEEPSEEK_API_KEY" ]; then
  echo "Error: DEEPSEEK_API_KEY environment variable is not set."
  echo "Please run the Docker container using: -e DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi

# 2. Run all assignment steps
# N_SAMPLES and PROVIDER are set by the Dockerfile but can be overridden
echo "--- Running Baseline (N_SAMPLES=${N_SAMPLES}, PROVIDER=${PROVIDER}) ---"
python -c "from iems490.evaluate import run_eval; run_eval(split='test', csv_out='results_100_baseline.csv')"
python -c "import pandas as pd; df=pd.read_csv('results_100_baseline.csv'); print('Baseline acc:', df['correct'].mean())"

echo "--- Running Manual (StructCoT + Self-Check) ---"
python -c "from iems490.evaluate import run_eval; from iems490.prompts_manual import MANUAL_SYSTEM, MANUAL_USER_TEMPLATE; acc = run_eval(split='test', csv_out='results_100_manual_structcot.csv', system_prompt=MANUAL_SYSTEM, user_template=MANUAL_USER_TEMPLATE); print('Manual (StructCoT) acc:', acc)"

echo "--- Running Automated (search on validation, then test) ---"
python -c "from iems490.prompts_automated import auto_select_prompt, run_eval_with_prompt; best = auto_select_prompt(val_split='test', n_val=40); print(f'Selected prompt acc={best.acc:.2%} on n={best.n}'); acc = run_eval_with_prompt(best.system, best.user_template, split='test', csv_out='results_100_auto_selected.csv'); print('Auto-selected acc:', acc)"

echo "--- All evaluations complete ---"