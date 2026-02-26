#!/bin/bash
# Purpose: Recompute metrics from final per-sample generations (main model run).

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate your_env

REPO_DIR="${REPO_DIR:-your_path_to_repo}"
if [[ "$REPO_DIR" == your_path_to_repo* ]]; then
  echo "Please set REPO_DIR before running." >&2
  exit 1
fi
PER_SAMPLE_GENERATIONS="${REPO_DIR}/results/evaluation/llama_8b_multilabel_1000_path_logit_10pct_per_disease/per_sample_generations.jsonl"
DISEASE_FILTER="${REPO_DIR}/data/mimic/disease-filter_updated.json"

python "${REPO_DIR}/code/compute_metrics_from_jsonl.py" \
  --jsonl "$PER_SAMPLE_GENERATIONS" \
  --filter_map "$DISEASE_FILTER" \
  --filter_mode drop_neg_only
