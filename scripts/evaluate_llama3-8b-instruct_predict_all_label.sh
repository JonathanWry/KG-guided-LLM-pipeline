#!/bin/bash
# Purpose: Run KG-guided LLaMA-8B evaluation and save per-disease multilabel metrics.
#SBATCH --job-name=llama3_evaluation_predict_full_path_1000_logit_per_disease  # Job name
#SBATCH --account=general                           # Account
#SBATCH --partition=a100-8-gm320-c96-m1152         # Partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=16G                                    # Memory
#SBATCH --time=112:00:00                             # Max runtime
#SBATCH --output=your_path_to_repo/results/logs/llama3-8b_evaluation_1000_path_logit_per_disease.log
#SBATCH --error=your_path_to_repo/results/logs/llama3-8b_evaluation_1000_path_logit_per_disease.log
#SBATCH --mail-type=BEGIN,END,FAIL                  # Job status notifications
#SBATCH --mail-user=rwan388@emory.edu               # Notification recipient
#SBATCH --chdir=your_path_to_repo                   # Working directory

REPO_DIR="${REPO_DIR:-your_path_to_repo}"
MODEL="${MODEL:-your_path_to_model_checkpoint}"
SCRATCH_BASE="${SCRATCH_BASE:-your_path_to_cache_dir}"
if [[ "$REPO_DIR" == your_path_to_repo* || "$MODEL" == your_path_to_model_checkpoint* || "$SCRATCH_BASE" == your_path_to_cache_dir* ]]; then
  echo "Please set REPO_DIR, MODEL, and SCRATCH_BASE before running." >&2
  exit 1
fi

mkdir -p $SCRATCH_BASE/{hf,hf/datasets,hf/hub,xdg,tmp,.triton/autotune}

export HF_HOME=$SCRATCH_BASE/hf
export TRANSFORMERS_CACHE=$SCRATCH_BASE/hf
export HUGGINGFACE_HUB_CACHE=$SCRATCH_BASE/hf/hub
export HF_DATASETS_CACHE=$SCRATCH_BASE/hf/datasets
export XDG_CACHE_HOME=$SCRATCH_BASE/xdg
export TMPDIR=$SCRATCH_BASE/tmp
export WANDB_PROJECT="LLM-Finetuning"                # the project you want
export WANDB_ENTITY="rwan388-emory-university" 
export WANDB_DIR="$SCRATCH_BASE/wandb"
export WANDB_CACHE_DIR="$SCRATCH_BASE/wandb_cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"  # set in environment before submit
export PYTHONNOUSERSITE=1
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Keep Triton autotune cache on local scratch
export TRITON_CACHE_DIR=$SCRATCH_BASE/.triton/autotune


# Initialize conda
conda init bash > /dev/null 2>&1
source ~/.bashrc

# Activate conda environment
conda activate your_env   # Replace with your environment name


# MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

UNSEEN_IDS_JSON="${REPO_DIR}/dataset/unseen_exclude_1000.json"
EDGE_LABELS_TXT="${REPO_DIR}/data/mimic/edge-labels-mimic3_updated_truncated.txt"
EDGE_TXT_JSON="${REPO_DIR}/data/mimic/edge_text.json"
NODE_TEXT_JSON="${REPO_DIR}/data/mimic/node_text.json"
HYPEREDGES_TXT="${REPO_DIR}/data/mimic/hyperedges-mimic3_truncated.txt"

# Sampling controls
TOTAL_PATIENTS=12353
RATIO=0.1              # 10% sample
SEED=42

# Optional canonical disease order (JSON list of 13 names, one per label/column)
# LABEL_LIST_JSON="your_path_to_label_list_json"  # uncomment if you have it

# Output directory for this run
OUT_DIR="${REPO_DIR}/results/evaluation/llama_8b_multilabel_1000_path_logit_10pct_per_disease"
mkdir -p "$OUT_DIR"

# Generation controls
MAX_NEW_TOKENS=2048

# Evaluation controls
THRESHOLD=0.5
# REQUIRE_COMPLETE=--require_complete_patient   # drop this flag if you want to allow incomplete patients

# --------------- Run ----------------

python "${REPO_DIR}/code/evaluate_llm_predict_noisy.py" \
  --unseen_ids_json "$UNSEEN_IDS_JSON" \
  --total_patients "$TOTAL_PATIENTS" \
  --ratio "$RATIO" \
  --seed "$SEED" \
  --edge_labels_txt "$EDGE_LABELS_TXT" \
  --edge_txt_json "$EDGE_TXT_JSON" \
  --node_text_json "$NODE_TEXT_JSON" \
  --hyperedges_txt "$HYPEREDGES_TXT" \
  --model_dir "$MODEL" \
  --out_dir "$OUT_DIR" \
  --device cuda \
  --verbose \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --threshold "$THRESHOLD" \
  --disease_whitelist_json "${REPO_DIR}/scripts/available_disease.json" \
  --relevance_txt "${REPO_DIR}/data/relevence.txt" \
  --paths_json "${REPO_DIR}/data/filtered_path_mappings.json"
  # $REQUIRE_COMPLETE
  # If you have a fixed canonical order, also pass:
  # --label_list_json "$LABEL_LIST_JSON"
  # To cap rows for a smoke test, add:
  # --max_eval_rows 2000
