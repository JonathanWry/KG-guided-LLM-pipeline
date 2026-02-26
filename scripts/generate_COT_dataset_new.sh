#!/bin/bash
# Purpose: Generate KG-guided CoT dataset (`cot_dataset_with_paths.jsonl`) for 1,000 sampled patients.
#SBATCH --job-name=cot_dataset_gen_with_paths           # Job name
#SBATCH --account=general                               # Account
#SBATCH --partition=a100-8-gm320-c96-m1152             # Partition
#SBATCH --nodes=1                                       # Node count
#SBATCH --ntasks=1                                      # Tasks per node
#SBATCH --gpus=1                                        # Number of GPUs
#SBATCH --mem=32G                                       # Memory requirement
#SBATCH --time=96:00:00                                 # Max runtime
#SBATCH --output=your_path_to_repo/results/logs/cot_dataset_with_paths.log
#SBATCH --error=your_path_to_repo/results/logs/cot_dataset_with_paths.log
#SBATCH --mail-type=BEGIN,END,FAIL                      # Email notifications
#SBATCH --mail-user=rwan388@emory.edu                   # Notification recipient
#SBATCH --chdir=your_path_to_repo                       # Working directory

REPO_DIR="${REPO_DIR:-your_path_to_repo}"
if [[ "$REPO_DIR" == your_path_to_repo* ]]; then
  echo "Please set REPO_DIR before running." >&2
  exit 1
fi
mkdir -p "${REPO_DIR}/results/logs" "${REPO_DIR}/results/progress" "${REPO_DIR}/cot_dataset"

# Initialize conda
conda init bash > /dev/null 2>&1
source ~/.bashrc

# Activate environment (adjust if needed)
conda activate your_env

# Azure OpenAI configuration
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-your_azure_openai_endpoint}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4o}"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"  # set in environment before submit
if [[ "$AZURE_OPENAI_ENDPOINT" == "your_azure_openai_endpoint" ]]; then
  echo "Please set AZURE_OPENAI_ENDPOINT before running." >&2
  exit 1
fi
if [[ -z "${AZURE_OPENAI_API_KEY}" ]]; then
  echo "AZURE_OPENAI_API_KEY is empty. Export it before running." >&2
  exit 1
fi

# Run CoT generation
python "${REPO_DIR}/code/construct_COT_dataset_new.py" \
  --graph_mode \
  --hyperedges_txt "${REPO_DIR}/data/mimic/hyperedges-mimic3_truncated.txt" \
  --node_text_json "${REPO_DIR}/data/mimic/node_text.json" \
  --edge_labels_txt "${REPO_DIR}/data/mimic/edge-labels-mimic3_updated_truncated.txt" \
  --edge_text_json "${REPO_DIR}/data/mimic/edge_text.json" \
  --disease_filter_json "${REPO_DIR}/data/mimic/disease-filter_updated.json" \
  --relevance_txt "${REPO_DIR}/data/relevence.txt" \
  --paths_json "${REPO_DIR}/data/filtered_path_mappings.json" \
  --output_jsonl "${REPO_DIR}/cot_dataset/cot_dataset_with_paths.jsonl" \
  --progress_file "${REPO_DIR}/results/progress/cot_dataset_with_paths.progress" \
  --checkpoint_every 10 \
  --sample_patients 1000 \
  --sample_seed 42 \
  --model gpt-4o

# Optional smoke test:
# --early_stop_patients 40 \
