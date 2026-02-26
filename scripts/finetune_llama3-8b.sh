#!/bin/bash
# Purpose: Fine-tune LLaMA-3.1-8B-Instruct on KG-guided filtered CoT data.
#SBATCH --job-name=llama3-8b_sft_full_1000_path            # Job name
#SBATCH --account=general                                   # Account
#SBATCH --partition=h200-8-gm1128-c192-m2048               # Partition (adjust per cluster)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=512G                 # bump to 512G if using CPU offload
#SBATCH --time=72:00:00                       # Max runtime
#SBATCH --output=your_path_to_repo/results/logs/llama3-8b_sft.log    # Log file
#SBATCH --error=your_path_to_repo/results/logs/llama3-8b_sft.log     # Error log file
#SBATCH --mail-type=BEGIN,END,FAIL            # Job status notifications
#SBATCH --mail-user=rwan388@emory.edu         # Notification recipient
#SBATCH --chdir=your_path_to_repo             # Working directory

REPO_DIR="${REPO_DIR:-your_path_to_repo}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-your_path_to_LlamaFactory}"
TRAIN_CFG="${TRAIN_CFG:-${REPO_DIR}/llm_training_config/llama3-8b_cot_noisy_1000_path.yaml}"
SCRATCH_BASE="${SCRATCH_BASE:-your_path_to_cache_dir}"
if [[ "$REPO_DIR" == your_path_to_repo* || "$LLAMAFACTORY_DIR" == your_path_to_LlamaFactory* || "$SCRATCH_BASE" == your_path_to_cache_dir* ]]; then
  echo "Please set REPO_DIR, LLAMAFACTORY_DIR, and SCRATCH_BASE before running." >&2
  exit 1
fi

mkdir -p $SCRATCH_BASE/{hf,hf/datasets,hf/hub,xdg,tmp,.triton/autotune}

export HF_HOME=$SCRATCH_BASE/hf
export TRANSFORMERS_CACHE=$SCRATCH_BASE/hf
export HUGGINGFACE_HUB_CACHE=$SCRATCH_BASE/hf/hub
export HF_DATASETS_CACHE=$SCRATCH_BASE/hf/datasets
export XDG_CACHE_HOME=$SCRATCH_BASE/xdg
export TMPDIR=$SCRATCH_BASE/tmp
export WANDB_PROJECT="LLM-Finetuning"                # W&B project
export WANDB_ENTITY="rwan388-emory-university" 
export WANDB_DIR="$SCRATCH_BASE/wandb"
export WANDB_CACHE_DIR="$SCRATCH_BASE/wandb_cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"  # set in environment before submit
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Keep Triton autotune cache on local scratch
export TRITON_CACHE_DIR=$SCRATCH_BASE/.triton/autotune

cd "$LLAMAFACTORY_DIR"
# Initialize conda
conda init bash > /dev/null 2>&1
source ~/.bashrc

# Activate conda environment
conda activate your_env   # Replace with your environment name

export HF_TOKEN="${HF_TOKEN:-}"  # set in environment before submit
if [[ -z "${HF_TOKEN}" ]]; then
  echo "HF_TOKEN is empty. Export HF_TOKEN before running." >&2
  exit 1
fi

# Persist credential for all processes (writes to ~/.cache/huggingface / ~/.huggingface)
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

export FORCE_TORCHRUN=1
# Run LLaMA-Factory training
llamafactory-cli train "$TRAIN_CFG"
