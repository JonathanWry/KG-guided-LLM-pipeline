#!/bin/bash
# Purpose: Generate per-patient disease paths from MIMIC features with retry-on-failure handling.
#SBATCH --job-name=mimicToPath                                  # Job name
#SBATCH --account=general                                       # Account
#SBATCH --partition=a100-8-gm320-c96-m1152                     # Partition
#SBATCH --nodes=1                                               # One node
#SBATCH --ntasks=1                                              # One task
#SBATCH --gpus=1                                                # One GPU
#SBATCH --mem=64G                                               # Memory
#SBATCH --time=96:00:00                                         # Max runtime
#SBATCH --output=your_path_to_repo/results/logs/mimicToPath.log
#SBATCH --error=your_path_to_repo/results/logs/mimicToPath.log
#SBATCH --mail-type=BEGIN,END,FAIL                              # Email notifications
#SBATCH --mail-user=rwan388@emory.edu                           # Notification recipient
#SBATCH --chdir=your_path_to_repo                               # Working directory

REPO_DIR="${REPO_DIR:-your_path_to_repo}"
if [[ "$REPO_DIR" == your_path_to_repo* ]]; then
  echo "Please set REPO_DIR before running." >&2
  exit 1
fi
PY="${REPO_DIR}/code/mimicToPath.py"
CHECKPOINT_FILE="${REPO_DIR}/results/kg_paths/checkpoint.json"
mkdir -p "${REPO_DIR}/results/logs" "${REPO_DIR}/results/kg_paths"

# Avoid relying on --requeue by handling retries in this script.

############################
# Environment initialization
############################
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u 2>/dev/null || true
else
  set +u
  source ~/.bashrc 2>/dev/null || true
  set -u 2>/dev/null || true
fi

# Activate environment
conda activate your_env

################################
# Strict mode + retry main loop
################################
set -euo pipefail

MAX_RETRY=999999          # Effectively unlimited; reduce if desired
SLEEP_BASE=10             # Initial backoff seconds
i=0
rc=1

echo "[$(date '+%F %T')] START loop runner for $PY"
echo "[$(date '+%F %T')] CONDA_BASE=${CONDA_BASE}, ENV=${CONDA_DEFAULT_ENV:-unknown}"

while [[ $rc -ne 0 && $i -lt $MAX_RETRY ]]; do
  i=$((i+1))
  echo "[$(date '+%F %T')] ---- Attempt #$i ----" >&2

  # Optional resource snapshots
  nvidia-smi || true
  free -h || true

  # Optional checkpoint snapshot
  [[ -f "$CHECKPOINT_FILE" ]] && echo "checkpoint: $(cat "$CHECKPOINT_FILE")" || true

  # Wrap srun in `if` so set -e does not terminate early on non-zero exit.
  if srun -N1 -n1 --exclusive python "$PY"; then
    rc=0
  else
    rc=$?
  fi

  echo "[$(date '+%F %T')] Exit code: $rc" >&2

  if [[ $rc -ne 0 ]]; then
    # Exponential backoff: 10, 20, 40, ... capped at 300s.
    backoff=$(( SLEEP_BASE * (1 << (i-1)) ))
    (( backoff > 300 )) && backoff=300
    echo "[$(date '+%F %T')] Retrying after ${backoff}s ..." >&2
    sleep "$backoff"
  fi
done

if [[ $rc -eq 0 ]]; then
  echo "[$(date '+%F %T')] Completed successfully."
  exit 0
else
  echo "[$(date '+%F %T')] Reached retry limit ($MAX_RETRY) with non-zero rc=$rc." >&2
  exit "$rc"
fi
