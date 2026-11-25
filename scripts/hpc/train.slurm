#!/bin/bash
# SLURM template for full training on a GPU cluster (e.g. University of Calgary TALC).
# The partition/account/module names below are PLACEHOLDERS: check the cluster documentation
# (`sinfo`, `module avail`) and edit them before submitting. Usage:
#   sbatch scripts/hpc/train.slurm E1_cnn          # or E2_resnet18
#SBATCH --job-name=ensf519-train
#SBATCH --partition=REPLACE_WITH_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
EXPERIMENT="${1:-E1_cnn}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# module load python/3.11 cuda/12.1        # <- adjust to the cluster's module names
# python -m venv .venv && .venv/bin/pip install -r requirements.txt   (first time only)
source .venv/bin/activate

: "${GENIMAGE_ROOT:?Set GENIMAGE_ROOT to the extracted GenImage folder (e.g. on scratch storage)}"
python scripts/reproduce.py status
# Resumes automatically from checkpoints/<experiment>/last.pt if a previous job timed out.
python scripts/run_experiments.py "${EXPERIMENT}" --set data.num_workers=8
