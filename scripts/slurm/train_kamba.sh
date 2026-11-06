#!/bin/bash
#SBATCH --job-name=mt-kamba
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

bash scripts/slurm/train_common.sh "$1" "kamba"
