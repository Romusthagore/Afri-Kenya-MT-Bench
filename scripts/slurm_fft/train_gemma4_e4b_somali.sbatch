#!/bin/bash
#SBATCH --job-name=fft-gemma4_e4b-somali
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/MT_Project/NEW/logs/%x-%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/MT_Project/NEW/logs/%x-%j.err

set -euo pipefail

cd /home/mila/a/ahomagnr/MT_Project/NEW

source /network/scratch/a/ahomagnr/venvs/mt-full-ft/bin/activate

export HF_HOME=/network/scratch/a/ahomagnr/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
accelerate launch --main_process_port "$MASTER_PORT" \
  --config_file configs/accelerate/fsdp2_gemma4_2gpu.yaml \
  -m src.training.train_fft \
  --config configs/models/gemma4_e4b_fft.yaml \
  --language somali \
  --output_dir /network/scratch/a/ahomagnr/MT_Project/NEW/outputs/gemma4_e4b/somali
