#!/bin/bash
#SBATCH --job-name=fft-test
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fft-test-%j.out
#SBATCH --error=logs/fft-test-%j.err

set -e

cd /home/mila/a/ahomagnr/MT_Project/NEW

source /network/scratch/a/ahomagnr/venvs/mt-full-ft/bin/activate

export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/network/scratch/a/ahomagnr/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet

mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"

accelerate launch \
  --config_file configs/accelerate/fsdp2_2gpu.yaml \
  -m src.training.train_fft \
  --config configs/models/qwen3_5_4b_fft.yaml \
  --language luhya \
  --max_steps 2 \
  --output_dir /network/scratch/a/ahomagnr/MT_Project/NEW/outputs/test_fft_bleu_chrf
