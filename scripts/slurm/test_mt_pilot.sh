#!/bin/bash
#SBATCH --job-name=mt-pilot
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/MT_Project/logs/mt-pilot-%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/MT_Project/logs/mt-pilot-%j.err

set -e

cd /home/mila/a/ahomagnr/MT_Project
source MT_Project/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=/network/scratch/a/ahomagnr/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet

python -m src.training.train_mt_pilot \
    --config configs/models/qwen3_5_4b_lora.yaml \
    --language gusii \
    --max_steps 110 \
    --output_dir /network/scratch/a/ahomagnr/MT_Project/outputs/pilot_eval_loss
