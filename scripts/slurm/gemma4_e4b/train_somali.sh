#!/bin/bash
#SBATCH --job-name=mt-gemma4_e4b-somali
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/MT_Project/logs/%x-%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/MT_Project/logs/%x-%j.err

set -euo pipefail

cd /home/mila/a/ahomagnr/MT_Project
source MT_Project/bin/activate

export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME=/network/scratch/a/ahomagnr/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export TOKENIZERS_PARALLELISM=false

python -m src.training.train \
    --config configs/models/gemma4_e4b_lora.yaml \
    --language somali \
    --output_dir /network/scratch/a/ahomagnr/MT_Project/outputs/gemma4_e4b/somali
