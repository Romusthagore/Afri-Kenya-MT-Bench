#!/bin/bash
set -euo pipefail

cd /home/mila/a/ahomagnr/MT_Project
source MT_Project/bin/activate

export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME=/network/scratch/a/ahomagnr/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_CACHE=$HF_HOME/xet
export TOKENIZERS_PARALLELISM=false

MODEL=${1:?Model configuration required}
LANGUAGE=${2:?Language required}

python -m src.training.train \
    --config "configs/models/${MODEL}.yaml" \
    --language "$LANGUAGE"
