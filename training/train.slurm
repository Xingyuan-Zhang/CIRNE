#!/bin/bash
#SBATCH --job-name=cirne-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=cirne-train-%j.log

set -euo pipefail

# Activate the environment containing CIRNE and cellpose==4.0.6 here.
python training/train_cellposesam.py \
  --data-root /path/to/training-patches \
  --output-dir /path/to/cirne-training-output \
  --epochs 300 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --weight-decay 0.1 \
  --save-every 10 \
  --expected-train-count 5200 \
  --expected-validation-count 1300

