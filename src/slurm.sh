#!/bin/bash
#
#SBATCH --job-name=marker_translator_training
#SBATCH --time=24:00:00
#SBATCH -c 32
#SBATCH --mem=64000M
#SBATCH --partition=bioe
#SBATCH -G 1

ml python/3.9.0

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
python3 main.py train --device "cuda" --checkpoint-dir "$GROUP_HOME/marker-labeler/checkpoint-$TIMESTAMP" --dataset-home "$GROUP_HOME/addb_dataset_publication" --epochs 500 --data-loading-workers 30 --batch-size 16 --transformer-dim 256 --transformer-nheads 8 --transformer-nlayers 6