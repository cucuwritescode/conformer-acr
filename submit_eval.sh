#!/bin/bash
#SBATCH --job-name=eval_acr
#SBATCH --account=bdyrk27
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_%j.log

source ~/miniconda3/bin/activate
conda activate conformer_env

REPO_DIR="/users/ffranchino/conformer-acr"
DATA_DIR="/nobackup/projects/bdyrk27/slakh_workspace/slakh_audio"

python evaluate.py \
    --checkpoint "$REPO_DIR/checkpoints/checkpoint_epoch0100.pt" \
    --index-file "$REPO_DIR/val_index.csv" \
    --data-dir "$DATA_DIR" \
    --batch-size 16 \
    --num-workers 4
