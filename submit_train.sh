#!/bin/bash
#SBATCH --job-name=conformer_train
#SBATCH --account=bdyrk27
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log

#Bede auto-allocates 40 CPUs + 512GB RAM with 4 GPUs (full node)

#boot up the environment
source ~/miniconda3/bin/activate
conda activate conformer_env

#NCCL tuning for single-node NVLink on POWER9
export NCCL_IB_DISABLE=1           #skip InfiniBand probing, use NVLink
export NCCL_P2P_LEVEL=NVL          #prefer NVLink for GPU-to-GPU
export NCCL_DEBUG=WARN             #log warnings (use INFO to debug hangs)

#define absolute paths
REPO_DIR="/users/ffranchino/conformer-acr"
DATA_DIR="/nobackup/projects/bdyrk27/slakh_workspace/slakh_audio"
TRAIN_INDEX="$REPO_DIR/train_index.csv"
VAL_INDEX="$REPO_DIR/val_index.csv"
CHECKPOINT_DIR="$REPO_DIR/checkpoints"

echo "=========================================================="
echo "ConformerACR Distributed Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPUs: 4 x V100-32GB | CPUs: 40 | RAM: 512GB"
echo "=========================================================="

#torchrun handles LOCAL_RANK/RANK/WORLD_SIZE
torchrun --standalone --nproc_per_node=4 train.py \
    --data-dir "$DATA_DIR" \
    --index-file "$TRAIN_INDEX" \
    --val-index-file "$VAL_INDEX" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size 16 \
    --num-workers 10 \
    --epochs 100

echo "=========================================================="
echo "Training job completed."
echo "=========================================================="
