#!/bin/bash
#SBATCH --job-name=acr_data_prep
#SBATCH --account=bdyrk27
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=data_prep_%j.log

source ~/miniconda3/bin/activate
conda activate conformer_env

WORKSPACE="/nobackup/projects/bdyrk27/slakh_workspace"
RAW_MIDI_DIR="$WORKSPACE/slakh_raw/slakh2100_flac_redux"
MUTATED_MIDI_DIR="$WORKSPACE/slakh_mutated"
FINAL_AUDIO_DIR="$WORKSPACE/slakh_audio"

mkdir -p $MUTATED_MIDI_DIR
mkdir -p $FINAL_AUDIO_DIR

echo "[STEP 1] Running Active Synthetic Mutation (mutate.py)..."
python mutate.py --input-dir $RAW_MIDI_DIR --output-dir $MUTATED_MIDI_DIR

echo "[STEP 2] Running Hybrid Rendering & CQT Extraction (render_hybrid.py)..."
python render_hybrid.py

echo "[STEP 3] Generating Dataset Index (prep_dataset.py)..."
python prep_dataset.py --data-dir $FINAL_AUDIO_DIR --output index.csv

echo "Pipeline finished successfully."
