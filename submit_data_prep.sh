#!/bin/bash
#SBATCH --job-name=acr_data_prep
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
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

#step 1: run active synthetic mutation
echo "[STEP 1] Running mutate.py..."
python mutate.py --input-dir $RAW_MIDI_DIR --output-dir $MUTATED_MIDI_DIR

#step 2: run hybrid rendering and cqt extraction
echo "[STEP 2] Running render_hybrid.py..."
python render_hybrid.py

#step 3: generate dataset index
echo "[STEP 3] Running prep_dataset.py..."
python prep_dataset.py --data-dir $FINAL_AUDIO_DIR --output index.csv

echo "Pipeline finished successfully."
