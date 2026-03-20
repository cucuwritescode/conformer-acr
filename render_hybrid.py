#!/usr/bin/env python3
"""
render_hybrid.py - HPC multi-core hybrid audio renderer

Renders MIDI files to audio using a hybrid approach:
- Complex instruments (piano, strings): DDSP neural synthesis
- Simple instruments (drums, etc): FluidSynth soundfont

Also extracts CQT spectrograms for training.
"""

import os
import glob
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import librosa
from joblib import Parallel, delayed
import pretty_midi
import torch


# ============================================================================
#constants
# ============================================================================

N_CORES = 40
SR = 22050
SOUNDFONT = "FluidR3_GM.sf2"  #ensure you have a basic .sf2 file or update this path

#complex instrument programs (piano, strings) - use DDSP
COMPLEX_PROGRAMS = list(range(0, 8)) + list(range(40, 56))


# ============================================================================
#rendering functions
# ============================================================================

def _fluid_render(midi_path: str, out_wav: str) -> None:
    """Render MIDI to audio using FluidSynth."""
    cmd = [
        "fluidsynth", "-ni", SOUNDFONT, midi_path,
        "-F", out_wav, "-r", str(SR), "-q"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ddsp_render(midi_path: str, out_wav: str) -> None:
    """Render MIDI to audio using DDSP neural synthesis."""
    cmd = [
        "midi_ddsp_synthesize",
        "--midi_path", midi_path,
        "--output_dir", os.path.dirname(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_track(midi_file: str, output_dir: str) -> None:
    """
    Process a single MIDI file: render to audio and extract CQT.

    Parameters
    ----------
    midi_file : str
        Path to input MIDI file.
    output_dir : str
        Directory to save output .flac and .pt files.
    """
    track_name = os.path.basename(midi_file).replace('.mid', '')
    flac_out = os.path.join(output_dir, f"{track_name}_mix.flac")
    cqt_out = os.path.join(output_dir, f"{track_name}_cqt.pt")

    #skip if already processed
    if os.path.exists(cqt_out):
        return

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        with tempfile.TemporaryDirectory() as tmpdir:
            stem_arrays = []

            for i, inst in enumerate(midi_data.instruments):
                if len(inst.notes) == 0:
                    continue

                #create single-instrument midi
                stem_midi = pretty_midi.PrettyMIDI()
                stem_midi.instruments.append(inst)
                tmp_mid = os.path.join(tmpdir, f"stem_{i}.mid")
                tmp_wav = os.path.join(tmpdir, f"stem_{i}.wav")
                stem_midi.write(tmp_mid)

                #choose renderer based on instrument complexity
                is_complex = not inst.is_drum and inst.program in COMPLEX_PROGRAMS
                if is_complex:
                    _ddsp_render(tmp_mid, tmp_wav)
                else:
                    _fluid_render(tmp_mid, tmp_wav)

                #load rendered audio
                if os.path.exists(tmp_wav):
                    y, _ = librosa.load(tmp_wav, sr=SR)
                    stem_arrays.append(y)

            if not stem_arrays:
                return

            #mix all stems
            max_len = max(len(a) for a in stem_arrays)
            master_mix = np.zeros(max_len)
            for a in stem_arrays:
                master_mix[:len(a)] += a

            #normalize and save
            master_mix = master_mix / (np.max(np.abs(master_mix)) + 1e-7)
            sf.write(flac_out, master_mix, SR, format='FLAC')

            #extract and save CQT
            C = np.abs(librosa.cqt(
                master_mix,
                sr=SR,
                hop_length=512,
                fmin=librosa.note_to_hz('C1'),
                n_bins=252,
                bins_per_octave=36
            ))
            C_db = librosa.amplitude_to_db(C, ref=np.max)
            torch.save(torch.tensor(C_db).T, cqt_out)

    except Exception as e:
        print(f"Error {track_name}: {e}", flush=True)


# ============================================================================
#main
# ============================================================================

if __name__ == "__main__":
    MUTATED_MIDI_DIR = "/nobackup/projects/bdyrk27/slakh_workspace/slakh_mutated"
    OUTPUT_AUDIO_DIR = "/nobackup/projects/bdyrk27/slakh_workspace/slakh_audio"

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    midi_files = glob.glob(os.path.join(MUTATED_MIDI_DIR, "**/*.mid"), recursive=True)
    print(f"Found {len(midi_files)} MIDI files to process", flush=True)

    Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_track)(f, OUTPUT_AUDIO_DIR) for f in midi_files
    )

    print("Rendering complete.", flush=True)
