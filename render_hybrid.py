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
    label_out = os.path.join(output_dir, f"{track_name}_labels.csv")

    #skip if already processed
    if os.path.exists(cqt_out) and os.path.exists(label_out):
        return

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        #generate ground truth labels from midi
        with open(label_out, 'w') as f:
            f.write("start_time,end_time,chord\n")

            #pool all polyphonic notes to extract the full harmonic context
            all_notes = []
            for inst in midi_data.instruments:
                if not inst.is_drum:
                    all_notes.extend(inst.notes)

            duration = midi_data.get_end_time()
            hop = 0.5  #0.5 second resolution for chord boundaries

            for start_t in np.arange(0, duration, hop):
                end_t = min(start_t + hop, duration)
                active_pitches = set()

                for note in all_notes:
                    if note.start < end_t and note.end > start_t:
                        active_pitches.add(note.pitch)

                if len(active_pitches) < 3:
                    f.write(f"{start_t:.3f},{end_t:.3f},N\n")
                    continue

                pitches = sorted(list(active_pitches))
                root = pitches[0] % 12
                intervals = set((p - root) % 12 for p in pitches)

                #match against mutated intervals (7, dim7, hdim7, min, maj)
                if {4, 7, 10}.issubset(intervals):
                    quality = "7"
                elif {3, 6, 9}.issubset(intervals):
                    quality = "dim7"
                elif {3, 6, 10}.issubset(intervals):
                    quality = "hdim7"
                elif {3, 7}.issubset(intervals):
                    quality = "min"
                elif {4, 7}.issubset(intervals):
                    quality = "maj"
                else:
                    quality = "N"

                if quality == "N":
                    f.write(f"{start_t:.3f},{end_t:.3f},N\n")
                else:
                    root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root]
                    f.write(f"{start_t:.3f},{end_t:.3f},{root_name}:{quality}\n")

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
