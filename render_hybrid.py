#!/usr/bin/env python3
"""
render_hybrid.py - HPC multi-core audio renderer

Renders MIDI files to audio using FluidSynth and extracts CQT spectrograms.
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

N_CORES = 32
SR = 22050
SOUNDFONT = "FluidR3_GM.sf2"


# ============================================================================
#rendering functions
# ============================================================================

def _fluid_render(midi_path: str, out_wav: str) -> bool:
    """Render MIDI to audio using FluidSynth."""
    cmd = [
        "fluidsynth", "-ni", SOUNDFONT, midi_path,
        "-F", out_wav, "-r", str(SR), "-q"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FluidSynth failed for {midi_path}: {e.stderr}", flush=True)
        return False
    except FileNotFoundError:
        print(f"FluidSynth not found. Install with: apt install fluidsynth", flush=True)
        return False


def process_track_dir(track_dir: str, output_dir: str, split_name: str) -> None:
    """
    Process a track directory: combine all stem MIDIs, render to audio, extract CQT.

    Parameters
    ----------
    track_dir : str
        Path to track directory (e.g., .../train/Track00001/)
    output_dir : str
        Directory to save output .flac and .pt files.
    split_name : str
        Split name (train/test/validation/omitted) for output filename.
    """
    track_name = os.path.basename(track_dir)
    out_name = f"{split_name}_{track_name}"
    flac_out = os.path.join(output_dir, f"{out_name}_mix.flac")
    cqt_out = os.path.join(output_dir, f"{out_name}_cqt.pt")
    label_out = os.path.join(output_dir, f"{out_name}_labels.csv")

    #skip if already processed
    if os.path.exists(cqt_out) and os.path.exists(label_out):
        return

    try:
        #find all stem MIDI files in this track
        midi_dir = os.path.join(track_dir, "MIDI")
        if not os.path.exists(midi_dir):
            return

        stem_files = sorted(glob.glob(os.path.join(midi_dir, "S*.mid")))
        if not stem_files:
            return

        #combine all stems into one PrettyMIDI object
        combined_midi = pretty_midi.PrettyMIDI()
        all_notes = []

        for stem_file in stem_files:
            stem_midi = pretty_midi.PrettyMIDI(stem_file)
            for inst in stem_midi.instruments:
                combined_midi.instruments.append(inst)
                if not inst.is_drum:
                    all_notes.extend(inst.notes)

        #generate ground truth labels from combined midi
        with open(label_out, 'w') as f:
            f.write("start_time,end_time,chord\n")

            duration = combined_midi.get_end_time()
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

                #match against mutated intervals - check complex chords first
                #altered dominants (7b9, 7#9) - dominant 7th with b9 or #9
                if {4, 7, 10}.issubset(intervals):
                    if 1 in intervals:  # b9 = minor 2nd
                        quality = "7b9"
                    elif 3 in intervals and 4 in intervals:  # #9 = augmented 2nd (enharmonic to minor 3rd)
                        quality = "7#9"
                    else:
                        quality = "7"
                elif {3, 6, 9}.issubset(intervals):
                    quality = "dim7"
                elif {3, 6, 10}.issubset(intervals):
                    quality = "hdim7"
                #extensions (11ths, 13ths) on triads
                elif {4, 7}.issubset(intervals):
                    if 9 in intervals:  # 13th = major 6th
                        quality = "maj13"
                    elif 5 in intervals:  # 11th = perfect 4th
                        quality = "maj11"
                    else:
                        quality = "maj"
                elif {3, 7}.issubset(intervals):
                    if 9 in intervals:  # 13th
                        quality = "min13"
                    elif 5 in intervals:  # 11th
                        quality = "min11"
                    else:
                        quality = "min"
                else:
                    quality = "N"

                if quality == "N":
                    f.write(f"{start_t:.3f},{end_t:.3f},N\n")
                else:
                    root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root]
                    f.write(f"{start_t:.3f},{end_t:.3f},{root_name}:{quality}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            stem_arrays = []

            #render each stem MIDI file (S00.mid, S01.mid, etc.)
            for i, stem_file in enumerate(stem_files):
                tmp_wav = os.path.join(tmpdir, f"stem_{i}.wav")
                _fluid_render(stem_file, tmp_wav)

                #load rendered audio
                if os.path.exists(tmp_wav):
                    y, _ = librosa.load(tmp_wav, sr=SR)
                    if np.abs(y).max() > 1e-6:  #skip silent stems
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
        print(f"Error {out_name}: {e}", flush=True)


# ============================================================================
#main
# ============================================================================

if __name__ == "__main__":
    #use the RAW slakh dataset (not mutated stems)
    SLAKH_RAW_DIR = "/nobackup/projects/bdyrk27/slakh_workspace/slakh_raw/slakh2100_flac_redux"
    OUTPUT_AUDIO_DIR = "/nobackup/projects/bdyrk27/slakh_workspace/slakh_audio_v2"

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    #collect all track directories across splits
    track_jobs = []
    for split in ["train", "test", "validation", "omitted"]:
        split_dir = os.path.join(SLAKH_RAW_DIR, split)
        if os.path.exists(split_dir):
            for track_name in os.listdir(split_dir):
                track_dir = os.path.join(split_dir, track_name)
                if os.path.isdir(track_dir):
                    track_jobs.append((track_dir, split))

    print(f"Found {len(track_jobs)} tracks to process", flush=True)

    Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_track_dir)(track_dir, OUTPUT_AUDIO_DIR, split)
        for track_dir, split in track_jobs
    )

    print("Rendering complete.", flush=True)
