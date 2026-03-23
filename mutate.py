#!/usr/bin/env python3
"""
mutate.py - MIDI Mutation Pipeline for Deep Long-Tail Chord Distribution

Two-Pass Quota System:
- Pass 1: Scan dataset, build candidate ledger, calculate quotas
- Pass 2: Surgically inject mutations to hit exact frame targets

Target quotas:
- Altered Dominants (7b9, 7#9): 10.0%
- Symmetrical (Diminished 7ths): 7.5%
- Upper Extensions (11ths, 13ths): 7.5%
"""

import os
import glob
import random
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pretty_midi


# ============================================================================
# Constants
# ============================================================================

FRAMES_PER_SECOND = 10

#quota percentages
QUOTA_ALTERED = 0.10      # 10%
QUOTA_SYMMETRICAL = 0.075  # 7.5%
QUOTA_EXTENSIONS = 0.075   # 7.5%

#chord interval patterns (in semitones from root)
#major triad: [0, 4, 7]
#minor triad: [0, 3, 7]
#dominant 7: [0, 4, 7, 10]
#half-dim (m7b5): [0, 3, 6, 10]
#diminished 7: [0, 3, 6, 9]

INTERVAL_DOM7 = {0, 4, 7, 10}
INTERVAL_HALFDIM = {0, 3, 6, 10}
INTERVAL_MAJOR = {0, 4, 7}
INTERVAL_MINOR = {0, 3, 7}

#polyphonic instrument programs (piano, strings, pads)
POLYPHONIC_PROGRAMS = set(range(0, 8)) | set(range(24, 32)) | set(range(40, 56)) | set(range(88, 96))


# ============================================================================
#pass 1: global audit functions
# ============================================================================

def calculate_total_frames(input_dir: str, frames_per_second: int = FRAMES_PER_SECOND) -> int:
    """
    Calculate total harmonic frames across all MIDI files in directory.

    Parameters
    ----------
    input_dir : str
        Directory containing MIDI files.
    frames_per_second : int
        Frame resolution (default 10 fps).

    Returns
    -------
    int
        Total number of frames across all files.
    """
    total_frames = 0
    #recursive to dig into Slakh subfolders
    midi_files = glob.glob(os.path.join(input_dir, "**", "*.mid"), recursive=True)

    for filepath in midi_files:
        pm = pretty_midi.PrettyMIDI(filepath)
        duration = pm.get_end_time()
        total_frames += int(duration * frames_per_second)

    return total_frames


def calculate_quotas(total_frames: int) -> Dict[str, int]:
    """
    Calculate exact frame quotas for each mutation type.

    Parameters
    ----------
    total_frames : int
        Total harmonic frames in dataset.

    Returns
    -------
    dict
        Frame quotas: {"altered": int, "symmetrical": int, "extensions": int}
    """
    return {
        "altered": int(total_frames * QUOTA_ALTERED),
        "symmetrical": int(total_frames * QUOTA_SYMMETRICAL),
        "extensions": int(total_frames * QUOTA_EXTENSIONS),
    }


def get_outer_voices(pitches: List[int]) -> Tuple[int, int]:
    """
    Get the highest (melody) and lowest (bass) pitches from a chord.

    Parameters
    ----------
    pitches : list of int
        MIDI pitch values.

    Returns
    -------
    tuple
        (highest_pitch, lowest_pitch)
    """
    if not pitches:
        return (0, 0)
    return (max(pitches), min(pitches))


def _get_intervals_from_pitches(pitches: List[int]) -> set:
    """Convert pitches to interval set relative to lowest note."""
    if len(pitches) < 3:
        return set()

    root = min(pitches)
    intervals = set()
    for p in pitches:
        interval = (p - root) % 12
        intervals.add(interval)
    return intervals


def _identify_chord_type(pitches: List[int]) -> Optional[str]:
    """
    Identify chord type from pitches.

    Returns
    -------
    str or None
        "dom7", "halfdim", "triad", or None if unrecognized.
    """
    if len(pitches) < 3:
        return None

    intervals = _get_intervals_from_pitches(pitches)

    #check for 4-note chords first
    if len(pitches) >= 4:
        if intervals == INTERVAL_DOM7:
            return "dom7"
        if intervals == INTERVAL_HALFDIM:
            return "halfdim"

    #check for triads (candidates for extensions)
    if INTERVAL_MAJOR.issubset(intervals) or INTERVAL_MINOR.issubset(intervals):
        return "triad"

    return None


def _extract_chords_from_track(
    instrument: pretty_midi.Instrument,
    filepath: str,
    track_idx: int,
    time_resolution: float = 0.1
) -> List[Dict]:
    """
    Extract chord events from an instrument track.

    Groups simultaneous notes into chords and identifies their types.
    """
    if instrument.is_drum:
        return []

    #skip non-polyphonic instruments
    if instrument.program not in POLYPHONIC_PROGRAMS:
        return []

    if not instrument.notes:
        return []

    #find time range
    start_time = min(n.start for n in instrument.notes)
    end_time = max(n.end for n in instrument.notes)

    chords = []
    current_time = start_time

    while current_time < end_time:
        #find notes active at this time
        active_notes = [
            n for n in instrument.notes
            if n.start <= current_time < n.end
        ]

        if len(active_notes) >= 3:
            pitches = [n.pitch for n in active_notes]
            chord_type = _identify_chord_type(pitches)

            if chord_type:
                #find end of this chord (when any note ends)
                chord_end = min(n.end for n in active_notes)

                chords.append({
                    "filepath": filepath,
                    "track_idx": track_idx,
                    "start_time": current_time,
                    "end_time": chord_end,
                    "chord_type": chord_type,
                    "pitches": pitches,
                })

                #skip to end of this chord
                current_time = chord_end
                continue

        current_time += time_resolution

    return chords


def build_candidate_ledger(input_dir: str) -> List[Dict]:
    """
    Build global ledger of all mutation candidate locations.

    Scans all MIDI files and identifies:
    - Dominant 7th chords (for altered mutation)
    - Half-diminished chords (for symmetrical conversion)
    - Triads (for extension injection)

    Parameters
    ----------
    input_dir : str
        Directory containing MIDI files.

    Returns
    -------
    list of dict
        Ledger entries with: filepath, track_idx, start_time, end_time, chord_type, pitches
    """
    ledger = []
    #recursive to dig into Slakh subfolders
    midi_files = glob.glob(os.path.join(input_dir, "**", "*.mid"), recursive=True)

    for filepath in midi_files:
        pm = pretty_midi.PrettyMIDI(filepath)

        for track_idx, instrument in enumerate(pm.instruments):
            chords = _extract_chords_from_track(
                instrument, filepath, track_idx
            )
            ledger.extend(chords)

    return ledger


def audit_dataset(input_dir: str, frames_per_second: int = FRAMES_PER_SECOND) -> Dict:
    """
    Complete Pass 1 audit of the dataset.

    Returns
    -------
    dict
        {
            "total_frames": int,
            "quotas": {"altered": int, "symmetrical": int, "extensions": int},
            "ledger": list of candidate dicts,
            "candidates_by_type": {"dom7": list, "halfdim": list, "triad": list}
        }
    """
    print("Pass 1: Auditing dataset...", flush=True)

    #calculate total frames
    total_frames = calculate_total_frames(input_dir, frames_per_second)
    print(f"  Total frames: {total_frames}", flush=True)

    #calculate quotas
    quotas = calculate_quotas(total_frames)
    print(f"  Quotas:", flush=True)
    print(f"    Altered (10.0%): {quotas['altered']} frames", flush=True)
    print(f"    Symmetrical (7.5%): {quotas['symmetrical']} frames", flush=True)
    print(f"    Extensions (7.5%): {quotas['extensions']} frames", flush=True)

    #build candidate ledger
    ledger = build_candidate_ledger(input_dir)
    print(f"  Found {len(ledger)} candidate chord locations", flush=True)

    #group by type
    candidates_by_type = {
        "dom7": [e for e in ledger if e["chord_type"] == "dom7"],
        "halfdim": [e for e in ledger if e["chord_type"] == "halfdim"],
        "triad": [e for e in ledger if e["chord_type"] == "triad"],
    }

    print(f"    Dominant 7ths: {len(candidates_by_type['dom7'])}", flush=True)
    print(f"    Half-diminished: {len(candidates_by_type['halfdim'])}", flush=True)
    print(f"    Triads: {len(candidates_by_type['triad'])}", flush=True)

    return {
        "total_frames": total_frames,
        "quotas": quotas,
        "ledger": ledger,
        "candidates_by_type": candidates_by_type,
    }


# ============================================================================
#pass 2: mutation functions
# ============================================================================

def inject_altered_dominant(pitches: List[int], alteration: str = "b9") -> List[int]:
    """
    Inject b9 or #9 into a dominant 7th chord.

    Preserves root, 3rd, and 7th. The 5th may be replaced or the alteration added.

    Parameters
    ----------
    pitches : list of int
        Original chord pitches (must be dominant 7th).
    alteration : str
        "b9" for flat 9, "#9" for sharp 9.

    Returns
    -------
    list of int
        Mutated pitches with alteration added.
    """
    if len(pitches) < 4:
        return pitches

    pitches = pitches.copy()
    root = min(pitches)
    high, low = get_outer_voices(pitches)

    #calculate alteration pitch
    if alteration == "b9":
        #b9 = 1 semitone above root (enharmonic)
        alt_pitch = root + 13  # octave + 1
    else:  # #9
        #sharp 9 = 3 semitones above root (enharmonic to minor 3rd)
        alt_pitch = root + 15  # octave + 3

    #ensure alteration doesn't exceed melody
    while alt_pitch > high:
        alt_pitch -= 12

    #ensure alteration doesn't go below bass
    while alt_pitch < low:
        alt_pitch += 12

    #if still conflicts with outer voices, place in middle
    if alt_pitch >= high or alt_pitch <= low:
        alt_pitch = root + 13 if alteration == "b9" else root + 15
        while alt_pitch >= high:
            alt_pitch -= 12
        while alt_pitch <= low:
            alt_pitch += 12

    #add alteration (don't remove existing notes to preserve chord identity)
    if alt_pitch not in pitches and low < alt_pitch < high:
        pitches.append(alt_pitch)

    return pitches


def convert_to_diminished(pitches: List[int]) -> List[int]:
    """
    Convert half-diminished to fully diminished by flattening the 7th.

    Half-dim: root, b3, b5, b7 (minor 7th = 10 semitones)
    Full dim: root, b3, b5, bb7 (diminished 7th = 9 semitones)

    Parameters
    ----------
    pitches : list of int
        Original half-diminished chord pitches.

    Returns
    -------
    list of int
        Fully diminished chord pitches.
    """
    if len(pitches) < 4:
        return pitches

    pitches = pitches.copy()
    root = min(pitches)

    #find the minor 7th (10 semitones from root) and flatten it
    for i, p in enumerate(pitches):
        interval = (p - root) % 12
        if interval == 10:  # minor 7th
            #flatten to diminished 7th (9 semitones)
            pitches[i] = p - 1
            break

    return pitches


def inject_extension(pitches: List[int], extension_type: str = "13") -> List[int]:
    """
    Inject 11th or 13th into a triad while preserving outer voices.

    The extension is placed between bass and melody to preserve outer voices.

    Parameters
    ----------
    pitches : list of int
        Original triad pitches.
    extension_type : str
        "11" for 11th, "13" for 13th.

    Returns
    -------
    list of int
        Chord with extension added.
    """
    if len(pitches) < 3:
        return pitches

    pitches = pitches.copy()
    root = min(pitches)
    high, low = get_outer_voices(pitches)

    #calculate extension interval from root
    if extension_type == "11":
        ext_interval = 5  # perfect 4th
    else:  # "13"
        ext_interval = 9  # major 6th

    #find valid octave placement: must be strictly between outer voices
    ext_pitch = root + ext_interval

    #adjust octave to fit between bass and melody
    while ext_pitch <= low:
        ext_pitch += 12
    while ext_pitch >= high:
        ext_pitch -= 12

    #verify it's valid: strictly between outer voices and not duplicate
    if low < ext_pitch < high and ext_pitch not in pitches:
        pitches.append(ext_pitch)

    return pitches


def _apply_mutation_to_track(
    instrument: pretty_midi.Instrument,
    mutations: List[Dict]
) -> None:
    """
    Apply mutations to notes in an instrument track.

    Modifies the instrument in place.
    """
    for mutation in mutations:
        start_time = mutation["start_time"]
        end_time = mutation["end_time"]
        mutation_type = mutation["mutation_type"]
        original_pitches = mutation["pitches"]

        #find notes in this time range
        affected_notes = [
            n for n in instrument.notes
            if n.start <= start_time < n.end or n.start < end_time <= n.end
            or (start_time <= n.start and n.end <= end_time)
        ]

        if not affected_notes:
            continue

        #get outer voices to preserve
        current_pitches = [n.pitch for n in affected_notes]
        high, low = get_outer_voices(current_pitches)

        #calculate new pitches based on mutation type
        if mutation_type == "altered":
            alteration = random.choice(["b9", "#9"])
            new_pitches = inject_altered_dominant(current_pitches, alteration)
        elif mutation_type == "symmetrical":
            new_pitches = convert_to_diminished(current_pitches)
        elif mutation_type == "extension":
            ext_type = random.choice(["11", "13"])
            new_pitches = inject_extension(current_pitches, ext_type)
        else:
            continue

        #find pitches to add (new ones not in original)
        pitches_to_add = [p for p in new_pitches if p not in current_pitches]

        #find pitches to modify (changed pitches)
        for note in affected_notes:
            #check if this pitch was modified (for diminished conversion)
            old_interval = (note.pitch - low) % 12
            for new_p in new_pitches:
                new_interval = (new_p - low) % 12
                #if interval changed by 1 semitone, this is the modified note
                if abs(old_interval - new_interval) == 1 or abs(old_interval - new_interval) == 11:
                    if note.pitch != high and note.pitch != low:  # preserve outer voices
                        note.pitch = new_p
                        break

        #add new notes for extensions/alterations
        for pitch in pitches_to_add:
            if low < pitch < high:  # only add between outer voices
                new_note = pretty_midi.Note(
                    velocity=affected_notes[0].velocity,
                    pitch=pitch,
                    start=start_time,
                    end=min(end_time, affected_notes[0].end)
                )
                instrument.notes.append(new_note)


def mutate_dataset(
    input_dir: str,
    output_dir: str,
    frames_per_second: int = FRAMES_PER_SECOND
) -> None:
    """
    Full mutation pipeline: audit, select candidates, inject mutations.

    Parameters
    ----------
    input_dir : str
        Directory containing original MIDI files.
    output_dir : str
        Directory to save mutated files.
    frames_per_second : int
        Frame resolution for quota calculation.
    """
    os.makedirs(output_dir, exist_ok=True)

    #pass 1: audit
    audit = audit_dataset(input_dir, frames_per_second)
    quotas = audit["quotas"]
    candidates = audit["candidates_by_type"]

    print("\nPass 2: Selecting mutation targets...", flush=True)

    #track mutations by file
    mutations_by_file: Dict[str, List[Dict]] = {}

    #select altered dominant candidates
    dom7_candidates = candidates["dom7"].copy()
    random.shuffle(dom7_candidates)
    altered_frames = 0

    for candidate in dom7_candidates:
        if altered_frames >= quotas["altered"]:
            break
        duration = candidate["end_time"] - candidate["start_time"]
        frames = int(duration * frames_per_second)

        filepath = candidate["filepath"]
        if filepath not in mutations_by_file:
            mutations_by_file[filepath] = []

        mutations_by_file[filepath].append({
            **candidate,
            "mutation_type": "altered"
        })
        altered_frames += frames

    print(f"  Altered dominants: {altered_frames}/{quotas['altered']} frames", flush=True)

    #select symmetrical (half-dim -> dim7) candidates
    halfdim_candidates = candidates["halfdim"].copy()
    random.shuffle(halfdim_candidates)
    symmetrical_frames = 0

    for candidate in halfdim_candidates:
        if symmetrical_frames >= quotas["symmetrical"]:
            break
        duration = candidate["end_time"] - candidate["start_time"]
        frames = int(duration * frames_per_second)

        filepath = candidate["filepath"]
        if filepath not in mutations_by_file:
            mutations_by_file[filepath] = []

        mutations_by_file[filepath].append({
            **candidate,
            "mutation_type": "symmetrical"
        })
        symmetrical_frames += frames

    print(f"  Symmetrical: {symmetrical_frames}/{quotas['symmetrical']} frames", flush=True)

    #select extension candidates
    triad_candidates = candidates["triad"].copy()
    random.shuffle(triad_candidates)
    extension_frames = 0

    for candidate in triad_candidates:
        if extension_frames >= quotas["extensions"]:
            break
        duration = candidate["end_time"] - candidate["start_time"]
        frames = int(duration * frames_per_second)

        filepath = candidate["filepath"]
        if filepath not in mutations_by_file:
            mutations_by_file[filepath] = []

        mutations_by_file[filepath].append({
            **candidate,
            "mutation_type": "extension"
        })
        extension_frames += frames

    print(f"  Extensions: {extension_frames}/{quotas['extensions']} frames", flush=True)

    #pass 2: apply mutations and save
    print("\nPass 2: Applying mutations...", flush=True)

    #recursive to dig into Slakh subfolders
    midi_files = glob.glob(os.path.join(input_dir, "**", "*.mid"), recursive=True)

    for filepath in midi_files:
        pm = pretty_midi.PrettyMIDI(filepath)

        #recreate the directory tree so files don't overwrite each other
        #(Slakh names every stem S01.mid, S02.mid, etc.)
        rel_path = os.path.relpath(filepath, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if filepath in mutations_by_file:
            #group mutations by track
            mutations_by_track: Dict[int, List[Dict]] = {}
            for mutation in mutations_by_file[filepath]:
                track_idx = mutation["track_idx"]
                if track_idx not in mutations_by_track:
                    mutations_by_track[track_idx] = []
                mutations_by_track[track_idx].append(mutation)

            #apply mutations to each track
            for track_idx, track_mutations in mutations_by_track.items():
                if track_idx < len(pm.instruments):
                    _apply_mutation_to_track(pm.instruments[track_idx], track_mutations)

        #save mutated file
        pm.write(output_path)

    print(f"\nMutated files saved to {output_dir}/", flush=True)
    print("Mutation complete.", flush=True)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Mutation Pipeline")
    parser.add_argument("--input-dir", type=str, default="tests/mock_data",
                        help="Input directory with MIDI files")
    parser.add_argument("--output-dir", type=str, default="tests/mock_data_mutated",
                        help="Output directory for mutated files")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only run Pass 1 audit, don't mutate")
    args = parser.parse_args()

    if args.audit_only:
        audit = audit_dataset(args.input_dir)
        print("\nAudit complete.", flush=True)
    else:
        mutate_dataset(args.input_dir, args.output_dir)
