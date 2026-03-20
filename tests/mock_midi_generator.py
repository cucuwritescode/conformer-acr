"""
Mock MIDI Generator for TDD testing of mutate.py

Generates 5 toy MIDI files with specific chord structures for testing
the mutation pipeline logic.
"""

import os
import pretty_midi
import numpy as np


def generate_mock_midi(output_dir: str = "tests/mock_data") -> None:
    """
    Generate 5 mock MIDI files for testing the mutation pipeline.

    Files created:
    - file1_triads.mid: Major and Minor triads
    - file2_dom7.mid: Dominant 7th chords
    - file3_halfdim.mid: Half-Diminished chords
    - file4_polyphonic.mid: Polyphonic with melody and bass
    - file5_mixed.mid: Mixed chord types for quota testing
    """
    os.makedirs(output_dir, exist_ok=True)

    #file 1: major and minor triads
    _create_triads_file(os.path.join(output_dir, "file1_triads.mid"))

    #file 2: dominant 7th chords
    _create_dom7_file(os.path.join(output_dir, "file2_dom7.mid"))

    #file 3: half-diminished chords
    _create_halfdim_file(os.path.join(output_dir, "file3_halfdim.mid"))

    #file 4: polyphonic with melody and bass
    _create_polyphonic_file(os.path.join(output_dir, "file4_polyphonic.mid"))

    #file 5: mixed for quota testing
    _create_mixed_file(os.path.join(output_dir, "file5_mixed.mid"))

    print(f"Generated 5 mock MIDI files in {output_dir}/")


def _add_chord(instrument: pretty_midi.Instrument, pitches: list,
               start: float, end: float, velocity: int = 100) -> None:
    """Helper to add a chord (multiple notes) to an instrument."""
    for pitch in pitches:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)


def _create_triads_file(filepath: str) -> None:
    """File 1: C major and A minor triads."""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")

    #C major triad: C4-E4-G4 (60-64-67)
    _add_chord(piano, [60, 64, 67], start=0.0, end=1.0)

    #A minor triad: A3-C4-E4 (57-60-64)
    _add_chord(piano, [57, 60, 64], start=1.0, end=2.0)

    #F major triad: F3-A3-C4 (53-57-60)
    _add_chord(piano, [53, 57, 60], start=2.0, end=3.0)

    #D minor triad: D4-F4-A4 (62-65-69)
    _add_chord(piano, [62, 65, 69], start=3.0, end=4.0)

    pm.instruments.append(piano)
    pm.write(filepath)


def _create_dom7_file(filepath: str) -> None:
    """File 2: Dominant 7th chords (candidates for altered mutation)."""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")

    #G7: G3-B3-D4-F4 (55-59-62-65)
    _add_chord(piano, [55, 59, 62, 65], start=0.0, end=1.0)

    #C7: C3-E3-G3-Bb3 (48-52-55-58)
    _add_chord(piano, [48, 52, 55, 58], start=1.0, end=2.0)

    #D7: D3-F#3-A3-C4 (50-54-57-60)
    _add_chord(piano, [50, 54, 57, 60], start=2.0, end=3.0)

    #E7: E3-G#3-B3-D4 (52-56-59-62)
    _add_chord(piano, [52, 56, 59, 62], start=3.0, end=4.0)

    pm.instruments.append(piano)
    pm.write(filepath)


def _create_halfdim_file(filepath: str) -> None:
    """File 3: Half-Diminished chords (candidates for symmetrical conversion)."""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")

    #Bm7b5 (B half-dim): B3-D4-F4-A4 (59-62-65-69)
    _add_chord(piano, [59, 62, 65, 69], start=0.0, end=1.0)

    #Em7b5 (E half-dim): E3-G3-Bb3-D4 (52-55-58-62)
    _add_chord(piano, [52, 55, 58, 62], start=1.0, end=2.0)

    #Am7b5 (A half-dim): A3-C4-Eb4-G4 (57-60-63-67)
    _add_chord(piano, [57, 60, 63, 67], start=2.0, end=3.0)

    #F#m7b5 (F# half-dim): F#3-A3-C4-E4 (54-57-60-64)
    _add_chord(piano, [54, 57, 60, 64], start=3.0, end=4.0)

    pm.instruments.append(piano)
    pm.write(filepath)


def _create_polyphonic_file(filepath: str) -> None:
    """File 4: Polyphonic with distinct melody (high) and bass (low) voices."""
    pm = pretty_midi.PrettyMIDI()

    #piano for chords (middle voices)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    #melody instrument (high)
    melody = pretty_midi.Instrument(program=73, name="Flute")

    #bass instrument (low)
    bass = pretty_midi.Instrument(program=32, name="Bass")

    #bar 1: C major context
    #bass: C2 (36)
    bass.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=1.0))
    #chord: E4-G4 (64-67) - inner voices only
    _add_chord(piano, [64, 67], start=0.0, end=1.0)
    #melody: C5 (72)
    melody.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0.0, end=1.0))

    #bar 2: G7 context (dominant 7th - mutation candidate)
    #bass: G2 (43)
    bass.notes.append(pretty_midi.Note(velocity=100, pitch=43, start=1.0, end=2.0))
    #chord: B3-D4-F4 (59-62-65) - inner voices
    _add_chord(piano, [59, 62, 65], start=1.0, end=2.0)
    #melody: G5 (79)
    melody.notes.append(pretty_midi.Note(velocity=100, pitch=79, start=1.0, end=2.0))

    #bar 3: Am context
    #bass: A2 (45)
    bass.notes.append(pretty_midi.Note(velocity=100, pitch=45, start=2.0, end=3.0))
    #chord: C4-E4 (60-64)
    _add_chord(piano, [60, 64], start=2.0, end=3.0)
    #melody: A5 (81)
    melody.notes.append(pretty_midi.Note(velocity=100, pitch=81, start=2.0, end=3.0))

    #bar 4: Bm7b5 context (half-dim - mutation candidate)
    #bass: B2 (47)
    bass.notes.append(pretty_midi.Note(velocity=100, pitch=47, start=3.0, end=4.0))
    #chord: D4-F4-A4 (62-65-69)
    _add_chord(piano, [62, 65, 69], start=3.0, end=4.0)
    #melody: B5 (83)
    melody.notes.append(pretty_midi.Note(velocity=100, pitch=83, start=3.0, end=4.0))

    #add drum track (should be ignored by mutate.py)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for beat in range(8):
        drums.notes.append(pretty_midi.Note(
            velocity=100, pitch=36, start=beat * 0.5, end=beat * 0.5 + 0.25
        ))

    pm.instruments.extend([piano, melody, bass, drums])
    pm.write(filepath)


def _create_mixed_file(filepath: str) -> None:
    """File 5: Mixed chord types for comprehensive quota testing."""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")

    #10 seconds of mixed content at 10 frames/sec = 100 frames total
    chords = [
        #triads (not mutation candidates) - 3 seconds
        ([60, 64, 67], 0.0, 1.0),   # C major
        ([57, 60, 64], 1.0, 2.0),   # A minor
        ([53, 57, 60], 2.0, 3.0),   # F major

        #dominant 7ths (altered candidates) - 3 seconds
        ([55, 59, 62, 65], 3.0, 4.0),   # G7
        ([48, 52, 55, 58], 4.0, 5.0),   # C7
        ([50, 54, 57, 60], 5.0, 6.0),   # D7

        #half-diminished (symmetrical candidates) - 2 seconds
        ([59, 62, 65, 69], 6.0, 7.0),   # Bm7b5
        ([52, 55, 58, 62], 7.0, 8.0),   # Em7b5

        #basic triads for extension candidates - 2 seconds
        ([60, 64, 67], 8.0, 9.0),   # C major (can add 11th/13th)
        ([55, 59, 62], 9.0, 10.0),  # G major (can add 11th/13th)
    ]

    for pitches, start, end in chords:
        _add_chord(piano, pitches, start, end)

    pm.instruments.append(piano)
    pm.write(filepath)


if __name__ == "__main__":
    generate_mock_midi()
