"""
Test suite for mutate.py - MIDI mutation pipeline.

Tests the Two-Pass Quota logic for engineering the Deep Long-Tail
chord distribution in the Slakh2100 dataset.
"""

import os
import sys
import pytest
import pretty_midi
import numpy as np

#add paths for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _THIS_DIR)

from mock_midi_generator import generate_mock_midi

#import mutate functions (will be implemented in Step 2/3)
#from mutate import (
#    calculate_total_frames,
#    calculate_quotas,
#    build_candidate_ledger,
#    inject_altered_dominant,
#    convert_to_diminished,
#    inject_extension,
#    get_outer_voices,
#    mutate_dataset,
#)


#constants
FRAMES_PER_SECOND = 10
MOCK_DATA_DIR = "tests/mock_data"


@pytest.fixture(scope="module", autouse=True)
def setup_mock_data():
    """Generate mock MIDI files before running tests."""
    generate_mock_midi(MOCK_DATA_DIR)
    yield
    #cleanup could go here if needed


class TestGlobalFrameCount:
    """Tests for calculating total harmonic frames across MIDI files."""

    def test_global_frame_count(self):
        """
        Asserts the script correctly calculates the total active
        harmonic frames across all MIDI files.

        Expected: Each file has specific durations:
        - file1_triads.mid: 4 seconds = 40 frames
        - file2_dom7.mid: 4 seconds = 40 frames
        - file3_halfdim.mid: 4 seconds = 40 frames
        - file4_polyphonic.mid: 4 seconds = 40 frames
        - file5_mixed.mid: 10 seconds = 100 frames
        Total: 260 frames
        """
        from mutate import calculate_total_frames

        total_frames = calculate_total_frames(MOCK_DATA_DIR, FRAMES_PER_SECOND)
        assert total_frames == 260, f"Expected 260 frames, got {total_frames}"

    def test_frame_calculation_single_file(self):
        """Test frame calculation for a single known file."""
        from mutate import calculate_total_frames

        #file5 is exactly 10 seconds = 100 frames
        pm = pretty_midi.PrettyMIDI(os.path.join(MOCK_DATA_DIR, "file5_mixed.mid"))
        duration = pm.get_end_time()
        expected_frames = int(duration * FRAMES_PER_SECOND)
        assert expected_frames == 100


class TestQuotaCalculation:
    """Tests for quota calculation logic."""

    def test_quota_calculation(self):
        """
        Asserts that given 10,000 total frames, the targets are:
        - Altered Dominants: 10.0% = 1,000 frames
        - Symmetrical: 7.5% = 750 frames
        - Extensions: 7.5% = 750 frames
        """
        from mutate import calculate_quotas

        quotas = calculate_quotas(total_frames=10000)

        assert quotas["altered"] == 1000, f"Altered should be 1000, got {quotas['altered']}"
        assert quotas["symmetrical"] == 750, f"Symmetrical should be 750, got {quotas['symmetrical']}"
        assert quotas["extensions"] == 750, f"Extensions should be 750, got {quotas['extensions']}"

    def test_quota_calculation_small(self):
        """Test quota calculation with smaller frame count."""
        from mutate import calculate_quotas

        quotas = calculate_quotas(total_frames=1000)

        assert quotas["altered"] == 100
        assert quotas["symmetrical"] == 75
        assert quotas["extensions"] == 75

    def test_quota_calculation_rounding(self):
        """Test that quotas round to integers correctly."""
        from mutate import calculate_quotas

        #333 frames: 10% = 33.3, 7.5% = 24.975
        quotas = calculate_quotas(total_frames=333)

        assert isinstance(quotas["altered"], int)
        assert isinstance(quotas["symmetrical"], int)
        assert isinstance(quotas["extensions"], int)


class TestOuterVoicePreservation:
    """Tests for the guardrail that preserves melody and bass."""

    def test_outer_voice_preservation(self):
        """
        Asserts that when an extension pitch is injected into a chord,
        the highest and lowest original notes remain unchanged.
        """
        from mutate import inject_extension, get_outer_voices

        #original chord with wide voicing: C3-E4-G4-C5 (48-64-67-72)
        #this gives room for a 13th (A4=69) to fit between E4 and C5
        original_pitches = [48, 64, 67, 72]
        original_high, original_low = get_outer_voices(original_pitches)

        assert original_high == 72, "Highest pitch should be C5 (72)"
        assert original_low == 48, "Lowest pitch should be C3 (48)"

        #inject 13th (A4 = 69, fits between G4=67 and C5=72)
        mutated_pitches = inject_extension(original_pitches.copy(), extension_type="13")

        #verify outer voices unchanged
        mutated_high, mutated_low = get_outer_voices(mutated_pitches)
        assert mutated_high == original_high, "Melody (highest) must not change"
        assert mutated_low == original_low, "Bass (lowest) must not change"

        #verify extension was added (A in some octave: 57, 69, 81...)
        a_pitches = [p for p in mutated_pitches if p % 12 == 9]  # A = 9 mod 12
        assert len(a_pitches) > 0, "13th (A) should be added"

    def test_outer_voice_with_extension_higher_than_melody(self):
        """
        If the extension pitch would be higher than melody,
        it should be placed an octave lower.
        """
        from mutate import inject_extension, get_outer_voices

        #chord with high melody: C4-E4-G5 (60-64-79)
        original_pitches = [60, 64, 79]
        original_high = 79

        mutated_pitches = inject_extension(original_pitches.copy(), extension_type="13")

        mutated_high, _ = get_outer_voices(mutated_pitches)
        assert mutated_high == original_high, "Melody must remain highest"


class TestAlteredDominantInjection:
    """Tests for altered dominant mutation (7b9, 7#9)."""

    def test_altered_dominant_injection(self):
        """
        Asserts that a standard Dominant 7th (e.g., G-B-D-F) correctly
        receives the flat-9 and/or sharp-9 pitches without deleting
        the root, third, or seventh.
        """
        from mutate import inject_altered_dominant

        #G7: G3-B3-D4-F4 (55-59-62-65)
        #root=G(55), 3rd=B(59), 5th=D(62), 7th=F(65)
        original_pitches = [55, 59, 62, 65]

        #inject b9 (Ab = 56 or 68)
        mutated_pitches = inject_altered_dominant(original_pitches.copy(), alteration="b9")

        #verify essential chord tones preserved
        assert 55 in mutated_pitches, "Root (G) must be preserved"
        assert 59 in mutated_pitches, "3rd (B) must be preserved"
        assert 65 in mutated_pitches, "7th (F) must be preserved"

        #verify b9 added (Ab in some octave)
        ab_pitches = [p for p in mutated_pitches if p % 12 == 8]  # Ab = 8 mod 12
        assert len(ab_pitches) > 0, "b9 (Ab) should be added"

    def test_altered_dominant_sharp9(self):
        """Test #9 injection on dominant 7th."""
        from mutate import inject_altered_dominant

        #C7: C3-E3-G3-Bb3 (48-52-55-58)
        original_pitches = [48, 52, 55, 58]

        mutated_pitches = inject_altered_dominant(original_pitches.copy(), alteration="#9")

        #verify root, 3rd, 7th preserved
        assert 48 in mutated_pitches, "Root (C) must be preserved"
        assert 52 in mutated_pitches, "3rd (E) must be preserved"
        assert 58 in mutated_pitches, "7th (Bb) must be preserved"

        #verify #9 added (D# = 3 mod 12)
        ds_pitches = [p for p in mutated_pitches if p % 12 == 3]
        assert len(ds_pitches) > 0, "#9 (D#) should be added"


class TestSymmetricalConversion:
    """Tests for half-diminished to fully-diminished conversion."""

    def test_symmetrical_conversion(self):
        """
        Asserts that a half-diminished chord (e.g., B-D-F-A) has its
        7th flattened (A -> Ab) to become fully diminished.

        Bm7b5: B-D-F-A (59-62-65-69)
        Bdim7: B-D-F-Ab (59-62-65-68)
        """
        from mutate import convert_to_diminished

        #Bm7b5: B3-D4-F4-A4 (59-62-65-69)
        original_pitches = [59, 62, 65, 69]

        mutated_pitches = convert_to_diminished(original_pitches.copy())

        #verify root, b3, b5 preserved
        assert 59 in mutated_pitches, "Root (B) must be preserved"
        assert 62 in mutated_pitches, "b3 (D) must be preserved"
        assert 65 in mutated_pitches, "b5 (F) must be preserved"

        #verify 7th flattened: A(69) -> Ab(68)
        assert 69 not in mutated_pitches, "Original 7th (A) should be removed"
        assert 68 in mutated_pitches, "Flattened 7th (Ab) should be present"

    def test_symmetrical_conversion_em7b5(self):
        """Test conversion on Em7b5."""
        from mutate import convert_to_diminished

        #Em7b5: E3-G3-Bb3-D4 (52-55-58-62)
        #Edim7: E3-G3-Bb3-Db4 (52-55-58-61)
        original_pitches = [52, 55, 58, 62]

        mutated_pitches = convert_to_diminished(original_pitches.copy())

        assert 52 in mutated_pitches, "Root (E) preserved"
        assert 55 in mutated_pitches, "b3 (G) preserved"
        assert 58 in mutated_pitches, "b5 (Bb) preserved"
        assert 62 not in mutated_pitches, "Original 7th (D) removed"
        assert 61 in mutated_pitches, "bb7 (Db) added"


class TestDrumGuardrail:
    """Tests that drum tracks are never mutated."""

    def test_drums_ignored(self):
        """Verify drum tracks are skipped during mutation."""
        from mutate import build_candidate_ledger

        ledger = build_candidate_ledger(MOCK_DATA_DIR)

        #file4 has a drum track - verify no candidates from it
        for entry in ledger:
            if entry["filepath"].endswith("file4_polyphonic.mid"):
                pm = pretty_midi.PrettyMIDI(entry["filepath"])
                track = pm.instruments[entry["track_idx"]]
                assert not track.is_drum, "Drum tracks should never appear in ledger"


class TestCandidateLedger:
    """Tests for Pass 1 - building the global candidate ledger."""

    def test_ledger_structure(self):
        """Verify ledger entries have required fields."""
        from mutate import build_candidate_ledger

        ledger = build_candidate_ledger(MOCK_DATA_DIR)

        assert len(ledger) > 0, "Ledger should not be empty"

        for entry in ledger:
            assert "filepath" in entry
            assert "track_idx" in entry
            assert "start_time" in entry
            assert "end_time" in entry
            assert "chord_type" in entry
            assert entry["chord_type"] in ["dom7", "halfdim", "triad"]

    def test_ledger_finds_dom7_candidates(self):
        """Verify ledger identifies dominant 7th chords."""
        from mutate import build_candidate_ledger

        ledger = build_candidate_ledger(MOCK_DATA_DIR)

        dom7_entries = [e for e in ledger if e["chord_type"] == "dom7"]
        assert len(dom7_entries) >= 4, "Should find at least 4 dom7 chords (from file2)"

    def test_ledger_finds_halfdim_candidates(self):
        """Verify ledger identifies half-diminished chords."""
        from mutate import build_candidate_ledger

        ledger = build_candidate_ledger(MOCK_DATA_DIR)

        halfdim_entries = [e for e in ledger if e["chord_type"] == "halfdim"]
        assert len(halfdim_entries) >= 4, "Should find at least 4 half-dim chords (from file3)"


class TestFullPipeline:
    """Integration tests for the complete mutation pipeline."""

    def test_mutate_dataset_creates_output(self):
        """Verify the full pipeline creates mutated files."""
        from mutate import mutate_dataset

        output_dir = "tests/mock_data_mutated"
        mutate_dataset(
            input_dir=MOCK_DATA_DIR,
            output_dir=output_dir,
            frames_per_second=FRAMES_PER_SECOND
        )

        #verify output files exist
        assert os.path.exists(output_dir)
        output_files = [f for f in os.listdir(output_dir) if f.endswith(".mid")]
        assert len(output_files) == 5, "Should create 5 mutated files"

    def test_mutate_preserves_filenames(self):
        """Verify output files have same names as input."""
        from mutate import mutate_dataset

        output_dir = "tests/mock_data_mutated"
        mutate_dataset(
            input_dir=MOCK_DATA_DIR,
            output_dir=output_dir,
            frames_per_second=FRAMES_PER_SECOND
        )

        input_files = set(os.listdir(MOCK_DATA_DIR))
        output_files = set(os.listdir(output_dir))

        assert input_files == output_files, "Output filenames should match input"
