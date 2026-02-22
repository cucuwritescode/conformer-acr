"""
conformer_acr.theory.vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chord label ↔ integer mappings for root, quality, and bass.

Handles enharmonic equivalence (Db == C#), the "N" (no-chord) class,
and structured parsing of MIREX-style chord strings into the three
integer targets the Conformer heads expect (root, bass, quality).

Keeps all *musical* logic (what *is* a "Cmaj7"?) isolated from the
neural-network and signal-processing code so it can be extended
toward neuro-symbolic reasoning later.
"""

from __future__ import annotations

import re
from typing import Final, Tuple

# ── Pitch-class mapping (0–11, 12 = No Chord) ───────────────────────
# Enharmonic equivalence: Db and C# both map to 1, etc.
PITCH_CLASSES: Final[dict[str, int]] = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    'N': 12,  # No Chord / Silence
}

# ── Canonical labels (for reverse-mapping) ───────────────────────────
ROOT_LABELS: Final[list[str]] = [
    "C", "C#", "D", "Eb", "E", "F",
    "F#", "G", "Ab", "A", "Bb", "B",
    "N",
]
"""Canonical 12-class root labels (sharps for C♯/F♯, flats elsewhere) + N."""

BASS_LABELS: Final[list[str]] = [
    "C", "C#", "D", "Eb", "E", "F",
    "F#", "G", "Ab", "A", "Bb", "B",
    "N",
]
"""Bass note labels (12 pitch classes + 'N' for root-position / no bass)."""

# ── Quality mapping (15 classes for the quality head) ────────────────
QUALITIES: Final[dict[str, int]] = {
    'maj': 0, 'min': 1, 'dim': 2, 'aug': 3,
    'maj7': 4, 'min7': 5, '7': 6, 'dim7': 7,
    'hdim7': 8, 'sus2': 9, 'sus4': 10,
    'maj6': 11, 'min6': 12, '9': 13,
    'N': 14,  # No Chord
}

QUALITY_LABELS: Final[list[str]] = [
    "maj", "min", "dim", "aug",
    "maj7", "min7", "7", "dim7",
    "hdim7", "sus2", "sus4",
    "maj6", "min6", "9",
    "N",
]
"""Supported chord quality labels in vocabulary order."""

# ── Full chord label list (Root:Quality, all combinations) ───────────
CHORD_LABELS: Final[list[str]] = [
    *(f"{root}:{qual}"
      for root in ROOT_LABELS[:-1]          # 12 pitch classes (exclude N)
      for qual in QUALITY_LABELS[:-1]),      # 14 qualities (exclude N)
    "N",                                     # No-chord sentinel
]
"""Flat list of all chord labels in vocabulary order (168 + N = 169)."""

# ── Lookup dicts (built once at import time) ─────────────────────────
_CHORD_TO_IDX: Final[dict[str, int]] = {
    label: idx for idx, label in enumerate(CHORD_LABELS)
}
_IDX_TO_CHORD: Final[dict[int, str]] = {
    idx: label for label, idx in _CHORD_TO_IDX.items()
}
_ROOT_TO_IDX: Final[dict[str, int]] = {
    root: idx for idx, root in enumerate(ROOT_LABELS)
}


# ── Core parser ─────────────────────────────────────────────────────
def parse_chord_string(chord_string: str) -> Tuple[int, int, int]:
    """Convert a MIREX-style chord string into ``(root_idx, bass_idx, quality_idx)``.

    Examples
    --------
    >>> parse_chord_string('C:maj/G')
    (0, 7, 0)
    >>> parse_chord_string('N')
    (12, 12, 14)
    >>> parse_chord_string('F#:min7/b7')
    (6, 6, 5)

    Parameters
    ----------
    chord_string : str
        Chord label in MIREX format, e.g. ``'C:maj'``, ``'Db:min7/5'``,
        ``'N'``, or ``'X'``.

    Returns
    -------
    tuple[int, int, int]
        ``(root_idx, bass_idx, quality_idx)`` ready for the three
        classification heads.
    """
    # Handle No-Chord / silence / unknown
    if chord_string == 'N' or chord_string.startswith('X'):
        return PITCH_CLASSES['N'], PITCH_CLASSES['N'], QUALITIES['N']

    # Regex: Root:Quality/Bass  or  Root:Quality  or  Root/Bass  or  Root
    match = re.match(r'^([A-G][#b]?)(?::(.+?))?(?:/(.+))?$', chord_string)

    if not match:
        # Fallback for unrecognised formatting → treat as no-chord
        return PITCH_CLASSES['N'], PITCH_CLASSES['N'], QUALITIES['N']

    root_str, qual_str, bass_str = match.groups()

    # 1. Parse Root
    root_idx = PITCH_CLASSES.get(root_str, PITCH_CLASSES['N'])

    # 2. Parse Quality
    # If no quality is specified, default to major (standard MIREX convention)
    qual_str = qual_str if qual_str else 'maj'
    # Simplify complex extensions if they aren't in our vocab
    qual_idx = QUALITIES.get(qual_str, QUALITIES['maj'])

    # 3. Parse Bass
    # If no bass is specified, the bass is the root
    if not bass_str:
        bass_idx = root_idx
    else:
        # Sometimes bass is written as an interval (e.g., '3' or 'b7').
        # For this setup, we map absolute pitch-class bass notes if
        # provided; otherwise default to root.
        bass_idx = PITCH_CLASSES.get(bass_str, root_idx)

    return root_idx, bass_idx, qual_idx


# ── Convenience helpers ─────────────────────────────────────────────
def chord_to_index(label: str) -> int:
    """Map a chord string (e.g. ``'C:maj'``) to its vocabulary index.

    Parameters
    ----------
    label : str
        Chord label in ``Root:Quality`` format, or ``'N'``.

    Returns
    -------
    int

    Raises
    ------
    KeyError
        If *label* is not in the current vocabulary.
    """
    return _CHORD_TO_IDX[label]


def index_to_chord(idx: int) -> str:
    """Map a vocabulary index back to its chord string.

    Parameters
    ----------
    idx : int
        Integer index into :data:`CHORD_LABELS`.

    Returns
    -------
    str

    Raises
    ------
    KeyError
        If *idx* is out of range.
    """
    return _IDX_TO_CHORD[idx]


def root_to_index(root: str) -> int:
    """Map a root name (e.g. ``'C#'`` or ``'Db'``) to 0–12.

    Parameters
    ----------
    root : str
        Root pitch-class name.

    Returns
    -------
    int
    """
    return PITCH_CLASSES.get(root, PITCH_CLASSES['N'])


def reduce_chord(label: str) -> str:
    """Collapse a complex chord label to a major/minor triad.

    ``'N'`` (no-chord) is returned as-is.  Anything with ``'min'`` in the
    quality becomes ``Root:min``; everything else becomes ``Root:maj``.

    Parameters
    ----------
    label : str
        Raw chord label, e.g. ``'C:min7'``, ``'G:maj'``, ``'N'``.

    Returns
    -------
    str
        Reduced label in ``{Root}:{maj|min}`` format, or ``'N'``.
    """
    if label == "N":
        return "N"
    parts = label.split(":")
    root = parts[0].split("/")[0]
    if len(parts) == 1:
        return f"{root}:maj"
    quality = parts[1].split("/")[0]
    return f"{root}:min" if "min" in quality else f"{root}:maj"
