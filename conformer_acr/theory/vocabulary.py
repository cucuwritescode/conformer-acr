"""
conformer_acr.theory.vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chord label ↔ integer mappings for root, quality, and bass.

Keeps all *musical* logic (what *is* a "Cmaj7"?) isolated from the
neural-network and signal-processing code so it can be extended
toward neuro-symbolic reasoning later.
"""

from __future__ import annotations

from typing import Final

# ── Pitch-class constants ────────────────────────────────────────────
ROOT_LABELS: Final[list[str]] = [
    "C", "C#", "D", "Eb", "E", "F",
    "F#", "G", "Ab", "A", "Bb", "B",
]
"""Canonical 12-class root labels (sharps for C♯/F♯, flats elsewhere)."""

QUALITY_LABELS: Final[list[str]] = [
    "maj", "min", "7", "maj7", "min7",
    "dim", "aug", "sus2", "sus4",
    "dim7", "hdim7", "minmaj7",
]
"""Supported chord quality labels."""

BASS_LABELS: Final[list[str]] = [*ROOT_LABELS, "none"]
"""Bass note labels (12 pitch classes + 'none' for root-position)."""

# ── Full chord label list (Root:Quality, 24 basic for now) ───────────
CHORD_LABELS: Final[list[str]] = [
    f"{root}:{qual}"
    for qual in ("maj", "min")
    for root in ROOT_LABELS
]
"""Flat list of all chord labels in vocabulary order."""

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


# ── Public helpers ───────────────────────────────────────────────────
def chord_to_index(label: str) -> int:
    """Map a chord string (e.g. ``'C:maj'``) to its vocabulary index.

    Parameters
    ----------
    label : str
        Chord label in ``Root:Quality`` format.

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
    """Map a root name (e.g. ``'C#'``) to 0-11.

    Parameters
    ----------
    root : str
        Root pitch-class name.

    Returns
    -------
    int
    """
    return _ROOT_TO_IDX[root]


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
