"""conformer_acr.theory — Musical-domain knowledge and chord vocabulary."""

from conformer_acr.theory.vocabulary import (
    CHORD_LABELS,
    PITCH_CLASSES,
    QUALITIES,
    QUALITY_LABELS,
    ROOT_LABELS,
    BASS_LABELS,
    chord_to_index,
    index_to_chord,
    parse_chord_string,
    reduce_chord,
    root_to_index,
)

__all__: list[str] = [
    "CHORD_LABELS",
    "PITCH_CLASSES",
    "QUALITIES",
    "QUALITY_LABELS",
    "ROOT_LABELS",
    "BASS_LABELS",
    "chord_to_index",
    "index_to_chord",
    "parse_chord_string",
    "reduce_chord",
    "root_to_index",
]
