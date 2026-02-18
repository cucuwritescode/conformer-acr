"""conformer_acr.theory — Musical-domain knowledge and chord vocabulary."""

from conformer_acr.theory.vocabulary import (
    CHORD_LABELS,
    ROOT_LABELS,
    chord_to_index,
    index_to_chord,
    reduce_chord,
)

__all__: list[str] = [
    "CHORD_LABELS",
    "ROOT_LABELS",
    "chord_to_index",
    "index_to_chord",
    "reduce_chord",
]
