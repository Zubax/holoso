"""
One of a pair of same-shaped helper modules; the pair exists so a diagnostic's choice between two stores at
identical source positions in DIFFERENT files is observable. Keep the two byte-identical in
layout; a test asserts it.
"""


def put(owner: object, value: float) -> None:
    owner.s = value  # type: ignore[attr-defined]
