"""
The base scalar annotations, shared by the interpreter conforming a declared parameter and the registry selecting
a lowering. A refinement narrowing one -- ``StaticNonNegative[int]`` and its twin -- is the registry's own matching
language and is read there, so the tree never learns what a refinement is.
"""

from ._ir import ScalarType

# The subset's one implicit conversion, as a set inclusion so the promotion rule has a single owner.
_ACCEPTED: dict[ScalarType, frozenset[ScalarType]] = {
    ScalarType.BOOL: frozenset({ScalarType.BOOL}),
    ScalarType.INT: frozenset({ScalarType.INT}),
    ScalarType.FLOAT: frozenset({ScalarType.INT, ScalarType.FLOAT}),
}


def accepted_stypes(declared: ScalarType) -> frozenset[ScalarType]:
    return _ACCEPTED[declared]


def annotation_stype(annotation: object) -> ScalarType | None:
    if annotation is bool:
        return ScalarType.BOOL
    if annotation is int:
        return ScalarType.INT
    if annotation is float:
        return ScalarType.FLOAT
    return None


def host_type(stype: ScalarType) -> type:
    return {ScalarType.BOOL: bool, ScalarType.INT: int, ScalarType.FLOAT: float}[stype]
