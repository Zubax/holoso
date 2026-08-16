"""
The base scalar annotations, shared by the interpreter conforming a declared parameter and the registry selecting
a lowering. A refinement narrowing one -- `StaticNonNegative[int]` and its twin -- is the registry's own matching
language and is read there, so the tree never learns what a refinement is.
"""

from typing import TypeAliasType

from ._ir import ScalarType

# The subset's one implicit conversion, as a set inclusion so the promotion rule has a single owner.
_ACCEPTED: dict[ScalarType, frozenset[ScalarType]] = {
    ScalarType.BOOL: frozenset({ScalarType.BOOL}),
    ScalarType.INT: frozenset({ScalarType.INT}),
    ScalarType.FLOAT: frozenset({ScalarType.INT, ScalarType.FLOAT}),
}


def accepted_stypes(declared: ScalarType) -> frozenset[ScalarType]:
    return _ACCEPTED[declared]


def unaliased(annotation: object) -> object:
    """
    The annotation a PEP 695 `type` alias names, aliases of aliases included. Reading `__value__` executes the
    alias's lazily deferred body, so the caller owns any exception it raises -- including the cycle refusal, since
    laziness is what lets a `type` alias name itself.
    """
    seen: list[object] = []
    while isinstance(annotation, TypeAliasType):
        if any(annotation is alias for alias in seen):
            raise ValueError(f"the type alias {annotation.__name__!r} is part of a reference cycle")
        seen.append(annotation)
        annotation = annotation.__value__
    return annotation


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
