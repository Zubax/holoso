"""
Record classes: admissibility, field annotations, and host-instance admission. Admissibility is the
structural-construction contract, not input validation: construction never runs host code, so a dataclass
feature that makes construction RUN code (a user `__init__`, `__post_init__`, an `InitVar`, a
`default_factory`) must refuse rather than silently diverge from the host. The generated-method detector is
the `"<string>"` co_filename dataclasses exec()s its methods from; a user `__init__` surviving under
`init=True` carries a real filename, and `init=False` leaves object's C-level `__init__` with no `__code__`.
"""

import dataclasses
import functools
import inspect

from .._annotations import unaliased


def inadmissible_reason(cls: type) -> str | None:
    """Why a dataclass class cannot be a record, or None when it can; non-dataclasses are not judged here."""
    assert dataclasses.is_dataclass(cls)
    name = cls.__name__
    params = getattr(cls, "__dataclass_params__")
    if not params.frozen:
        return f"the dataclass {name} is not frozen; only @dataclass(frozen=True) records are supported"
    if not params.init:
        return f"the dataclass {name} is declared with init=False, so it has no generated __init__ to mirror"
    code = getattr(cls.__init__, "__code__", None)
    if code is None or code.co_filename != "<string>":
        return f"the dataclass {name} defines its own __init__, which the compiler cannot run"
    if hasattr(cls, "__post_init__"):
        return f"the dataclass {name} defines __post_init__, which the compiler cannot run"
    for field in dataclasses.fields(cls):
        if field.default_factory is not dataclasses.MISSING:
            return f"the field {name}.{field.name} uses default_factory, which the compiler cannot run"
    # Set equality, not order: kw_only fields legitimately reorder the generated signature, and binding is
    # by name anyway; an InitVar (extra parameter) or an init=False field (missing one) fails here.
    declared = set(inspect.signature(cls).parameters)
    if declared != {field.name for field in dataclasses.fields(cls)}:
        return f"the __init__ signature of {name} does not mirror its fields (an InitVar or a non-field parameter)"
    return None


@functools.cache
def field_annotations(cls: type) -> dict[str, object]:
    """
    Field name -> unaliased annotation for every dataclass field, merged over the MRO child-wins
    (`inspect.get_annotations` is own-class-only). Raises whatever a lazy annotation body raises;
    the caller owns locating it.
    """
    merged: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        merged.update(inspect.get_annotations(base, eval_str=True))
    return {field.name: unaliased(merged[field.name]) for field in dataclasses.fields(cls)}
