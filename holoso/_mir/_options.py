from dataclasses import dataclass

from .._operators import OperatorOptions
from .._type import FloatFormat


@dataclass(frozen=True, slots=True)
class MirOptions:
    """
    The integer width is not given but decided at lowering from `wint_min`, the float format, and whether any float
    survives optimization, so a kernel without floats need not size its wide register for them.
    """

    operator: OperatorOptions
    float_format: FloatFormat
    wint_min: int
    wmultiplier: int
    ifconv_max_ops: int
