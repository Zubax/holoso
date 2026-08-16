from dataclasses import dataclass

from .._operators import OperatorOptions
from .._type import FloatFormat


@dataclass(frozen=True, slots=True)
class MirOptions:
    """
    The integer width is not specified but decided by the MIR based on the wint_min, the float format,
    and the actually used operators.
    The idea is that a kernel that needs no floats does not need to size its register file for them, and vice versa.
    """

    operator: OperatorOptions
    float_format: FloatFormat
    wint_min: int
    wmultiplier: int
    ifconv_max_ops: int
