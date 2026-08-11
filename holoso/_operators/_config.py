"""The user's operator selection, once it has become hardware."""

from dataclasses import dataclass
from typing import TypeVar

from .._errors import UnsupportedConstruct
from ._common import HardwareOperator
from ._float import *
from ._int import IMulOperator


@dataclass(frozen=True)
class OpConfig:
    """
    This class only contains operators that are configurable.
    Operators that don't have tunable parameters can be constructed ad-hoc instead.
    An integer operator is never optional, only tuned, so it is always here and never goes through :func:`require`.
    """

    fadd: FAddOperator | None
    fmul: FMulOperator | None
    fdiv: FDivOperator | None
    fmul_ilog2: FMulILog2Operator | None
    fcmp: FCmpOperator | None
    fround: FRoundOperator | None
    ffma: FFmaOperator | None
    fsort: FSortOperator | None
    fexp2: FExp2Operator | None
    flog2: FLog2Operator | None
    fsincos: FSincosOperator | None
    fatan2: FAtan2Operator | None
    ffromint: FFromIntOperator | None
    ftoint: FToIntOperator | None

    imul: IMulOperator


_CONFIGURED = TypeVar("_CONFIGURED", bound=HardwareOperator)


def require(operator: _CONFIGURED | None, name: str) -> _CONFIGURED:
    """The configured operator with its exact type, or a refusal naming what needs configuring."""
    if operator is None:
        raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
    return operator
