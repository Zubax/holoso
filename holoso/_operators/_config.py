"""The user's operator selection, once it has become hardware."""

from dataclasses import dataclass
from typing import Any

from .._errors import UnsupportedConstruct
from ._float import *
from ._int import IMulOperator


@dataclass(frozen=True)
class OpConfig:
    """
    This class only contains operators that are configurable.
    Operators that don't have tunable parameters can be constructed ad-hoc instead.
    An integer operator is never optional, only tuned, so it is always here and never goes through :meth:`require`.
    """

    fadd: FAddOperator | None
    fmul: FMulOperator | None
    fdiv: FDivOperator | None
    fmul_ilog2: FMulILog2OperatorFamily | None
    fmul_ilog2_var: FMulILog2VarOperator | None
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

    def require(self, name: str) -> Any:
        """The named operator, or a refusal naming what needs configuring."""
        operator = getattr(self, name)
        if operator is None:
            raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
        return operator
