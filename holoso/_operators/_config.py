"""The user's operator selection, once it has become hardware."""

from dataclasses import dataclass, fields
from typing import TypeVar

from .._errors import UnsupportedConstruct
from .._type import FloatFormat, FloatType, IntFormat, IntType
from ._common import HardwareOperator
from ._float import *
from ._int import IMulOperator


@dataclass(frozen=True)
class OpConfig:
    """
    The machine's formats and every configurable operator built for them; construction validates the pairing, so an
    OpConfig cannot name an operator whose ports disagree with its own formats.
    Operators that don't have tunable parameters can be constructed ad-hoc instead.
    An integer operator is never optional, only tuned, so it is always here and never goes through :func:`require`.
    """

    float_format: FloatFormat
    int_format: IntFormat

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

    def __post_init__(self) -> None:
        assert self.int_format.width >= self.float_format.width, "one wide register holds either family whole"
        # Read off the signature rather than off the operator's ``fmt``: a conversion operator carries one format per
        # side, and asking a format which family it belongs to can only confirm that it matches its own kind.
        for field in fields(self):
            operator = getattr(self, field.name)
            if not isinstance(operator, HardwareOperator):
                continue
            signature = operator.signature
            assert all(
                (ty.fmt == self.float_format if isinstance(ty, FloatType) else True)
                and (ty.fmt == self.int_format if isinstance(ty, IntType) else True)
                for ty in signature.operand_types + signature.result_types
            ), f"the configured {field.name!r} is not built for the machine's format of every family its ports name"


_CONFIGURED = TypeVar("_CONFIGURED", bound=HardwareOperator)


def require(operator: _CONFIGURED | None, name: str) -> _CONFIGURED:
    """The configured operator with its exact type, or a refusal naming what needs configuring."""
    if operator is None:
        raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
    return operator
