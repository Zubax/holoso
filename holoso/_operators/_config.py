"""The user's operator selection, once it has become hardware."""

from dataclasses import dataclass, fields
from typing import TypeVar

from .._errors import UnsupportedConstruct
from .._type import FloatFormat, FloatType, IntFormat, IntType
from ._common import HardwareOperator
from ._float import *
from ._int import IMulOperator


@dataclass(frozen=True, slots=True)
class OperatorOptions:
    """
    ``None`` is not built, and a kernel needing it is refused by name; a configured but unused operator costs nothing.
    Integer operators are always available, so only their knobs appear here.
    """

    fadd: FAddOperator.Options | None = None
    fmul: FMulOperator.Options | None = None
    fdiv: FDivOperator.Options | None = None
    fmul_ilog2: FMulILog2Operator.Options | None = None
    fcmp: FCmpOperator.Options | None = None
    fround: FRoundOperator.Options | None = None
    ffma: FFmaOperator.Options | None = None
    fsort: FSortOperator.Options | None = None
    fexp2: FExp2Operator.Options | None = None
    flog2: FLog2Operator.Options | None = None
    fsincos: FSincosOperator.Options | None = None
    fatan2: FAtan2Operator.Options | None = None
    ffromint: FFromIntOperator.Options | None = None
    ftoint: FToIntOperator.Options | None = None

    imul: IMulOperator.Options = IMulOperator.Options()


@dataclass(frozen=True)
class OpConfig:
    """
    The machine's formats and every configurable operator built for them; construction validates the pairing, so an
    OpConfig cannot name an operator whose ports disagree with its own formats.
    Operators that don't have tunable parameters can be constructed ad-hoc instead.
    An integer operator is never optional, only tuned, so it is always here and never goes through :func:`require`.

    The integer format is an answer rather than a knob, settled against the kernel at selection, so one
    configuration builds more than one of these and nothing here may assume a relation between the two formats:
    a kernel carrying no float leaves the word answering to ``wint_min`` alone, below the float's width.
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

    @staticmethod
    def build(opts: OperatorOptions, fmt: FloatFormat, wmultiplier: int, ifmt: IntFormat) -> OpConfig:
        """The one place the user's configuration becomes hardware, once the machine word is known."""
        return OpConfig(
            float_format=fmt,
            int_format=ifmt,
            fadd=FAddOperator(fmt, opts.fadd) if opts.fadd is not None else None,
            fmul=FMulOperator(fmt, opts.fmul, wmultiplier) if opts.fmul is not None else None,
            fdiv=FDivOperator(fmt, opts.fdiv) if opts.fdiv is not None else None,
            fmul_ilog2=(FMulILog2Operator(fmt, ifmt, opts.fmul_ilog2) if opts.fmul_ilog2 is not None else None),
            fcmp=FCmpOperator(fmt, opts.fcmp) if opts.fcmp is not None else None,
            fround=FRoundOperator(fmt, opts.fround) if opts.fround is not None else None,
            ffma=FFmaOperator(fmt, opts.ffma, wmultiplier) if opts.ffma is not None else None,
            fsort=FSortOperator(fmt, opts.fsort) if opts.fsort is not None else None,
            fexp2=FExp2Operator(fmt, opts.fexp2, wmultiplier) if opts.fexp2 is not None else None,
            flog2=FLog2Operator(fmt, opts.flog2, wmultiplier) if opts.flog2 is not None else None,
            fsincos=FSincosOperator(fmt, opts.fsincos, wmultiplier) if opts.fsincos is not None else None,
            fatan2=FAtan2Operator(fmt, opts.fatan2, wmultiplier) if opts.fatan2 is not None else None,
            ffromint=FFromIntOperator(fmt, ifmt, opts.ffromint) if opts.ffromint is not None else None,
            ftoint=FToIntOperator(fmt, ifmt, opts.ftoint) if opts.ftoint is not None else None,
            imul=IMulOperator(ifmt, opts.imul),
        )

    def __post_init__(self) -> None:
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
