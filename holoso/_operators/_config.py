"""The user's operator selection, once it has become hardware."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from .._errors import UnsupportedConstruct
from .._type import FloatFormat, FloatType, IntFormat, IntType
from ._common import HardwareOperator
from ._float import *
from ._int import *


@dataclass(frozen=True, slots=True)
class OperatorOptions:
    """
    Every pooled operator appears here with its own options. A float one may be absent, and a kernel needing it is
    refused by name; an integer one is never optional, only tuned, so it carries its options rather than `None`.
    """

    fadd: FAddOperator.Options | None = None
    fmul: FMulOperator.Options | None = None
    fdiv: FDivOperator.Options | None = None
    fmul_ilog2: FMulILog2Operator.Options | None = None
    filog2: FILog2Operator.Options | None = None
    fcmp: FCmpOperator.Options | None = None
    fround: FRoundOperator.Options | None = None
    ffma: FFmaOperator.Options | None = None
    fsort: FSortOperator.Options | None = None
    fexp2: FExp2Operator.Options | None = None
    flog2: FLog2Operator.Options | None = None
    fsqrt: FSqrtOperator.Options | None = None
    fsincos: FSincosOperator.Options | None = None
    fatan2: FAtan2Operator.Options | None = None
    ffromint: FFromIntOperator.Options | None = None
    ftoint: FToIntOperator.Options | None = None

    iadd: IAddOperator.Options = IAddOperator.Options()
    isub: ISubOperator.Options = ISubOperator.Options()
    imul: IMulOperator.Options = IMulOperator.Options()
    idiv: IDivOperator.Options = IDivOperator.Options()
    iabs: IAbsOperator.Options = IAbsOperator.Options()
    ishl: IShlOperator.Options = IShlOperator.Options()
    ishr: IShrOperator.Options = IShrOperator.Options()
    ipopcnt: IPopcntOperator.Options = IPopcntOperator.Options()
    icmp: ICmpOperator.Options = ICmpOperator.Options()


@dataclass(frozen=True)
class OpConfig:
    """
    The machine's formats and the operators configured for them, built on demand.

    Demand is what makes this sound: the word is settled against the kernel, so one configuration yields several of
    these and nothing here may assume a relation between the two formats. A float-free kernel leaves the word at
    `wint_min`, where an operator holding an exponent cannot be built -- and need not be, nothing there demanding it.
    `cached_property` needs an instance dictionary, hence no slots.
    """

    options: OperatorOptions
    float_format: FloatFormat
    int_format: IntFormat
    wmultiplier: int

    @cached_property
    def fadd(self) -> FAddOperator | None:
        opt = self.options.fadd
        return None if opt is None else self._built(FAddOperator, self.float_format, opt)

    @cached_property
    def fmul(self) -> FMulOperator | None:
        opt = self.options.fmul
        return None if opt is None else self._built(FMulOperator, self.float_format, opt, self.wmultiplier)

    @cached_property
    def fdiv(self) -> FDivOperator | None:
        opt = self.options.fdiv
        return None if opt is None else self._built(FDivOperator, self.float_format, opt)

    @cached_property
    def fmul_ilog2(self) -> FMulILog2Operator | None:
        opt = self.options.fmul_ilog2
        return None if opt is None else self._built(FMulILog2Operator, self.float_format, self.int_format, opt)

    @cached_property
    def filog2(self) -> FILog2Operator | None:
        opt = self.options.filog2
        return None if opt is None else self._built(FILog2Operator, self.float_format, self.int_format, opt)

    @cached_property
    def fcmp(self) -> FCmpOperator | None:
        opt = self.options.fcmp
        return None if opt is None else self._built(FCmpOperator, self.float_format, opt)

    @cached_property
    def fround(self) -> FRoundOperator | None:
        opt = self.options.fround
        return None if opt is None else self._built(FRoundOperator, self.float_format, opt)

    @cached_property
    def ffma(self) -> FFmaOperator | None:
        opt = self.options.ffma
        return None if opt is None else self._built(FFmaOperator, self.float_format, opt, self.wmultiplier)

    @cached_property
    def fsort(self) -> FSortOperator | None:
        opt = self.options.fsort
        return None if opt is None else self._built(FSortOperator, self.float_format, opt)

    @cached_property
    def fexp2(self) -> FExp2Operator | None:
        opt = self.options.fexp2
        return None if opt is None else self._built(FExp2Operator, self.float_format, opt, self.wmultiplier)

    @cached_property
    def flog2(self) -> FLog2Operator | None:
        opt = self.options.flog2
        return None if opt is None else self._built(FLog2Operator, self.float_format, opt, self.wmultiplier)

    @cached_property
    def fsqrt(self) -> FSqrtOperator | None:
        opt = self.options.fsqrt
        return None if opt is None else self._built(FSqrtOperator, self.float_format, opt)

    @cached_property
    def fsincos(self) -> FSincosOperator | None:
        opt = self.options.fsincos
        return None if opt is None else self._built(FSincosOperator, self.float_format, opt, self.wmultiplier)

    @cached_property
    def fatan2(self) -> FAtan2Operator | None:
        opt = self.options.fatan2
        return None if opt is None else self._built(FAtan2Operator, self.float_format, opt, self.wmultiplier)

    @cached_property
    def ffromint(self) -> FFromIntOperator | None:
        opt = self.options.ffromint
        return None if opt is None else self._built(FFromIntOperator, self.float_format, self.int_format, opt)

    @cached_property
    def ftoint(self) -> FToIntOperator | None:
        opt = self.options.ftoint
        return None if opt is None else self._built(FToIntOperator, self.float_format, self.int_format, opt)

    @cached_property
    def iadd(self) -> IAddOperator:
        return self._built(IAddOperator, self.int_format, self.options.iadd)

    @cached_property
    def isub(self) -> ISubOperator:
        return self._built(ISubOperator, self.int_format, self.options.isub)

    @cached_property
    def imul(self) -> IMulOperator:
        return self._built(IMulOperator, self.int_format, self.options.imul)

    @cached_property
    def idiv(self) -> IDivOperator:
        return self._built(IDivOperator, self.int_format, self.options.idiv)

    @cached_property
    def iabs(self) -> IAbsOperator:
        return self._built(IAbsOperator, self.int_format, self.options.iabs)

    @cached_property
    def ishl(self) -> IShlOperator:
        return self._built(IShlOperator, self.int_format, self.options.ishl)

    @cached_property
    def ishr(self) -> IShrOperator:
        return self._built(IShrOperator, self.int_format, self.options.ishr)

    @cached_property
    def ipopcnt(self) -> IPopcntOperator:
        return self._built(IPopcntOperator, self.int_format, self.options.ipopcnt)

    @cached_property
    def icmp(self) -> ICmpOperator:
        return self._built(ICmpOperator, self.int_format, self.options.icmp)

    def _built[**P, T: HardwareOperator](self, build: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Read off the signature rather than off the operator's `fmt`: a conversion operator carries one format per
        side, and asking a format which family it belongs to can only confirm that it matches its own kind.
        """
        try:
            operator = build(*args, **kwargs)
            latency = operator.latency  # the float library loads a format's tables lazily, some only on this question
        except KeyError as ex:  # the float library's refusal of a format it holds no precomputed table for
            (reason,) = ex.args
            raise UnsupportedConstruct(f"a needed operator cannot be built at {self.float_format}: {reason}") from ex
        assert latency >= 0
        signature = operator.signature
        assert all(
            (ty.fmt == self.float_format if isinstance(ty, FloatType) else True)
            and (ty.fmt == self.int_format if isinstance(ty, IntType) else True)
            for ty in signature.operand_types + signature.result_types
        ), f"the configured {operator.mnemonic!r} is not built for the machine's formats of every family its ports name"
        return operator


def require[T: HardwareOperator](operator: T | None, name: str) -> T:
    """The configured operator with its exact type, or a refusal naming what needs configuring."""
    if operator is None:
        raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
    return operator
