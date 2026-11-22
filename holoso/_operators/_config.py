"""The user's operator selection, once it has become hardware."""

import logging
from dataclasses import dataclass
from functools import cached_property

from .._errors import UnsupportedConstruct
from .._type import FloatFormat, FloatType, IntFormat, IntType
from ._common import BaseOperatorOptions, HardwareOperator
from ._float import *
from ._int import *

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class OperatorOptions:
    """
    Every kind of operator appears here with its own options, in a field named after the operator's mnemonic.
    A float one may be absent, and a kernel needing it is refused; an integer one is never optional, only tuned.
    """

    fadd: FAddOptions | None = None
    """float `y=a+b`"""
    fmul: FMulOptions | None = None
    """float `y=a*b`"""
    fdivsqrt: FDivsqrtOptions | None = None
    """float `y=a/b` and `y=sqrt(a)`"""
    fmul_ilog2: FMulILog2Options | None = None
    """float `y=a*2**k` (ldexp) for a signed integer `k`"""
    filog2: FILog2Options | None = None
    """
    `y = ⌊log2(|(-1)^s * 1.fraction * 2^(exp-bias)|)⌋ = exp-bias`,
    `bias = 2^(wexp-1)-1`, fraction is discarded; like C `ilogb(a)`, or `frexp(a).exp-1`.
    """
    fcmp: FCmpOptions | None = None
    """float comparison `y=a<b` and sorting."""
    frint: FRintOptions | None = None
    """float `y=round(a)`, `y=ceil(a)`, `y=floor(a)`, `y=trunc(a)`; integer and float output"""
    ffma: FFmaOptions | None = None
    """float `y=a*b+c`"""
    fexp2: FExp2Options | None = None
    """float `y=2**a`"""
    flog2: FLog2Options | None = None
    """float `y=log2(a)`"""
    fcordic: FCordicOptions | None = None
    """float `y=sin(a)`, `y=cos(a)`, `y=atan2(y,x)`, `y=sqrt(a^2+b^2)`"""
    ffromint: FFromIntOptions | None = None
    """float `y=float(a)` for an integer `a`"""

    iadds: IAddsOptions = IAddsOptions()
    """saturated signed integer `y=a+b`"""
    imuls: IMulsOptions = IMulsOptions()
    """saturated signed integer `y=a*b`"""
    idivs: IDivsOptions = IDivsOptions()
    """saturated signed integer `y=a/b` and `y=a%b`"""
    iabss: IAbssOptions = IAbssOptions()
    """saturated signed integer `y=abs(a)`"""
    ishft: IShftOptions = IShftOptions()
    """integer `y=a<<b` (non-saturated) and `y=a>>b` (sign-extended)"""
    ipopcnt: IPopcntOptions = IPopcntOptions()
    """count of nonzero bits in an integer"""


@dataclass(frozen=True)
class OpConfig:
    """
    The machine's formats and the operators configured for them, built on demand; lowering builds the primitives that
    run on them.

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
    def fadd(self) -> FAddOperator:
        return self._checked(FAddOperator.build(self.float_format, self._configured(self.options.fadd, "fadd")))

    @cached_property
    def fmul(self) -> FMulOperator:
        return self._checked(
            FMulOperator.build(self.float_format, self._configured(self.options.fmul, "fmul"), self.wmultiplier)
        )

    @cached_property
    def fdivsqrt(self) -> FDivsqrtOperator:
        return self._checked(
            FDivsqrtOperator.build(self.float_format, self._configured(self.options.fdivsqrt, "fdivsqrt"))
        )

    @cached_property
    def fmul_ilog2(self) -> FMulILog2Operator:
        return self._checked(
            FMulILog2Operator.build(
                self.float_format,
                self.int_format,
                self._configured(self.options.fmul_ilog2, "fmul_ilog2"),
            )
        )

    @cached_property
    def filog2(self) -> FILog2Operator:
        return self._checked(
            FILog2Operator.build(self.float_format, self.int_format, self._configured(self.options.filog2, "filog2"))
        )

    @cached_property
    def fcmp(self) -> FCmpOperator:
        return self._checked(FCmpOperator.build(self.float_format, self._configured(self.options.fcmp, "fcmp")))

    @cached_property
    def frint(self) -> FRintOperator:
        return self._checked(
            FRintOperator.build(self.float_format, self.int_format, self._configured(self.options.frint, "frint"))
        )

    @cached_property
    def ffma(self) -> FFmaOperator:
        return self._checked(
            FFmaOperator.build(self.float_format, self._configured(self.options.ffma, "ffma"), self.wmultiplier)
        )

    @cached_property
    def fexp2(self) -> FExp2Operator:
        return self._checked(
            FExp2Operator.build(self.float_format, self._configured(self.options.fexp2, "fexp2"), self.wmultiplier)
        )

    @cached_property
    def flog2(self) -> FLog2Operator:
        return self._checked(
            FLog2Operator.build(self.float_format, self._configured(self.options.flog2, "flog2"), self.wmultiplier)
        )

    @cached_property
    def fcordic(self) -> FCordicOperator:
        return self._checked(
            FCordicOperator.build(
                self.float_format,
                self._configured(self.options.fcordic, "fcordic"),
                self.wmultiplier,
            )
        )

    @cached_property
    def ffromint(self) -> FFromIntOperator:
        return self._checked(
            FFromIntOperator.build(
                self.float_format,
                self.int_format,
                self._configured(self.options.ffromint, "ffromint"),
            )
        )

    @cached_property
    def iadds(self) -> IAddOperator:
        return self._checked(IAddOperator.build(self.int_format, self.options.iadds))

    @cached_property
    def imuls(self) -> IMulOperator:
        return self._checked(IMulOperator.build(self.int_format, self.options.imuls, self.wmultiplier))

    @cached_property
    def idivs(self) -> IDivOperator:
        return self._checked(IDivOperator.build(self.int_format, self.options.idivs))

    @cached_property
    def iabss(self) -> IAbsOperator:
        return self._checked(IAbsOperator.build(self.int_format, self.options.iabss))

    @cached_property
    def ishft(self) -> IShftOperator:
        return self._checked(IShftOperator.build(self.int_format, self.options.ishft))

    @cached_property
    def ipopcnt(self) -> IPopcntOperator:
        return self._checked(IPopcntOperator.build(self.int_format, self.options.ipopcnt))

    def serves(self, kind: type[HardwareOperator]) -> bool:
        """Whether this machine is configured with operators of `kind`, answered without building one."""
        options = self.options
        configured: dict[type[HardwareOperator], BaseOperatorOptions | None] = {
            FAddOperator: options.fadd,
            FMulOperator: options.fmul,
            FDivsqrtOperator: options.fdivsqrt,
            FMulILog2Operator: options.fmul_ilog2,
            FILog2Operator: options.filog2,
            FCmpOperator: options.fcmp,
            FRintOperator: options.frint,
            FFmaOperator: options.ffma,
            FExp2Operator: options.fexp2,
            FLog2Operator: options.flog2,
            FCordicOperator: options.fcordic,
            FFromIntOperator: options.ffromint,
            IAddOperator: options.iadds,
            IMulOperator: options.imuls,
            IDivOperator: options.idivs,
            IAbsOperator: options.iabss,
            IShftOperator: options.ishft,
            IPopcntOperator: options.ipopcnt,
        }
        return configured[kind] is not None

    @staticmethod
    def _configured[T](options: T | None, name: str) -> T:
        """A float operator is optional, and a kernel needing one left unconfigured is refused by name."""
        if options is None:
            raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
        return options

    def _checked[O: HardwareOperator](self, operator: O) -> O:
        """
        Read off the ports rather than off a format argument: a conversion operator carries one format per side, and
        asking a format which family it belongs to can only confirm that it matches its own kind.
        """
        assert all(
            (ty.fmt == self.float_format if isinstance(ty, FloatType) else True)
            and (ty.fmt == self.int_format if isinstance(ty, IntType) else True)
            for ty in [*operator.operand_ports.values(), *operator.output_ports.values()]
        ), f"the configured {operator.name!r} is not built for the machine's formats of every family its ports name"
        _logger.info("Operator %s: %d instance(s), %s", operator.name, operator.instances, operator.block.timing)
        return operator
