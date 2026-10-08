"""The user's operator selection, once it has become hardware."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import zkf

from .._errors import UnsupportedConstruct
from .._type import FloatFormat, FloatType, IntFormat, IntType
from ._common import HardwareOperator
from ._float import *
from ._int import *

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OperatorOptions:
    """
    Every kind of operator appears here with its own options, in a field named after the operator's mnemonic. A float
    one may be absent, and a kernel needing it is refused by name; an integer one is never optional, only tuned, so it
    carries its options rather than `None`.
    """

    fadd: FAddOptions | None = None
    fmul: FMulOptions | None = None
    fdivsqrt: FDivsqrtOptions | None = None
    fmul_ilog2: FMulILog2Options | None = None
    filog2: FILog2Options | None = None
    fcmp: FCmpOptions | None = None
    frint: FRintOptions | None = None
    ffma: FFmaOptions | None = None
    fsort: FSortOptions | None = None
    fexp2: FExp2Options | None = None
    flog2: FLog2Options | None = None
    fcordic: FCordicOptions | None = None
    ffromint: FFromIntOptions | None = None

    iadds: IAddOptions = IAddOptions()
    isubs: ISubOptions = ISubOptions()
    imuls: IMulOptions = IMulOptions()
    idivs: IDivOptions = IDivOptions()
    iabss: IAbsOptions = IAbsOptions()
    ishft: IShftOptions = IShftOptions()
    ipopcnt: IPopcntOptions = IPopcntOptions()
    icmp: ICmpOptions = ICmpOptions()


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
        return self._built(FAddOperator.build, self.float_format, self._configured(self.options.fadd, "fadd"))

    @cached_property
    def fmul(self) -> FMulOperator:
        return self._built(
            FMulOperator.build, self.float_format, self._configured(self.options.fmul, "fmul"), self.wmultiplier
        )

    @cached_property
    def fdivsqrt(self) -> FDivsqrtOperator:
        return self._built(
            FDivsqrtOperator.build, self.float_format, self._configured(self.options.fdivsqrt, "fdivsqrt")
        )

    @cached_property
    def fmul_ilog2(self) -> FMulILog2Operator:
        return self._built(
            FMulILog2Operator.build,
            self.float_format,
            self.int_format,
            self._configured(self.options.fmul_ilog2, "fmul_ilog2"),
        )

    @cached_property
    def filog2(self) -> FILog2Operator:
        return self._built(
            FILog2Operator.build, self.float_format, self.int_format, self._configured(self.options.filog2, "filog2")
        )

    @cached_property
    def fcmp(self) -> FCmpOperator:
        return self._built(FCmpOperator.build, self.float_format, self._configured(self.options.fcmp, "fcmp"))

    @cached_property
    def frint(self) -> FRintOperator:
        return self._built(
            FRintOperator.build, self.float_format, self.int_format, self._configured(self.options.frint, "frint")
        )

    @cached_property
    def ffma(self) -> FFmaOperator:
        return self._built(
            FFmaOperator.build, self.float_format, self._configured(self.options.ffma, "ffma"), self.wmultiplier
        )

    @cached_property
    def fsort(self) -> FSortOperator:
        return self._built(FSortOperator.build, self.float_format, self._configured(self.options.fsort, "fsort"))

    @cached_property
    def fexp2(self) -> FExp2Operator:
        return self._built(
            FExp2Operator.build, self.float_format, self._configured(self.options.fexp2, "fexp2"), self.wmultiplier
        )

    @cached_property
    def flog2(self) -> FLog2Operator:
        return self._built(
            FLog2Operator.build, self.float_format, self._configured(self.options.flog2, "flog2"), self.wmultiplier
        )

    @cached_property
    def fcordic(self) -> FCordicOperator:
        return self._built(
            FCordicOperator.build,
            self.float_format,
            self._configured(self.options.fcordic, "fcordic"),
            self.wmultiplier,
        )

    @cached_property
    def ffromint(self) -> FFromIntOperator:
        return self._built(
            FFromIntOperator.build,
            self.float_format,
            self.int_format,
            self._configured(self.options.ffromint, "ffromint"),
        )

    @cached_property
    def iadds(self) -> IAddOperator:
        return self._built(IAddOperator.build, self.int_format, self.options.iadds)

    @cached_property
    def isubs(self) -> ISubOperator:
        return self._built(ISubOperator.build, self.int_format, self.options.isubs)

    @cached_property
    def imuls(self) -> IMulOperator:
        return self._built(IMulOperator.build, self.int_format, self.options.imuls)

    @cached_property
    def idivs(self) -> IDivOperator:
        return self._built(IDivOperator.build, self.int_format, self.options.idivs)

    @cached_property
    def iabss(self) -> IAbsOperator:
        return self._built(IAbsOperator.build, self.int_format, self.options.iabss)

    @cached_property
    def ishft(self) -> IShftOperator:
        return self._built(IShftOperator.build, self.int_format, self.options.ishft)

    @cached_property
    def ipopcnt(self) -> IPopcntOperator:
        return self._built(IPopcntOperator.build, self.int_format, self.options.ipopcnt)

    @cached_property
    def icmp(self) -> ICmpOperator:
        return self._built(ICmpOperator.build, self.int_format, self.options.icmp)

    def serves(self, kind: type[HardwareOperator]) -> bool:
        """Whether this machine is configured with operators of `kind`, answered without building one."""
        return getattr(self.options, kind.name) is not None

    @staticmethod
    def _configured[T](options: T | None, name: str) -> T:
        """A float operator is optional, and a kernel needing one left unconfigured is refused by name."""
        if options is None:
            raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
        return options

    def _built[O: HardwareOperator, **P](self, build: Callable[P, O], *args: P.args, **kwargs: P.kwargs) -> O:
        """
        Read off the ports rather than off a format argument: a conversion operator carries one format per side, and
        asking a format which family it belongs to can only confirm that it matches its own kind.
        """
        try:
            operator = build(*args, **kwargs)
        except zkf.UnsupportedFormat as ex:
            raise UnsupportedConstruct(f"a needed operator cannot be built for this machine's formats: {ex}") from ex
        assert all(
            (port.scalar_type.fmt == self.float_format if isinstance(port.scalar_type, FloatType) else True)
            and (port.scalar_type.fmt == self.int_format if isinstance(port.scalar_type, IntType) else True)
            for port in operator.operand_ports + operator.output_ports
        ), f"the configured {operator.name!r} is not built for the machine's formats of every family its ports name"
        _logger.info(
            "Operator %s: %d instance(s), %s",
            operator.name,
            operator.instances,
            ", ".join(
                ("" if mode.code is None else f"mode {mode.code} ")
                + f"latency {operator.latency(mode)} II {mode.initiation_interval}"
                for mode in operator.modes
            ),
        )
        return operator
