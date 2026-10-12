"""
The float operators and primitives. Every operator's timing and parameters come from the external ZKF library, and
every primitive's reference arithmetic from its value model. An operator whose format parameterizes it is a float one
whatever else it touches, so the casts across the type boundary live here too.
"""

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import ClassVar, Self

import zkf

from .._util import Relation
from .._value import FloatValue, IntValue, RoundMode, ScalarValue
from .._type import BoolType, FloatFormat, FloatType, IntFormat, IntType
from ._common import (
    BaseOperatorOptions,
    BoolInversion,
    ComparatorPrimitive,
    HardwareOperator,
    InlinePrimitive,
    ModePort,
    OperatorMode,
    OperatorPort,
    PooledPrimitive,
    ScalarSignature,
)

_ROUND_LABEL: dict[RoundMode, str] = {
    RoundMode.NEAREST_EVEN: "round",
    RoundMode.FLOOR: "floor",
    RoundMode.CEIL: "ceil",
    RoundMode.TRUNC: "trunc",
}
"""Rendered into the report and the ROM comments, so it is pinned here rather than taken from zkf's member names."""


def _knobs(options: BaseOperatorOptions) -> dict[str, int]:
    """Options name their knobs as the ZKF model does; the instance budget is the machine's, not the core's."""
    return {f.name: getattr(options, f.name) for f in fields(options) if f.name != "instances"}


def _floats(fmt: FloatFormat, *names: str) -> tuple[OperatorPort, ...]:
    return tuple(OperatorPort(name, FloatType(fmt)) for name in names)


def _timing(name: str, model: zkf.OperatorModel) -> zkf.Timing:
    """
    The model's single timing, which a mode port selecting only the arithmetic leaves undivided. ZKF reads a zero
    `LATENCY` as "do not check", so a combinational core is refused rather than elaborated with its latency unverified.
    """
    timing = model.timing
    assert isinstance(timing, zkf.Timing), model.module
    assert timing.latency == model.params["LATENCY"], model.module
    if timing.latency < 1:
        raise ValueError(f"{name} needs at least one register stage (an operator must have latency >= 1)")
    return timing


def _zkf_operator[O: HardwareOperator](
    cls: type[O],
    model: zkf.OperatorModel,
    options: BaseOperatorOptions,
    operands: tuple[OperatorPort, ...],
    outputs: tuple[OperatorPort, ...],
) -> O:
    """An operator with a single timing, taken from its model; reading the parameters loads a format's tables."""
    interval = _timing(cls.name, model).initiation_interval
    return cls.of_one_mode(model.params, options.instances, operands, outputs, interval)


def _format(fmt: FloatFormat) -> zkf.ZkfFormat:
    return zkf.ZkfFormat(fmt.wexp, fmt.wman)


@dataclass(frozen=True, slots=True)
class FloatPrimitive(PooledPrimitive, ABC):
    """Float operands throughout, which is what makes the narrowed operand validator below sound."""

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[FloatValue, ...]:
        validated: list[FloatValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, FloatValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class FAddOptions(BaseOperatorOptions):
    stage_input: int = 0  # takes any count of input register stages (extra stages relieve routing congestion)
    stage_decode: int = 0
    stage_align: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0


class FAddOperator(HardwareOperator):
    __slots__ = ()
    name = "fadd"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FAddOptions) -> Self:
        model = zkf.AddModel(_format(fmt), **_knobs(options))
        return _zkf_operator(cls, model, options, _floats(fmt, "a", "b"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FAddPrimitive(FloatPrimitive):
    operator: FAddOperator
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # signed sum: a+b == b+a bit-for-bit

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return (a + b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class FMulOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_product: int = 0  # splitting the product is rarely useful unless wman exceeds the DSP slice input width
    stage_pack: int = 0
    stage_output: int = 0


class FMulOperator(HardwareOperator):
    __slots__ = ()
    name = "fmul"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FMulOptions, wmultiplier: int) -> Self:
        model = zkf.MulModel(_format(fmt), wmultiplier=wmultiplier, **_knobs(options))
        return _zkf_operator(cls, model, options, _floats(fmt, "a", "b"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FMulPrimitive(FloatPrimitive):
    operator: FMulOperator
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # product: a*b == b*a bit-for-bit

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return (a * b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class FDivsqrtOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_decode: int = 0  # takes effect at an even wman only
    stage_pack: int = 0
    stage_output: int = 0


class DivsqrtMode(IntEnum):
    """The value driven on `op_sqrt`, which is also the `MODE` elaborating the core for that operation alone."""

    DIVISION = 0
    SQRT = 1

    @property
    def operand_count(self) -> int:
        return 2 if self is DivsqrtMode.DIVISION else 1


class FDivsqrtOperator(HardwareOperator):
    """
    One digit-recurrence pipeline serving the quotient `a / b` and the square root of `a`, chosen per firing on
    `op_sqrt`, at one latency. `error` reports a zero divisor in the first and a negative operand in the second.
    """

    __slots__ = ()
    name = "fdivsqrt"
    mode_port = ModePort("op_sqrt", 1)
    error_ports = ("error",)

    @classmethod
    def build(cls, fmt: FloatFormat, options: FDivsqrtOptions) -> Self:
        model = zkf.DivsqrtModel(_format(fmt), **_knobs(options))
        timing = _timing(cls.name, model)
        modes: list[OperatorMode] = []
        single_mode_params: dict[OperatorMode, dict[str, int]] = {}
        for mode in DivsqrtMode:
            operator_mode = OperatorMode(int(mode), "LATENCY", timing.initiation_interval, mode.operand_count, (0,))
            modes.append(operator_mode)
            fixed = zkf.DivsqrtModel(_format(fmt), mode=int(mode), **_knobs(options))
            assert fixed.timing == timing, "a fixed mode must keep its timing, which the schedule was built on"
            single_mode_params[operator_mode] = fixed.params
        ports = _floats(fmt, "a", "b"), _floats(fmt, "y")
        return cls(model.params, options.instances, *ports, tuple(modes), single_mode_params)

    def mode_of_divsqrt(self, mode: DivsqrtMode) -> OperatorMode:
        return self.mode_of(int(mode))


@dataclass(frozen=True, slots=True)
class FDivPrimitive(FloatPrimitive):
    operator: FDivsqrtOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_divsqrt(DivsqrtMode.DIVISION)

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return (a / b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"


@dataclass(frozen=True, slots=True)
class FSqrtPrimitive(FloatPrimitive):
    """Correctly-rounded square root; a negative operand yields -inf and raises `error`."""

    operator: FDivsqrtOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_divsqrt(DivsqrtMode.SQRT)

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.sqrt(),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"√{a}"


@dataclass(frozen=True, slots=True)
class FILog2Options(BaseOperatorOptions):
    stage_input: int = 0


class FILog2Operator(HardwareOperator):
    __slots__ = ()
    name = "filog2"
    operands_without_sideband = frozenset({0})

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FILog2Options) -> Self:
        model = zkf.Ilog2Model(_format(fmt), wint=ifmt.width, **_knobs(options))
        outputs = (OperatorPort("y", IntType(ifmt)),)
        return _zkf_operator(cls, model, options, _floats(fmt, "a"), outputs)


@dataclass(frozen=True, slots=True)
class FILog2Primitive(PooledPrimitive):
    """
    The extraction half of the exponent pair, `FMulILog2Primitive` being the scaling half: the two compose with no
    arithmetic between them, sharing the binade convention `FloatValue.ilog2` states. Sign-invariant: the operand's
    conditioner reaches no logic at all, sign conditioning rewriting only the bit the exponent field does not read.
    """

    operator: FILog2Operator
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset({0})

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        (result,) = self.signature.result_types
        assert isinstance(result, IntType)
        return (IntValue.from_int(result.fmt, a.ilog2()),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"ilog2({a})"


@dataclass(frozen=True, slots=True)
class FMulILog2Options(BaseOperatorOptions):
    stage_input: int = 0
    stage_decode: int = 0


class FMulILog2Operator(HardwareOperator):
    __slots__ = ()
    name = "fmul_ilog2"

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FMulILog2Options) -> Self:
        model = zkf.MulIlog2Model(_format(fmt), wk=ifmt.width, **_knobs(options))
        # The wrapper sizes the exponent port by the machine's integer format, so it spells the core's WK as WINT.
        params = {("WINT" if name == "WK" else name): value for name, value in model.params.items()}
        operands = (OperatorPort("a", FloatType(fmt)), OperatorPort("k", IntType(ifmt)))
        outputs = _floats(fmt, "y")
        interval = _timing(cls.name, model).initiation_interval
        return cls.of_one_mode(params, options.instances, operands, outputs, interval)


@dataclass(frozen=True, slots=True)
class FMulILog2Primitive(PooledPrimitive):
    """Exact scaling by a power of two: `a * 2**k`; every `k` is legal."""

    operator: FMulILog2Operator

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        a, k = self._validated_operands(operands)
        assert isinstance(a, FloatValue) and isinstance(k, IntValue)
        return (a.scale_pow2(k.value),)

    def render(self, *operands: str) -> str:
        a, k = operands
        return f"{a}×2^{k}"


@dataclass(frozen=True, slots=True)
class FCmpOptions(BaseOperatorOptions):
    """
    The comparator runs every float relation, min and max, one firing per cycle, so a kernel issuing several of them at
    once shortens by raising `instances`. The zkf core is combinational, hence the nonzero default: an operator needs
    latency >= 1.
    """

    stage_input: int = 0
    stage_output: int = 1


class FCmpOperator(HardwareOperator):
    """
    The comparator, which orders its operands on the flags and answers them sorted beside. A comparison reads the
    flags alone; a sort reads the sorted pair and the flags, so a comparison of the operands a min or max already
    orders taps that firing. Firings of both contend for the one comparator.
    """

    __slots__ = ()
    name = "fcmp"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FCmpOptions) -> Self:
        model = zkf.CmpModel(_format(fmt), **_knobs(options))
        interval = _timing(cls.name, model).initiation_interval
        flag = BoolType()
        flags = (OperatorPort("a_gt_b", flag), OperatorPort("a_eq_b", flag), OperatorPort("a_lt_b", flag))
        comparison = OperatorMode(None, "LATENCY", interval, 2, (0, 1, 2))
        sorting = OperatorMode(None, "LATENCY", interval, 2, (3, 4, 0, 1, 2))
        outputs = (*flags, *_floats(fmt, "min", "max"))
        return cls(model.params, options.instances, _floats(fmt, "a", "b"), outputs, (comparison, sorting))

    @property
    def comparison(self) -> OperatorMode:
        return self.modes[0]

    @property
    def sorting(self) -> OperatorMode:
        return self.modes[1]


@dataclass(frozen=True, slots=True)
class FCmpPrimitive(FloatPrimitive, ComparatorPrimitive):
    """ZKF has no NaN, so the ordering is total."""

    operator: FCmpOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.comparison

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class FSortPrimitive(FloatPrimitive):
    """
    The ascending `(min, max)` of the sign-conditioned operands, `min(a,b)` result 0 and `max(a,b)` result 1, so a min
    and a max over one operand pair fuse into a single firing. The order flags of the operands leave beside them, so a
    comparison of the same two operands taps this firing instead of taking one of its own.
    NOT commutative: min/max preserve the selected operand's exact bits, and on a tie the comparator answers the second
    operand as the min and the first as the max, so swapping operands can flip the sign of a zero result (a -0
    conditioned from a zero magnitude).
    """

    operator: FCmpOperator
    output_labels: ClassVar[tuple[str, ...]] = ("min", "max")

    @property
    def mode(self) -> OperatorMode:
        return self.operator.sorting

    def flag_of(self, relation: Relation) -> tuple[int, BoolInversion]:
        """The result carrying `relation`, and the inversion reading it: the comparison's tap, found by its port."""
        tap, inversion = FCmpPrimitive.tap_of(relation)
        return self.mode.outputs.index(self.operator.comparison.outputs[tap]), inversion

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue | bool, ...]:
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return *FloatValue.sort(a, b), ordering > 0, ordering == 0, ordering < 0

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"sort({a},{b})"

    def render_output(self, result: int, inversion: BoolInversion | None, *operands: str) -> str:
        if inversion is None:
            return super().render_output(result, inversion, *operands)
        tap = self.operator.comparison.outputs.index(self.mode.outputs[result])
        return FCmpPrimitive(self.operator).render_output(tap, inversion, *operands)


@dataclass(frozen=True, slots=True)
class FRintOptions(BaseOperatorOptions):
    """The zkf core is combinational, hence the nonzero default: an operator needs latency >= 1."""

    stage_input: int = 0
    stage_shift: int = 1
    stage_round: int = 0
    stage_output: int = 0


class FRintOperator(HardwareOperator):
    """
    Rounds a float to an integer and answers it twice at once, as a float of the same format and as a signed integer.
    The rounding is one mode, selected on the `round_mode` port by its own encoding.
    """

    __slots__ = ()
    name = "frint"
    mode_port = ModePort("round_mode", 2)

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FRintOptions) -> Self:
        model = zkf.RintModel(_format(fmt), wint=ifmt.width, **_knobs(options))
        # Every rounding runs at one timing: the mode selects the arithmetic, not the pipeline.
        interval = _timing(cls.name, model).initiation_interval
        modes = tuple(OperatorMode(int(mode), "LATENCY", interval, 1, (0, 1)) for mode in RoundMode)
        outputs = (OperatorPort("y_float", FloatType(fmt)), OperatorPort("y_int", IntType(ifmt)))
        return cls(model.params, options.instances, _floats(fmt, "a"), outputs, modes)

    def mode_of_rounding(self, rounding: RoundMode) -> OperatorMode:
        return self.mode_of(int(rounding))


@dataclass(frozen=True, slots=True)
class FRintPrimitive(FloatPrimitive):
    """
    One rounding of a float (nearest-even, floor, ceil or trunc), answered as an integral-valued float (result 0) and
    as a signed integer saturating at the rails, an infinity reaching one of them (result 1). A float use and an
    integer use of one rounding of one operand therefore fuse into a single firing.
    """

    operator: FRintOperator
    rounding: RoundMode

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_rounding(self.rounding)

    _EVAL: ClassVar[dict[RoundMode, Callable[[FloatValue], FloatValue]]] = {
        RoundMode.NEAREST_EVEN: FloatValue.round,
        RoundMode.FLOOR: FloatValue.floor,
        RoundMode.CEIL: FloatValue.ceil,
        RoundMode.TRUNC: FloatValue.trunc,
    }
    _EVAL_INT: ClassVar[dict[RoundMode, Callable[[FloatValue, IntFormat], IntValue]]] = {
        RoundMode.NEAREST_EVEN: FloatValue.round_int,
        RoundMode.FLOOR: FloatValue.floor_int,
        RoundMode.CEIL: FloatValue.ceil_int,
        RoundMode.TRUNC: FloatValue.trunc_int,
    }

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, IntValue]:
        (a,) = self._validated_operands(operands)
        integer = self.signature.result_types[1]
        assert isinstance(integer, IntType)
        return self._EVAL[self.rounding](a), self._EVAL_INT[self.rounding](a, integer.fmt)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{_ROUND_LABEL[self.rounding]}({a})"

    def render_output(self, result: int, inversion: BoolInversion | None, *operands: str) -> str:
        assert inversion is None
        text = self.render(*operands)
        return f"i{text}" if result == 1 else text


@dataclass(frozen=True, slots=True)
class FFmaOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_product: int = 0
    stage_decode: int = 0
    stage_align: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0


class FFmaOperator(HardwareOperator):
    __slots__ = ()
    name = "ffma"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FFmaOptions, wmultiplier: int) -> Self:
        model = zkf.FmaModel(_format(fmt), wmultiplier=wmultiplier, **_knobs(options))
        return _zkf_operator(cls, model, options, _floats(fmt, "a", "b", "c"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FFmaPrimitive(FloatPrimitive):
    """
    Fused multiply-add `a*b + c`, single-rounded (full-width product rounded once with `c`). Not commutative under
    operand reversal (gives `c*b+a`).
    """

    operator: FFmaOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        a, b, c = self._validated_operands(operands)
        return (FloatValue.fma(a, b, c),)

    def render(self, *operands: str) -> str:
        a, b, c = operands
        return f"{a}×{b}+{c}"


@dataclass(frozen=True, slots=True)
class FExp2Options(BaseOperatorOptions):
    """
    The four product knobs stage the parallel products of the table kernel, whose latency is that of its longest
    branch: a stage on a shorter one costs no cycle. `zkf_exp2.v` says which product each stages.
    """

    stage_input: int = 0
    stage_reduce: int = 0
    stage_product: int = 0
    stage_product_square: int = 0
    stage_product_odd: int = 0  # takes effect from a 27-bit significand up
    stage_product_chain: int = 0
    stage_pack: int = 0
    stage_output: int = 0


class FExp2Operator(HardwareOperator):
    __slots__ = ()
    name = "fexp2"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FExp2Options, wmultiplier: int) -> Self:
        model = zkf.Exp2Model(_format(fmt), wmultiplier=wmultiplier, **_knobs(options))
        return _zkf_operator(cls, model, options, _floats(fmt, "a"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FExp2Primitive(FloatPrimitive):
    operator: FExp2Operator

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.exp2(),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"2^{a}"


@dataclass(frozen=True, slots=True)
class FLog2Options(BaseOperatorOptions):
    """The product knobs are those of `FExp2Options`; `stage_product_final` stages the multiply after the kernel."""

    stage_input: int = 0
    stage_product: int = 0
    stage_product_square: int = 0
    stage_product_odd: int = 0  # takes effect from a 27-bit significand up
    stage_product_chain: int = 0
    stage_align: int = 0
    stage_product_final: int = 0
    stage_pack: int = 0
    stage_output: int = 0


class FLog2Operator(HardwareOperator):
    __slots__ = ()
    name = "flog2"
    error_ports = ("domain_error", "pole")

    @classmethod
    def build(cls, fmt: FloatFormat, options: FLog2Options, wmultiplier: int) -> Self:
        model = zkf.Log2Model(_format(fmt), wmultiplier=wmultiplier, **_knobs(options))
        ports = _floats(fmt, "a"), _floats(fmt, "y")
        return _zkf_operator(cls, model, options, *ports)


@dataclass(frozen=True, slots=True)
class FLog2Primitive(FloatPrimitive):
    operator: FLog2Operator

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.log2(),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"log2({a})"


@dataclass(frozen=True, slots=True)
class FCordicOptions(BaseOperatorOptions):
    """A nearby hypot over the operands of an atan2 folds into the vectoring magnitude for free."""

    unroll100: int = 100
    stage_input: int = 0
    stage_product: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0


class CordicMode(IntEnum):
    """
    The value driven on `vectoring`, which is also how ZKF keys each mode's timing; the member name spells the mode's
    latency parameter.
    """

    ROTATION = 0
    VECTORING = 1

    @property
    def operand_count(self) -> int:
        """Rotation reads the phase alone, vectoring `(y, x)`."""
        return 1 if self is CordicMode.ROTATION else 2


class FCordicOperator(HardwareOperator):
    """
    The CORDIC core, its mode chosen per firing on `vectoring`: rotation reads a phase in turns and yields its sine and
    cosine, vectoring reads `(y, x)` and yields their angle in turns and their magnitude. NOT throughput-1: the core
    holds one transaction in flight and re-accepts one cycle after retiring, as each mode's ZKF timing states.
    """

    __slots__ = ()
    name = "fcordic"
    mode_port = ModePort("vectoring", 1)

    @classmethod
    def build(cls, fmt: FloatFormat, options: FCordicOptions, wmultiplier: int) -> Self:
        knobs = _knobs(options)
        model = zkf.CordicModel(_format(fmt), wmultiplier=wmultiplier, **knobs)
        timing = model.timing
        assert not isinstance(timing, zkf.Timing) and set(timing) == set(CordicMode)
        # The wrapper carries the knobs the options offer; one the core has beside them stays at the core's default.
        unoffered = {f.name.upper() for f in fields(model)} - {name.upper() for name in knobs} - {"WMULTIPLIER", "MODE"}
        modes: list[OperatorMode] = []
        single_mode_params: dict[OperatorMode, dict[str, int]] = {}
        for mode in CordicMode:
            latency_param = f"LATENCY_{mode.name}"
            assert timing[mode].latency == model.params[latency_param]
            interval = timing[mode].initiation_interval
            operator_mode = OperatorMode(int(mode), latency_param, interval, mode.operand_count, (0, 1))
            modes.append(operator_mode)
            fixed = zkf.CordicModel(_format(fmt), wmultiplier=wmultiplier, mode=int(mode), **knobs)
            assert fixed.timing == timing[mode], "a fixed mode must keep its timing, which the schedule was built on"
            single_mode_params[operator_mode] = {k: v for k, v in fixed.params.items() if k not in unoffered}
        params = {k: v for k, v in model.params.items() if k not in unoffered}
        ports = _floats(fmt, "a", "b"), _floats(fmt, "r0", "r1")
        return cls(params, options.instances, *ports, tuple(modes), single_mode_params)

    def mode_of_cordic(self, mode: CordicMode) -> OperatorMode:
        return self.mode_of(int(mode))


@dataclass(frozen=True, slots=True)
class FSincosPrimitive(FloatPrimitive):
    operator: FCordicOperator
    output_labels: ClassVar[tuple[str, ...]] = ("sin", "cos")

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_cordic(CordicMode.ROTATION)

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return a.sincos()

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"sincos({a})"


@dataclass(frozen=True, slots=True)
class FAtan2Primitive(FloatPrimitive):
    operator: FCordicOperator
    output_labels: ClassVar[tuple[str, ...]] = ("theta", "mag")

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_cordic(CordicMode.VECTORING)

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        y, x = self._validated_operands(operands)
        return FloatValue.atan2(y, x)

    def render(self, *operands: str) -> str:
        y, x = operands
        return f"atan2({y},{x})"


@dataclass(frozen=True, slots=True)
class FloatClassificationPrimitive(InlinePrimitive, ABC):
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (BoolType(),))


@dataclass(frozen=True, slots=True)
class FloatIsFinitePrimitive(FloatClassificationPrimitive):
    mnemonic: ClassVar[str] = "fisfinite"
    # The exponent field alone decides finiteness; the directional classifiers below read the sign and so declare
    # nothing, which is why this sits here rather than on the shared base.
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset({0})

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"isfinite({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisfinite({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (a.fmt.is_finite(a.bits),)


@dataclass(frozen=True, slots=True)
class FloatIsPosInfPrimitive(FloatClassificationPrimitive):
    mnemonic: ClassVar[str] = "fisposinf"

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"isposinf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisposinf({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (not a.fmt.is_finite(a.bits) and not a.negative,)


@dataclass(frozen=True, slots=True)
class FloatIsNegInfPrimitive(FloatClassificationPrimitive):
    mnemonic: ClassVar[str] = "fisneginf"

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"isneginf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisneginf({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (not a.fmt.is_finite(a.bits) and a.negative,)


@dataclass(frozen=True, slots=True)
class FloatToBoolPrimitive(InlinePrimitive):
    mnemonic: ClassVar[str] = "ftobool"
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset({0})  # a zero test reads the exponent alone
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (BoolType(),))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ftobool({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (a.exponent != 0,)


@dataclass(frozen=True, slots=True)
class BoolToFloatPrimitive(InlinePrimitive):
    mnemonic: ClassVar[str] = "ffrombool"
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), (FloatType(self.fmt),))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"float({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ffrombool({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, bool)
        return (FloatValue.from_float(self.fmt, 1.0 if a else 0.0),)


@dataclass(frozen=True, slots=True)
class FFromIntOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0


class FFromIntOperator(HardwareOperator):
    __slots__ = ()
    name = "ffromint"

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FFromIntOptions) -> Self:
        model = zkf.FromIntModel(_format(fmt), wint=ifmt.width, **_knobs(options))
        return _zkf_operator(cls, model, options, (OperatorPort("a", IntType(ifmt)),), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FFromIntPrimitive(PooledPrimitive):
    """Signed integer to float, nearest with ties to even; a magnitude past the finite range becomes an infinity."""

    operator: FFromIntOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, IntValue)
        (result,) = self.signature.result_types
        assert isinstance(result, FloatType)
        return (a.to_float(result.fmt),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"float({a})"
