"""Hardware operator models and folded port conditioners."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from hashlib import blake2s
from typing import Any, ClassVar

import zkf

from ._errors import UnsupportedConstruct
from ._value import FloatValue
from ._type import BoolType, FloatFormat, FloatType, IntType, ScalarSignature, ScalarType


def _instance_stem_text(text: str) -> str:
    return re.sub(r"[^0-9a-z_]+", "_", text.lower()).strip("_") or "x"


def _instance_stem_hash(params: dict[str, int]) -> str:
    payload = "\n".join(f"{name}={value}" for name, value in sorted(params.items())).encode("ascii")
    return blake2s(payload, digest_size=4).hexdigest()


def _hashed_instance_stem(mnemonic: str, params: dict[str, int]) -> str:
    return f"{_instance_stem_text(mnemonic)}_{_instance_stem_hash(params)}"


@dataclass(frozen=True, slots=True)
class FloatSignControl:
    """A hardware-side floating-point sign conditioner: absolute value first, then optional negation."""

    negate: bool = False
    absolute: bool = False

    def then(self, outer: "FloatSignControl") -> "FloatSignControl":
        if outer.absolute:
            return FloatSignControl(negate=outer.negate, absolute=True)
        return FloatSignControl(negate=self.negate ^ outer.negate, absolute=self.absolute)

    def apply_value(self, value: FloatValue) -> FloatValue:
        return value.apply_sign(negate=self.negate, absolute=self.absolute)

    def decorate(self, text: str) -> str:
        if self.absolute:
            text = f"|{text}|"
        if self.negate:
            text = f"-{text}"
        return text

    @property
    def is_identity(self) -> bool:
        return not self.negate and not self.absolute

    @property
    def encoded(self) -> int:
        return (1 if self.negate else 0) | (2 if self.absolute else 0)


@dataclass(frozen=True, slots=True)
class IntIdentity:
    """
    The conditioner of an integer port, which is always the identity: two's-complement negation is not free in fabric
    the way ``holoso_fsgnop`` is, so an integer port folds nothing into a sideband. Deliberately without ``then``
    (nothing composes), ``encoded`` (there is no sign opcode to ride the wrapper), and any value application (there is
    no integer runtime value yet) -- a wide port's conditioner is not a sign algebra.
    """

    @property
    def is_identity(self) -> bool:
        return True

    def decorate(self, text: str) -> str:
        return text


@dataclass(frozen=True, slots=True)
class BoolInversion:
    """
    A hardware-side boolean conditioner: an optional inversion, the single-bit dual of :class:`FloatSignControl`.
    Free in fabric (it folds into whatever LUT consumes or produces the bit); it is what lets one comparator output
    port serve two relations (e.g. ``a<b`` is the ``lt`` flag, ``a>=b`` the same flag inverted).
    """

    invert: bool = False

    def then(self, outer: "BoolInversion") -> "BoolInversion":
        return BoolInversion(invert=self.invert ^ outer.invert)

    def apply(self, value: bool) -> bool:
        return value ^ self.invert

    def decorate(self, text: str) -> str:
        return f"~{text}" if self.invert else text

    @property
    def is_identity(self) -> bool:
        return not self.invert

    @property
    def encoded(self) -> int:
        return 1 if self.invert else 0


# The wide bank is shared across scalar families, so what a wide port may fold depends on the family it holds.
type WideConditioner = FloatSignControl | IntIdentity
type PortConditioner = WideConditioner | BoolInversion


class Relation(Enum):
    """
    The relations a comparator serves, shared by every comparator family. Naming them here rather than in the
    semantic IR keeps this layer below the HIR, whose operators carry the relation in their own identity; MIR maps
    the two. The value is the symbol used when rendering a tapped flag back as the relation it implements.
    """

    LT = "<"
    LE = "≤"
    GT = ">"
    GE = "≥"
    EQ = "="
    NE = "≠"


@dataclass(frozen=True, slots=True)
class ImmediateField:
    """
    A per-firing immediate input port: a small microcode-driven constant on a named wrapper port, the data-carrying
    dual of the sign sidebands. Lets one shared instance serve several per-firing modes, not one instance per mode.
    """

    name: str  # wrapper port name
    width: int  # bit width


def identity_conditioner(scalar_type: ScalarType) -> PortConditioner:
    if isinstance(scalar_type, FloatType):
        return FloatSignControl()
    if isinstance(scalar_type, IntType):
        return IntIdentity()
    if isinstance(scalar_type, BoolType):
        return BoolInversion()
    raise TypeError(f"no conditioner is defined for ports of {scalar_type!r}")


def value_class(scalar_type: ScalarType) -> type[FloatValue] | type[bool]:
    """
    The runtime value class a port of this scalar type carries, letting a type-polymorphic operator assert its
    payloads without naming a concrete class.
    """
    if isinstance(scalar_type, FloatType):
        return FloatValue
    if isinstance(scalar_type, BoolType):
        return bool
    raise TypeError(f"no runtime value class is defined for {scalar_type!r}")


@dataclass(frozen=True)
class HardwareOperator(ABC):
    """
    A fully specified hardware operator configuration.
    Frozen-dataclass equality makes an instance the resource-sharing key: equal operators time-share one physical
    module. Each concrete operator owns its timing, reference semantics, notation, and port types -- possibly several
    typed output ports (a comparator's three one-hot order flags, a sorter's min and max).
    Commutative operators allow port assignment orient each use's operands to shrink the per-port read muxes.
    The two structural families are :class:`PooledHardwareOperator` (a physical streaming module) and
    :class:`InlineHardwareOperator` (a pure expression folded into a register write).
    """

    mnemonic: ClassVar[str]

    # Per-firing immediate input ports (empty for most operators; ``fround`` declares its 2-bit ``round_mode``). The
    # value rides the MIR operation, not the operator identity, so one shared instance serves every mode.
    immediate_ports: ClassVar[list[ImmediateField]] = []

    # Commutation symmetry: swapping the two operands permutes the output ports through this map (``new_port =
    # swap_output_permutation[old_port]``); ``None`` means non-commutative. Single-output commutative operators use
    # the identity ``(0,)``; the comparator's order flags transpose (``gt`` and ``lt`` exchange, ``eq`` is fixed).
    # The permutation must preserve each port's type, so a swapped firing's taps stay in their banks.
    swap_output_permutation: ClassVar[tuple[int, ...] | None] = None

    @property
    @abstractmethod
    def latency(self) -> int: ...

    @property
    def initiation_interval(self) -> int:
        """
        Minimum cycles between successive issues on one physical instance (1 = fully pipelined) -- the per-operator
        sense of II. Distinct from the module-level ``Lir.initiation_interval``, the whole-transaction cost, which is
        this project's deliberate usage (see DESIGN.md, Direction).
        """
        return 1

    @abstractmethod
    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str: ...

    @property
    def is_commutative(self) -> bool:
        return self.swap_output_permutation is not None

    def render_output(
        self, port: int, conditioner: "PortConditioner", *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        """
        Human-friendly form of one tapped output port. The default covers single-output operators only; a
        multi-output operator must override it (silently rendering every tap as the whole-operator expression would
        mislabel the report). ``immediates`` is forwarded so a mode-bearing operator renders the firing's actual mode.
        """
        assert len(self.signature.result_types) == 1 and port == 0, f"{self.mnemonic} must override render_output"
        return conditioner.decorate(self.render(*operands, immediates=immediates))

    @property
    @abstractmethod
    def signature(self) -> ScalarSignature: ...

    @property
    def arity(self) -> int:
        return self.signature.arity

    @abstractmethod
    def evaluate(
        self, *operands: "FloatValue | bool", immediates: tuple[int, ...] = ()
    ) -> tuple["FloatValue | bool", ...]:
        """
        Bit-exact reference semantics: one value per output port, aligned with ``signature.result_types``.
        ``immediates`` carries the per-firing immediate values (empty for most operators).
        """


@dataclass(frozen=True)
class PooledHardwareOperator(HardwareOperator, ABC):
    """
    An operator backed by a physical streaming module instance (in_valid/out_valid, per-float-port sign conditioners).
    The scheduler pools and contends equal operators over shared instances; every port is microcode-driven in the
    generated RTL (a per-operand read opcode selects each operand, a per-register write opcode installs each result).
    """

    error_ports: ClassVar[list[str]] = []
    output_hdl_ports: ClassVar[list[str]] = ["y"]  # module port name per output, aligned with result_types

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"

    @property
    def instance_stem(self) -> str:
        """
        Verilog-safe physical instance stem, compactly identifying this operator family and its RTL parameters.
        Override this if the operator's hardware identity is not fully captured by its mnemonic and parameters.
        """
        return _hashed_instance_stem(self.mnemonic, self.params)

    @property
    @abstractmethod
    def params(self) -> dict[str, int]:
        """The complete RTL ``#(.NAME(value))`` parameter set (WEXP/WMAN, LATENCY, and every operator knob)."""


@dataclass(frozen=True)
class InlineHardwareOperator(HardwareOperator, ABC):
    """
    A pure combinational operator folded into a register write: each firing is one PC-gated statement that reads its
    operands and writes its single result on one step. No module, no pooling, no contention.
    """

    @property
    def latency(self) -> int:
        # It reads and writes on one step; the register's write-then-read cost is the bank's READ_FIRST_EDGE in the
        # landing helper, not a pipeline stage.
        return 0

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        return self.verilog_expr(*operands).replace(" ", "")

    @abstractmethod
    def verilog_expr(self, *operand_nets: str) -> str: ...


class ParameterizedHardwareOperator(ABC):
    """
    A family of hardware operators needing per-node parameters.
    It carries only config-time values; the concrete :class:`HardwareOperator` it produces owns the hardware metadata.
    """

    @abstractmethod
    def instantiate(self, *params: int) -> HardwareOperator: ...


@dataclass(frozen=True, slots=True)
class FloatHardwareOperator(PooledHardwareOperator, ABC):
    fmt: FloatFormat

    _model: zkf.OperatorModel = field(init=False, compare=False, repr=False)

    @property
    def latency(self) -> int:
        return self._model.latency

    @property
    def initiation_interval(self) -> int:
        return self._model.initiation_interval

    @property
    def params(self) -> dict[str, int]:
        return self._model.params

    def float_signature(self, arity: int) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty,) * arity, (ty,))

    def _validated_operands(self, operands: tuple["FloatValue | bool", ...], arity: int) -> tuple[FloatValue, ...]:
        if len(operands) != arity:
            raise ValueError(f"{self.mnemonic} expected {arity} operands, got {len(operands)}")
        validated: list[FloatValue] = []
        for index, operand in enumerate(operands):
            if not isinstance(operand, FloatValue):
                raise TypeError(f"{self.mnemonic} operand {index} must be FloatValue, got {type(operand).__name__}")
            if operand.fmt != self.fmt:
                raise ValueError(f"{self.mnemonic} operand {index} has {operand.fmt}, expected {self.fmt}")
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class FloatParameterizedHardwareOperator(ParameterizedHardwareOperator, ABC):
    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FAddOperator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0  # takes any count of input register stages (extra stages relieve routing congestion)
        stage_decode: int = 0
        stage_align: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fadd"
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # signed sum: a+b == b+a bit-for-bit
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.AddModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
            stage_align=self.opt.stage_align,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (a + b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class FMulOperator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_product: int = 0  # splitting the product is rarely useful unless wman exceeds the DSP slice input width
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fmul"
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # product: a*b == b*a bit-for-bit
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.MulModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (a * b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class FDivOperator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fdiv"
    error_ports: ClassVar[list[str]] = ["div0"]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.DivModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            stage_input=self.opt.stage_input,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (a / b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}/{b}"


@dataclass(frozen=True, slots=True)
class FMulILog2Operator(FloatHardwareOperator):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete operator the family returns."""

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_decode: int = 0

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    k: int
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.MulIlog2ConstModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            k=self.k,
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        return (a.scale_pow2(self.k),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"


@dataclass(frozen=True, slots=True)
class FMulILog2OperatorFamily(FloatParameterizedHardwareOperator):
    """The ilog2 family: a factory whose stage knobs are baked into every concrete operator it instantiates."""

    opt: FMulILog2Operator.Options

    def instantiate(self, *params: int) -> FMulILog2Operator:
        (k,) = params
        return FMulILog2Operator(fmt=self.fmt, k=k, opt=self.opt)


@dataclass(frozen=True, slots=True)
class FCmpOperator(FloatHardwareOperator):
    """
    A floating-point comparator: a pooled streaming module producing the three mutually-exclusive one-hot order flags
    (a>b, a==b, a<b) with input sign conditioning. A comparison ``a <relation> b`` taps exactly one flag with an
    optional inversion (ZKF has no NaN, so the ordering is total and every relation is one flag or its complement);
    one instance therefore serves every relation, and several relations over the same operands fuse into one firing.
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0

    mnemonic: ClassVar[str] = "fcmp"
    output_hdl_ports: ClassVar[list[str]] = ["a_gt_b", "a_eq_b", "a_lt_b"]

    # Relation -> (output port 0..2 = gt/eq/lt, inversion): the single place the relation/flag mapping is defined.
    # A relation maps onto exactly one port with an optional inversion (consumers go through `tap_of`):
    # gt, eq, lt directly; le = ~gt, ne = ~eq, ge = ~lt.
    _TAP_OF_RELATION: ClassVar[dict[Relation, tuple[int, BoolInversion]]] = {
        Relation.GT: (0, BoolInversion()),
        Relation.EQ: (1, BoolInversion()),
        Relation.LT: (2, BoolInversion()),
        Relation.LE: (0, BoolInversion(invert=True)),
        Relation.NE: (1, BoolInversion(invert=True)),
        Relation.GE: (2, BoolInversion(invert=True)),
    }
    _RELATION_OF_TAP: ClassVar[dict[tuple[int, BoolInversion], Relation]] = {
        tap: rel for rel, tap in _TAP_OF_RELATION.items()
    }
    # The ZKF ordering is total and compare is antisymmetric, so cmp(b,a) is cmp(a,b) with gt and lt transposed
    # (eq fixed) -- the comparator is commutative under that flag exchange, which lets port assignment orient its
    # operands freely.
    swap_output_permutation: ClassVar[tuple[int, ...]] = (2, 1, 0)
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.CmpModel(zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman), stage_input=self.opt.stage_input)
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty, ty), (BoolType(), BoolType(), BoolType()))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"cmp({a},{b})"

    @classmethod
    def tap_of(cls, relation: Relation) -> tuple[int, BoolInversion]:
        """The (output port, inversion) pair implementing a relation; every relation is one flag or its complement."""
        return cls._TAP_OF_RELATION[relation]

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        """Human-friendly form of one tapped flag, recovered as the relation it implements (e.g. ``a≥b``)."""
        assert isinstance(conditioner, BoolInversion)
        a, b = operands
        return f"{a}{self._RELATION_OF_TAP[(port, conditioner)].value}{b}"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands, 2)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class FRoundOperator(FloatHardwareOperator):
    """
    Round a float to an integral-valued float. One pooled instance serves all four modes (nearest-even, floor, ceil,
    trunc) via the 2-bit ``round_mode`` immediate, as one comparator serves every relation.
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        """The zkf core is combinational, hence the nonzero default: a pooled operator needs latency >= 1."""

        stage_input: int = 1
        stage_decode: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fround"
    immediate_ports: ClassVar[list[ImmediateField]] = [ImmediateField("round_mode", 2)]
    opt: Options

    class Mode(IntEnum):
        """Matches the mode encoding in holoso_fround"""

        ROUND = 0
        FLOOR = 1
        CEIL = 2
        TRUNC = 3

    _EVAL: ClassVar[dict[Mode, Callable[[FloatValue], FloatValue]]] = {
        Mode.ROUND: FloatValue.round,
        Mode.FLOOR: FloatValue.floor,
        Mode.CEIL: FloatValue.ceil,
        Mode.TRUNC: FloatValue.trunc,
    }

    def __post_init__(self) -> None:
        model = zkf.RoundModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)
        if self.latency < 1:
            raise ValueError("fround needs at least one register stage (a pooled operator must have latency >= 1)")

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        (mode,) = immediates
        return (self._EVAL[self.Mode(mode)](a),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        (mode,) = immediates
        return f"{self.Mode(mode).name.lower()}({a})"


@dataclass(frozen=True, slots=True)
class FFmaOperator(FloatHardwareOperator):
    """
    Fused multiply-add ``a*b + c``, single-rounded (full-width product rounded once with ``c``). Arity 3; serves the
    explicit ``math.fma`` and the implicit ``a*b+c`` fusion. Not commutative under operand reversal (gives ``c*b+a``).
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_product: int = 0
        stage_decode: int = 0
        stage_align: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "ffma"
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.FmaModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_decode=self.opt.stage_decode,
            stage_align=self.opt.stage_align,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(3)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b, c = self._validated_operands(operands, 3)
        return (FloatValue.fma(a, b, c),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b, c = operands
        return f"{a}×{b}+{c}"


@dataclass(frozen=True, slots=True)
class FSortOperator(FloatHardwareOperator):
    """
    A 2-element float sorter emitting the ascending ``(min, max)`` of its operands, with input and per-output sign
    conditioning. ``min(a,b)`` taps port 0 and ``max(a,b)`` port 1; one instance serves both, and a min and a max over
    one operand pair fuse into a single firing (as the comparator's relations do).
    NOT commutative: min/max preserve the selected operand's exact bits, and the sorter breaks a tie toward the second
    operand, so swapping operands can flip the sign of a zero result (a -0 conditioned from a zero magnitude).
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0

    mnemonic: ClassVar[str] = "fsort"
    output_hdl_ports: ClassVar[list[str]] = ["min", "max"]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.SortModel(zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman), stage_input=self.opt.stage_input)
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty, ty), (ty, ty))

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return FloatValue.sort(a, b)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{self.mnemonic}({a},{b})"

    def render_output(
        self, port: int, conditioner: "PortConditioner", *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, FloatSignControl)
        a, b = operands
        return conditioner.decorate(f"{self.output_hdl_ports[port]}({a}, {b})")


@dataclass(frozen=True, slots=True)
class FExp2Operator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_reduce: int = 0
        stage_product: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fexp2"
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.Exp2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_reduce=self.opt.stage_reduce,
            stage_product=self.opt.stage_product,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        return (a.exp2(),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"2^{a}"


@dataclass(frozen=True, slots=True)
class FLog2Operator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_decode: int = 0
        stage_product: int = 0
        stage_product_final: int = 0
        stage_normalize: int = 0
        stage_normalize_output: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "flog2"
    error_ports: ClassVar[list[str]] = ["domain_error", "pole"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.Log2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
            stage_product=self.opt.stage_product,
            stage_product_final=self.opt.stage_product_final,
            stage_normalize=self.opt.stage_normalize,
            stage_normalize_output=self.opt.stage_normalize_output,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        return (a.log2(),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"log2({a})"


@dataclass(frozen=True, slots=True)
class _FCordicOperator(FloatHardwareOperator, ABC):
    """
    These are NOT throughput-1: the core holds one transaction in flight and re-accepts one cycle after retiring.
    """

    def render_output(
        self, port: int, conditioner: "PortConditioner", *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, FloatSignControl)
        return conditioner.decorate(f"{self.output_hdl_ports[port]}({', '.join(operands)})")


@dataclass(frozen=True, slots=True)
class FSincosOperator(_FCordicOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        unroll100: int = 100
        stage_input: int = 0
        stage_product: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fsincos"
    output_hdl_ports: ClassVar[list[str]] = ["sin", "cos"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.SincosModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            unroll100=self.opt.unroll100,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty,), (ty, ty))

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        return a.sincos()

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"sincos({a})"


@dataclass(frozen=True, slots=True)
class FAtan2Operator(_FCordicOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        """A nearby hypot over the same operands folds into the magnitude port for free."""

        unroll100: int = 100
        stage_input: int = 0
        stage_product: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fatan2"
    output_hdl_ports: ClassVar[list[str]] = ["theta", "mag"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.Atan2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            unroll100=self.opt.unroll100,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty, ty), (ty, ty))

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        y, x = self._validated_operands(operands, 2)
        return FloatValue.atan2(y, x)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        y, x = operands
        return f"atan2({y},{x})"


@dataclass(frozen=True, slots=True)
class BoolLogicOperator(InlineHardwareOperator, ABC):
    """
    A boolean-logic operator (AND/OR/XOR): a plain ``& | ^`` gate folded into its boolean register's write.
    Never added to :class:`OpConfig` -- it has no module and no configuration.
    """


@dataclass(frozen=True, slots=True)
class BoolAndOperator(BoolLogicOperator):
    mnemonic: ClassVar[str] = "band"

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(), BoolType()), (BoolType(),))

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"{a} & {b}"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = operands
        return (bool(a) and bool(b),)


@dataclass(frozen=True, slots=True)
class BoolOrOperator(BoolLogicOperator):
    mnemonic: ClassVar[str] = "bor"

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(), BoolType()), (BoolType(),))

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"{a} | {b}"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = operands
        return (bool(a) or bool(b),)


@dataclass(frozen=True, slots=True)
class BoolXorOperator(BoolLogicOperator):
    mnemonic: ClassVar[str] = "bxor"

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(), BoolType()), (BoolType(),))

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"{a} ^ {b}"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = operands
        return (bool(a) != bool(b),)


@dataclass(frozen=True, slots=True)
class FloatClassificationOperator(InlineHardwareOperator, ABC):
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (BoolType(),))


@dataclass(frozen=True, slots=True)
class FloatIsFiniteOperator(FloatClassificationOperator):
    mnemonic: ClassVar[str] = "fisfinite"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"isfinite({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisfinite({a})"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = operands
        assert isinstance(a, FloatValue)
        return (a.fmt.is_finite(a.bits),)


@dataclass(frozen=True, slots=True)
class FloatIsPosInfOperator(FloatClassificationOperator):
    mnemonic: ClassVar[str] = "fisposinf"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"isposinf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisposinf({a})"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = operands
        assert isinstance(a, FloatValue)
        return (not a.fmt.is_finite(a.bits) and not a.negative,)


@dataclass(frozen=True, slots=True)
class FloatIsNegInfOperator(FloatClassificationOperator):
    mnemonic: ClassVar[str] = "fisneginf"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"isneginf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisneginf({a})"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = operands
        assert isinstance(a, FloatValue)
        return (not a.fmt.is_finite(a.bits) and a.negative,)


@dataclass(frozen=True, slots=True)
class FloatToBoolOperator(InlineHardwareOperator):
    mnemonic: ClassVar[str] = "ftobool"
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (BoolType(),))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ftobool({a})"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = operands
        assert isinstance(a, FloatValue)
        return (a.exponent != 0,)


@dataclass(frozen=True, slots=True)
class SelectOperator(InlineHardwareOperator):
    """
    A data mux ``cond ? a : b`` over same-typed values, folded into the destination register write as a ternary over
    the operand nets. Produced by HIR if-conversion and by selected MIR composite lowerings.
    Each operand is a dedicated direct (unlatched) register read -- an area/timing characteristic of inline operators;
    the cost is one mux per merged value, the same order as the per-arm phi-copy installs the branch would otherwise
    need.
    """

    mnemonic: ClassVar[str] = "select"
    scalar_type: ScalarType

    @property
    def signature(self) -> ScalarSignature:
        ty = self.scalar_type
        return ScalarSignature((BoolType(), ty, ty), (ty,))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        cond, a, b = operands
        return f"{cond}?{a}:{b}"

    def verilog_expr(self, *operand_nets: str) -> str:
        cond, a, b = operand_nets
        return f"({cond} ? {a} : {b})"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue | bool, ...]:
        cond, a, b = operands
        assert isinstance(cond, bool)
        assert isinstance(a, value_class(self.scalar_type)) and isinstance(b, value_class(self.scalar_type))
        return (a if cond else b,)


@dataclass(frozen=True, slots=True)
class BoolToFloatOperator(InlineHardwareOperator):
    mnemonic: ClassVar[str] = "ffrombool"
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), (FloatType(self.fmt),))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"float({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ffrombool({a})"

    def evaluate(self, *operands: FloatValue | bool, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = operands
        return (FloatValue.from_float(self.fmt, 1.0 if a else 0.0),)


class IMulOperator:
    """Not implemented yet; this placeholder carries only its knobs."""

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_product: int = 0


@dataclass(frozen=True)
class OpConfig:
    """
    This class only contains operators that are configurable.
    Operators that don't have tunable parameters can be constructed ad-hoc instead.
    """

    fadd: FAddOperator | None
    fmul: FMulOperator | None
    fdiv: FDivOperator | None
    fmul_ilog2: FMulILog2OperatorFamily | None
    fcmp: FCmpOperator | None
    fround: FRoundOperator | None
    ffma: FFmaOperator | None
    fsort: FSortOperator | None
    fexp2: FExp2Operator | None
    flog2: FLog2Operator | None
    fsincos: FSincosOperator | None
    fatan2: FAtan2Operator | None

    def require(self, name: str) -> Any:
        """The named operator, or a refusal naming what needs configuring."""
        operator = getattr(self, name)
        if operator is None:
            raise UnsupportedConstruct(f"the kernel needs the {name!r} operator, which is not configured")
        return operator
