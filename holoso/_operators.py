"""Hardware operator models and folded floating-point sign controls."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import blake2s
from typing import ClassVar

from ._value import (
    FloatValue,
    add_float_values,
    compare_float_values,
    div_float_values,
    mul_float_values,
    mul_ilog2_float_value,
)
from ._type import BoolType, FloatFormat, FloatType, ScalarSignature, ScalarType
from ._hir import RelationalOp


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
        """Compose two controls where ``self`` is applied first and ``outer`` after."""
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
    def encoded(self) -> int:
        return (1 if self.negate else 0) | (2 if self.absolute else 0)


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
    def encoded(self) -> int:
        return 1 if self.invert else 0


type PortConditioner = FloatSignControl | BoolInversion


def identity_conditioner(scalar_type: ScalarType) -> PortConditioner:
    """The identity conditioner of a port's type: a float port carries a sign control, a boolean port an inversion."""
    if isinstance(scalar_type, FloatType):
        return FloatSignControl()
    if isinstance(scalar_type, BoolType):
        return BoolInversion()
    raise TypeError(f"no conditioner is defined for ports of {scalar_type!r}")


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
    # Commutation symmetry: swapping the two operands permutes the output ports through this map (``new_port =
    # swap_output_permutation[old_port]``); ``None`` means non-commutative. Single-output commutative operators use
    # the identity ``(0,)``; the comparator's order flags transpose (``gt`` and ``lt`` exchange, ``eq`` is fixed).
    # The permutation must preserve each port's type, so a swapped firing's taps stay in their banks.
    swap_output_permutation: ClassVar[tuple[int, ...] | None] = None

    @property
    @abstractmethod
    def latency(self) -> int:
        """Exact cycle latency of this fully specified operator instance."""

    @property
    def initiation_interval(self) -> int:
        """
        Minimum cycles between successive issues on one physical instance (1 = fully pipelined) -- the per-operator
        sense of II. Distinct from the module-level ``Lir.initiation_interval``, the whole-transaction cost, which is
        this project's deliberate usage (see DESIGN.md, Direction).
        """
        return 1

    @abstractmethod
    def render(self, *operands: str) -> str:
        """Human-friendly expression for the report and trace comments."""

    @property
    def is_commutative(self) -> bool:
        """Whether operand swap (with the induced output-port permutation) preserves bit-exact semantics."""
        return self.swap_output_permutation is not None

    def render_output(self, port: int, conditioner: "PortConditioner", *operands: str) -> str:
        """
        Human-friendly form of one tapped output port. The default covers single-output operators only; a
        multi-output operator must override it (silently rendering every tap as the whole-operator expression would
        mislabel the report).
        """
        assert len(self.signature.result_types) == 1 and port == 0, f"{self.mnemonic} must override render_output"
        return conditioner.decorate(self.render(*operands))

    @property
    @abstractmethod
    def signature(self) -> ScalarSignature:
        """Concrete operand- and result-port types."""

    @property
    def arity(self) -> int:
        return self.signature.arity

    @abstractmethod
    def evaluate(self, *operands: "FloatValue | bool") -> tuple["FloatValue | bool", ...]:
        """Bit-exact reference semantics: one value per output port, aligned with ``signature.result_types``."""


@dataclass(frozen=True)
class PooledHardwareOperator(HardwareOperator, ABC):
    """
    An operator backed by a physical streaming module instance (in_valid/out_valid, per-float-port sign conditioners).
    The scheduler pools and contends equal operators over shared instances; every port is microcode-driven in the
    generated RTL (read-address lanes for operands, write-enable lanes per output port).
    """

    error_ports: ClassVar[list[str]] = []
    output_hdl_ports: ClassVar[list[str]] = ["y"]  # module port name per output, aligned with result_types

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"

    @property
    def instance_stem(self) -> str:
        """
        Verilog-safe physical instance stem, compactly identifying this operator family and its HDL params.
        Override this if the operator's hardware identity is not fully captured by its mnemonic and HDL params.
        """
        return _hashed_instance_stem(self.mnemonic, self.hdl_params())

    @abstractmethod
    def hdl_params(self) -> dict[str, int]:
        """Operator-specific ``#(.NAME(v))`` params; the backend prepends ``WEXP``/``WMAN``."""


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

    def render(self, *operands: str) -> str:
        """Defaults to the Verilog expression with the whitespace squeezed out; override where that reads poorly."""
        return self.verilog_expr(*operands).replace(" ", "")

    @abstractmethod
    def verilog_expr(self, *operand_nets: str) -> str:
        """The combinational RHS of this operator over the given operand net expressions."""


class ParameterizedHardwareOperator(ABC):
    """
    A family of hardware operators needing per-node parameters.
    It carries only config-time values; the concrete :class:`HardwareOperator` it produces owns the hardware metadata.
    """

    @abstractmethod
    def instantiate(self, *params: int) -> HardwareOperator: ...


@dataclass(frozen=True, slots=True)
class FloatHardwareOperator(PooledHardwareOperator, ABC):
    """A fully specified floating-point operator bound to one ZKF format."""

    fmt: FloatFormat

    @property
    def instance_stem(self) -> str:
        params = {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman}
        params.update(self.hdl_params())
        return _hashed_instance_stem(self.mnemonic, params)

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
    """A floating-point operator family bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FAddOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fadd"
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # signed sum: a+b == b+a bit-for-bit
    stage_input: int = 0
    stage_decode: int = 0
    stage_align: int = 0
    stage_normalize: int = 0  # close-cancellation normshift barriers, 0..2 (forwarded to _zkf_normshift.STAGE_SPLIT)
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_decode", "stage_align", "stage_pack", "stage_output"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")
        if self.stage_normalize not in (0, 1, 2):
            raise ValueError(f"stage_normalize must be 0, 1, or 2; got {self.stage_normalize!r}")

    @property
    def latency(self) -> int:
        return (
            4
            + self.stage_input
            + self.stage_decode
            + self.stage_align
            + self.stage_normalize
            + self.stage_pack
            + self.stage_output
        )

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue | bool) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (add_float_values(a, b),)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"

    def hdl_params(self) -> dict[str, int]:
        return {
            "STAGE_INPUT": self.stage_input,
            "STAGE_DECODE": self.stage_decode,
            "STAGE_ALIGN": self.stage_align,
            "STAGE_NORMALIZE": self.stage_normalize,
            "STAGE_PACK": self.stage_pack,
            "STAGE_OUTPUT": self.stage_output,
        }


@dataclass(frozen=True, slots=True)
class FMulOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fmul"
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # product: a*b == b*a bit-for-bit
    stage_input: int = 0
    stage_product: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_product", "stage_pack", "stage_output"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_product + self.stage_pack + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue | bool) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (mul_float_values(a, b),)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"

    def hdl_params(self) -> dict[str, int]:
        return {
            "STAGE_INPUT": self.stage_input,
            "STAGE_PRODUCT": self.stage_product,
            "STAGE_PACK": self.stage_pack,
            "STAGE_OUTPUT": self.stage_output,
        }


@dataclass(frozen=True, slots=True)
class FDivOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fdiv"
    error_ports: ClassVar[list[str]] = ["div0"]
    stage_input: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_pack", "stage_output"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    @property
    def latency(self) -> int:
        w = self.fmt.wman
        return 2 + self.stage_input + ((w + 2 + ((w + 2) % 2)) // 2) + self.stage_pack + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue | bool) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (div_float_values(a, b),)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": self.stage_input, "STAGE_PACK": self.stage_pack, "STAGE_OUTPUT": self.stage_output}


@dataclass(frozen=True, slots=True)
class FMulILog2Operator(FloatHardwareOperator):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete operator the family returns."""

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    k: int
    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        limit = (1 << self.fmt.wexp) - 2
        if self.k < -limit or self.k >= limit:
            raise ValueError(f"k must satisfy {-limit} <= k < {limit} for {self.fmt}; got {self.k!r}")
        for field in ("stage_input", "stage_decode"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_decode

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: FloatValue | bool) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        return (mul_ilog2_float_value(a, self.k),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"

    def hdl_params(self) -> dict[str, int]:
        return {"K": self.k, "STAGE_INPUT": self.stage_input, "STAGE_DECODE": self.stage_decode}


@dataclass(frozen=True, slots=True)
class FMulILog2OperatorFamily(FloatParameterizedHardwareOperator):
    """The ilog2 family: a factory whose stage knobs are baked into every concrete operator it instantiates."""

    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_decode"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    def instantiate(self, *params: int) -> FMulILog2Operator:
        (k,) = params
        return FMulILog2Operator(fmt=self.fmt, k=k, stage_input=self.stage_input, stage_decode=self.stage_decode)


@dataclass(frozen=True, slots=True)
class FCmpOperator(FloatHardwareOperator):
    """
    A floating-point comparator: a pooled streaming module producing the three mutually-exclusive one-hot order flags
    (a>b, a==b, a<b) with input sign conditioning. A comparison ``a <relation> b`` taps exactly one flag with an
    optional inversion (ZKF has no NaN, so the ordering is total and every relation is one flag or its complement);
    one instance therefore serves every relation, and several relations over the same operands fuse into one firing.
    """

    mnemonic: ClassVar[str] = "fcmp"
    output_hdl_ports: ClassVar[list[str]] = ["a_gt_b", "a_eq_b", "a_lt_b"]

    # RelationalOp -> (output port 0..2 = gt/eq/lt, inversion): the single place the relation/flag mapping is defined.
    # A relation maps onto exactly one port with an optional inversion (consumers go through `tap_of`):
    # gt, eq, lt directly; le = ~gt, ne = ~eq, ge = ~lt.
    _TAP_OF_RELATION: ClassVar[dict[RelationalOp, tuple[int, BoolInversion]]] = {
        RelationalOp.GT: (0, BoolInversion()),
        RelationalOp.EQ: (1, BoolInversion()),
        RelationalOp.LT: (2, BoolInversion()),
        RelationalOp.LE: (0, BoolInversion(invert=True)),
        RelationalOp.NE: (1, BoolInversion(invert=True)),
        RelationalOp.GE: (2, BoolInversion(invert=True)),
    }
    _RELATION_OF_TAP: ClassVar[dict[tuple[int, BoolInversion], RelationalOp]] = {
        tap: rel for rel, tap in _TAP_OF_RELATION.items()
    }
    _RELATION_SYMBOL: ClassVar[dict[RelationalOp, str]] = {
        RelationalOp.LT: "<",
        RelationalOp.LE: "≤",
        RelationalOp.GT: ">",
        RelationalOp.GE: "≥",
        RelationalOp.EQ: "=",
        RelationalOp.NE: "≠",
    }
    # The ZKF ordering is total and compare is antisymmetric, so cmp(b,a) is cmp(a,b) with gt and lt transposed
    # (eq fixed) -- the comparator is commutative under that flag exchange, which lets port assignment orient its
    # operands freely.
    swap_output_permutation: ClassVar[tuple[int, ...]] = (2, 1, 0)
    stage_input: int = 0

    def __post_init__(self) -> None:
        if self.stage_input not in (0, 1):
            raise ValueError(f"stage_input must be 0 or 1; got {self.stage_input!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_input

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty, ty), (BoolType(), BoolType(), BoolType()))

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"cmp({a},{b})"

    @classmethod
    def tap_of(cls, relation: RelationalOp) -> tuple[int, "BoolInversion"]:
        """The (output port, inversion) pair implementing a relation; every relation is one flag or its complement."""
        return cls._TAP_OF_RELATION[relation]

    def render_output(self, port: int, conditioner: "PortConditioner", *operands: str) -> str:
        """Human-friendly form of one tapped flag, recovered as the relation it implements (e.g. ``a≥b``)."""
        assert isinstance(conditioner, BoolInversion)
        a, b = operands
        return f"{a}{self._RELATION_SYMBOL[self._RELATION_OF_TAP[(port, conditioner)]]}{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": self.stage_input}

    def evaluate(self, *operands: FloatValue | bool) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands, 2)
        ordering = compare_float_values(a, b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class BoolLogicOperator(InlineHardwareOperator, ABC):
    """
    A boolean-logic operator (AND/OR): a plain ``& |`` gate folded into its boolean register's write. Never
    added to :class:`OpConfig` -- it has no module and no configuration.
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

    def evaluate(self, *operands: FloatValue | bool) -> tuple[bool, ...]:
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

    def evaluate(self, *operands: FloatValue | bool) -> tuple[bool, ...]:
        a, b = operands
        return (bool(a) or bool(b),)


@dataclass(frozen=True, slots=True)
class FloatToBoolOperator(InlineHardwareOperator):
    """
    A float->bool cast ``bool(x)``: true iff the operand is nonzero, i.e. its ZKF exponent field is nonzero (sign- and
    mantissa-agnostic). Folded into the boolean register write as a call to the shared ``holoso_ftobool`` function;
    never added to :class:`OpConfig`.
    """

    mnemonic: ClassVar[str] = "ftobool"
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

    def evaluate(self, *operands: FloatValue | bool) -> tuple[bool, ...]:
        (a,) = operands
        assert isinstance(a, FloatValue)
        return (a.exponent != 0,)


@dataclass(frozen=True, slots=True)
class SelectOperator(InlineHardwareOperator):
    """
    A data mux ``cond ? a : b`` over wide values, folded into the destination register write as a ternary over the
    operand nets. Produced exclusively by HIR if-conversion; never added to :class:`OpConfig`. Each operand is a
    dedicated direct (unlatched) register read -- an area/timing characteristic of inline operators; the cost is one
    mux per merged value, the same order as the per-arm phi-copy installs the branch would otherwise need.
    """

    mnemonic: ClassVar[str] = "select"
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((BoolType(), ty, ty), (ty,))

    def render(self, *operands: str) -> str:
        cond, a, b = operands
        return f"{cond}?{a}:{b}"

    def verilog_expr(self, *operand_nets: str) -> str:
        cond, a, b = operand_nets
        return f"({cond} ? {a} : {b})"

    def evaluate(self, *operands: "FloatValue | bool") -> tuple[FloatValue]:
        cond, a, b = operands
        assert isinstance(cond, bool) and isinstance(a, FloatValue) and isinstance(b, FloatValue)
        return (a if cond else b,)


@dataclass(frozen=True, slots=True)
class BoolSelectOperator(InlineHardwareOperator):
    """
    A boolean mux ``cond ? a : b`` over 1-bit values, the dual of :class:`SelectOperator`, folded into the destination
    boolean register write as a ternary over the operand nets. Format-agnostic (no ``fmt``); produced exclusively by
    HIR if-conversion of a boolean-phi diamond; never added to :class:`OpConfig`.
    """

    mnemonic: ClassVar[str] = "bool_select"

    @property
    def signature(self) -> ScalarSignature:
        ty = BoolType()
        return ScalarSignature((ty, ty, ty), (ty,))

    def render(self, *operands: str) -> str:
        cond, a, b = operands
        return f"{cond}?{a}:{b}"

    def verilog_expr(self, *operand_nets: str) -> str:
        cond, a, b = operand_nets
        return f"({cond} ? {a} : {b})"

    def evaluate(self, *operands: FloatValue | bool) -> tuple[bool, ...]:
        cond, a, b = operands
        return (bool(a) if bool(cond) else bool(b),)


@dataclass(frozen=True, slots=True)
class BoolToFloatOperator(InlineHardwareOperator):
    """
    A bool->float cast ``float(cond)``: ZKF ``1.0`` when true, ``+0.0`` when false. Folded into the wide register
    write as a call to the shared ``holoso_ffrombool`` function; it reads a boolean register and writes a wide
    register, the one operator that crosses from the boolean bank into the wide bank. Never added to
    :class:`OpConfig`.
    """

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

    def evaluate(self, *operands: FloatValue | bool) -> tuple[FloatValue, ...]:
        (a,) = operands
        return (FloatValue.from_float(self.fmt, 1.0 if a else 0.0),)


@dataclass(frozen=True)
class OpConfig:
    """
    The hardware operator configuration threaded into synthesis. Constructed by the user before synthesis.
    Each field fixes one operator's format and parameters.
    """

    fadd: FAddOperator
    fmul: FMulOperator
    fdiv: FDivOperator
    fmul_ilog2: FMulILog2OperatorFamily
    fcmp: FCmpOperator

    @property
    def float_format(self) -> FloatFormat:
        formats = {self.fadd.fmt, self.fmul.fmt, self.fdiv.fmt, self.fmul_ilog2.fmt, self.fcmp.fmt}
        if len(formats) != 1:
            ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(f"all floating-point operators must use the same format; got {ordered}")
        return self.fadd.fmt
