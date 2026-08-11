"""
The vocabulary every hardware operator family shares: the port conditioners, the abstract operator hierarchy, and the
two operators that belong to no one family.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, assert_never

from .._value import FloatValue, IntValue, ScalarValue
from .._type import BoolType, FloatType, IntType, ScalarType


@dataclass(frozen=True, slots=True)
class FloatSignControl:
    """A hardware-side floating-point sign conditioner: absolute value first, then optional negation."""

    negate: bool = False
    absolute: bool = False

    def then(self, outer: FloatSignControl) -> FloatSignControl:
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
    the way ``holoso_fsgnop`` is, so an integer port folds nothing into a sideband.
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

    def then(self, outer: BoolInversion) -> BoolInversion:
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


def apply_conditioner(conditioner: PortConditioner, value: ScalarValue) -> ScalarValue:
    """Apply a port's folded sideband: a sign control on a float value, an inversion on a boolean one."""
    match conditioner:
        case FloatSignControl():
            assert isinstance(value, FloatValue), "a float sign control applies only to a FloatValue"
            return conditioner.apply_value(value)
        case IntIdentity():
            assert isinstance(value, IntValue), "an integer identity applies only to an IntValue"
            return value
        case BoolInversion():
            assert isinstance(value, bool), "a boolean inversion applies only to a bool"
            return conditioner.apply(value)
        case _:
            assert_never(conditioner)


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


def has_sign_control(scalar_type: ScalarType) -> bool:
    """Whether a port carries a sign sideband, which only a float does; read off the conditioner so the two agree."""
    return isinstance(identity_conditioner(scalar_type), FloatSignControl)


@dataclass(frozen=True, slots=True)
class ScalarSignature:
    """
    Operand- and result-port types for a concrete hardware operator. An operator may produce several results (e.g. a
    comparator's three one-hot order flags, a sorter's min and max), one per output port, each independently typed.
    """

    operand_types: tuple[ScalarType, ...]
    result_types: tuple[ScalarType, ...]

    @property
    def arity(self) -> int:
        return len(self.operand_types)


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
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
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

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[ScalarValue, ...]:
        """Driven by the signature's per-port type, so a cross-family operator needs no check of its own."""
        signature = self.signature
        if len(operands) != signature.arity:
            raise ValueError(f"{self.mnemonic} expected {signature.arity} operands, got {len(operands)}")
        for index, (operand, ty) in enumerate(zip(operands, signature.operand_types, strict=True)):
            match ty, operand:
                case FloatType(), FloatValue() if operand.fmt == ty.fmt:
                    pass
                case IntType(), IntValue() if operand.fmt == ty.fmt:
                    pass
                case BoolType(), bool():
                    pass
                case _:
                    raise TypeError(f"{self.mnemonic} operand {index} must be {ty}, got {operand!r}")
        return operands

    @abstractmethod
    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[ScalarValue, ...]:
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
    operand_hdl_ports: ClassVar[list[str]]  # module port name per operand, aligned with the arity
    output_hdl_ports: ClassVar[list[str]]  # module port name per output, aligned with result_types

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"

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


@dataclass(frozen=True, slots=True)
class ComparatorOperator(PooledHardwareOperator, ABC):
    """
    A pooled comparator over one scalar family, producing the three mutually-exclusive one-hot order flags. Every
    family Holoso compares is totally ordered, so each relation is one flag or its complement; one instance therefore
    serves them all, and several relations over the same operands fuse into one firing.
    """

    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["a_gt_b", "a_eq_b", "a_lt_b"]

    # The single place the relation/flag mapping is defined; consumers go through ``tap_of``.
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
    # A total order makes compare antisymmetric, so cmp(b,a) transposes gt and lt: commutative under that exchange,
    # which lets port assignment orient the operands freely.
    swap_output_permutation: ClassVar[tuple[int, ...]] = (2, 1, 0)

    @property
    @abstractmethod
    def scalar_type(self) -> ScalarType:
        """The family being compared -- the one pooled place where operands and results belong to different ones."""

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (BoolType(), BoolType(), BoolType()))

    @classmethod
    def tap_of(cls, relation: Relation) -> tuple[int, BoolInversion]:
        return cls._TAP_OF_RELATION[relation]

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}⇔{b}"

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        """Recovers the tapped flag as the relation it implements, e.g. ``a≥b``."""
        assert isinstance(conditioner, BoolInversion)
        a, b = operands
        return f"{a}{self._RELATION_OF_TAP[(port, conditioner)].value}{b}"


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
        return f"(({cond}) ? ({a}) : ({b}))"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[ScalarValue, ...]:
        cond, a, b = self._validated_operands(operands)
        assert isinstance(cond, bool)
        return (a if cond else b,)
