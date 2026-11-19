"""
The vocabulary every family shares: the port conditioners, hardware operators, the abstract primitive hierarchy, and
the two primitives that belong to no one family.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from typing import Any, ClassVar, Self, assert_never

import zkf
from amaranth.hdl import Shape, ShapeLike
from amaranth.lib import wiring

from .._value import FloatValue, IntValue, ScalarValue
from .._type import BoolType, FloatType, IntType, ScalarType
from .._util import Relation


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
        """On the sign bit alone, as the kernel conditions an operand: a zero keeps its conditioned sign too."""
        sign = (value.negative and not self.absolute) != self.negate
        return FloatValue(value.fmt, value.bits & ~(1 << value.fmt.sign_shift) | int(sign) << value.fmt.sign_shift)

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
    the way a float's sign flip is, so an integer port folds nothing into a sign field.
    """

    @property
    def is_identity(self) -> bool:
        return True

    def decorate(self, text: str) -> str:
        return text


@dataclass(frozen=True, slots=True)
class BoolInversion:
    """
    A hardware-side boolean conditioner: an optional inversion, the single-bit dual of FloatSignControl.
    Free in fabric (it folds into whatever LUT consumes or produces the bit); it is what lets one comparator output
    port serve two relations (e.g. `a<b` is the `lt` flag, `a>=b` the same flag inverted).
    """

    invert: bool = False

    def apply(self, value: bool) -> bool:
        return value ^ self.invert

    def decorate(self, text: str) -> str:
        return f"~{text}" if self.invert else text

    @property
    def is_identity(self) -> bool:
        return not self.invert


# The wide bank is shared across scalar families, so what a wide port may fold depends on the family it holds.
type WideConditioner = FloatSignControl | IntIdentity
type PortConditioner = WideConditioner | BoolInversion


def apply_conditioner(conditioner: PortConditioner, value: ScalarValue) -> ScalarValue:
    match conditioner:
        case FloatSignControl():
            assert isinstance(value, FloatValue)
            return conditioner.apply_value(value)
        case IntIdentity():
            assert isinstance(value, IntValue)
            return value
        case BoolInversion():
            assert isinstance(value, bool)
            return conditioner.apply(value)
        case _:
            assert_never(conditioner)


def identity_conditioner(scalar_type: ScalarType) -> PortConditioner:
    if isinstance(scalar_type, FloatType):
        return FloatSignControl()
    if isinstance(scalar_type, IntType):
        return IntIdentity()
    assert isinstance(scalar_type, BoolType), scalar_type
    return BoolInversion()


def has_sign_control(scalar_type: ScalarType) -> bool:
    """Whether a port of this type can carry a sign field; read off the conditioner so the two agree."""
    return isinstance(identity_conditioner(scalar_type), FloatSignControl)


@dataclass(frozen=True, slots=True)
class ScalarSignature:
    """Operand- and result-port types for a concrete primitive, which answers one result."""

    operand_types: tuple[ScalarType, ...]
    result_type: ScalarType

    @property
    def arity(self) -> int:
        return len(self.operand_types)


_HANDSHAKE_INPUTS = ("in_valid", "out_ready")

type ImmediateValues = tuple[tuple[str, int], ...]
"""Immediates by name, in name order: the key of a drive."""


def _immediate_values(immediates: Mapping[str, int]) -> ImmediateValues:
    return tuple(sorted(immediates.items()))


@dataclass(frozen=True, slots=True)
class Drive:
    """
    One way of activating an operator's block: the operator with a value on each of its immediates, and what the block
    an instance running one primitive of it alone renders says of it. A firing is one drive over one set of operands,
    and every primitive of a drive reads the same operand ports at the same timing; equal drives carry equal facts.
    """

    operator: HardwareOperator
    immediates: ImmediateValues
    timing: zkf.Timing
    operand_ports: tuple[str, ...]
    """The operand ports the drive reads, in operand order: those the block keeps once the immediates are frozen."""


def float_operand(value: ScalarValue) -> FloatValue:
    assert isinstance(value, FloatValue)
    return value


def int_operand(value: ScalarValue) -> int:
    assert isinstance(value, IntValue)
    return value.value


def agreed[T](primitives: Iterable[PooledPrimitive], immediate: str, kind: Callable[[int], T]) -> T | None:
    """The value every one of `primitives` gives `immediate`, as `kind`, or None where they disagree on it."""
    values = {primitive.immediates[immediate] for primitive in primitives}
    assert values
    value, *others = values
    return None if others else kind(value)


@dataclass(frozen=True, slots=True)
class HardwareOperator[B: zkf.Block = zkf.Block]:
    """
    One kind of physical streaming module as configured, and the resource-sharing key: equal operators time-share its
    instances. It owns the hardware -- the block, built for every drive, the instance budget and the typed ports the
    register file reads and writes -- while each primitive running on it is one result of one of its drives. An
    instance renders the block its own primitives need (`instance_block`), and a primitive's drive is read off the
    block an instance running it alone would render. An operator may have any number of immediates, of any shape.

    Each kind is a subclass that adds no fields, only the facts shared by every configuration of the kind and a typed
    `build` from the machine's formats and the kind's options; it stays undecorated, since a regenerated field-based
    hash would hash the block.
    """

    name: ClassVar[str]
    error_ports: ClassVar[tuple[str, ...]] = ()

    block: B
    instances: int
    operand_ports: Mapping[str, ScalarType]
    output_ports: Mapping[str, ScalarType]
    """The block's data ports by name, typed as the register file sees them; the operand ports in operand order."""
    immediate_ports: tuple[str, ...] = field(init=False, compare=False, repr=False)
    """The block's immediates by name, in signature order, read once."""

    def __hash__(self) -> int:
        # Consistent with equality, since equal operators are of one kind, and cheap in the allocator's hot loops.
        return hash(self.name)

    def __post_init__(self) -> None:
        assert self.name and self.instances >= 1, self.name
        assert self.output_ports, self.name
        data_ports = [*self.operand_ports.items(), *self.output_ports.items()]
        names = [*(name for name, _ in data_ports), *self.error_ports]
        assert len(set(names)) == len(names), self.name
        members = self.block.signature.members
        assert set(names) <= set(members), self.name
        assert all(Shape.cast(members[name].shape).width == ty.width for name, ty in data_ports), self.name
        assert all(ty.is_wide for ty in self.operand_ports.values()), "an operator reads only wide operands"
        inputs = tuple(
            name for name, member in members.items() if member.flow == wiring.In and name not in _HANDSHAKE_INPUTS
        )
        operands = set(self.operand_ports)
        assert operands <= set(inputs), self.name
        object.__setattr__(self, "immediate_ports", tuple(name for name in inputs if name not in operands))

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> B:
        """
        The block an instance renders for the primitives its firings run. An operator with immediates answers this
        itself: for every pattern of agreement among `primitives`, the block holds inside each immediate they all give
        one value and keeps each other one, and every port it keeps has the shape it has on the unspecialized block. It
        reads only what the primitives declare -- their kind, immediates and output port -- never their drive, which is
        derived from this very block, nor anything read off one; nor does it construct a primitive of its own operator,
        whose drive would ask it again.
        """
        assert primitives
        return self.block


def _carries(shape: ShapeLike, value: int) -> bool:
    """
    Whether an immediate port of `shape` holds `value` as itself: an integer within the port's range, and a value the
    enumeration gives back unchanged where the shape is one, its flag combinations included.
    """
    if not isinstance(value, int):
        return False
    cast = Shape.cast(shape)
    width: int = cast.width
    low, high = (-(1 << (width - 1)), 1 << (width - 1)) if cast.signed else (0, 1 << width)
    if not low <= value < high:
        return False
    if isinstance(shape, type) and issubclass(shape, Enum):
        try:
            return bool(shape(value).value == value)
        except ValueError:
            return False
    return True


@dataclass(frozen=True)
class Primitive(ABC):
    """What an operation computes: a signature with one result, bit-exact reference semantics, and a rendering."""

    # Float operand positions whose sign the primitive cannot observe: their conditioner is fixed to the identity, since
    # offering one would buy a second firing for one answer.
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset()

    def __post_init__(self) -> None:
        """
        The declarations assert what the primitive reads, which nothing can derive, so these rules close the ways they
        can be wrong.
        """
        signature, name = self.signature, type(self).__name__
        declined = self.unconditioned_operands
        # Declining a sideband the port never had would say nothing: only a port carrying a sign field can decline it.
        assert all(0 <= position < signature.arity for position in declined), name
        assert all(has_sign_control(signature.operand_types[position]) for position in declined), name

    @property
    @abstractmethod
    def latency(self) -> int: ...

    @abstractmethod
    def render(self, *operands: str) -> str: ...

    def render_output(self, inversion: BoolInversion | None, *operands: str) -> str:
        """Human-friendly form of the result, read through `inversion`, which is present exactly for a boolean one."""
        text = self.render(*operands)
        return text if inversion is None else inversion.decorate(text)

    @property
    @abstractmethod
    def signature(self) -> ScalarSignature: ...

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[ScalarValue, ...]:
        """Driven by the signature's per-port type, so a cross-family primitive needs no check of its own."""
        for index, (operand, ty) in enumerate(zip(operands, self.signature.operand_types, strict=True)):
            assert (
                (isinstance(ty, FloatType) and isinstance(operand, FloatValue) and operand.fmt == ty.fmt)
                or (isinstance(ty, IntType) and isinstance(operand, IntValue) and operand.fmt == ty.fmt)
                or (isinstance(ty, BoolType) and isinstance(operand, bool))
            ), f"{type(self).__name__} operand {index} must be {ty}, got {operand!r}"
        return operands

    @abstractmethod
    def evaluate(self, *operands: ScalarValue) -> ScalarValue:
        """Bit-exact reference semantics."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseOperatorOptions:
    """
    The knobs of one kind of operator. Every kind's options subclass this.
    """

    instances: int = 1
    """How many physical copies of this operator the machine may emit. A cap: only the copies the schedule uses."""

    def __post_init__(self) -> None:
        if self.instances < 1:
            raise ValueError(f"instances must be >= 1, got {self.instances}")


@dataclass(frozen=True)
class PooledPrimitive[B: zkf.Block = Any](Primitive, ABC):
    """
    One result of one drive of an operator: the primitive is the semantics, the operator the hardware, and everything
    physical -- ports and timing -- is read off the drive, never redeclared; its reference arithmetic calls the block's.
    Its operands are the drive's operand ports in order. Primitives of one drive tapping distinct output ports over the
    same operands are one firing. The scheduler pools and contends firings over the operator's instances.
    """

    operator: HardwareOperator[B]

    _commutative: ClassVar[bool] = False
    """Whether exchanging the two operands leaves the result in place, bit for bit."""

    def __post_init__(self) -> None:
        operator = self.operator
        # An error sideband is an observable output that "unchanged across every result port" does not cover, so a
        # primitive raising one may decline no operand's sign.
        assert not (operator.error_ports and self.unconditioned_operands), type(self).__name__
        # A value on every immediate of the block and on nothing else, each one its port holds as itself: an input a
        # later block adds fails here rather than going undriven, and a value past its port would encode as another.
        members = operator.block.signature.members
        assert set(self.immediates) == set(operator.immediate_ports), type(self).__name__
        assert all(_carries(members[name].shape, value) for name, value in self.immediates.items()), type(self).__name__
        assert self.output_name in operator.output_ports, type(self).__name__
        # A busy window ending no later than the step after the commit cannot outlive the block that issued it.
        timing = self.drive.timing
        assert 1 <= timing.initiation_interval <= timing.latency + 1, type(self).__name__
        # `err_pc` reserves 0 for no error, and only a combinational firing can commit on the entry's first step.
        assert not operator.error_ports or self.latency >= 1, type(self).__name__
        super().__post_init__()

    @property
    def immediates(self) -> Mapping[str, int]:
        """
        The value this primitive gives each block input that is not an operand port, by input name, such as the
        subtraction's `sub`; most blocks have none.
        """
        return {}

    @property
    def output_name(self) -> str:
        """The operator output port this primitive answers: the operator's only one, unless a subclass names another."""
        (name,) = self.operator.output_ports
        return name

    @property
    def mirrored(self) -> PooledPrimitive | None:
        """
        The primitive of the same drive answering this one's value once the two operands are exchanged -- itself where
        the operands commute -- or None where none does. Flipping a firing's operands to shrink the read multiplexers
        replaces each of its primitives by its mirror.
        """
        return self if self._commutative else None

    @cached_property
    def drive(self) -> Drive:
        """
        What the block an instance running this primitive alone renders says of it: its timing, whose initiation
        interval is the cycles until the instance accepts its next firing, of any primitive (1 = fully pipelined) --
        the per-firing sense of II, distinct from the module-level `Lir.min_initiation_interval`, the whole-transaction
        cost, which is this project's deliberate usage (see DESIGN.md, Direction) -- and the operand ports it keeps.
        Derived once, since ZKF computes timing and ports on every read and the allocator asks in its inner loop.
        """
        operator = self.operator
        block = operator.instance_block(frozenset({self}))
        timing = block.timing
        assert isinstance(timing, zkf.Timing), type(self).__name__
        members = block.signature.members
        assert not any(name in members for name in operator.immediate_ports), type(self).__name__
        assert self.output_name in members, type(self).__name__
        operand_ports = tuple(name for name in operator.operand_ports if name in members)
        return Drive(operator, _immediate_values(self.immediates), timing, operand_ports)

    @property
    def latency(self) -> int:
        return self.drive.timing.latency

    @property
    def signature(self) -> ScalarSignature:
        operator = self.operator
        operands = tuple(operator.operand_ports[name] for name in self.drive.operand_ports)
        return ScalarSignature(operands, operator.output_ports[self.output_name])

    @abstractmethod
    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> ScalarValue:
        """
        The bit-exact reference, computed by the operator's block, or by `block`, a specialization of it that an
        instance renders. Either takes the inputs of the block before it is specialized.
        """

    def _int_result(self, value: int) -> IntValue:
        result_type = self.signature.result_type
        assert isinstance(result_type, IntType)
        return IntValue.from_int(result_type.fmt, value)


@dataclass(frozen=True)
class InlinePrimitive(Primitive, ABC):
    """
    A pure combinational primitive folded into a register write: each firing is one PC-gated statement that reads its
    operands and writes its single result on one step. No module, no pooling, no contention.
    """

    @property
    def latency(self) -> int:
        # It reads and writes on one step; the register's write-then-read cost is the bank's READ_FIRST_EDGE in the
        # landing helper, not a pipeline stage.
        return 0

    def render(self, *operands: str) -> str:
        return self.verilog_expr(*operands).replace(" ", "")

    @abstractmethod
    def verilog_expr(self, *operand_nets: str) -> str: ...


class TotalOrdering(Enum):
    """
    Where the first operand stands against the second under a total order, each the one-hot flag a comparing drive
    answers on the output port of that name. Exchanging the operands exchanges `a > b` and `a < b` and keeps `a == b`.
    """

    GT = "a_gt_b"
    EQ = "a_eq_b"
    LT = "a_lt_b"

    @property
    def mirror(self) -> TotalOrdering:
        return {
            TotalOrdering.GT: TotalOrdering.LT,
            TotalOrdering.EQ: TotalOrdering.EQ,
            TotalOrdering.LT: TotalOrdering.GT,
        }[self]


_ORDERING_OF_RELATION: dict[Relation, tuple[TotalOrdering, BoolInversion]] = {
    Relation.GT: (TotalOrdering.GT, BoolInversion()),
    Relation.EQ: (TotalOrdering.EQ, BoolInversion()),
    Relation.LT: (TotalOrdering.LT, BoolInversion()),
    Relation.LE: (TotalOrdering.GT, BoolInversion(invert=True)),
    Relation.NE: (TotalOrdering.EQ, BoolInversion(invert=True)),
    Relation.GE: (TotalOrdering.LT, BoolInversion(invert=True)),
}
"""
Every family Holoso compares is totally ordered, so each relation is one ordering's flag or its complement, and the
relations over one operand pair tap one firing, but for a flag tapped under both inversions, which a firing can carry
only once.
"""

_RELATION_OF_ORDERING: dict[tuple[TotalOrdering, BoolInversion], Relation] = {
    tap: rel for rel, tap in _ORDERING_OF_RELATION.items()
}


@dataclass(frozen=True, slots=True)
class ComparatorPrimitive[B: zkf.Block](PooledPrimitive[B], ABC):
    """Whether the operands of a comparing drive stand in one ordering, beside whatever else that drive answers."""

    ordering: TotalOrdering

    @classmethod
    def of(cls, operator: HardwareOperator[B], relation: Relation) -> tuple[Self, BoolInversion]:
        """The primitive carrying `relation`, and the inversion reading it."""
        ordering, inversion = _ORDERING_OF_RELATION[relation]
        return cls(operator, ordering), inversion

    @property
    def output_name(self) -> str:
        return self.ordering.value

    @property
    def mirrored(self) -> PooledPrimitive:
        return replace(self, ordering=self.ordering.mirror)

    def _flag(self, a_gt_b: bool, a_eq_b: bool, a_lt_b: bool) -> bool:
        """This ordering's flag among the three a comparing block answers."""
        return {TotalOrdering.GT: a_gt_b, TotalOrdering.EQ: a_eq_b, TotalOrdering.LT: a_lt_b}[self.ordering]

    def render(self, *operands: str) -> str:
        return self.render_output(BoolInversion(), *operands)

    def render_output(self, inversion: BoolInversion | None, *operands: str) -> str:
        """The flag as the relation it implements, e.g. `a≥b`."""
        assert inversion is not None
        a, b = operands
        return f"{a}{_RELATION_OF_ORDERING[(self.ordering, inversion)].value}{b}"


@dataclass(frozen=True, slots=True)
class SelectPrimitive(InlinePrimitive):
    """
    A data mux `cond ? a : b` over same-typed values, folded into the destination register write as a ternary over
    the operand nets. Produced by HIR if-conversion and by selected MIR composite lowerings.
    Each operand is a dedicated direct (unlatched) register read; the cost is one mux per merged value, the same order
    as the per-arm phi-copy installs the branch would otherwise need.
    """

    scalar_type: ScalarType

    @property
    def signature(self) -> ScalarSignature:
        ty = self.scalar_type
        return ScalarSignature((BoolType(), ty, ty), ty)

    def render(self, *operands: str) -> str:
        cond, a, b = operands
        return f"{cond}?{a}:{b}"

    def verilog_expr(self, *operand_nets: str) -> str:
        cond, a, b = operand_nets
        return f"(({cond}) ? ({a}) : ({b}))"

    def evaluate(self, *operands: ScalarValue) -> ScalarValue:
        cond, a, b = self._validated_operands(operands)
        assert isinstance(cond, bool)
        return a if cond else b
