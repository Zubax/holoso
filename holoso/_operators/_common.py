"""
The vocabulary every family shares: the port conditioners, hardware operators and their modes, the abstract primitive
hierarchy, and the two primitives that belong to no one family.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import ClassVar, Self, assert_never

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
    the way `holoso_fsgnop` is, so an integer port folds nothing into a sideband.
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


@dataclass(frozen=True, slots=True)
class ModePort:
    """The wrapper input that selects each firing's mode, driven from the microcode like a sign sideband."""

    name: str
    width: int


def identity_conditioner(scalar_type: ScalarType) -> PortConditioner:
    if isinstance(scalar_type, FloatType):
        return FloatSignControl()
    if isinstance(scalar_type, IntType):
        return IntIdentity()
    if isinstance(scalar_type, BoolType):
        return BoolInversion()
    raise TypeError(f"no conditioner is defined for ports of {scalar_type!r}")


def has_sign_control(scalar_type: ScalarType) -> bool:
    """Whether a port of this type can carry a sign sideband; read off the conditioner so the two agree."""
    return isinstance(identity_conditioner(scalar_type), FloatSignControl)


@dataclass(frozen=True, slots=True)
class ScalarSignature:
    """
    Operand- and result-port types for a concrete primitive. A primitive may produce several results (e.g. a
    comparator's three one-hot order flags, a sorter's min and max), one per output port, each independently typed.
    """

    operand_types: tuple[ScalarType, ...]
    result_types: tuple[ScalarType, ...]

    @property
    def arity(self) -> int:
        return len(self.operand_types)


@dataclass(frozen=True, slots=True)
class OperatorPort:
    """
    One port of an operator's RTL wrapper, by HDL name, typed as the register file sees it: a wrapper may present a
    narrower physical port, which the connection zero-extends.
    """

    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class OperatorMode:
    """
    One mode of an operator, bound to what it decides: the code it drives on the operator's mode port (none on an
    operator without one), the RTL parameter carrying its latency, its initiation interval, how many leading operand
    ports it reads, and which output ports carry its results, in result order. Every observable output of the mode,
    error ports included, is independent of the operand ports past that count, so their read and sign fields may be
    left don't-care. Modes sharing a code are one operation of the module read through different output ports.
    """

    code: int | None
    latency_param: str
    initiation_interval: int
    operand_count: int
    outputs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HardwareOperator:
    """
    One kind of physical streaming module as configured, and the resource-sharing key: equal operators time-share one
    module. It is the single home of every physical fact; the primitives that run on it hold only semantics. A float
    operand port carries a sign sideband unless listed in `operands_without_sideband`.

    `single_mode_params` holds, for each mode the module can also be elaborated for alone, the parameters doing so: the
    module's parameter names less the other modes' latency parameters, the mode's own latency unchanged. An operator
    offering them asserts that the mode keeps its whole timing alone, so an instance whose firings all drive that
    mode's code can shed the rest without the schedule noticing.

    A firing's busy window is its own mode's initiation interval. Requiring `L_a - L_b < II_a` for every ordered pair
    of modes that share an output port makes the earliest next firing on that port commit strictly after its
    predecessor, so the busy window stays the only per-instance constraint; modes on disjoint ports interleave freely.
    The error ports are one sideband every mode drives, so they tie all modes together.

    Each kind is a subclass that adds no fields, only the facts shared by every configuration of the kind and a typed
    `build` from the machine's formats and the kind's options; it stays undecorated, since a regenerated field-based
    hash would fail on `params`.
    """

    name: ClassVar[str]
    mode_port: ClassVar[ModePort | None] = None
    error_ports: ClassVar[tuple[str, ...]] = ()
    operands_without_sideband: ClassVar[frozenset[int]] = frozenset()

    params: Mapping[str, int]  # in instantiation order
    instances: int
    operand_ports: tuple[OperatorPort, ...]
    output_ports: tuple[OperatorPort, ...]
    modes: tuple[OperatorMode, ...]
    single_mode_params: Mapping[OperatorMode, Mapping[str, int]] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Consistent with equality, since equal operators are of one kind, and cheap in the allocator's hot loops.
        return hash(self.name)

    @classmethod
    def of_one_mode(
        cls,
        params: Mapping[str, int],
        instances: int,
        operand_ports: tuple[OperatorPort, ...],
        output_ports: tuple[OperatorPort, ...],
        initiation_interval: int,
    ) -> Self:
        """An operator with no mode port, whose one mode reads every operand port at the `LATENCY` parameter."""
        mode = OperatorMode(None, "LATENCY", initiation_interval, len(operand_ports), tuple(range(len(output_ports))))
        return cls(params, instances, operand_ports, output_ports, (mode,))

    def __post_init__(self) -> None:
        assert self.name and self.instances >= 1, self.name
        assert self.output_ports and (self.mode_port is None or self.mode_port.width >= 1), self.name
        ports = [port.name for port in (*self.operand_ports, *self.output_ports)]
        names = [*ports, *([] if self.mode_port is None else [self.mode_port.name]), *self.error_ports]
        assert len(set(names)) == len(names), self.name
        assert all(port.scalar_type.is_wide for port in self.operand_ports), "an operator reads only wide operands"
        assert all(
            0 <= position < len(self.operand_ports) and has_sign_control(self.operand_ports[position].scalar_type)
            for position in self.operands_without_sideband
        ), self.name
        assert self.modes and len(set(self.modes)) == len(self.modes), self.name
        named = {mode.latency_param for mode in self.modes}
        assert named == {name for name in self.params if name.startswith("LATENCY")}, (self.name, named)
        for mode in self.modes:
            if self.mode_port is None:
                assert mode.code is None, self.name
            else:
                assert mode.code is not None and 0 <= mode.code < 1 << self.mode_port.width, self.name
            assert 1 <= mode.operand_count <= len(self.operand_ports), self.name
            assert mode.outputs and len(set(mode.outputs)) == len(mode.outputs), self.name
            assert all(0 <= port < len(self.output_ports) for port in mode.outputs), self.name
            # A result commits at issue + latency; latency >= 1 keeps its write opcode off the held accept-dwell word,
            # and a busy window ending no later than the step after it cannot outlive the block that issued it.
            assert self.latency(mode) >= 1 and 1 <= mode.initiation_interval <= self.latency(mode) + 1, self.name
        for a in self.modes:
            shared = [b for b in self.modes if self.error_ports or set(a.outputs) & set(b.outputs)]
            assert all(self.latency(a) - self.latency(b) < a.initiation_interval for b in shared), self.name
        for a, b in combinations(self.modes, 2):
            if a.code == b.code:
                assert (a.latency_param, a.initiation_interval, a.operand_count) == (
                    b.latency_param,
                    b.initiation_interval,
                    b.operand_count,
                ), self.name
                assert self.single_mode_params.get(a) == self.single_mode_params.get(b), self.name
        assert not self.single_mode_params or self.mode_port is not None, self.name
        for mode, narrowed in self.single_mode_params.items():
            assert mode in self.modes, self.name
            others = {other.latency_param for other in self.modes} - {mode.latency_param}
            assert set(narrowed) == set(self.params) - others, self.name
            assert narrowed[mode.latency_param] == self.params[mode.latency_param], self.name

    @property
    def module_name(self) -> str:
        return f"holoso_{self.name}"

    def conditions_operand(self, position: int) -> bool:
        """Whether operand port `position` has a sign sideband."""
        port = self.operand_ports[position]
        return has_sign_control(port.scalar_type) and position not in self.operands_without_sideband

    def mode_of(self, code: int) -> OperatorMode:
        modes = [mode for mode in self.modes if mode.code == code]
        assert len(modes) == 1, (self.name, code)
        return modes[0]

    @property
    def sole_mode(self) -> OperatorMode:
        (mode,) = self.modes
        return mode

    def latency(self, mode: OperatorMode) -> int:
        assert mode in self.modes, (self.name, mode)
        return self.params[mode.latency_param]

    def params_for(self, modes: frozenset[OperatorMode]) -> Mapping[str, int]:
        """The parameters elaborating an instance whose firings run `modes`."""
        assert modes and modes <= set(self.modes), self.name
        if len({mode.code for mode in modes}) == 1:
            return self.single_mode_params.get(next(iter(modes)), self.params)
        return self.params

    @property
    def latencies(self) -> list[int]:
        return [self.latency(mode) for mode in self.modes]


@dataclass(frozen=True)
class Primitive(ABC):
    """
    What a firing computes: a signature, bit-exact reference semantics, and a rendering. Commutative primitives let
    port assignment orient each use's operands to shrink the per-port read muxes.
    """

    # Commutation symmetry: swapping the two operands permutes the output ports through this map (`new_port =
    # swap_output_permutation[old_port]`); `None` means non-commutative. Single-output commutative primitives use
    # the identity `(0,)`; the comparator's order flags transpose (`gt` and `lt` exchange, `eq` is fixed).
    # The permutation must preserve each port's type, so a swapped firing's taps stay in their banks.
    swap_output_permutation: ClassVar[tuple[int, ...] | None] = None

    # Float operand positions whose sign the primitive cannot observe: their conditioner is fixed to the identity, since
    # offering one would buy a second firing for one answer.
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset()

    # One label per result port of a multi-output primitive whose taps render as `label(operands)`.
    output_labels: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        """
        The declarations assert what the primitive reads, which nothing can derive, so these rules close the ways they
        can be wrong.
        """
        signature, name = self.signature, type(self).__name__
        declined = self.unconditioned_operands
        # Declining a sideband the port never had would say nothing; only a float port carries one to decline.
        assert all(0 <= position < signature.arity for position in declined), name
        assert all(has_sign_control(signature.operand_types[position]) for position in declined), name
        permutation = self.swap_output_permutation
        if permutation is not None:
            # A flip exchanges the two operands together with their conditioners, so the two ports must be
            # interchangeable, and a declaration covering one of them would migrate onto the other.
            assert signature.arity == 2 and signature.operand_types[0] == signature.operand_types[1], name
            assert declined in (frozenset(), frozenset({0, 1})), name
            result_types = signature.result_types
            assert sorted(permutation) == list(range(len(result_types))), name
            assert all(result_types[permutation[p]] == result_types[p] for p in range(len(permutation))), name

    @property
    @abstractmethod
    def latency(self) -> int: ...

    @abstractmethod
    def render(self, *operands: str) -> str: ...

    @property
    def is_commutative(self) -> bool:
        return self.swap_output_permutation is not None

    def render_output(self, result: int, inversion: BoolInversion | None, *operands: str) -> str:
        """
        Human-friendly form of one tapped result, `inversion` present exactly for a boolean one. A multi-output
        primitive declares its `output_labels` or overrides this (silently rendering every tap as the whole-primitive
        expression would mislabel the report).
        """
        if self.output_labels:
            text = f"{self.output_labels[result]}({', '.join(operands)})"
        else:
            assert (
                len(self.signature.result_types) == 1 and result == 0
            ), f"{type(self).__name__} must label its outputs"
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
    def evaluate(self, *operands: ScalarValue) -> tuple[ScalarValue, ...]:
        """Bit-exact reference semantics: one value per output port, aligned with `signature.result_types`."""


@dataclass(frozen=True, slots=True)
class BaseOperatorOptions:
    """
    The knobs of one kind of operator. Every kind's options subclass this.
    """

    instances: int = 1
    """How many physical copies of this operator the machine may emit. A cap: only the copies the schedule uses."""

    def __post_init__(self) -> None:
        if self.instances < 1:
            raise ValueError(f"instances must be >= 1, got {self.instances}")


@dataclass(frozen=True, slots=True)
class PooledPrimitive(Primitive, ABC):
    """
    A primitive that runs on an operator, in the one mode it selects there: the primitive is the semantics, the
    operator the hardware, and everything physical -- ports, parameters, timing -- is read off the operator, never
    redeclared. Operand position i is the operator's operand port i, a mode reading a prefix of them; result q leaves on
    the q-th output port its mode drives. The scheduler pools and contends firings over the operator's instances.
    """

    operator: HardwareOperator

    def __post_init__(self) -> None:
        # An error sideband is an observable output that "unchanged across every result port" does not cover, so a
        # primitive raising one may decline no operand's sign.
        name = type(self).__name__
        assert not (self.operator.error_ports and self.unconditioned_operands), name
        # A port without a sideband can only be driven by the identity, which lowering guarantees for declined operands.
        assert all(
            self.operator.conditions_operand(position) or position in self.unconditioned_operands
            for position in range(self.mode.operand_count)
            if has_sign_control(self.operator.operand_ports[position].scalar_type)
        ), name
        super().__post_init__()

    @property
    def mode(self) -> OperatorMode:
        """A primitive on an operator of several modes selects its own through that operator's typed accessor."""
        return self.operator.sole_mode

    @property
    def latency(self) -> int:
        return self.operator.latency(self.mode)

    @property
    def initiation_interval(self) -> int:
        """
        Minimum cycles until the instance accepts its next firing, of any mode (1 = fully pipelined) -- the per-firing
        sense of II. Distinct from the module-level `Lir.min_initiation_interval`, the whole-transaction cost, which is
        this project's deliberate usage (see DESIGN.md, Direction).
        """
        return self.mode.initiation_interval

    @property
    def signature(self) -> ScalarSignature:
        operands = self.operator.operand_ports[: self.mode.operand_count]
        outputs = [self.operator.output_ports[port] for port in self.mode.outputs]
        return ScalarSignature(
            tuple(port.scalar_type for port in operands), tuple(port.scalar_type for port in outputs)
        )

    def physical_port(self, result: int) -> int:
        """The operator output port carrying the `result`-th result, which a mode need not start at port 0."""
        return self.mode.outputs[result]


@dataclass(frozen=True)
class InlinePrimitive(Primitive, ABC):
    """
    A pure combinational primitive folded into a register write: each firing is one PC-gated statement that reads its
    operands and writes its single result on one step. No module, no pooling, no contention.
    """

    mnemonic: ClassVar[str]

    @property
    def latency(self) -> int:
        # It reads and writes on one step; the register's write-then-read cost is the bank's READ_FIRST_EDGE in the
        # landing helper, not a pipeline stage.
        return 0

    def render(self, *operands: str) -> str:
        return self.verilog_expr(*operands).replace(" ", "")

    @abstractmethod
    def verilog_expr(self, *operand_nets: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ComparatorPrimitive(PooledPrimitive, ABC):
    """
    A pooled comparator over one scalar family, producing the three mutually-exclusive one-hot order flags. Every
    family Holoso compares is totally ordered, so each relation is one flag or its complement; one instance therefore
    serves them all, and several relations over the same operands fuse into one firing.
    """

    # The single place the relation/flag mapping is defined; consumers go through `tap_of`.
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

    @classmethod
    def tap_of(cls, relation: Relation) -> tuple[int, BoolInversion]:
        return cls._TAP_OF_RELATION[relation]

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}⇔{b}"

    def render_output(self, result: int, inversion: BoolInversion | None, *operands: str) -> str:
        """Recovers the tapped flag as the relation it implements, e.g. `a≥b`."""
        assert inversion is not None
        a, b = operands
        return f"{a}{self._RELATION_OF_TAP[(result, inversion)].value}{b}"


@dataclass(frozen=True, slots=True)
class SelectPrimitive(InlinePrimitive):
    """
    A data mux `cond ? a : b` over same-typed values, folded into the destination register write as a ternary over
    the operand nets. Produced by HIR if-conversion and by selected MIR composite lowerings.
    Each operand is a dedicated direct (unlatched) register read; the cost is one mux per merged value, the same order
    as the per-arm phi-copy installs the branch would otherwise need.
    """

    mnemonic: ClassVar[str] = "select"
    scalar_type: ScalarType

    @property
    def signature(self) -> ScalarSignature:
        ty = self.scalar_type
        return ScalarSignature((BoolType(), ty, ty), (ty,))

    def render(self, *operands: str) -> str:
        cond, a, b = operands
        return f"{cond}?{a}:{b}"

    def verilog_expr(self, *operand_nets: str) -> str:
        cond, a, b = operand_nets
        return f"(({cond}) ? ({a}) : ({b}))"

    def evaluate(self, *operands: ScalarValue) -> tuple[ScalarValue, ...]:
        cond, a, b = self._validated_operands(operands)
        assert isinstance(cond, bool)
        return (a if cond else b,)
