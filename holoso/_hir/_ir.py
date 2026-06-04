"""HIR data model."""

from dataclasses import dataclass

from ._const import Const, FloatConst
from ._operators import Operator
from ._types import FloatType, Type

type ValueId = int


@dataclass(frozen=True, slots=True)
class InPort:
    """A module input port (a function parameter)."""

    name: str
    type: Type


@dataclass(frozen=True, slots=True)
class Operation:
    """A semantic operation occurrence in HIR."""

    operator: Operator
    operands: tuple[ValueId, ...]

    @property
    def type(self) -> Type:
        return self.operator.signature.result_type


@dataclass(frozen=True, slots=True)
class StateRead:
    """
    The live-in value of a persistent state slot: the content of its register at the start of an initiation, carried
    over from the previous initiation (or the reset snapshot on the first one). Interned per slot, so every read of the
    same attribute before it is first rewritten shares one value.
    """

    slot: str
    type: Type


type Node = InPort | Const | Operation | StateRead


@dataclass(frozen=True, slots=True)
class OutputPort:
    name: str
    value: ValueId


@dataclass(frozen=True, slots=True)
class StateSlot:
    """
    A persistent state register backing a written instance attribute. ``reset_value`` is the snapshot taken from the
    instance at synthesis time and loaded at module reset; ``live_out`` is the value the attribute holds at method exit
    and that must reside in the slot's register at the initiation boundary. Observability is not a slot property: a
    public attribute is exposed by a separate ``state_<attr>`` output port that the frontend emits alongside the slot.
    """

    name: str
    reset_value: float
    live_out: ValueId


@dataclass(frozen=True, slots=True)
class Hir:
    """A complete single-block HIR: the value DAG, ordered inputs, ordered named outputs, and persistent state slots."""

    nodes: dict[ValueId, Node]
    input_ids: list[ValueId]
    outputs: list[OutputPort]
    state_slots: list[StateSlot]

    def input_names(self) -> list[str]:
        names: list[str] = []
        for vid in self.input_ids:
            node = self.nodes[vid]
            assert isinstance(node, InPort)
            names.append(node.name)
        return names


class HirBuilder:
    """Builds an :class:`Hir`, interning pure nodes so identical subexpressions share one value ID."""

    def __init__(self) -> None:
        self._nodes: dict[ValueId, Node] = {}
        self._intern: dict[Node, ValueId] = {}
        self._input_ids: list[ValueId] = []
        self._outputs: list[OutputPort] = []
        self._state_slots: list[StateSlot] = []

    def _fresh(self, node: Node) -> ValueId:
        vid = len(self._nodes)
        self._nodes[vid] = node
        return vid

    def _interned(self, node: Node) -> ValueId:
        vid = self._intern.get(node)
        if vid is None:
            vid = self._fresh(node)
            self._intern[node] = vid
        return vid

    def _type_of(self, vid: ValueId) -> Type:
        node = self._nodes[vid]
        match node:
            case InPort(type=type):
                return type
            case StateRead(type=type):
                return type
            case Const():
                return node.type
            case Operation():
                return node.type
        raise TypeError(f"HIR node {vid} has no semantic type")

    def input(self, name: str, type: Type) -> ValueId:
        # Input ports are never interned: each parameter is a distinct, ordered port.
        vid = self._fresh(InPort(name, type))
        self._input_ids.append(vid)
        return vid

    def float_input(self, name: str) -> ValueId:
        return self.input(name, FloatType())

    def state_read(self, slot: str, type: Type) -> ValueId:
        # Interned: repeated reads of a slot before it is first rewritten share one live-in value.
        return self._interned(StateRead(slot, type))

    def float_state_read(self, slot: str) -> ValueId:
        return self.state_read(slot, FloatType())

    def state_slot(self, name: str, reset_value: float, live_out: ValueId) -> None:
        self._state_slots.append(StateSlot(name, float(reset_value), live_out))

    def float_const(self, value: float) -> ValueId:
        return self.const_node(FloatConst(float(value)))

    def const_node(self, const: Const) -> ValueId:
        return self._interned(const)

    def operation(self, operator: Operator, operands: list[ValueId]) -> ValueId:
        signature = operator.signature
        if len(operands) != signature.arity:
            raise ValueError(f"{operator.mnemonic} expects {signature.arity} operand(s), got {len(operands)}")
        operand_types = tuple(self._type_of(operand) for operand in operands)
        if operand_types != signature.operand_types:
            raise ValueError(f"{operator.mnemonic} expects operands of {signature.operand_types}, got {operand_types}")
        return self._interned(Operation(operator, tuple(operands)))

    def output(self, name: str, value: ValueId) -> None:
        self._outputs.append(OutputPort(name, value))

    def finish(self) -> Hir:
        return Hir(
            nodes=dict(self._nodes),
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
            state_slots=list(self._state_slots),
        )
