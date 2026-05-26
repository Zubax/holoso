"""HIR data model."""

from dataclasses import dataclass

from ._operators import Operator

type ValueId = int


@dataclass(frozen=True, slots=True)
class InPort:
    """A module input port (a function parameter)."""

    name: str


@dataclass(frozen=True, slots=True)
class Const:
    """A floating-point constant."""

    value: float


@dataclass(frozen=True, slots=True)
class Operation:
    """A semantic operation occurrence in HIR."""

    operator: Operator
    operands: tuple[ValueId, ...]


type Node = InPort | Const | Operation


@dataclass(frozen=True, slots=True)
class OutputPort:
    name: str
    value: ValueId


@dataclass(frozen=True, slots=True)
class Hir:
    """A complete single-block HIR: the value DAG, ordered inputs, and ordered named outputs."""

    nodes: dict[ValueId, Node]
    input_ids: list[ValueId]
    outputs: list[OutputPort]

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

    def input(self, name: str) -> ValueId:
        # Input ports are never interned: each parameter is a distinct, ordered port.
        vid = self._fresh(InPort(name))
        self._input_ids.append(vid)
        return vid

    def const(self, value: float) -> ValueId:
        return self._interned(Const(float(value)))

    def operation(self, operator: Operator, operands: list[ValueId]) -> ValueId:
        if len(operands) != operator.arity:
            raise ValueError(f"{operator.mnemonic} expects {operator.arity} operand(s), got {len(operands)}")
        return self._interned(Operation(operator, tuple(operands)))

    def output(self, name: str, value: ValueId) -> None:
        self._outputs.append(OutputPort(name, value))

    def finish(self) -> Hir:
        return Hir(
            nodes=dict(self._nodes),
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
        )
