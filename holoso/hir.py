"""The high-level IR (HIR): an SSA value DAG of scalar floating-point operations."""

import enum
from dataclasses import dataclass

from .format import FloatFormat
from .operators import OpKind, Sgnop

type ValueId = int


class ArithOp(enum.Enum):
    ADD = "add"
    MUL = "mul"
    DIV = "div"


class SignOp(enum.Enum):
    NEG = "neg"
    ABS = "abs"


@dataclass(frozen=True, slots=True)
class InPort:
    """A module input port (a function parameter)."""

    name: str


@dataclass(frozen=True, slots=True)
class Const:
    """A floating-point constant."""

    value: float


@dataclass(frozen=True, slots=True)
class Arith:
    """A binary arithmetic operation. Subtraction is ``ADD`` with a sign-flipped operand."""

    op: ArithOp
    a: ValueId
    b: ValueId


@dataclass(frozen=True, slots=True)
class SignFix:
    """Combinational sign manipulation (negate / absolute). Folds onto a consuming operator's sign-op port."""

    op: SignOp
    a: ValueId


@dataclass(frozen=True, slots=True)
class Fmul2K:
    """Exact scaling by a power of two: ``a * 2**k`` (introduced by strength reduction)."""

    a: ValueId
    k: int


@dataclass(frozen=True, slots=True)
class OpNode:
    """
    A selected hardware operator use with folded sign-ops. ``b`` is ``None`` for unary ``FMUL_ILOG2``;
    ``k`` is the exponent for ``FMUL_ILOG2`` (else ``None``).
    """

    kind: OpKind
    a: ValueId
    b: ValueId | None
    a_sgnop: Sgnop
    b_sgnop: Sgnop
    y_sgnop: Sgnop
    k: int | None
    latency: int


type Node = InPort | Const | Arith | SignFix | Fmul2K | OpNode


@dataclass(frozen=True, slots=True)
class OutputPort:
    name: str
    value: ValueId
    sgnop: Sgnop = Sgnop.NONE  # combinational sign-op applied to the output wire (folded residual sign)


@dataclass(frozen=True, slots=True)
class Hir:
    """A complete single-block HIR: the value DAG, ordered inputs, and ordered named outputs."""

    fmt: FloatFormat
    nodes: dict[ValueId, Node]
    input_ids: tuple[ValueId, ...]
    outputs: tuple[OutputPort, ...]

    def input_names(self) -> list[str]:
        names: list[str] = []
        for vid in self.input_ids:
            node = self.nodes[vid]
            assert isinstance(node, InPort)
            names.append(node.name)
        return names

    def count(self, predicate: type[Node] | tuple[type[Node], ...]) -> int:
        return sum(1 for node in self.nodes.values() if isinstance(node, predicate))

    def arith_count(self, op: ArithOp) -> int:
        return sum(1 for node in self.nodes.values() if isinstance(node, Arith) and node.op == op)

    def op_count(self, kind: OpKind) -> int:
        return sum(1 for node in self.nodes.values() if isinstance(node, OpNode) and node.kind == kind)

    def dump(self) -> str:
        lines = [f"hir {self.fmt}"]
        for vid in sorted(self.nodes):
            lines.append(f"  v{vid} = {self.nodes[vid]}")
        for out in self.outputs:
            suffix = "" if out.sgnop is Sgnop.NONE else f" (sgnop={out.sgnop.name})"
            lines.append(f"  {out.name} <- v{out.value}{suffix}")
        return "\n".join(lines)


class HirBuilder:
    """Builds an :class:`Hir`, interning pure nodes so identical subexpressions share one value id (structural CSE)."""

    def __init__(self, fmt: FloatFormat) -> None:
        self.fmt = fmt
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

    def arith(self, op: ArithOp, a: ValueId, b: ValueId) -> ValueId:
        return self._interned(Arith(op, a, b))

    def signfix(self, op: SignOp, a: ValueId) -> ValueId:
        return self._interned(SignFix(op, a))

    def fmul2k(self, a: ValueId, k: int) -> ValueId:
        return self._interned(Fmul2K(a, k))

    def opnode(
        self,
        kind: OpKind,
        a: ValueId,
        b: ValueId | None,
        a_sgnop: Sgnop,
        b_sgnop: Sgnop,
        y_sgnop: Sgnop,
        k: int | None,
        latency: int,
    ) -> ValueId:
        return self._interned(OpNode(kind, a, b, a_sgnop, b_sgnop, y_sgnop, k, latency))

    def output(self, name: str, value: ValueId, sgnop: Sgnop = Sgnop.NONE) -> None:
        self._outputs.append(OutputPort(name, value, sgnop))

    def finish(self) -> Hir:
        return Hir(
            fmt=self.fmt,
            nodes=dict(self._nodes),
            input_ids=tuple(self._input_ids),
            outputs=tuple(self._outputs),
        )
