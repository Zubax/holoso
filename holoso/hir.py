"""The high-level IR (HIR): an SSA value DAG of scalar floating-point operations."""

from collections.abc import Callable
from dataclasses import dataclass

from .format import FloatFormat
from .operators import Op, Sgnop

type ValueId = int


@dataclass(frozen=True, slots=True)
class ArithOp:
    """A binary arithmetic operator before hardware selection. ``evaluate`` returns None on /0 so const-fold aborts."""

    name: str
    evaluate: Callable[[float, float], float | None]


ADD = ArithOp("add", lambda a, b: a + b)
MUL = ArithOp("mul", lambda a, b: a * b)
DIV = ArithOp("div", lambda a, b: a / b if b != 0 else None)


@dataclass(frozen=True, slots=True)
class SignOp:
    """A combinational sign manipulation; ``sgnop`` is the folded 2-bit encoding it contributes to a consumer."""

    name: str
    evaluate: Callable[[float], float]
    sgnop: Sgnop


NEG = SignOp("neg", lambda x: -x, Sgnop.NEG)
ABS = SignOp("abs", abs, Sgnop.ABS)


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
    A selected hardware operator use with folded sign-ops. ``op`` is the fully-specified operator;
    ``b`` is ``None`` for a unary operator. ``operands``/``operand_sgnops`` expose the filled positions
    (length ``op.arity``) so consumers iterate positionally without branching on operator identity.
    Latency is ``op.latency(fmt)``.
    """

    op: Op
    a: ValueId
    b: ValueId | None
    a_sgnop: Sgnop
    b_sgnop: Sgnop
    y_sgnop: Sgnop

    @property
    def operands(self) -> list[ValueId]:
        b = self.b
        return [self.a] if b is None else [self.a, b]

    @property
    def operand_sgnops(self) -> list[Sgnop]:
        return [self.a_sgnop] if self.b is None else [self.a_sgnop, self.b_sgnop]


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
    input_ids: list[ValueId]
    outputs: list[OutputPort]

    def input_names(self) -> list[str]:
        names: list[str] = []
        for vid in self.input_ids:
            node = self.nodes[vid]
            assert isinstance(node, InPort)
            names.append(node.name)
        return names

    def count(self, predicate: type[Node] | list[type[Node]]) -> int:
        classes = (predicate,) if isinstance(predicate, type) else tuple(predicate)
        return sum(1 for node in self.nodes.values() if isinstance(node, classes))

    def arith_count(self, op: ArithOp) -> int:
        return sum(1 for node in self.nodes.values() if isinstance(node, Arith) and node.op is op)

    def op_count(self, cls: type[Op]) -> int:
        return sum(1 for node in self.nodes.values() if isinstance(node, OpNode) and isinstance(node.op, cls))

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
        op: Op,
        a: ValueId,
        b: ValueId | None,
        a_sgnop: Sgnop,
        b_sgnop: Sgnop,
        y_sgnop: Sgnop,
    ) -> ValueId:
        return self._interned(OpNode(op, a, b, a_sgnop, b_sgnop, y_sgnop))

    def output(self, name: str, value: ValueId, sgnop: Sgnop = Sgnop.NONE) -> None:
        self._outputs.append(OutputPort(name, value, sgnop))

    def finish(self) -> Hir:
        return Hir(
            fmt=self.fmt,
            nodes=dict(self._nodes),
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
        )
