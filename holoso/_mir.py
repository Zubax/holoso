"""Selected mid-level IR (MIR): concrete hardware operators with folded sign controls."""

from dataclasses import dataclass

from ._hir import ValueId
from ._operators import HardwareOperator, SignControl


@dataclass(frozen=True, slots=True)
class MirInput:
    """A module input port."""

    name: str


@dataclass(frozen=True, slots=True)
class MirConst:
    """A floating-point constant."""

    value: float


@dataclass(frozen=True, slots=True)
class MirOperation:
    """A selected hardware-operator use with folded sign controls."""

    operator: HardwareOperator
    operands: list[ValueId]
    operand_signs: list[SignControl]
    result_sign: SignControl


type MirNode = MirInput | MirConst | MirOperation


@dataclass(frozen=True, slots=True)
class MirOutput:
    name: str
    value: ValueId
    sign: SignControl = SignControl()


@dataclass(frozen=True, slots=True)
class Mir:
    """A single-block selected graph ready for scheduling."""

    nodes: dict[ValueId, MirNode]
    input_ids: list[ValueId]
    outputs: list[MirOutput]


class MirBuilder:
    """Builds a selected graph, preserving structural CSE for constants and selected operations."""

    def __init__(self) -> None:
        self._nodes: dict[ValueId, MirNode] = {}
        self._intern: dict[object, ValueId] = {}
        self._input_ids: list[ValueId] = []
        self._outputs: list[MirOutput] = []

    def _fresh(self, node: MirNode) -> ValueId:
        vid = len(self._nodes)
        self._nodes[vid] = node
        return vid

    def input(self, name: str) -> ValueId:
        vid = self._fresh(MirInput(name))
        self._input_ids.append(vid)
        return vid

    def const(self, value: float) -> ValueId:
        key = ("const", float(value))
        vid = self._intern.get(key)
        if vid is None:
            vid = self._fresh(MirConst(float(value)))
            self._intern[key] = vid
        return vid

    def operation(
        self,
        operator: HardwareOperator,
        operands: list[ValueId],
        operand_signs: list[SignControl],
        result_sign: SignControl = SignControl(),
    ) -> ValueId:
        if len(operands) != operator.arity:
            raise ValueError(f"{operator.mnemonic} expects {operator.arity} operand(s), got {len(operands)}")
        if len(operand_signs) != operator.arity:
            raise ValueError(f"{operator.mnemonic} expects {operator.arity} sign control(s), got {len(operand_signs)}")
        key = (operator, tuple(operands), tuple(operand_signs), result_sign)
        vid = self._intern.get(key)
        if vid is None:
            vid = self._fresh(MirOperation(operator, list(operands), list(operand_signs), result_sign))
            self._intern[key] = vid
        return vid

    def output(self, name: str, value: ValueId, sign: SignControl = SignControl()) -> None:
        self._outputs.append(MirOutput(name, value, sign))

    def finish(self) -> Mir:
        return Mir(
            nodes=dict(self._nodes),
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
        )
