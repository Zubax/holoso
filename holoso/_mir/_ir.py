"""Selected mid-level IR (MIR): concrete hardware operators with typed scalar sidebands."""

from dataclasses import dataclass

from .._hir import ValueId
from .._operators import FloatHardwareOperator, FloatSignControl, HardwareOperator
from .._errors import UnsupportedConstruct
from .._type import FloatFormat, FloatType, ScalarType


@dataclass(frozen=True, slots=True)
class MirInput:
    """A typed module input value."""

    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirConst:
    """A typed scalar constant value."""

    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirOperation:
    """A selected hardware-operator use with the fields shared by all scalar domains."""

    operator: HardwareOperator
    operands: list[ValueId]

    @property
    def scalar_type(self) -> ScalarType:
        return self.operator.signature.result_type


@dataclass(frozen=True, slots=True)
class MirOutput:
    """A named module output driven by an internal value."""

    name: str
    value: ValueId


@dataclass(frozen=True, slots=True)
class MirFloatInput(MirInput):
    """A floating-point module input port."""

    scalar_type: FloatType

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, FloatType):
            raise TypeError(f"MirFloatInput scalar_type must be FloatType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirFloatConst(MirConst):
    """A floating-point constant."""

    scalar_type: FloatType
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, FloatType):
            raise TypeError(f"MirFloatConst scalar_type must be FloatType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirFloatOperation(MirOperation):
    """A selected floating-point hardware-operator use with folded sign controls."""

    operator: FloatHardwareOperator
    operands: list[ValueId]
    operand_signs: list[FloatSignControl]
    result_sign: FloatSignControl

    def __post_init__(self) -> None:
        if not isinstance(self.operator, FloatHardwareOperator):
            raise TypeError(f"MirFloatOperation operator must be FloatHardwareOperator, got {self.operator!r}")
        signature = self.operator.signature
        if not isinstance(signature.result_type, FloatType):
            raise TypeError(f"MirFloatOperation result type must be FloatType, got {signature.result_type!r}")
        if any(not isinstance(operand_type, FloatType) for operand_type in signature.operand_types):
            raise TypeError(f"MirFloatOperation operand types must all be FloatType, got {signature.operand_types!r}")
        if len(self.operands) != signature.arity:
            raise ValueError(f"{self.operator.mnemonic} expects {signature.arity} operand(s), got {len(self.operands)}")
        if len(self.operand_signs) != signature.arity:
            raise ValueError(
                f"{self.operator.mnemonic} expects {signature.arity} sign control(s), got {len(self.operand_signs)}"
            )
        if any(not isinstance(sign, FloatSignControl) for sign in self.operand_signs):
            raise TypeError(f"operand_signs must contain FloatSignControl values, got {self.operand_signs!r}")
        if not isinstance(self.result_sign, FloatSignControl):
            raise TypeError(f"result_sign must be FloatSignControl, got {self.result_sign!r}")

    @property
    def scalar_type(self) -> FloatType:
        scalar_type = self.operator.signature.result_type
        if not isinstance(scalar_type, FloatType):
            raise TypeError(f"MirFloatOperation result type must be FloatType, got {scalar_type!r}")
        return scalar_type


type MirNode = MirInput | MirConst | MirOperation
type MirFloatNode = MirFloatInput | MirFloatConst | MirFloatOperation


@dataclass(frozen=True, slots=True)
class MirFloatOutput(MirOutput):
    """A floating-point module output with a folded output sign control."""

    sign: FloatSignControl = FloatSignControl()

    def __post_init__(self) -> None:
        if not isinstance(self.sign, FloatSignControl):
            raise TypeError(f"MirFloatOutput sign must be FloatSignControl, got {self.sign!r}")


@dataclass(frozen=True, slots=True)
class Mir:
    """A single-block selected graph ready for scheduling."""

    nodes: dict[ValueId, MirNode]
    input_ids: list[ValueId]
    outputs: list[MirOutput]


@dataclass(frozen=True, slots=True)
class MirFloatView:
    """A MIR graph narrowed once to the float-only resource family implemented by LIR today."""

    nodes: dict[ValueId, MirFloatNode]
    input_ids: list[ValueId]
    outputs: list[MirFloatOutput]
    fmt: FloatFormat

    @property
    def input_nodes(self) -> dict[ValueId, MirFloatInput]:
        result: dict[ValueId, MirFloatInput] = {}
        for vid in self.input_ids:
            node = self.nodes[vid]
            if isinstance(node, MirFloatInput):
                result[vid] = node
        return result

    @property
    def const_nodes(self) -> dict[ValueId, MirFloatConst]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirFloatConst)}

    @property
    def operation_nodes(self) -> dict[ValueId, MirFloatOperation]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirFloatOperation)}

    @classmethod
    def from_mir(cls, mir: Mir) -> "MirFloatView":
        nodes: dict[ValueId, MirFloatNode] = {}
        formats: set[FloatFormat] = set()
        for vid, node in mir.nodes.items():
            match node:
                case MirFloatInput(scalar_type=scalar_type):
                    nodes[vid] = node
                    formats.add(scalar_type.fmt)
                case MirFloatConst(scalar_type=scalar_type):
                    nodes[vid] = node
                    formats.add(scalar_type.fmt)
                case MirFloatOperation(scalar_type=scalar_type):
                    nodes[vid] = node
                    formats.add(scalar_type.fmt)
                case MirInput():
                    raise UnsupportedConstruct(f"LIR construction does not support non-float MIR input {vid}")
                case MirConst():
                    raise UnsupportedConstruct(f"LIR construction does not support non-float MIR constant {vid}")
                case MirOperation():
                    raise UnsupportedConstruct(f"LIR construction does not support non-float MIR operation {vid}")
        outputs: list[MirFloatOutput] = []
        for out in mir.outputs:
            if not isinstance(out, MirFloatOutput):
                raise UnsupportedConstruct(f"LIR construction does not support non-float MIR output {out.name!r}")
            outputs.append(out)
        for vid in mir.input_ids:
            input_node = nodes.get(vid)
            if input_node is None:
                raise ValueError(f"MIR input ID {vid} does not reference a MIR node")
            if not isinstance(input_node, MirFloatInput):
                raise ValueError(f"MIR input ID {vid} must reference a MirFloatInput, got {input_node!r}")
        if len(formats) != 1:
            ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(f"LIR requires exactly one floating-point format; got {ordered or 'none'}")
        return cls(nodes=nodes, input_ids=list(mir.input_ids), outputs=outputs, fmt=next(iter(formats)))


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

    def _type_of(self, vid: ValueId) -> ScalarType:
        node = self._nodes[vid]
        match node:
            case MirInput(scalar_type=scalar_type):
                return scalar_type
            case MirConst(scalar_type=scalar_type):
                return scalar_type
            case MirOperation() as operation:
                return operation.scalar_type
        raise TypeError(f"MIR node {vid} has no scalar type")

    def float_input(self, name: str, scalar_type: FloatType) -> ValueId:
        vid = self._fresh(MirFloatInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def float_const(self, value: float, scalar_type: FloatType) -> ValueId:
        key = ("float_const", float(value), scalar_type)
        vid = self._intern.get(key)
        if vid is None:
            vid = self._fresh(MirFloatConst(scalar_type=scalar_type, value=float(value)))
            self._intern[key] = vid
        return vid

    def float_operation(
        self,
        operator: FloatHardwareOperator,
        operands: list[ValueId],
        operand_signs: list[FloatSignControl],
        result_sign: FloatSignControl = FloatSignControl(),
    ) -> ValueId:
        signature = operator.signature
        if len(operands) != signature.arity:
            raise ValueError(f"{operator.mnemonic} expects {signature.arity} operand(s), got {len(operands)}")
        if len(operand_signs) != signature.arity:
            raise ValueError(f"{operator.mnemonic} expects {signature.arity} sign control(s), got {len(operand_signs)}")
        for operand, expected_type in zip(operands, signature.operand_types, strict=True):
            if self._type_of(operand) != expected_type:
                raise ValueError(
                    f"operator {operator.mnemonic} expects operands of {signature.operand_types}, "
                    f"got {tuple(self._type_of(operand) for operand in operands)}"
                )
        key = (operator, tuple(operands), tuple(operand_signs), result_sign)
        vid = self._intern.get(key)
        if vid is None:
            vid = self._fresh(
                MirFloatOperation(
                    operator=operator,
                    operands=list(operands),
                    operand_signs=list(operand_signs),
                    result_sign=result_sign,
                )
            )
            self._intern[key] = vid
        return vid

    def float_output(self, name: str, value: ValueId, sign: FloatSignControl = FloatSignControl()) -> None:
        if not isinstance(self._type_of(value), FloatType):
            raise ValueError(f"float output {name!r} must be driven by a floating-point value")
        self._outputs.append(MirFloatOutput(name, value, sign))

    def finish(self) -> Mir:
        return Mir(
            nodes=dict(self._nodes),
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
        )
