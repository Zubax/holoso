"""Selected mid-level IR (MIR): concrete hardware operators with typed scalar sidebands, arranged into a CFG."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from .._operators import (
    BoolInversion,
    FloatSignControl,
    HardwareOperator,
    InlineHardwareOperator,
    IntIdentity,
    PortConditioner,
    identity_conditioner,
)
from .._errors import UnsupportedConstruct
from .._type import BoolType, FloatFormat, FloatType, IntFormat, IntType, ScalarType
from .._util import BlockId, ValueId


@dataclass(frozen=True, slots=True)
class MirInput:
    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirStateRead:
    """The slot's live-in: its register content at the initiation start."""

    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirConst:
    scalar_type: ScalarType


def _check_conditioner(conditioner: PortConditioner, port_type: ScalarType) -> None:
    """Port conditioner must be the type's own: sign control for floats, inversion for bools, identity for ints."""
    assert isinstance(conditioner, type(identity_conditioner(port_type)))


@dataclass(frozen=True, slots=True)
class MirOperation:
    """
    A selected hardware-operator use producing ONE value: the ``output_port``-th result, conditioned by
    ``output_conditioner``. Operations sharing one block, operator, operands, and operand conditioners while tapping
    DISTINCT output ports fuse into a single firing at LIR build -- a multi-output module computes all its results at
    once. The operation belongs to the resource family of its tapped port's type; operands may reference either family
    (a comparison reads float operands and produces booleans; the bool->float cast the reverse).
    """

    operator: HardwareOperator
    operands: list[ValueId]
    operand_conditioners: list[PortConditioner]
    output_port: int
    output_conditioner: PortConditioner
    immediates: tuple[int, ...]  # per-firing immediate values, aligned with operator.immediate_ports

    def __post_init__(self) -> None:
        signature = self.operator.signature
        ports = self.operator.immediate_ports
        assert not isinstance(self.operator, InlineHardwareOperator) or len(signature.result_types) == 1
        assert len(self.immediates) == len(ports)
        assert all(0 <= value < (1 << port.width) for value, port in zip(self.immediates, ports, strict=True))
        assert len(self.operands) == signature.arity
        assert len(self.operand_conditioners) == signature.arity
        for conditioner, operand_type in zip(self.operand_conditioners, signature.operand_types, strict=True):
            _check_conditioner(conditioner, operand_type)
        assert 0 <= self.output_port < len(signature.result_types)
        _check_conditioner(self.output_conditioner, signature.result_types[self.output_port])

    @property
    def scalar_type(self) -> ScalarType:
        return self.operator.signature.result_types[self.output_port]


@dataclass(frozen=True, slots=True)
class MirOutput:
    name: str
    value: ValueId
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class MirStateSlot:
    """A persistent slot register holding ``live_out`` at the transaction boundary."""

    name: str
    reset_value: float | int | bool
    live_out: ValueId
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class MirFloatInput(MirInput):
    scalar_type: FloatType

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, FloatType)


@dataclass(frozen=True, slots=True)
class MirBoolInput(MirInput):
    scalar_type: BoolType

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, BoolType)


@dataclass(frozen=True, slots=True)
class MirFloatStateRead(MirStateRead):
    scalar_type: FloatType

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, FloatType)


@dataclass(frozen=True, slots=True)
class MirFloatConst(MirConst):
    scalar_type: FloatType
    value: float

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, FloatType)
        assert isinstance(self.value, float)


@dataclass(frozen=True, slots=True)
class MirBoolStateRead(MirStateRead):
    scalar_type: BoolType

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, BoolType)


@dataclass(frozen=True, slots=True)
class MirBoolConst(MirConst):
    scalar_type: BoolType
    value: bool

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, BoolType)
        assert isinstance(self.value, bool)


@dataclass(frozen=True, slots=True)
class MirIntInput(MirInput):
    scalar_type: IntType

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, IntType)


@dataclass(frozen=True, slots=True)
class MirIntStateRead(MirStateRead):
    scalar_type: IntType

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, IntType)


@dataclass(frozen=True, slots=True)
class MirIntConst(MirConst):
    scalar_type: IntType
    value: int

    def __post_init__(self) -> None:
        assert isinstance(self.scalar_type, IntType)
        assert isinstance(self.value, int) and not isinstance(self.value, bool)
        # Saturation is what the arithmetic does, never what a literal does, so this is a refusal, not an invariant.
        if not self.scalar_type.fmt.fits(self.value):
            raise UnsupportedConstruct(f"{self.value} does not fit {self.scalar_type.fmt}; raise wint_min")


@dataclass(frozen=True, slots=True)
class MirPhi:
    """
    An SSA merge at a block's entry: one ``(predecessor_block, value, conditioner)`` arm per incoming edge, of one
    scalar type. The arm conditioner is the type's own sideband, applied when the arm value is installed: a folded
    sign control on a float arm (``y = -x`` on one branch), an optional inversion on a boolean arm (``f = not g``),
    and nothing at all on an integer arm, which has no free sideband to fold into.
    """

    scalar_type: ScalarType
    arms: tuple[tuple[BlockId, ValueId, PortConditioner], ...]

    def __post_init__(self) -> None:
        for _pred, _value, conditioner in self.arms:
            _check_conditioner(conditioner, self.scalar_type)


type MirNode = MirInput | MirStateRead | MirConst | MirOperation | MirPhi
type MirBoolNode = MirBoolInput | MirBoolStateRead | MirBoolConst | MirPhi | MirOperation

# The wide bank is shared, so each of its leaf kinds is a union; operations and phis join it on scalar width.
type MirWideInput = MirFloatInput | MirIntInput
type MirWideStateRead = MirFloatStateRead | MirIntStateRead
type MirWideConst = MirFloatConst | MirIntConst
type MirWideNode = MirWideInput | MirWideStateRead | MirWideConst | MirOperation | MirPhi


@dataclass(frozen=True, slots=True)
class MirFloatOutput(MirOutput):
    conditioner: FloatSignControl = FloatSignControl()

    def __post_init__(self) -> None:
        assert isinstance(self.conditioner, FloatSignControl)


@dataclass(frozen=True, slots=True)
class MirIntOutput(MirOutput):
    conditioner: IntIdentity = IntIdentity()

    def __post_init__(self) -> None:
        assert isinstance(self.conditioner, IntIdentity)


@dataclass(frozen=True, slots=True)
class MirBoolOutput(MirOutput):
    conditioner: BoolInversion = BoolInversion()

    def __post_init__(self) -> None:
        assert isinstance(self.conditioner, BoolInversion)


@dataclass(frozen=True, slots=True)
class MirFloatStateSlot(MirStateSlot):
    reset_value: float
    conditioner: FloatSignControl = FloatSignControl()

    def __post_init__(self) -> None:
        assert isinstance(self.reset_value, float)
        assert isinstance(self.conditioner, FloatSignControl)


@dataclass(frozen=True, slots=True)
class MirIntStateSlot(MirStateSlot):
    reset_value: int
    conditioner: IntIdentity = IntIdentity()

    def __post_init__(self) -> None:
        assert isinstance(self.reset_value, int) and not isinstance(self.reset_value, bool)
        assert isinstance(self.conditioner, IntIdentity)


@dataclass(frozen=True, slots=True)
class MirBoolStateSlot(MirStateSlot):
    reset_value: bool
    conditioner: BoolInversion = BoolInversion()

    def __post_init__(self) -> None:
        assert isinstance(self.reset_value, bool)
        assert isinstance(self.conditioner, BoolInversion)


type MirWideOutput = MirFloatOutput | MirIntOutput
type MirWideStateSlot = MirFloatStateSlot | MirIntStateSlot


@dataclass(frozen=True, slots=True)
class MirJump:
    target: BlockId


@dataclass(frozen=True, slots=True)
class MirBranch:
    cond: ValueId
    if_true: BlockId
    if_false: BlockId


@dataclass(frozen=True, slots=True)
class MirRet:
    """The sole function exit: commit state-writes and outputs."""


type MirTerminator = MirJump | MirBranch | MirRet


@dataclass(frozen=True, slots=True)
class MirBlock:
    """``operations`` is straight-line in evaluation order; ``phis`` merge at block entry."""

    id: BlockId
    phis: tuple[ValueId, ...]
    operations: tuple[ValueId, ...]
    terminator: MirTerminator


@dataclass(frozen=True, slots=True)
class Mir:
    """A selected graph arranged into a CFG of basic blocks; ``blocks[0]`` is the entry."""

    float_format: FloatFormat
    int_format: IntFormat
    nodes: dict[ValueId, MirNode]
    blocks: list[MirBlock]
    input_ids: list[ValueId]
    outputs: list[MirOutput]
    state_slots: list[MirStateSlot]

    @property
    def entry(self) -> BlockId:
        return self.blocks[0].id

    @property
    def ret_block(self) -> BlockId:
        """The id of the sole function-exit block (a kernel has exactly one ``MirRet``)."""
        return next(block.id for block in self.blocks if isinstance(block.terminator, MirRet))


class _MirBankView(ABC):
    """
    Common structure of a single-bank MIR resource view: the phi-arm subset and the per-block operation listing, both
    derived identically from the bank's narrowed node table. Each concrete view fixes the element types of ``nodes``
    and supplies ``operation_nodes`` -- the bank's defining membership predicate over the operations.
    """

    __slots__ = ()

    # Each concrete view narrows this to its bank's node element type (e.g. ``dict[ValueId, MirWideNode]``).
    nodes: Mapping[ValueId, MirNode]

    @property
    @abstractmethod
    def operation_nodes(self) -> dict[ValueId, MirOperation]: ...

    @property
    def phi_nodes(self) -> dict[ValueId, MirPhi]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirPhi)}

    def block_operations(self, block: MirBlock) -> list[ValueId]:
        """The bank's operation ids defined in ``block``, in evaluation order."""
        return [vid for vid in block.operations if vid in self.operation_nodes]


@dataclass(frozen=True, slots=True)
class MirWideView(_MirBankView):
    """
    The wide data-bank resource family narrowed out of a MIR graph, carrying the shared CFG so scheduling runs per
    block. The bank is physical, not a scalar family -- it holds floats and integers alike: :meth:`from_mir` admits
    operations and phis structurally, on ``scalar_type.is_wide``, and only the leaves nominally.
    """

    nodes: dict[ValueId, MirWideNode]
    blocks: list[MirBlock]
    entry: BlockId
    input_ids: list[ValueId]
    outputs: list[MirWideOutput]
    state_slots: list[MirWideStateSlot]
    float_format: FloatFormat
    int_format: IntFormat

    @property
    def input_nodes(self) -> dict[ValueId, MirWideInput]:
        return {
            vid: node for vid in self.input_ids if isinstance(node := self.nodes[vid], (MirFloatInput, MirIntInput))
        }

    @property
    def state_read_nodes(self) -> dict[ValueId, MirWideStateRead]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, (MirFloatStateRead, MirIntStateRead))}

    @property
    def const_nodes(self) -> dict[ValueId, MirWideConst]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, (MirFloatConst, MirIntConst))}

    @property
    def operation_nodes(self) -> dict[ValueId, MirOperation]:
        return {
            vid: node for vid, node in self.nodes.items() if isinstance(node, MirOperation) and node.scalar_type.is_wide
        }

    def scalar_type_of(self, vid: ValueId) -> FloatType | IntType:
        """Which family a wide value belongs to: the bank is physical, so only the value itself names its type."""
        scalar_type = self.nodes[vid].scalar_type
        assert isinstance(scalar_type, (FloatType, IntType))
        return scalar_type

    @classmethod
    def from_mir(cls, mir: Mir) -> MirWideView:
        nodes: dict[ValueId, MirWideNode] = {}
        float_formats: set[FloatFormat] = set()
        int_formats: set[IntFormat] = set()
        for vid, node in mir.nodes.items():
            match node:
                case (
                    MirFloatInput(scalar_type=float_type)
                    | MirFloatConst(scalar_type=float_type)
                    | MirFloatStateRead(scalar_type=float_type)
                ):
                    nodes[vid] = node
                    float_formats.add(float_type.fmt)
                case (
                    MirIntInput(scalar_type=int_type)
                    | MirIntConst(scalar_type=int_type)
                    | MirIntStateRead(scalar_type=int_type)
                ):
                    nodes[vid] = node
                    int_formats.add(int_type.fmt)
                case MirOperation(scalar_type=scalar_type) | MirPhi(scalar_type=scalar_type) if scalar_type.is_wide:
                    nodes[vid] = node
                    if isinstance(scalar_type, FloatType):
                        float_formats.add(scalar_type.fmt)
                    elif isinstance(scalar_type, IntType):
                        int_formats.add(scalar_type.fmt)
                case MirBoolInput() | MirBoolStateRead() | MirBoolConst() | MirPhi() | MirOperation():
                    pass  # the bool resource family (bool state/const/phi and bool-result ops), handled by MirBoolView
        outputs = [out for out in mir.outputs if isinstance(out, (MirFloatOutput, MirIntOutput))]
        state_slots = [slot for slot in mir.state_slots if isinstance(slot, (MirFloatStateSlot, MirIntStateSlot))]
        for vid in mir.input_ids:
            if not isinstance(mir.nodes.get(vid), (MirFloatInput, MirIntInput, MirBoolInput)):
                raise ValueError(f"MIR input ID {vid} must reference a float, integer, or boolean input")
        input_ids = [vid for vid in mir.input_ids if isinstance(nodes.get(vid), (MirFloatInput, MirIntInput))]
        if stray_floats := float_formats - {mir.float_format}:
            ordered = ", ".join(str(fmt) for fmt in sorted(stray_floats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(
                f"LIR requires MIR float values to use configured format {mir.float_format}; got {ordered}"
            )
        if stray_ints := int_formats - {mir.int_format}:
            ordered = ", ".join(str(fmt) for fmt in sorted(stray_ints, key=lambda fmt: fmt.width))
            raise ValueError(
                f"LIR requires MIR integer values to use configured format {mir.int_format}; got {ordered}"
            )
        return cls(
            nodes=nodes,
            blocks=mir.blocks,
            entry=mir.entry,
            input_ids=input_ids,
            outputs=outputs,
            state_slots=state_slots,
            float_format=mir.float_format,
            int_format=mir.int_format,
        )


@dataclass(frozen=True, slots=True)
class MirBoolView(_MirBankView):
    """
    The boolean resource family narrowed out of a MIR graph: bool state reads, constants, phis, and bool-result
    operations (comparator taps, boolean logic, and float->bool casts), plus the bool state slots and the shared CFG.
    """

    nodes: dict[ValueId, MirBoolNode]
    blocks: list[MirBlock]
    entry: BlockId
    input_ids: list[ValueId]
    outputs: list[MirBoolOutput]
    state_slots: list[MirBoolStateSlot]

    @property
    def input_nodes(self) -> dict[ValueId, MirBoolInput]:
        return {vid: node for vid in self.input_ids if isinstance(node := self.nodes[vid], MirBoolInput)}

    @property
    def state_read_nodes(self) -> dict[ValueId, MirBoolStateRead]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirBoolStateRead)}

    @property
    def const_nodes(self) -> dict[ValueId, MirBoolConst]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirBoolConst)}

    @property
    def operation_nodes(self) -> dict[ValueId, MirOperation]:
        return {
            vid: node
            for vid, node in self.nodes.items()
            if isinstance(node, MirOperation) and isinstance(node.scalar_type, BoolType)
        }

    @classmethod
    def from_mir(cls, mir: Mir) -> MirBoolView:
        nodes: dict[ValueId, MirBoolNode] = {}
        for vid, node in mir.nodes.items():
            if isinstance(node, (MirBoolInput, MirBoolStateRead, MirBoolConst)):
                nodes[vid] = node
            elif isinstance(node, MirOperation) and isinstance(node.scalar_type, BoolType):
                nodes[vid] = node
            elif isinstance(node, MirPhi) and isinstance(node.scalar_type, BoolType):
                nodes[vid] = node
        outputs = [out for out in mir.outputs if isinstance(out, MirBoolOutput)]
        state_slots = [slot for slot in mir.state_slots if isinstance(slot, MirBoolStateSlot)]
        input_ids = [vid for vid in mir.input_ids if isinstance(nodes.get(vid), MirBoolInput)]
        return cls(
            nodes=nodes,
            blocks=mir.blocks,
            entry=mir.entry,
            input_ids=input_ids,
            outputs=outputs,
            state_slots=state_slots,
        )


@dataclass
class _MirBlockUC:
    phis: list[ValueId]
    operations: list[ValueId]
    terminator: MirTerminator | None


class MirBuilder:
    """
    Builds a selected CFG. The first :meth:`block` is the entry. Inputs, constants, and state reads are entry-global
    (constants and state reads interned); operations are interned within their block; phis are never interned. A block
    is sealed by :meth:`jump` / :meth:`branch` / :meth:`ret`.
    """

    def __init__(self, float_format: FloatFormat, int_format: IntFormat) -> None:
        self._float_format = float_format
        self._int_format = int_format
        self._nodes: dict[ValueId, MirNode] = {}
        self._global_intern: dict[object, ValueId] = {}
        self._block_intern: dict[object, ValueId] = {}
        self._blocks: list[_MirBlockUC] = []
        self._cur: BlockId | None = None
        self._input_ids: list[ValueId] = []
        self._outputs: list[MirOutput] = []
        self._state_slots: list[MirStateSlot] = []

    def block(self) -> BlockId:
        bid = len(self._blocks)
        self._blocks.append(_MirBlockUC(phis=[], operations=[], terminator=None))
        if self._cur is None:
            self._cur = bid
        return bid

    def position_at(self, block: BlockId) -> None:
        self._cur = block

    @property
    def current_block(self) -> BlockId:
        if self._cur is None:
            raise RuntimeError("no current block; call block() first")
        return self._cur

    def set_terminator(self, block: BlockId, terminator: MirTerminator) -> None:
        self._blocks[block].terminator = terminator

    def jump(self, target: BlockId) -> None:
        self.set_terminator(self.current_block, MirJump(target))

    def branch(self, cond: ValueId, if_true: BlockId, if_false: BlockId) -> None:
        assert isinstance(self._type_of(cond), BoolType)
        self.set_terminator(self.current_block, MirBranch(cond, if_true, if_false))

    def ret(self) -> None:
        self.set_terminator(self.current_block, MirRet())

    def _fresh(self, node: MirNode) -> ValueId:
        vid = len(self._nodes)
        self._nodes[vid] = node
        return vid

    def _global(self, key: object, node: MirNode) -> ValueId:
        vid = self._global_intern.get(key)
        if vid is None:
            vid = self._fresh(node)
            self._global_intern[key] = vid
        return vid

    def _type_of(self, vid: ValueId) -> ScalarType:
        node = self._nodes[vid]
        match node:
            case MirInput(scalar_type=scalar_type):
                return scalar_type
            case MirStateRead(scalar_type=scalar_type):
                return scalar_type
            case MirConst(scalar_type=scalar_type):
                return scalar_type
            case MirPhi(scalar_type=scalar_type):
                return scalar_type
            case MirOperation() as operation:
                return operation.scalar_type
        assert False, f"MIR node {vid} has no scalar type"

    def float_input(self, name: str, scalar_type: FloatType) -> ValueId:
        vid = self._fresh(MirFloatInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def int_input(self, name: str, scalar_type: IntType) -> ValueId:
        vid = self._fresh(MirIntInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def bool_input(self, name: str, scalar_type: BoolType) -> ValueId:
        vid = self._fresh(MirBoolInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def float_state_read(self, name: str, scalar_type: FloatType) -> ValueId:
        return self._global(("float_state_read", name), MirFloatStateRead(name, scalar_type))

    def int_state_read(self, name: str, scalar_type: IntType) -> ValueId:
        return self._global(("int_state_read", name), MirIntStateRead(name, scalar_type))

    def bool_state_read(self, name: str, scalar_type: BoolType) -> ValueId:
        return self._global(("bool_state_read", name), MirBoolStateRead(name, scalar_type))

    def float_const(self, value: float, scalar_type: FloatType) -> ValueId:
        return self._global(
            ("float_const", float(value), scalar_type), MirFloatConst(scalar_type=scalar_type, value=float(value))
        )

    def int_const(self, value: int, scalar_type: IntType) -> ValueId:
        return self._global(("int_const", value, scalar_type), MirIntConst(scalar_type=scalar_type, value=value))

    def bool_const(self, value: bool, scalar_type: BoolType) -> ValueId:
        return self._global(
            ("bool_const", bool(value), scalar_type), MirBoolConst(scalar_type=scalar_type, value=bool(value))
        )

    def operation(
        self,
        operator: HardwareOperator,
        operands: list[ValueId],
        operand_conditioners: list[PortConditioner],
        output_port: int = 0,
        output_conditioner: PortConditioner | None = None,
        immediates: tuple[int, ...] = (),
    ) -> ValueId:
        """
        Append a hardware-operator use producing the ``output_port``-th result, interned within the current block.
        The value's resource family follows the tapped port's type; operands are type-checked against the operator
        signature and may reference either resource family. ``output_conditioner`` defaults to the tapped port's
        identity conditioner. ``immediates`` carries the per-firing immediate values (e.g. a rounding mode). The intern
        key includes the tapped port, both conditioner sides, and the immediates, so two relations over one comparator
        firing -- or two rounding modes over one operand -- stay distinct values while identical taps collapse.
        """
        signature = operator.signature
        assert len(operands) == signature.arity
        assert len(operand_conditioners) == signature.arity
        assert all(
            self._type_of(operand) == expected
            for operand, expected in zip(operands, signature.operand_types, strict=True)
        )
        if output_conditioner is None:
            output_conditioner = identity_conditioner(signature.result_types[output_port])
        key = (
            self.current_block,
            operator,
            tuple(operands),
            tuple(operand_conditioners),
            output_port,
            output_conditioner,
            immediates,
        )
        vid = self._block_intern.get(key)
        if vid is None:
            vid = self._fresh(
                MirOperation(
                    operator=operator,
                    operands=list(operands),
                    operand_conditioners=list(operand_conditioners),
                    output_port=output_port,
                    output_conditioner=output_conditioner,
                    immediates=immediates,
                )
            )
            self._block_intern[key] = vid
            self._blocks[self.current_block].operations.append(vid)
        return vid

    def _check_phi_arms(self, scalar_type: ScalarType, arms: list[tuple[BlockId, ValueId, PortConditioner]]) -> None:
        """A mistyped arm escapes the conditioner check: the wrong family picks its matching conditioner too."""
        assert all(self._type_of(value) == scalar_type for _pred, value, _conditioner in arms)

    def phi(self, scalar_type: ScalarType, arms: list[tuple[BlockId, ValueId, PortConditioner]]) -> ValueId:
        self._check_phi_arms(scalar_type, arms)
        vid = self._fresh(MirPhi(scalar_type=scalar_type, arms=tuple(arms)))
        self._blocks[self.current_block].phis.append(vid)
        return vid

    def open_phi(self, scalar_type: ScalarType, entry_arm: tuple[BlockId, ValueId, PortConditioner]) -> ValueId:
        """
        Create a loop-header phi with only its entry arm; the latch arm is supplied later by set_phi_arms (the back
        edge references a body value defined after the header in the lowering order).
        """
        return self.phi(scalar_type, [entry_arm])

    def set_phi_arms(self, phi: ValueId, arms: list[tuple[BlockId, ValueId, PortConditioner]]) -> None:
        """Replace a phi's arms (closes a loop-header phi opened by open_phi once the latch value is lowered)."""
        node = self._nodes[phi]
        assert isinstance(node, MirPhi)
        self._check_phi_arms(node.scalar_type, arms)
        self._nodes[phi] = MirPhi(scalar_type=node.scalar_type, arms=tuple(arms))

    def float_output(self, name: str, value: ValueId, conditioner: FloatSignControl = FloatSignControl()) -> None:
        assert isinstance(self._type_of(value), FloatType)
        self._outputs.append(MirFloatOutput(name, value, conditioner))

    def int_output(self, name: str, value: ValueId, conditioner: IntIdentity = IntIdentity()) -> None:
        assert isinstance(self._type_of(value), IntType)
        self._outputs.append(MirIntOutput(name, value, conditioner))

    def bool_output(self, name: str, value: ValueId, conditioner: BoolInversion = BoolInversion()) -> None:
        assert isinstance(self._type_of(value), BoolType)
        self._outputs.append(MirBoolOutput(name, value, conditioner))

    def float_state_slot(
        self,
        name: str,
        reset_value: float,
        live_out: ValueId,
        conditioner: FloatSignControl = FloatSignControl(),
    ) -> None:
        assert isinstance(self._type_of(live_out), FloatType)
        self._state_slots.append(MirFloatStateSlot(name, float(reset_value), live_out, conditioner))

    def int_state_slot(
        self, name: str, reset_value: int, live_out: ValueId, conditioner: IntIdentity = IntIdentity()
    ) -> None:
        assert isinstance(self._type_of(live_out), IntType)
        if not self._int_format.fits(reset_value):  # the slot has no type of its own, so the machine format rules
            raise UnsupportedConstruct(f"slot {name!r} reset {reset_value} does not fit {self._int_format}")
        self._state_slots.append(MirIntStateSlot(name, reset_value, live_out, conditioner))

    def bool_state_slot(
        self, name: str, reset_value: bool, live_out: ValueId, conditioner: BoolInversion = BoolInversion()
    ) -> None:
        assert isinstance(self._type_of(live_out), BoolType)
        self._state_slots.append(MirBoolStateSlot(name, bool(reset_value), live_out, conditioner))

    def finish(self) -> Mir:
        if not self._blocks:
            raise RuntimeError("cannot finish a MIR with no blocks")
        blocks: list[MirBlock] = []
        for bid, ub in enumerate(self._blocks):
            if ub.terminator is None:
                raise RuntimeError(f"MIR block {bid} was not sealed with a terminator")
            blocks.append(MirBlock(bid, tuple(ub.phis), tuple(ub.operations), ub.terminator))
        return Mir(
            float_format=self._float_format,
            int_format=self._int_format,
            nodes=dict(self._nodes),
            blocks=blocks,
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
            state_slots=list(self._state_slots),
        )
