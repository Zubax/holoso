"""Selected mid-level IR (MIR): concrete hardware operators with typed scalar sidebands, arranged into a CFG."""

from dataclasses import dataclass

from .._hir import ValueId
from .._operators import (
    BoolInversion,
    FloatSignControl,
    HardwareOperator,
    InlineHardwareOperator,
    PortConditioner,
    identity_conditioner,
)
from .._errors import UnsupportedConstruct
from .._type import BoolType, FloatFormat, FloatType, ScalarType

type BlockId = int


@dataclass(frozen=True, slots=True)
class MirInput:
    """A typed module input value."""

    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirStateRead:
    """A typed read of a persistent state slot's live-in value (the slot's register content at the initiation start)."""

    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirConst:
    """A typed scalar constant value."""

    scalar_type: ScalarType


def _check_conditioner(conditioner: PortConditioner, port_type: ScalarType, role: str) -> None:
    """A port's conditioner must be the type's own: a sign control on a float port, an inversion on a boolean one."""
    expected = type(identity_conditioner(port_type))  # the single owner of the type -> conditioner-class mapping
    if not isinstance(conditioner, expected):
        raise TypeError(f"{role} conditioner must be {expected.__name__} for {port_type!r}, got {conditioner!r}")


@dataclass(frozen=True, slots=True)
class MirOperation:
    """
    A selected hardware-operator use producing ONE value: the ``output_port``-th result of ``operator``, conditioned
    by ``output_conditioner``. Every port carries its type's conditioner (a folded sign control on a float port, an
    optional inversion on a boolean port). Operations sharing one block, operator, operands, and operand conditioners
    while tapping DISTINCT output ports fuse into a single firing at LIR build -- a multi-output module computes all
    its results at once. The operation belongs to the resource family of its tapped port's type (float or bool); the
    views slice it by that type. Operands may reference either resource family (a comparison reads float operands and
    produces booleans; the bool->float cast the reverse).
    """

    operator: HardwareOperator
    operands: list[ValueId]
    operand_conditioners: list[PortConditioner]
    output_port: int
    output_conditioner: PortConditioner

    def __post_init__(self) -> None:
        signature = self.operator.signature
        if isinstance(self.operator, InlineHardwareOperator) and len(signature.result_types) != 1:
            raise TypeError(f"inline operator {self.operator.mnemonic} must have exactly one result port")
        if len(self.operands) != signature.arity:
            raise ValueError(f"{self.operator.mnemonic} expects {signature.arity} operand(s), got {len(self.operands)}")
        if len(self.operand_conditioners) != signature.arity:
            raise ValueError(
                f"{self.operator.mnemonic} expects {signature.arity} conditioner(s),"
                f" got {len(self.operand_conditioners)}"
            )
        for conditioner, operand_type in zip(self.operand_conditioners, signature.operand_types, strict=True):
            _check_conditioner(conditioner, operand_type, "operand")
        if not 0 <= self.output_port < len(signature.result_types):
            raise ValueError(
                f"{self.operator.mnemonic} has {len(signature.result_types)} output port(s);"
                f" port {self.output_port} does not exist"
            )
        _check_conditioner(self.output_conditioner, signature.result_types[self.output_port], "output")

    @property
    def scalar_type(self) -> ScalarType:
        return self.operator.signature.result_types[self.output_port]


@dataclass(frozen=True, slots=True)
class MirOutput:
    """A named module output driven by an internal value."""

    name: str
    value: ValueId


@dataclass(frozen=True, slots=True)
class MirStateSlot:
    """A persistent state slot: a register holding ``live_out`` at the boundary, reset to ``reset_value``."""

    name: str
    reset_value: float
    live_out: ValueId


@dataclass(frozen=True, slots=True)
class MirFloatInput(MirInput):
    """A floating-point module input port."""

    scalar_type: FloatType

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, FloatType):
            raise TypeError(f"MirFloatInput scalar_type must be FloatType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirBoolInput(MirInput):
    """A boolean module input port."""

    scalar_type: BoolType

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, BoolType):
            raise TypeError(f"MirBoolInput scalar_type must be BoolType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirFloatStateRead(MirStateRead):
    """A floating-point read of a persistent state slot's live-in value."""

    scalar_type: FloatType

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, FloatType):
            raise TypeError(f"MirFloatStateRead scalar_type must be FloatType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirFloatConst(MirConst):
    """A floating-point constant."""

    scalar_type: FloatType
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, FloatType):
            raise TypeError(f"MirFloatConst scalar_type must be FloatType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirBoolStateRead(MirStateRead):
    """A boolean read of a persistent state slot's live-in value."""

    scalar_type: BoolType

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, BoolType):
            raise TypeError(f"MirBoolStateRead scalar_type must be BoolType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirBoolConst(MirConst):
    """A boolean constant."""

    scalar_type: BoolType
    value: bool

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, BoolType):
            raise TypeError(f"MirBoolConst scalar_type must be BoolType, got {self.scalar_type!r}")


@dataclass(frozen=True, slots=True)
class MirPhi:
    """
    An SSA merge at a block's entry: one ``(predecessor_block, value, conditioner)`` arm per incoming edge, of one
    scalar type. The arm conditioner is the type's own sideband, applied when the arm value is installed: a folded
    sign control on a float arm (``y = -x`` on one branch), an optional inversion on a boolean arm (``f = not g``).
    """

    scalar_type: ScalarType
    arms: tuple[tuple[BlockId, ValueId, PortConditioner], ...]

    def __post_init__(self) -> None:
        for _pred, _value, conditioner in self.arms:
            _check_conditioner(conditioner, self.scalar_type, "phi arm")


type MirNode = MirInput | MirStateRead | MirConst | MirOperation | MirPhi
type MirFloatNode = MirFloatInput | MirFloatStateRead | MirFloatConst | MirOperation | MirPhi
type MirBoolNode = MirBoolInput | MirBoolStateRead | MirBoolConst | MirPhi | MirOperation


@dataclass(frozen=True, slots=True)
class MirFloatOutput(MirOutput):
    """A floating-point module output with a folded output sign control."""

    sign: FloatSignControl = FloatSignControl()

    def __post_init__(self) -> None:
        if not isinstance(self.sign, FloatSignControl):
            raise TypeError(f"MirFloatOutput sign must be FloatSignControl, got {self.sign!r}")


@dataclass(frozen=True, slots=True)
class MirBoolOutput(MirOutput):
    """A boolean module output with an optional folded inversion on the driven value."""

    inversion: BoolInversion = BoolInversion()

    def __post_init__(self) -> None:
        if not isinstance(self.inversion, BoolInversion):
            raise TypeError(f"MirBoolOutput inversion must be BoolInversion, got {self.inversion!r}")


@dataclass(frozen=True, slots=True)
class MirFloatStateSlot(MirStateSlot):
    """A floating-point persistent state slot with a folded sign control on its live-out value."""

    sign: FloatSignControl = FloatSignControl()

    def __post_init__(self) -> None:
        if not isinstance(self.sign, FloatSignControl):
            raise TypeError(f"MirFloatStateSlot sign must be FloatSignControl, got {self.sign!r}")


@dataclass(frozen=True, slots=True)
class MirBoolStateSlot(MirStateSlot):
    """A boolean persistent state slot with an optional folded inversion on its live-out value."""

    inversion: BoolInversion = BoolInversion()

    def __post_init__(self) -> None:
        if not isinstance(self.reset_value, bool):
            raise TypeError(f"MirBoolStateSlot reset_value must be bool, got {self.reset_value!r}")
        if not isinstance(self.inversion, BoolInversion):
            raise TypeError(f"MirBoolStateSlot inversion must be BoolInversion, got {self.inversion!r}")


@dataclass(frozen=True, slots=True)
class MirJump:
    """Unconditional control transfer to ``target``."""

    target: BlockId


@dataclass(frozen=True, slots=True)
class MirBranch:
    """Conditional control transfer on a boolean value ``cond``."""

    cond: ValueId
    if_true: BlockId
    if_false: BlockId


@dataclass(frozen=True, slots=True)
class MirRet:
    """The sole function exit: commit state-writes and outputs."""


type MirTerminator = MirJump | MirBranch | MirRet


@dataclass(frozen=True, slots=True)
class MirBlock:
    """One basic block: entry phis, straight-line operations in evaluation order, and a terminator."""

    id: BlockId
    phis: tuple[ValueId, ...]
    operations: tuple[ValueId, ...]
    terminator: MirTerminator


@dataclass(frozen=True, slots=True)
class Mir:
    """A selected graph arranged into a CFG of basic blocks (``blocks[0]`` is the entry), ready for scheduling."""

    float_format: FloatFormat
    nodes: dict[ValueId, MirNode]
    blocks: list[MirBlock]
    input_ids: list[ValueId]
    outputs: list[MirOutput]
    state_slots: list[MirStateSlot]

    @property
    def entry(self) -> BlockId:
        return self.blocks[0].id


@dataclass(frozen=True, slots=True)
class MirFloatView:
    """The float resource family narrowed out of a MIR graph, carrying the shared CFG so scheduling runs per block."""

    nodes: dict[ValueId, MirFloatNode]
    blocks: list[MirBlock]
    entry: BlockId
    input_ids: list[ValueId]
    outputs: list[MirFloatOutput]
    state_slots: list[MirFloatStateSlot]
    fmt: FloatFormat

    @property
    def input_nodes(self) -> dict[ValueId, MirFloatInput]:
        return {vid: node for vid in self.input_ids if isinstance(node := self.nodes[vid], MirFloatInput)}

    @property
    def state_read_nodes(self) -> dict[ValueId, MirFloatStateRead]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirFloatStateRead)}

    @property
    def const_nodes(self) -> dict[ValueId, MirFloatConst]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirFloatConst)}

    @property
    def operation_nodes(self) -> dict[ValueId, MirOperation]:
        return {
            vid: node
            for vid, node in self.nodes.items()
            if isinstance(node, MirOperation) and isinstance(node.scalar_type, FloatType)
        }

    @property
    def phi_nodes(self) -> dict[ValueId, MirPhi]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirPhi)}

    def block_operations(self, block: MirBlock) -> list[ValueId]:
        """The float-result operation ids defined in ``block``, in evaluation order."""
        return [vid for vid in block.operations if vid in self.operation_nodes]

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
                case MirFloatStateRead(scalar_type=scalar_type):
                    nodes[vid] = node
                    formats.add(scalar_type.fmt)
                case MirOperation(scalar_type=FloatType() as scalar_type):
                    nodes[vid] = node
                    formats.add(scalar_type.fmt)
                case MirPhi(scalar_type=FloatType() as scalar_type):
                    nodes[vid] = node
                    formats.add(scalar_type.fmt)
                case MirBoolInput() | MirBoolStateRead() | MirBoolConst() | MirPhi() | MirOperation():
                    pass  # the bool resource family (bool state/const/phi and bool-result ops), handled by MirBoolView
                case MirInput():
                    raise UnsupportedConstruct(f"LIR construction does not support MIR input {vid} of this type")
                case MirStateRead():
                    raise UnsupportedConstruct(f"LIR construction does not support MIR state read {vid} of this type")
                case MirConst():
                    raise UnsupportedConstruct(f"LIR construction does not support MIR constant {vid} of this type")
        outputs: list[MirFloatOutput] = []
        for out in mir.outputs:
            if isinstance(out, MirFloatOutput):
                outputs.append(out)
        state_slots = [slot for slot in mir.state_slots if isinstance(slot, MirFloatStateSlot)]
        for vid in mir.input_ids:
            if not isinstance(mir.nodes.get(vid), (MirFloatInput, MirBoolInput)):
                raise ValueError(f"MIR input ID {vid} must reference a MirFloatInput or MirBoolInput")
        input_ids = [vid for vid in mir.input_ids if isinstance(nodes.get(vid), MirFloatInput)]
        unexpected = formats - {mir.float_format}
        if unexpected:
            ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(
                f"LIR requires MIR float values to use configured format {mir.float_format}; got {ordered}"
            )
        return cls(
            nodes=nodes,
            blocks=mir.blocks,
            entry=mir.entry,
            input_ids=input_ids,
            outputs=outputs,
            state_slots=state_slots,
            fmt=mir.float_format,
        )


@dataclass(frozen=True, slots=True)
class MirBoolView:
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
    def phi_nodes(self) -> dict[ValueId, MirPhi]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirPhi)}

    @property
    def operation_nodes(self) -> dict[ValueId, MirOperation]:
        return {
            vid: node
            for vid, node in self.nodes.items()
            if isinstance(node, MirOperation) and isinstance(node.scalar_type, BoolType)
        }

    def block_operations(self, block: MirBlock) -> list[ValueId]:
        """The bool-result operation ids defined in ``block``, in evaluation order."""
        return [vid for vid in block.operations if vid in self.operation_nodes]

    @classmethod
    def from_mir(cls, mir: Mir) -> "MirBoolView":
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

    def __init__(self, float_format: FloatFormat) -> None:
        self._float_format = float_format
        self._nodes: dict[ValueId, MirNode] = {}
        self._global_intern: dict[object, ValueId] = {}
        self._block_intern: dict[object, ValueId] = {}
        self._blocks: list[_MirBlockUC] = []
        self._cur: BlockId | None = None
        self._input_ids: list[ValueId] = []
        self._outputs: list[MirOutput] = []
        self._state_slots: list[MirStateSlot] = []

    # -- block management ------------------------------------------------------------------------------------------

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
        if not isinstance(self._type_of(cond), BoolType):
            raise ValueError("a MIR branch condition must be a boolean value")
        self.set_terminator(self.current_block, MirBranch(cond, if_true, if_false))

    def ret(self) -> None:
        self.set_terminator(self.current_block, MirRet())

    # -- value construction ----------------------------------------------------------------------------------------

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
        raise TypeError(f"MIR node {vid} has no scalar type")

    def float_input(self, name: str, scalar_type: FloatType) -> ValueId:
        vid = self._fresh(MirFloatInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def bool_input(self, name: str, scalar_type: BoolType) -> ValueId:
        vid = self._fresh(MirBoolInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def float_state_read(self, name: str, scalar_type: FloatType) -> ValueId:
        return self._global(("float_state_read", name), MirFloatStateRead(name, scalar_type))

    def bool_state_read(self, name: str, scalar_type: BoolType) -> ValueId:
        return self._global(("bool_state_read", name), MirBoolStateRead(name, scalar_type))

    def float_const(self, value: float, scalar_type: FloatType) -> ValueId:
        return self._global(
            ("float_const", float(value), scalar_type), MirFloatConst(scalar_type=scalar_type, value=float(value))
        )

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
    ) -> ValueId:
        """
        Append a hardware-operator use producing the ``output_port``-th result, interned within the current block.
        The value's resource family follows the tapped port's type; operands are type-checked against the operator
        signature and may reference either resource family. ``output_conditioner`` defaults to the tapped port's
        identity conditioner. The intern key includes the tapped port and both conditioner sides, so two relations
        over one comparator firing stay distinct values while identical taps collapse.
        """
        signature = operator.signature
        if len(operands) != signature.arity:
            raise ValueError(f"{operator.mnemonic} expects {signature.arity} operand(s), got {len(operands)}")
        if len(operand_conditioners) != signature.arity:
            raise ValueError(
                f"{operator.mnemonic} expects {signature.arity} conditioner(s), got {len(operand_conditioners)}"
            )
        for operand, expected_type in zip(operands, signature.operand_types, strict=True):
            if self._type_of(operand) != expected_type:
                raise ValueError(
                    f"operator {operator.mnemonic} expects operands of {signature.operand_types}, "
                    f"got {tuple(self._type_of(operand) for operand in operands)}"
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
                )
            )
            self._block_intern[key] = vid
            self._blocks[self.current_block].operations.append(vid)
        return vid

    def phi(self, scalar_type: ScalarType, arms: list[tuple[BlockId, ValueId, PortConditioner]]) -> ValueId:
        vid = self._fresh(MirPhi(scalar_type=scalar_type, arms=tuple(arms)))
        self._blocks[self.current_block].phis.append(vid)
        return vid

    def open_phi(self, scalar_type: ScalarType, entry_arm: tuple[BlockId, ValueId, PortConditioner]) -> ValueId:
        """Create a loop-header phi with only its entry arm; the latch arm is supplied later by set_phi_arms (the back
        edge references a body value defined after the header in the lowering order)."""
        return self.phi(scalar_type, [entry_arm])

    def set_phi_arms(self, phi: ValueId, arms: list[tuple[BlockId, ValueId, PortConditioner]]) -> None:
        """Replace a phi's arms (closes a loop-header phi opened by open_phi once the latch value is lowered)."""
        node = self._nodes[phi]
        if not isinstance(node, MirPhi):
            raise ValueError(f"value {phi} is not a phi")
        self._nodes[phi] = MirPhi(scalar_type=node.scalar_type, arms=tuple(arms))

    def float_output(self, name: str, value: ValueId, sign: FloatSignControl = FloatSignControl()) -> None:
        if not isinstance(self._type_of(value), FloatType):
            raise ValueError(f"float output {name!r} must be driven by a floating-point value")
        self._outputs.append(MirFloatOutput(name, value, sign))

    def bool_output(self, name: str, value: ValueId, inversion: BoolInversion = BoolInversion()) -> None:
        if not isinstance(self._type_of(value), BoolType):
            raise ValueError(f"bool output {name!r} must be driven by a boolean value")
        self._outputs.append(MirBoolOutput(name, value, inversion))

    def float_state_slot(
        self,
        name: str,
        reset_value: float,
        live_out: ValueId,
        sign: FloatSignControl = FloatSignControl(),
    ) -> None:
        if not isinstance(self._type_of(live_out), FloatType):
            raise ValueError(f"float state slot {name!r} must hold a floating-point value")
        self._state_slots.append(MirFloatStateSlot(name, float(reset_value), live_out, sign))

    def bool_state_slot(
        self, name: str, reset_value: bool, live_out: ValueId, inversion: BoolInversion = BoolInversion()
    ) -> None:
        if not isinstance(self._type_of(live_out), BoolType):
            raise ValueError(f"bool state slot {name!r} must hold a boolean value")
        self._state_slots.append(MirBoolStateSlot(name, bool(reset_value), live_out, inversion))

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
            nodes=dict(self._nodes),
            blocks=blocks,
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
            state_slots=list(self._state_slots),
        )
