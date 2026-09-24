"""Selected mid-level IR (MIR): concrete hardware operators with typed scalar sidebands, arranged into a CFG."""

import math
from dataclasses import dataclass, field
from typing import assert_never

from .._operators import (
    HardwareOperator,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
    has_sign_control,
    identity_conditioner,
)
from .._errors import UnsupportedConstruct
from .._type import BoolType, FloatFormat, FloatType, IntFormat, IntType, ScalarType
from .._util import BlockId, ValueId, reverse_postorder_of


@dataclass(frozen=True, slots=True)
class MirInput:
    name: str
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class MirStateRead:
    """The slot's live-in: its register content at the initiation start."""

    name: str
    scalar_type: ScalarType


def degrades(value: float, fmt: FloatFormat) -> bool:
    """
    Whether a literal encodes to zero or infinity and so is not the number written. Asked of the encoded bits: a coarse
    mantissa can round a magnitude up past the double range, so `decode` saturates for a target value that is finite.
    """
    if not math.isfinite(value) or value == 0.0:
        return False
    bits = fmt.encode(value)
    return bits == 0 or not fmt.is_finite(bits)


def refuse_degrading(value: float, fmt: FloatFormat, what: str, remedy: str = "widen wexp or rescale") -> None:
    """`remedy` is the way out the caller knows of beyond the format itself."""
    if degrades(value, fmt):
        raise UnsupportedConstruct(f"{what} {value!r} degrades to {fmt.decode(fmt.encode(value))!r} in {fmt}; {remedy}")


def _refuse_unholdable(value: float | int | bool, scalar_type: ScalarType, what: str) -> None:
    """
    A value the machine must hold is refused where its encoding is not the number written; saturation is what the
    arithmetic does, never what a literal does.
    """
    match scalar_type:
        case FloatType(fmt=fmt):
            assert type(value) is float
            refuse_degrading(value, fmt, what)
        case IntType(fmt=fmt):
            assert type(value) is int
            if not fmt.fits(value):
                raise UnsupportedConstruct(f"{what} {value} does not fit {fmt}; raise wint_min")
        case BoolType():
            assert type(value) is bool


@dataclass(frozen=True, slots=True)
class MirConst:
    scalar_type: ScalarType
    value: float | int | bool

    def __post_init__(self) -> None:
        _refuse_unholdable(self.value, self.scalar_type, "constant")
        # ZKF has no negative zero, and neither does HIR; normalizing here keeps the pool from ever holding one.
        if isinstance(self.scalar_type, FloatType):
            object.__setattr__(self, "value", self.value + 0.0)


def _check_unconditioned_operands(operator: HardwareOperator) -> None:
    """
    The declaration asserts what the operator READS, which nothing can derive, so these three rules close the ways
    it can be wrong. They run per operation because an operator has no constructor of its own to hang them on.
    """
    declined = operator.unconditioned_operands
    if not declined:
        return
    signature = operator.signature
    # Declining a sideband the port never had would say nothing; only a float port carries one to decline.
    assert all(has_sign_control(signature.operand_types[position]) for position in declined)
    # An ERROR sideband is an observable output that "unchanged across every RESULT port" does not cover, so the
    # claim would be unsound rather than merely unverified; an operator raising one may decline nothing.
    assert not (isinstance(operator, PooledHardwareOperator) and operator.error_ports)
    # A firing may exchange a commutative operator's operands together with their conditioners, so a claim covering
    # one of them would migrate onto the other. `swap_output_permutation` maps RESULT ports and cannot say this.
    assert not operator.is_commutative or declined == frozenset(range(signature.arity))


def _check_conditioner(conditioner: PortConditioner, port_type: ScalarType) -> None:
    """Port conditioner must be the type's own: sign control for floats, inversion for bools, identity for ints."""
    assert isinstance(conditioner, type(identity_conditioner(port_type)))


@dataclass(frozen=True, slots=True)
class MirOperation:
    """
    A selected hardware-operator use producing ONE value: the `output_port`-th result, conditioned by
    `output_conditioner`. Operations sharing one block, operator, operands, and operand conditioners while tapping
    DISTINCT output ports fuse into a single firing at LIR build -- a multi-output module computes all its results at
    once. The operation belongs to the resource family of its tapped port's type; operands may reference either family
    (a comparison reads float operands and produces booleans; the bool->float cast the reverse).
    """

    operator: HardwareOperator
    operands: tuple[ValueId, ...]
    operand_conditioners: tuple[PortConditioner, ...]
    output_port: int
    output_conditioner: PortConditioner
    immediates: tuple[int, ...]  # per-firing immediate values, aligned with operator.immediate_ports
    scalar_type: ScalarType = field(init=False, compare=False)

    def __post_init__(self) -> None:
        signature = self.operator.signature
        ports = self.operator.immediate_ports
        assert not isinstance(self.operator, InlineHardwareOperator) or len(signature.result_types) == 1
        assert len(self.immediates) == len(ports)
        assert all(0 <= value < (1 << port.width) for value, port in zip(self.immediates, ports, strict=True))
        assert len(self.operands) == signature.arity
        assert len(self.operand_conditioners) == signature.arity
        _check_unconditioned_operands(self.operator)
        for position, (conditioner, operand_type) in enumerate(
            zip(self.operand_conditioners, signature.operand_types, strict=True)
        ):
            _check_conditioner(conditioner, operand_type)
            # The invariant the whole design rests on: a declined sideband leaves nothing to bind. Stated over the
            # declaration rather than over `conditions_operand`, which also answers no for the boolean and integer
            # ports -- and a boolean port carries a real inversion.
            assert position not in self.operator.unconditioned_operands or conditioner.is_identity
        assert 0 <= self.output_port < len(signature.result_types)
        _check_conditioner(self.output_conditioner, signature.result_types[self.output_port])
        object.__setattr__(self, "scalar_type", signature.result_types[self.output_port])


@dataclass(frozen=True, slots=True)
class MirOutput:
    name: str
    value: ValueId
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class MirStateSlot:
    """A persistent slot register holding `live_out` at the transaction boundary."""

    name: str
    reset_value: float | int | bool
    live_out: ValueId
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class MirPhi:
    """
    An SSA merge at a block's entry: one `(predecessor_block, value, conditioner)` arm per incoming edge, of one
    scalar type. The arm conditioner is the type's own sideband, applied when the arm value is installed: a folded
    sign control on a float arm (`y = -x` on one branch), an optional inversion on a boolean arm (`f = not g`),
    and nothing at all on an integer arm, which has no free sideband to fold into.
    """

    scalar_type: ScalarType
    arms: tuple[tuple[BlockId, ValueId, PortConditioner], ...]

    def __post_init__(self) -> None:
        for _pred, _value, conditioner in self.arms:
            _check_conditioner(conditioner, self.scalar_type)


type MirNode = MirInput | MirStateRead | MirConst | MirOperation | MirPhi


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
    """`operations` is straight-line in evaluation order; `phis` merge at block entry."""

    id: BlockId
    phis: tuple[ValueId, ...]
    operations: tuple[ValueId, ...]
    terminator: MirTerminator


@dataclass(frozen=True, slots=True)
class Mir:
    """A selected graph arranged into a CFG of basic blocks; `blocks[0]` is the entry."""

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
        """The id of the sole function-exit block (a kernel has exactly one `MirRet`)."""
        return next(block.id for block in self.blocks if isinstance(block.terminator, MirRet))


def successors(block: MirBlock) -> list[BlockId]:
    match block.terminator:
        case MirJump(target=target):
            return [target]
        case MirBranch(if_true=if_true, if_false=if_false):
            return [if_true, if_false]
        case MirRet():
            return []
        case _:
            assert_never(block.terminator)


def reverse_postorder(mir: Mir) -> list[BlockId]:
    return reverse_postorder_of(mir.entry, {block.id: successors(block) for block in mir.blocks})


class _MirBankView:
    """
    The phi-arm subset and the per-block operation listing of one bank; `operation_nodes` is filled once since the
    scheduler reads it per operation.
    """

    __slots__ = ()

    nodes: dict[ValueId, MirNode]
    operation_nodes: dict[ValueId, MirOperation]

    def __post_init__(self) -> None:
        operations = {vid: node for vid, node in self.nodes.items() if isinstance(node, MirOperation)}
        object.__setattr__(self, "operation_nodes", operations)

    @property
    def phi_nodes(self) -> dict[ValueId, MirPhi]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirPhi)}

    @property
    def state_read_nodes(self) -> dict[ValueId, MirStateRead]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirStateRead)}

    @property
    def const_nodes(self) -> dict[ValueId, MirConst]:
        return {vid: node for vid, node in self.nodes.items() if isinstance(node, MirConst)}

    def block_operations(self, block: MirBlock) -> list[ValueId]:
        """The bank's operation ids defined in `block`, in evaluation order."""
        return [vid for vid in block.operations if vid in self.operation_nodes]


def _bank_nodes(mir: Mir, wide: bool) -> dict[ValueId, MirNode]:
    """A bank is physical, not a scalar family, so it admits every node by the width of its type alone."""
    return {vid: node for vid, node in mir.nodes.items() if node.scalar_type.is_wide == wide}


@dataclass(frozen=True, slots=True)
class MirWideView(_MirBankView):
    """
    The wide data bank narrowed out of a MIR graph, carrying the shared CFG so scheduling runs per block. It holds
    floats and integers alike.
    """

    nodes: dict[ValueId, MirNode]
    blocks: list[MirBlock]
    entry: BlockId
    input_ids: list[ValueId]
    outputs: list[MirOutput]
    state_slots: list[MirStateSlot]
    float_format: FloatFormat
    int_format: IntFormat
    operation_nodes: dict[ValueId, MirOperation] = field(init=False, compare=False)

    def scalar_type_of(self, vid: ValueId) -> FloatType | IntType:
        """Which family a wide value belongs to: the bank is physical, so only the value itself names its type."""
        scalar_type = self.nodes[vid].scalar_type
        assert isinstance(scalar_type, (FloatType, IntType))
        return scalar_type

    @classmethod
    def from_mir(cls, mir: Mir) -> MirWideView:
        nodes = _bank_nodes(mir, wide=True)
        machine = {FloatType(mir.float_format), IntType(mir.int_format)}
        assert all(node.scalar_type in machine for node in nodes.values())
        return cls(
            nodes=nodes,
            blocks=mir.blocks,
            entry=mir.entry,
            input_ids=[vid for vid in mir.input_ids if vid in nodes],
            outputs=[out for out in mir.outputs if out.value in nodes],
            state_slots=[slot for slot in mir.state_slots if slot.live_out in nodes],
            float_format=mir.float_format,
            int_format=mir.int_format,
        )


@dataclass(frozen=True, slots=True)
class MirBoolView(_MirBankView):
    """The boolean bank narrowed out of a MIR graph, carrying the shared CFG."""

    nodes: dict[ValueId, MirNode]
    blocks: list[MirBlock]
    entry: BlockId
    input_ids: list[ValueId]
    outputs: list[MirOutput]
    state_slots: list[MirStateSlot]
    operation_nodes: dict[ValueId, MirOperation] = field(init=False, compare=False)

    @classmethod
    def from_mir(cls, mir: Mir) -> MirBoolView:
        nodes = _bank_nodes(mir, wide=False)
        return cls(
            nodes=nodes,
            blocks=mir.blocks,
            entry=mir.entry,
            input_ids=[vid for vid in mir.input_ids if vid in nodes],
            outputs=[out for out in mir.outputs if out.value in nodes],
            state_slots=[slot for slot in mir.state_slots if slot.live_out in nodes],
        )


@dataclass
class _MirBlockUC:
    phis: list[ValueId]
    operations: list[ValueId]
    terminator: MirTerminator | None


class MirBuilder:
    """
    Builds a selected CFG. The first block is the entry. Inputs, constants, and state reads are entry-global
    (constants and state reads interned); operations are interned within their block; phis are never interned. A block
    is sealed by jump / branch / ret.
    """

    def __init__(self, float_format: FloatFormat, int_format: IntFormat) -> None:
        self._float_format = float_format
        self._int_format = int_format
        self._nodes: dict[ValueId, MirNode] = {}
        self._global_intern: dict[MirStateRead | MirConst, ValueId] = {}
        self._block_intern: dict[tuple[BlockId, MirOperation], ValueId] = {}
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
        assert self._cur is not None
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

    def _global(self, node: MirStateRead | MirConst) -> ValueId:
        vid = self._global_intern.get(node)
        if vid is None:
            vid = self._fresh(node)
            self._global_intern[node] = vid
        return vid

    def _type_of(self, vid: ValueId) -> ScalarType:
        return self._nodes[vid].scalar_type

    def input(self, name: str, scalar_type: ScalarType) -> ValueId:
        vid = self._fresh(MirInput(name, scalar_type))
        self._input_ids.append(vid)
        return vid

    def state_read(self, name: str, scalar_type: ScalarType) -> ValueId:
        return self._global(MirStateRead(name, scalar_type))

    def const(self, value: float | int | bool, scalar_type: ScalarType) -> ValueId:
        return self._global(MirConst(scalar_type, value))

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
        Append a hardware-operator use producing the `output_port`-th result, interned within the current block.
        The value's resource family follows the tapped port's type; operands are type-checked against the operator
        signature and may reference either resource family. `output_conditioner` defaults to the tapped port's
        identity conditioner. `immediates` carries the per-firing immediate values (e.g. a rounding mode). Interned by
        the node, so two relations over one comparator firing -- or two rounding modes over one operand -- stay
        distinct values while identical taps collapse.
        """
        signature = operator.signature
        assert len(operands) == signature.arity
        assert len(operand_conditioners) == signature.arity
        assert all(
            self._type_of(operand) == expected
            for operand, expected in zip(operands, signature.operand_types, strict=True)
        )
        # Normalized BEFORE interning, so every conditioned view of one unconditioned operand interns to a
        # single value. Discarding the transform is what the declaration licenses: it cannot be observed.
        operand_conditioners = [
            identity_conditioner(operand_type) if position in operator.unconditioned_operands else conditioner
            for position, (conditioner, operand_type) in enumerate(
                zip(operand_conditioners, signature.operand_types, strict=True)
            )
        ]
        if output_conditioner is None:
            output_conditioner = identity_conditioner(signature.result_types[output_port])
        node = MirOperation(
            operator, tuple(operands), tuple(operand_conditioners), output_port, output_conditioner, immediates
        )
        key = (self.current_block, node)
        vid = self._block_intern.get(key)
        if vid is None:
            vid = self._fresh(node)
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

    def output(self, name: str, value: ValueId, conditioner: PortConditioner | None = None) -> None:
        scalar_type = self._type_of(value)
        conditioner = identity_conditioner(scalar_type) if conditioner is None else conditioner
        _check_conditioner(conditioner, scalar_type)
        self._outputs.append(MirOutput(name, value, conditioner))

    def state_slot(
        self, name: str, reset_value: float | int | bool, live_out: ValueId, conditioner: PortConditioner | None = None
    ) -> None:
        scalar_type = self._type_of(live_out)
        conditioner = identity_conditioner(scalar_type) if conditioner is None else conditioner
        _check_conditioner(conditioner, scalar_type)
        _refuse_unholdable(reset_value, scalar_type, f"state slot {name!r} reset")
        self._state_slots.append(MirStateSlot(name, reset_value, live_out, conditioner))

    def finish(self) -> Mir:
        assert self._blocks
        blocks: list[MirBlock] = []
        for bid, ub in enumerate(self._blocks):
            assert ub.terminator is not None, f"block {bid} is not sealed"
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
