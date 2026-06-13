"""HIR data model: an SSA value DAG arranged into a control-flow graph of basic blocks."""

from dataclasses import dataclass

from ._const import BoolConst, Const, FloatConst
from ._operators import Operator
from ._types import BoolType, FloatType, Type

type ValueId = int
type BlockId = int


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
    same attribute before it is first rewritten shares one value. A state read is an entry-block value: it is resident
    from the start of the initiation and therefore dominates every block.
    """

    slot: str
    type: Type


@dataclass(frozen=True, slots=True)
class Phi:
    """
    An SSA merge at a block's entry: one ``(predecessor_block, value)`` arm per incoming control edge. A phi is the one
    node permitted to reference values that do not dominate its own block -- each arm value is live-out of its
    predecessor. Phis are never interned; each merge is a distinct value.
    """

    type: Type
    arms: tuple[tuple[BlockId, ValueId], ...]


type Node = InPort | Const | Operation | StateRead | Phi


@dataclass(frozen=True, slots=True)
class Jump:
    """Unconditional control transfer to ``target``."""

    target: BlockId


@dataclass(frozen=True, slots=True)
class Branch:
    """Conditional control transfer: to ``if_true`` when ``cond`` (a BoolType value) is true, else ``if_false``."""

    cond: ValueId
    if_true: BlockId
    if_false: BlockId


@dataclass(frozen=True, slots=True)
class Ret:
    """Commit the persistent state-writes and outputs and finish the initiation. The sole function exit."""


type Terminator = Jump | Branch | Ret


def predecessors(blocks: list["Block"]) -> dict[BlockId, set[BlockId]]:
    """Per block, the set of CFG predecessor block ids (the one edge walk shared by validation and passes)."""
    preds: dict[BlockId, set[BlockId]] = {block.id: set() for block in blocks}
    for block in blocks:
        match block.terminator:
            case Jump(target=target):
                preds[target].add(block.id)
            case Branch(if_true=if_true, if_false=if_false):
                preds[if_true].add(block.id)
                preds[if_false].add(block.id)
            case Ret():
                pass
    return preds


@dataclass(frozen=True, slots=True)
class Block:
    """
    One basic block: its entry phis, its straight-line operations in evaluation order, and its terminator. Inputs,
    constants, and state reads are entry-global pure values and do not appear in any block's ``operations`` list.
    """

    id: BlockId
    phis: tuple[ValueId, ...]
    operations: tuple[ValueId, ...]
    terminator: Terminator


@dataclass(frozen=True, slots=True)
class OutputPort:
    name: str
    value: ValueId


@dataclass(frozen=True, slots=True)
class StateSlot:
    """
    A persistent state register backing a written instance attribute. ``reset_value`` is the typed snapshot taken from
    the instance at synthesis time and loaded at module reset; ``live_out`` is the value the attribute holds at the
    function exit (often a phi merging the value across paths) and that must reside in the slot's register at the
    initiation boundary. Observability is not a slot property: a public attribute is exposed by a separate
    ``state_<attr>`` output port that the frontend emits alongside the slot.
    """

    name: str
    reset_value: Const
    live_out: ValueId


@dataclass(frozen=True, slots=True)
class Hir:
    """
    A complete HIR: the value DAG, the CFG of basic blocks (``blocks[0]`` is the entry), ordered inputs, named
    outputs, and persistent state slots. The frontend emits structured, reducible control flow with a single exit.
    """

    nodes: dict[ValueId, Node]
    blocks: list[Block]
    input_ids: list[ValueId]
    outputs: list[OutputPort]
    state_slots: list[StateSlot]

    @property
    def entry(self) -> BlockId:
        return self.blocks[0].id

    def input_names(self) -> list[str]:
        names: list[str] = []
        for vid in self.input_ids:
            node = self.nodes[vid]
            assert isinstance(node, InPort)
            names.append(node.name)
        return names


@dataclass
class _BlockUnderConstruction:
    phis: list[ValueId]
    operations: list[ValueId]
    terminator: Terminator | None


class HirBuilder:
    """
    Builds an :class:`Hir`. The first :meth:`block` is the entry. Inputs, constants, and state reads are entry-global
    (constants and state reads are interned so identical ones share an id); operations are interned within their block,
    so an identical expression evaluated in two sibling arms stays two distinct values (only the taken arm runs).
    Phis are never interned. A block is sealed by :meth:`jump` / :meth:`branch` / :meth:`ret`.
    """

    def __init__(self) -> None:
        self._nodes: dict[ValueId, Node] = {}
        self._global_intern: dict[Node, ValueId] = {}  # constants and state reads (entry-global)
        self._block_intern: dict[tuple[BlockId, Node], ValueId] = {}  # operations, scoped to their block
        self._blocks: list[_BlockUnderConstruction] = []
        self._cur: BlockId | None = None
        self._input_ids: list[ValueId] = []
        self._outputs: list[OutputPort] = []
        self._state_slots: list[StateSlot] = []

    # -- block management ------------------------------------------------------------------------------------------

    def block(self) -> BlockId:
        """Create a fresh empty block and return its id; the first one created is the entry. Does not move position."""
        bid = len(self._blocks)
        self._blocks.append(_BlockUnderConstruction(phis=[], operations=[], terminator=None))
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

    def set_terminator(self, block: BlockId, terminator: Terminator) -> None:
        self._blocks[block].terminator = terminator

    def jump(self, target: BlockId) -> None:
        self.set_terminator(self.current_block, Jump(target))

    def branch(self, cond: ValueId, if_true: BlockId, if_false: BlockId) -> None:
        if not isinstance(self._type_of(cond), BoolType):
            raise ValueError("a branch condition must be a boolean value")
        self.set_terminator(self.current_block, Branch(cond, if_true, if_false))

    def ret(self) -> None:
        self.set_terminator(self.current_block, Ret())

    # -- value construction ----------------------------------------------------------------------------------------

    def _fresh(self, node: Node) -> ValueId:
        vid = len(self._nodes)
        self._nodes[vid] = node
        return vid

    def _global(self, node: Node) -> ValueId:
        vid = self._global_intern.get(node)
        if vid is None:
            vid = self._fresh(node)
            self._global_intern[node] = vid
        return vid

    def _type_of(self, vid: ValueId) -> Type:
        node = self._nodes[vid]
        match node:
            case InPort(type=type):
                return type
            case StateRead(type=type):
                return type
            case Phi(type=type):
                return type
            case Const():
                return node.type
            case Operation():
                return node.type
        raise TypeError(f"HIR node {vid} has no semantic type")

    def type_of(self, vid: ValueId) -> Type:
        """The semantic type of a value, used by the frontend when merging environments into phis."""
        return self._type_of(vid)

    def input(self, name: str, type: Type) -> ValueId:
        # Input ports are never interned: each parameter is a distinct, ordered port.
        vid = self._fresh(InPort(name, type))
        self._input_ids.append(vid)
        return vid

    def float_input(self, name: str) -> ValueId:
        return self.input(name, FloatType())

    def bool_input(self, name: str) -> ValueId:
        return self.input(name, BoolType())

    def state_read(self, slot: str, type: Type) -> ValueId:
        # Interned globally: repeated live-in reads of a slot share one value, resident from the initiation start.
        return self._global(StateRead(slot, type))

    def float_state_read(self, slot: str) -> ValueId:
        return self.state_read(slot, FloatType())

    def bool_state_read(self, slot: str) -> ValueId:
        return self.state_read(slot, BoolType())

    def state_slot(self, name: str, reset_value: Const, live_out: ValueId) -> None:
        self._state_slots.append(StateSlot(name, reset_value, live_out))

    def const_node(self, const: Const) -> ValueId:
        return self._global(const)

    def float_const(self, value: float) -> ValueId:
        return self.const_node(FloatConst(float(value)))

    def bool_const(self, value: bool) -> ValueId:
        return self.const_node(BoolConst(bool(value)))

    def operation(self, operator: Operator, operands: list[ValueId]) -> ValueId:
        signature = operator.signature
        if len(operands) != signature.arity:
            raise ValueError(f"{operator.mnemonic} expects {signature.arity} operand(s), got {len(operands)}")
        operand_types = tuple(self._type_of(operand) for operand in operands)
        if operand_types != signature.operand_types:
            raise ValueError(f"{operator.mnemonic} expects operands of {signature.operand_types}, got {operand_types}")
        node = Operation(operator, tuple(operands))
        key = (self.current_block, node)
        vid = self._block_intern.get(key)
        if vid is None:
            vid = self._fresh(node)
            self._block_intern[key] = vid
            self._blocks[self.current_block].operations.append(vid)
        return vid

    def phi(self, type: Type, arms: list[tuple[BlockId, ValueId]]) -> ValueId:
        for _, arm in arms:
            if arm in self._nodes and self._type_of(arm) != type:
                raise ValueError(f"phi arm {arm} has type {self._type_of(arm)}, expected {type}")
        vid = self._fresh(Phi(type, tuple(arms)))
        self._blocks[self.current_block].phis.append(vid)
        return vid

    def open_phi(self, type: Type, entry_arm: tuple[BlockId, ValueId]) -> ValueId:
        """
        Create a loop-header phi carrying only its entry (preheader) arm; the latch (back-edge) arm is supplied later
        by :meth:`set_phi_arms`, once the loop body -- which references this phi as the loop-carried value -- has been
        lowered. This is the one forward reference in HIR construction; ``finish`` validates that the phi was closed.
        """
        return self.phi(type, [entry_arm])

    def set_phi_arms(self, phi: ValueId, arms: list[tuple[BlockId, ValueId]]) -> None:
        """Replace a phi's arms (used to close a loop-header phi opened by :meth:`open_phi` once the latch is known)."""
        node = self._nodes[phi]
        if not isinstance(node, Phi):
            raise ValueError(f"value {phi} is not a phi")
        for _, arm in arms:
            if arm in self._nodes and self._type_of(arm) != node.type:
                raise ValueError(f"phi arm {arm} has type {self._type_of(arm)}, expected {node.type}")
        self._nodes[phi] = Phi(node.type, tuple(arms))

    def output(self, name: str, value: ValueId) -> None:
        self._outputs.append(OutputPort(name, value))

    def finish(self) -> Hir:
        if not self._blocks:
            raise RuntimeError("cannot finish an HIR with no blocks")
        blocks: list[Block] = []
        for bid, ub in enumerate(self._blocks):
            if ub.terminator is None:
                raise RuntimeError(f"block {bid} was not sealed with a terminator")
            blocks.append(Block(bid, tuple(ub.phis), tuple(ub.operations), ub.terminator))
        self._validate_phi_predecessors(blocks)
        return Hir(
            nodes=dict(self._nodes),
            blocks=blocks,
            input_ids=list(self._input_ids),
            outputs=list(self._outputs),
            state_slots=list(self._state_slots),
        )

    def _validate_phi_predecessors(self, blocks: list[Block]) -> None:
        """
        Every phi must carry exactly one arm per CFG predecessor of its block. This catches a loop-header phi opened
        by :meth:`open_phi` and never closed (a back-edge arm left missing) and a stale or invented arm predecessor --
        construction bugs that would otherwise produce a malformed merge.
        """
        preds = predecessors(blocks)
        for block in blocks:
            for phi_id in block.phis:
                phi = self._nodes[phi_id]
                assert isinstance(phi, Phi)
                arm_preds = sorted(pred for pred, _ in phi.arms)
                if arm_preds != sorted(preds[block.id]):
                    raise RuntimeError(
                        f"phi {phi_id} in block {block.id} has arms for predecessors {arm_preds}, "
                        f"expected {sorted(preds[block.id])}"
                    )
