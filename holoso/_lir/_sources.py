"""
The register file's steering endpoints, read off a finished Lir: per operand port the ordered sources its read mux
selects among, and per register the ordered sources its write select takes. The Verilog codebooks (the emitter's
per-endpoint tables from opcode to source) number their opcodes by position in these lists, and the steering metric
counts their arms, so the two cannot disagree on what the fabric multiplexes. The operand templates are the same
operands before their bank is allocated, with a value in place of the register the allocation decides; the allocator's
objective resolves them into the write sources the codebooks deduplicate, so it prices exactly the arms the emitter
builds.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .._operators import BoolInversion, InlineHardwareOperator, PortConditioner, WideConditioner
from .._util import ValueId
from ._ir import (
    BoolConstRef,
    BoolOperand,
    BoolRegRef,
    Lir,
    OperatorInstance,
    ReadPort,
    RegRef,
    WideConstRef,
    WideOperand,
    pooled_write_word,
)

type ReadSource = RegRef | WideConstRef


@dataclass(frozen=True, slots=True)
class WideOperandTemplate:
    """A wide operand whose register, if it has one, the wide bank's allocation decides (`ValueId` is the hole)."""

    source: ValueId | WideConstRef
    conditioner: WideConditioner

    @property
    def hole(self) -> ValueId | None:
        return self.source if isinstance(self.source, int) else None

    def substitute(self, fn: Callable[[ValueId], ValueId]) -> "WideOperandTemplate":
        return self if self.hole is None else WideOperandTemplate(fn(self.hole), self.conditioner)

    def resolve(self, register_of: Callable[[ValueId], int]) -> WideOperand:
        source = self.source if isinstance(self.source, WideConstRef) else RegRef(register_of(self.source))
        return WideOperand(source, self.conditioner)


@dataclass(frozen=True, slots=True)
class BoolOperandTemplate:
    """A boolean operand whose register, if any, the boolean bank's allocation decides (`ValueId` is the hole)."""

    source: ValueId | BoolConstRef
    inversion: BoolInversion

    @property
    def hole(self) -> ValueId | None:
        return self.source if isinstance(self.source, int) else None

    def substitute(self, fn: Callable[[ValueId], ValueId]) -> "BoolOperandTemplate":
        return self if self.hole is None else BoolOperandTemplate(fn(self.hole), self.inversion)

    def resolve(self, register_of: Callable[[ValueId], int]) -> BoolOperand:
        source = BoolRegRef(register_of(self.source)) if isinstance(self.source, int) else self.source
        return BoolOperand(source, self.inversion)


type OperandTemplate = WideOperandTemplate | BoolOperandTemplate


@dataclass(frozen=True, slots=True)
class OpWriteSource:
    """
    A pooled operator output lane. A boolean lane folds its fabric inversion into `invert`; a wide lane conditions on
    the wrapper instead (a float's sign on the `y*sgn` field, nothing at all for any other family), so its `invert`
    is always False and equal conditioners never split the opcode.
    """

    inst: OperatorInstance
    port: int
    invert: bool


@dataclass(frozen=True, slots=True)
class InlineWriteSource:
    """An inline-operator combinational result; structurally identical results dedup to one opcode (loop bodies)."""

    operator: InlineHardwareOperator
    operands: tuple[WideOperand | BoolOperand, ...]
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class MoveWriteSource:
    """A move of one operand into a register: a phi-arm copy/write, a constant install, or an early state writeback."""

    operand: WideOperand | BoolOperand


type WriteSource = OpWriteSource | InlineWriteSource | MoveWriteSource


@dataclass(frozen=True, slots=True)
class WriteEvent:
    """One microcode-driven register write: which register takes which `source` on which ROM (executing) `step`."""

    dst: RegRef | BoolRegRef
    source: WriteSource
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.source, OpWriteSource) and self.source.invert:
            assert isinstance(self.dst, BoolRegRef), "a wide lane conditions on the wrapper, so it never inverts"


def read_sources_per_port(lir: Lir) -> dict[ReadPort, list[ReadSource]]:
    """
    Per operand port `(instance, position)` of every instance, the distinct sources it reads across the schedule:
    the registers in ascending index order, then each constant in first-appearance order. A constant is an arm of the
    port's mux exactly like a register. A port reading one source drives it directly and needs no mux.
    """
    regs: dict[ReadPort, set[int]] = {}
    consts: dict[ReadPort, list[WideConstRef]] = {}
    for op in lir.ops:
        for pos, operand in enumerate(op.operands):
            source = operand.source
            if isinstance(source, RegRef):
                regs.setdefault((op.inst, pos), set()).add(source.index)
            elif isinstance(source, WideConstRef):
                book = consts.setdefault((op.inst, pos), [])
                if source not in book:
                    book.append(source)
    ports = [(inst, pos) for inst in lir.instances for pos in range(inst.operator.signature.arity)]
    assert set(regs) | set(consts) <= set(ports)
    return {port: [*(RegRef(index) for index in sorted(regs.get(port, ()))), *consts.get(port, [])] for port in ports}


def write_events(lir: Lir) -> list[WriteEvent]:
    """
    Every microcode-driven register write as `(dst, source, ROM step)`, in one deterministic traversal shared by the
    codebook builder and the packer so the code<->source mapping cannot drift. The ROM step is the source's executing
    step (the fetch PC it fires on, minus the fetch lag): a pooled write rides its commit cycle, an inline/copy/write
    rides `block_base + issue/commit`, an early state install rides `state_copy_step - fetch_lag`. Boundary state
    installs (and all boolean state installs, which are boundary-only) are handshake-gated special arms, not opcode
    sources, so they are excluded here.
    """
    events: list[WriteEvent] = []
    for op in lir.ops:
        for write in op.writes:
            if isinstance(write.dst, RegRef):
                invert = False  # a wide lane conditions on the wrapper, not on the opcode
            else:
                assert isinstance(write.conditioner, BoolInversion)
                invert = write.conditioner.invert
            events.append(
                WriteEvent(write.dst, OpWriteSource(op.inst, write.port, invert), pooled_write_word(op.commit_cycle))
            )
    for block in lir.blocks:
        base = lir.block_base[block.index]
        for inline_op in block.inline_ops:
            source = InlineWriteSource(inline_op.operator, tuple(inline_op.operands), inline_op.write.conditioner)
            events.append(WriteEvent(inline_op.write.dst, source, base + inline_op.commit_cycle))
        for copy in block.wide_copies:
            events.append(WriteEvent(copy.dst, MoveWriteSource(copy.source), base + copy.issue_cycle))
        for bwrite in block.bool_writes:
            events.append(WriteEvent(bwrite.dst, MoveWriteSource(bwrite.source), base + bwrite.issue_cycle))
    for slot in lir.wide_state_slots:
        if slot.needs_copy and not lir.wide_state_install_is_boundary(slot):
            events.append(WriteEvent(slot.reg, MoveWriteSource(slot.tap), lir.state_copy_step(slot) - lir.fetch_lag))
    return events


def read_arms(lir: Lir) -> dict[ReadPort, int]:
    """Per operand port, the arms of its read mux: every source the port ever reads, register or constant."""
    return {port: len(sources) for port, sources in read_sources_per_port(lir).items()}


def write_arms(lir: Lir) -> dict[RegRef | BoolRegRef, int]:
    """
    Per register, the arms of its write select: the structurally distinct opcode sources plus the handshake-gated
    arms the emitter places beside them (the input load; a wide slot's boundary install, a boolean slot's install).
    """
    arms: dict[RegRef | BoolRegRef, int] = {
        dst: len(sources) for dst, sources in write_sources_per_register(write_events(lir)).items()
    }
    special: list[RegRef | BoolRegRef] = [load.dst for load in lir.inputs]
    special += [
        slot.reg for slot in lir.wide_state_slots if slot.needs_copy and lir.wide_state_install_is_boundary(slot)
    ]
    special += [bslot.reg for bslot in lir.bool_state_slots if bslot.needs_copy]
    for dst in special:
        arms[dst] = arms.get(dst, 0) + 1
    return arms


def write_sources_per_register(events: list[WriteEvent]) -> dict[RegRef | BoolRegRef, list[WriteSource]]:
    """
    Per register, the distinct opcode-selected write sources in first-appearance order over the `write_events`: the
    arms of its write select, structurally deduplicated (one arm serves every step that writes the same source). The
    handshake-gated arms -- an input load, a boundary state install -- are not opcode sources and are not listed.
    """
    sources: dict[RegRef | BoolRegRef, list[WriteSource]] = {}
    for event in events:
        book = sources.setdefault(event.dst, [])
        if event.source not in book:
            book.append(event.source)
    return sources
