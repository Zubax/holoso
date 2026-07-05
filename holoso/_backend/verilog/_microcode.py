"""
The microcode model for the Verilog ZISC backend.

From a scheduled :class:`Lir` this derives the per-step VLIW control word. Datapath value routing is expressed as
dense per-endpoint opcodes parsed by ``case`` statements -- the two endpoints are duals:

- a READ endpoint is an operator operand port; its opcode selects one source from that port's read codebook;
- a WRITE endpoint is a register; its opcode selects one source from that register's write codebook,
  with code 0 reserved for NOP (hold).

Everything a register can take on a cycle is thus one tiny opcode, carrying its write-enable, destination select,
const-pool select, and boolean inversion together; PC is left to control flow alone. This module is pure data -- the
emitter owns the Verilog text and renders each source key.
"""

from dataclasses import dataclass
from string import ascii_letters

from ..._lir import (
    BoolOperand,
    BoolRegRef,
    FloatConstRef,
    FloatOperand,
    Lir,
    OperatorInstance,
    PooledScheduledOp,
    RegRef,
    pooled_write_word,
)
from ..._operators import BoolInversion, InlineHardwareOperator, PortConditioner

PORT_LETTERS = ascii_letters  # operand position -> wrapper port letter (a, b, ...)


@dataclass
class Field:
    """
    One scalar control field of the microcode word, with its value on every step (``None`` == don't-care).

    ``gated`` marks an effect trigger (an operator issue strobe, or a per-register write opcode): it is concrete every
    step (inert 0 when idle) and ANDed with ``transacting`` at the decode -- ``& {width{transacting}}`` -- so a held
    ``ucode[0]`` dwell, a fill bubble, or a stale pre-reset word decodes to 0 (NOP) and commits nothing. A write opcode
    reserves code 0 for NOP, so the mask is exactly a NOP. Every other field is qualifying data, a don't-care while its
    trigger is inactive, never masked.

    After :func:`finalize_fields`, a field is either constant across the program (``offset < 0``; driven by the
    constant net ``const_value``) or varying (stored at bit ``offset`` of the ROM word).
    """

    name: str
    width: int
    values: list[int | None]
    gated: bool = False
    offset: int = -1
    const_value: int = 0

    @property
    def default(self) -> int | None:
        """Resting value on a don't-care step: a gated field is inert (0); a plain field is unconstrained (None)."""
        return 0 if self.gated else None


def base_name(inst: OperatorInstance) -> str:
    return f"{inst.operator.instance_stem}_{inst.index}"


def code_width(count: int) -> int:
    """Bit width of a dense code enumerating ``count`` distinct values (at least 1 bit)."""
    return max(1, (count - 1).bit_length()) if count > 1 else 1


# Source descriptors: value-equal keys the codebooks dedup on, and which the emitter renders to an RHS net/expression.
# A read source reuses the LIR refs directly (``regs[i]`` / ``const_i``); the write sources below discriminate
# the three ways a register takes a value.
type ReadSource = RegRef | FloatConstRef  # a constant key is its nonnegative-pool magnitude; the sign rides uc_*sgn


@dataclass(frozen=True, slots=True)
class OpWriteSource:
    """
    A pooled operator output lane. A boolean lane folds its fabric inversion into ``invert``; a wide lane's sign rides
    the wrapper (the ``y*sgn`` field), so its ``invert`` is always False and equal signs never split the opcode.
    """

    inst: OperatorInstance
    port: int
    invert: bool


@dataclass(frozen=True, slots=True)
class InlineWriteSource:
    """An inline-operator combinational result; structurally identical results dedup to one opcode (loop bodies)."""

    operator: InlineHardwareOperator
    operands: tuple[FloatOperand | BoolOperand, ...]
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class MoveWriteSource:
    """A move of one operand into a register: a phi-arm copy/write, a constant install, or an early state writeback."""

    operand: FloatOperand | BoolOperand


type WriteSource = OpWriteSource | InlineWriteSource | MoveWriteSource


@dataclass(frozen=True, slots=True)
class WriteEvent:
    """One microcode-driven register write: which register takes which ``source`` on which ROM (executing) ``step``."""

    dst: RegRef | BoolRegRef
    source: WriteSource
    step: int

    def __post_init__(self) -> None:
        # A wide lane's sign rides the ysgn wrapper; only a boolean lane folds an inversion into OpWriteSource.invert.
        if isinstance(self.source, OpWriteSource) and self.source.invert:
            assert isinstance(self.dst, BoolRegRef)


@dataclass(frozen=True, slots=True)
class ReadCodebook:
    """
    An operand port's ordered read sources and the dense opcode that selects among them: code ``i`` picks
    ``sources[i]`` over ``opcode_width`` bits. A single-source port drives its source directly and carries no field.
    """

    sources: tuple[ReadSource, ...]

    @property
    def opcode_width(self) -> int:
        return code_width(len(self.sources))

    def code(self, source: ReadSource) -> int:
        return self.sources.index(source)

    def arms(self) -> list[tuple[int, ReadSource]]:
        return list(enumerate(self.sources))


@dataclass(frozen=True, slots=True)
class WriteCodebook:
    """
    A register's ordered write sources and the dense opcode that selects among them, code 0 reserved for the NOP hold:
    code ``i+1`` picks ``sources[i]`` over ``opcode_width`` bits. A register with no sources carries no write opcode.
    """

    sources: tuple[WriteSource, ...]

    @property
    def opcode_width(self) -> int:
        return code_width(len(self.sources) + 1)  # +1: code 0 is the NOP hold

    def code(self, source: WriteSource) -> int:
        return self.sources.index(source) + 1

    def arms(self) -> list[tuple[int, WriteSource]]:
        return [(index + 1, source) for index, source in enumerate(self.sources)]


# Microcode field names. Signal names (``s_<base>_*``) and field names (``uc_*``) live in disjoint namespaces.
def f_issue(base: str) -> str:
    return f"uc_issue_{base}"


def f_imm(base: str, name: str) -> str:
    return f"uc_{base}_imm_{name}"


def f_osgn(base: str, letter: str) -> str:
    return f"uc_{base}_{letter}sgn"


def f_ysgn(base: str, port: int) -> str:
    return f"uc_{base}_y{port}sgn"


def f_rd(base: str, letter: str) -> str:
    return f"uc_{base}_{letter}rd"


def f_op(dst: RegRef | BoolRegRef) -> str:
    return f"uc_op_{dst.stable_label}"


def tapped_lanes(lir: Lir) -> set[tuple[OperatorInstance, int]]:
    """The operator output ports some firing writes -- an untapped port gets no nets and is left unconnected."""
    return {(op.inst, write.port) for op in lir.ops for write in op.writes}


def read_codebook(lir: Lir) -> dict[tuple[OperatorInstance, int], ReadCodebook]:
    """
    Per operand port ``(instance, position)``, the ordered distinct read sources: the registers it reads (read-set
    order) then each distinct constant magnitude it reads (first-appearance over the schedule). The read opcode carries
    the position over ``code_width`` bits; a single-source port keeps its lone source and needs no opcode field.
    """
    sources: dict[tuple[OperatorInstance, int], list[ReadSource]] = {
        (inst, pos): [] for inst in lir.instances for pos in range(inst.operator.arity)
    }
    for key, regs in lir.read_set_per_port.items():
        sources[key] = [RegRef(reg) for reg in regs]
    for op in lir.ops:
        for pos, operand in enumerate(op.operands):
            if isinstance(operand, FloatOperand) and isinstance(operand.source, FloatConstRef):
                book = sources[(op.inst, pos)]
                if operand.source not in book:
                    book.append(operand.source)
    return {key: ReadCodebook(tuple(srcs)) for key, srcs in sources.items()}


def write_events(lir: Lir) -> list[WriteEvent]:
    """
    Every microcode-driven register write as ``(dst, source, ROM step)``, in one deterministic traversal shared by the
    codebook builder and the packer so the code<->source mapping cannot drift. The ROM step is the source's executing
    step (the fetch PC it fires on, minus the fetch lag): a pooled write rides its commit cycle, an inline/copy/write
    rides ``block_base + issue/commit``, an early state install rides ``state_copy_step - fetch_lag``. Boundary state
    installs (and all boolean state installs, which are boundary-only) are handshake-gated special arms, not opcode
    sources, so they are excluded here.
    """
    events: list[WriteEvent] = []
    for op in lir.ops:
        for write in op.writes:
            if isinstance(write.dst, RegRef):
                invert = False  # a wide lane's sign rides the wrapper, not the opcode
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
        for copy in block.copies:
            events.append(WriteEvent(copy.dst, MoveWriteSource(copy.source), base + copy.issue_cycle))
        for bwrite in block.bool_writes:
            events.append(WriteEvent(bwrite.dst, MoveWriteSource(bwrite.source), base + bwrite.issue_cycle))
    for slot in lir.float_state_slots:
        if slot.needs_copy and not lir.float_state_install_is_boundary(slot):
            events.append(WriteEvent(slot.reg, MoveWriteSource(slot.tap), lir.state_copy_step(slot) - lir.fetch_lag))
    return events


def write_codebook(
    events: list[WriteEvent],
) -> dict[RegRef | BoolRegRef, WriteCodebook]:
    """Per register, the ordered distinct write sources (first-appearance dedup over :func:`write_events`)."""
    sources: dict[RegRef | BoolRegRef, list[WriteSource]] = {}
    for event in events:
        book = sources.setdefault(event.dst, [])
        if event.source not in book:
            book.append(event.source)
    return {dst: WriteCodebook(tuple(srcs)) for dst, srcs in sources.items()}


def build_microcode(
    lir: Lir,
    read_books: dict[tuple[OperatorInstance, int], ReadCodebook],
    write_books: dict[RegRef | BoolRegRef, WriteCodebook],
    events: list[WriteEvent],
    tapped: set[tuple[OperatorInstance, int]],
) -> dict[str, Field]:
    """
    Build the per-step value table of every control field from the static schedule.

    Control is placed on the step each operation requires: the issue strobe, the operand signs, the read opcodes, and a
    wide result's sign ride the ISSUE step (the read is latch-free, so the datapath samples an operand a fetch lag
    later); each register's WRITE opcode rides the source's executing step (see :func:`write_events`). A write opcode
    carries ``code_width(N+1)`` bits over its ``N`` sources with code 0 reserved for the NOP hold; the read opcode
    carries ``code_width(K)`` over its ``K`` sources with no NOP (a don't-care idle read is harmless).
    """
    depth = lir.last_pc + 1  # one control word per fetch PC: blocks are laid out across 0..last_pc with NOP gaps
    fields: dict[str, Field] = {}

    def add(name: str, width: int, gated: bool = False) -> None:
        fields[name] = Field(name, width, [0 if gated else None] * depth, gated=gated)

    def put(name: str, step: int, value: int) -> None:
        # Single-writer rule: a field's step slot may be set once to a non-default value (or repeatedly to the same
        # value). For a write opcode this is exactly "<=1 landing per register per step" -- distinct sources take
        # distinct codes, so a genuine double-landing on one register trips this rather than silently clobbering.
        field = fields[name]
        current = field.values[step]
        assert (
            current == field.default or current == value
        ), f"microcode single-writer violation on field {name!r} step {step}: holds {current!r}, cannot write {value!r}"
        field.values[step] = value

    for inst in lir.instances:
        base = base_name(inst)
        add(f_issue(base), 1, gated=True)
        for imm in inst.operator.immediate_ports:
            add(f_imm(base, imm.name), imm.width)
        for pos in range(inst.operator.arity):
            add(f_osgn(base, PORT_LETTERS[pos]), 2)
            read_book = read_books[(inst, pos)]
            if len(read_book.sources) > 1:
                add(f_rd(base, PORT_LETTERS[pos]), read_book.opcode_width)
        for q, result_type in enumerate(inst.operator.signature.result_types):
            if result_type.is_wide and (inst, q) in tapped:
                add(f_ysgn(base, q), 2)
    for dst, book in write_books.items():
        add(f_op(dst), book.opcode_width, gated=True)

    for op in lir.ops:
        base = base_name(op.inst)
        ci = op.issue_cycle
        assert 0 <= ci < depth, f"microcode read/issue step out of range: ci={ci}, depth={depth}"
        put(f_issue(base), ci, 1)
        for value, imm in zip(op.immediates, op.operator.immediate_ports, strict=True):
            put(f_imm(base, imm.name), ci, value)
        for pos, operand in enumerate(op.operands):
            assert isinstance(operand, FloatOperand), "pooled operators read only wide operands today (no read lane)"
            put(f_osgn(base, PORT_LETTERS[pos]), ci, operand.sign.encoded)
            field = f_rd(base, PORT_LETTERS[pos])
            if field in fields:
                put(field, ci, read_books[(op.inst, pos)].code(operand.source))
        for write in op.writes:
            if isinstance(write.dst, RegRef):
                put(f_ysgn(base, write.port), ci, write.conditioner.encoded)  # wide result sign rides the wrapper

    for event in events:
        assert (
            0 <= event.step < lir.present_step
        ), f"microcode write step past present: step={event.step}, present={lir.present_step}"
        put(f_op(event.dst), event.step, write_books[event.dst].code(event.source))

    return fields


def finalize_fields(fields: dict[str, Field]) -> int:
    """Partition fields into constant (lifted out, ``offset = -1``) and varying (packed); return the ROM word width."""
    offset = 0
    for f in fields.values():
        concrete = [v for v in f.values if v is not None]
        if concrete and any(v != concrete[0] for v in concrete):
            f.offset = offset
            offset += f.width
        else:
            f.offset = -1
            f.const_value = concrete[0] if concrete else 0
    return max(1, offset)


def pack(fields: dict[str, Field], step: int) -> int:
    word = 0
    for f in fields.values():
        if f.offset < 0:
            continue
        v = f.values[step]
        word |= ((0 if v is None else v) & ((1 << f.width) - 1)) << f.offset
    return word


# ---- ROM-step annotations (human-readable summary shared with the HTML report's schedule view) ----
def _op_expr(op: PooledScheduledOp) -> str:
    dsts = "/".join(write.conditioner.decorate(write.dst.stable_label) for write in op.writes)
    operands = [operand.stable_label for operand in op.operands]
    return f"{dsts}={op.inst.operator.render(*operands, immediates=op.immediates)}"


def _landing_label(dst: RegRef | BoolRegRef, source: WriteSource) -> str:
    """A non-pooled write rendered ``dst=source`` for the ROM-step comment -- the dual of ``_write_source_rhs``."""
    match source:
        case InlineWriteSource(operator=operator, operands=operands, conditioner=conditioner):
            rendered = operator.render(*[operand.stable_label for operand in operands])
            return f"{conditioner.decorate(dst.stable_label)}={rendered}"
        case MoveWriteSource(operand=operand):
            return f"{dst.stable_label}={operand.stable_label}"
        case OpWriteSource():
            assert False, "pooled commits are named by cycle_summary's commit list, not as landings"


def landings_by_step(events: list[WriteEvent]) -> dict[int, list[str]]:
    """
    Per ROM step, the non-pooled writes that land there rendered ``dst=source`` -- inline firings, phi-arm copies,
    boolean writes, and early state installs. Derived from :func:`write_events` (pooled commits are excluded, being
    named by ``cycle_summary``'s commit list), so the ROM word comment names every value the opcode installs and the
    emitted RTL stays mappable onto the HTML schedule report.
    """
    landings: dict[int, list[str]] = {}
    for event in events:
        if not isinstance(event.source, OpWriteSource):
            landings.setdefault(event.step, []).append(_landing_label(event.dst, event.source))
    return landings


def cycle_summary(issues: list[PooledScheduledOp], commits: list[PooledScheduledOp], landings: list[str]) -> str:
    parts: list[str] = []
    if issues:
        parts.append("issue " + ", ".join(_op_expr(op) for op in issues))
    if commits:
        parts.append("commit " + ", ".join("/".join(write.dst.stable_label for write in op.writes) for op in commits))
    if landings:
        parts.append("land " + ", ".join(landings))
    return "; ".join(parts)
