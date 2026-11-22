"""
The microcode model for the Verilog ZISC backend.

From a scheduled Lir this derives the per-step VLIW control word. Datapath value routing is expressed as
dense per-endpoint opcodes parsed by `case` statements -- the two endpoints are duals:

- a READ endpoint is an operator operand port; its opcode selects one source from that port's read codebook;
- a WRITE endpoint is a register; its opcode selects one source from that register's write codebook,
  with code 0 reserved for NOP (hold).

Everything a register can take on a cycle is thus one tiny opcode, carrying its write-enable, destination select,
const-pool select, and boolean inversion together; PC is left to control flow alone. This module is pure data -- the
emitter owns the Verilog text and renders each source key. The source lists themselves are the Lir's
(`read_sources_per_port`, `write_sources_per_register`), so the codebooks encode exactly the steering that the
`steering` metric counts.
"""

from dataclasses import dataclass

from amaranth.hdl import Shape

from ..._lir import (
    BoolRegRef,
    InlineWriteSource,
    Lir,
    MoveWriteSource,
    OperatorInstance,
    OpWriteSource,
    PooledScheduledOp,
    ReadSource,
    RegRef,
    WideOperand,
    WriteEvent,
    WriteSource,
    read_sources_per_port,
    write_sources_per_register,
)
from ..._operators import FloatSignControl, has_sign_control


def _has_issue_strobe(lir: Lir, inst: OperatorInstance) -> bool:
    """Whether the instance's block takes `in_valid`: a block without it runs every cycle, its issues being implicit."""
    return "in_valid" in lir.instances[inst].signature.members


def immediate_inputs(lir: Lir, inst: OperatorInstance) -> list[str]:
    """
    The immediates of the instance's block, which each firing drives with a constant. A block specialized to one value
    of an immediate no longer has it.
    """
    members = lir.instances[inst].signature.members
    return [name for name in inst.operator.immediate_ports if name in members]


def operand_names(lir: Lir, inst: OperatorInstance) -> list[str]:
    """The operand ports the instance's block has: one built for fewer primitives lacks those only the others read."""
    members = lir.instances[inst].signature.members
    return [name for name in inst.operator.operand_ports if name in members]


@dataclass
class Field:
    """
    One scalar control field of the microcode word, with its value on every step (`None` == don't-care).

    `gated` marks an effect trigger (an operator issue strobe, or a per-register write opcode): it is concrete every
    step (inert 0 when idle) and ANDed with `transacting` at the decode -- `& {width{transacting}}` -- so a held
    `ucode[0]` dwell, a fill bubble, or a stale pre-reset word decodes to 0 (NOP) and commits nothing. A write opcode
    reserves code 0 for NOP, so the mask is exactly a NOP. Every other field is qualifying data, a don't-care while its
    trigger is inactive, never masked.

    After finalize_fields, a field is either constant across the program (`offset < 0`; driven by the
    constant net `const_value`) or varying (stored at bit `offset` of the ROM word).
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


def _code_width(count: int) -> int:
    """Bit width of a dense code enumerating `count` distinct values (at least 1 bit)."""
    return max(1, (count - 1).bit_length()) if count > 1 else 1


@dataclass(frozen=True, slots=True)
class ReadCodebook:
    """
    An operand port's ordered read sources and the dense opcode that selects among them: code `i` picks
    `sources[i]` over `opcode_width` bits. A single-source port drives its source directly and carries no field.
    """

    sources: tuple[ReadSource, ...]

    @property
    def opcode_width(self) -> int:
        return _code_width(len(self.sources))

    def code(self, source: ReadSource) -> int:
        return self.sources.index(source)

    def arms(self) -> list[tuple[int, ReadSource]]:
        return list(enumerate(self.sources))


@dataclass(frozen=True, slots=True)
class WriteCodebook:
    """
    A register's ordered write sources and the dense opcode that selects among them, code 0 reserved for the NOP hold:
    code `i+1` picks `sources[i]` over `opcode_width` bits. A register with no sources carries no write opcode.
    """

    sources: tuple[WriteSource, ...]

    @property
    def opcode_width(self) -> int:
        return _code_width(len(self.sources) + 1)  # +1: code 0 is the NOP hold

    def code(self, source: WriteSource) -> int:
        return self.sources.index(source) + 1

    def arms(self) -> list[tuple[int, WriteSource]]:
        return [(index + 1, source) for index, source in enumerate(self.sources)]


# Microcode field names, after the block ports they serve. Signal names (`s_<base>_*`) and field names (`uc_*`) live in
# disjoint namespaces.
def f_input(base: str, port: str) -> str:
    """The field driving the block input `port` itself: the issue strobe, an immediate."""
    return f"uc_{base}_{port}"


def f_osgn(base: str, port: str) -> str:
    return f"uc_{base}_{port}_sgn"


def f_rd(base: str, port: str) -> str:
    return f"uc_{base}_{port}_rd"


def f_op(dst: RegRef | BoolRegRef) -> str:
    return f"uc_op_{dst.stable_label}"


def tapped_lanes(events: list[WriteEvent]) -> set[tuple[OperatorInstance, str]]:
    """The operator output ports some firing writes -- an untapped port gets no nets and is left unconnected."""
    return {(event.source.inst, event.source.port) for event in events if isinstance(event.source, OpWriteSource)}


def read_codebook(lir: Lir) -> dict[tuple[OperatorInstance, str], ReadCodebook]:
    return {port: ReadCodebook(tuple(sources)) for port, sources in read_sources_per_port(lir).items()}


def write_codebook(events: list[WriteEvent]) -> dict[RegRef | BoolRegRef, WriteCodebook]:
    return {dst: WriteCodebook(tuple(sources)) for dst, sources in write_sources_per_register(events).items()}


def build_microcode(
    lir: Lir,
    read_books: dict[tuple[OperatorInstance, str], ReadCodebook],
    write_books: dict[RegRef | BoolRegRef, WriteCodebook],
    events: list[WriteEvent],
) -> dict[str, Field]:
    """
    Build the per-step value table of every control field from the static schedule.

    Control is placed on the step each operation requires: the issue strobe, the operand signs, and the read opcodes
    ride the ISSUE step (the read is latch-free, so the datapath samples an operand a fetch lag later); each register's
    WRITE opcode rides the source's executing step (see write_events). A read opcode has no NOP code because a
    don't-care idle read is harmless.
    """
    depth = lir.last_pc + 1  # one control word per fetch PC: blocks are laid out across 0..last_pc with NOP gaps
    fields: dict[str, Field] = {}

    def add(name: str, width: int, gated: bool = False) -> None:
        assert name not in fields, f"two microcode fields named {name!r}"
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
        base, operator = inst.name, inst.operator
        if _has_issue_strobe(lir, inst):
            add(f_input(base, "in_valid"), 1, gated=True)
        members = lir.instances[inst].signature.members
        for port in immediate_inputs(lir, inst):
            add(f_input(base, port), Shape.cast(members[port].shape).width)
        for port in operand_names(lir, inst):
            if has_sign_control(operator.operand_ports[port]):
                add(f_osgn(base, port), 2)
            read_book = read_books[(inst, port)]
            if len(read_book.sources) > 1:
                add(f_rd(base, port), read_book.opcode_width)
    for dst, book in write_books.items():
        add(f_op(dst), book.opcode_width, gated=True)

    for op, block in ((op, block) for block in lir.blocks for op in block.ops):
        ci = lir.block_base[block.index] + op.issue_cycle
        base, operator = op.inst.name, op.inst.operator
        assert 0 <= ci < depth, f"microcode read/issue step out of range: ci={ci}, depth={depth}"
        if _has_issue_strobe(lir, op.inst):
            put(f_input(base, "in_valid"), ci, 1)
        for port in immediate_inputs(lir, op.inst):
            put(f_input(base, port), ci, int(dict(op.drive.immediates)[port]))
        for port, operand in zip(op.drive.operand_ports, op.operands, strict=True):
            assert isinstance(operand, WideOperand), "operators read only wide operands (no read lane)"
            if has_sign_control(operator.operand_ports[port]):
                assert isinstance(operand.conditioner, FloatSignControl)
                put(f_osgn(base, port), ci, operand.conditioner.encoded)
            else:
                assert operand.conditioner.is_identity, "an integer port has no sign field"
            field = f_rd(base, port)
            if field in fields:
                put(field, ci, read_books[(op.inst, port)].code(operand.source))

    for event in events:
        assert 0 <= event.step < depth, f"microcode write step out of range: {event.step}, depth={depth}"
        put(f_op(event.dst), event.step, write_books[event.dst].code(event.source))

    return fields


def finalize_fields(fields: dict[str, Field]) -> int:
    """Partition fields into constant (lifted out, `offset = -1`) and varying (packed); return the ROM word width."""
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


def _op_expr(op: PooledScheduledOp) -> str:
    operands = [operand.stable_label for operand in op.operands]
    return " & ".join(
        f"{write.dst.stable_label}={primitive.render_output(write.inversion, *operands)}"
        for primitive, write in op.taps.items()
    )


def _landing_label(dst: RegRef | BoolRegRef, source: WriteSource) -> str:
    """A non-pooled write rendered `dst=source` for the ROM-step comment -- the dual of the emitter's write RHS."""
    match source:
        case InlineWriteSource(primitive=primitive, operands=operands, inversion=inversion):
            rendered = primitive.render(*[operand.stable_label for operand in operands])
            label = dst.stable_label if inversion is None else inversion.decorate(dst.stable_label)
            return f"{label}={rendered}"
        case MoveWriteSource(operand=operand):
            return f"{dst.stable_label}={operand.stable_label}"
        case OpWriteSource():
            assert False, "pooled commits are named by cycle_summary's commit list, not as landings"


def landings_by_step(events: list[WriteEvent]) -> dict[int, list[str]]:
    """
    Per ROM step, the non-pooled writes that land there rendered `dst=source` -- inline firings and copies. Derived from
    write_events (pooled commits are excluded, being named by `cycle_summary`'s commit list), so the ROM word comment
    names every value the opcode installs and the emitted RTL stays mappable onto the HTML schedule report.
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
