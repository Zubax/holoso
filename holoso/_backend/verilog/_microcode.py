"""
The microcode model for the Verilog ZISC backend.

From a scheduled :class:`Lir` this derives the per-step VLIW control word: the dedicated read/write port assignment,
every control field and its value on each step (``None`` == don't-care), and the partition into constant fields
(driven by constant nets) versus varying fields (packed into the ROM word). It also owns the Verilog-safe naming the
emitter wires up. It emits no Verilog text -- that is the emitter's job; this module is pure data.

Write control is per OUTPUT-PORT LANE, keyed ``(instance, port)``: a lane's write-enable/address ride the commit step
itself (both banks write combinationally). A wide lane carries its issue-step sign conditioner ``y_sgnop``; a boolean
lane carries a 1-bit inversion conditioner on the commit step (the boolean dual, applied as a fabric XOR at the register
write). A lane exists only if some firing taps it; a never-tapped module output gets no fields and is left unconnected.
"""

from dataclasses import dataclass
from string import ascii_letters

from ..._lir import (
    BoolConstRef,
    BoolRegRef,
    FloatConstRef,
    FloatCopy,
    FloatOperand,
    Lir,
    OperatorInstance,
    PooledScheduledOp,
    RegRef,
    pooled_write_word,
)
from ..._operators import FloatSignControl
from ..._type import is_wide_type

PORT_LETTERS = ascii_letters  # operand position -> wrapper port letter (a, b, ...)


@dataclass
class Field:
    """
    One scalar control field of the microcode word, with its value on every step (``None`` == don't-care).

    After :func:`finalize_fields`, a field is either constant across the program (``offset < 0``; driven by the
    constant net ``const_value``) or varying (stored at bit ``offset`` of the ROM word).
    """

    name: str
    width: int
    values: list[int | None]
    offset: int = -1
    const_value: int = 0


def base_name(inst: OperatorInstance) -> str:
    return f"{inst.operator.instance_stem}_{inst.index}"


def code_width(count: int) -> int:
    """Bit width of a dense code enumerating ``count`` distinct values (at least 1 bit)."""
    return max(1, (count - 1).bit_length()) if count > 1 else 1


def write_target_lists(lir: Lir) -> dict[tuple[OperatorInstance, int], list[int]]:
    """
    Per output-port lane ``(instance, port)``, the sorted distinct registers it ever writes -- the lane's
    write-address codebook (its bank is implied by the destinations' type). The write-address field carries the
    position in this list, not the raw register index, so its width is ``code_width(M)`` over the lane's ``M``
    targets rather than the whole register file. The per-register write selector compares against the same position,
    so the recode is transparent (no decode logic on the consumer) and mirrors the read side, where the read-address
    field carries the dense read-set index. Lanes never tapped by any firing are absent.
    """
    targets: dict[tuple[OperatorInstance, int], list[int]] = {}
    for op in lir.ops:
        for write in op.writes:
            regs = targets.setdefault((op.inst, write.port), [])
            if write.dst.index not in regs:
                regs.append(write.dst.index)
    for regs in targets.values():
        regs.sort()
    return targets


# Microcode field names. Signal names (``s_<base>_*``) and field names (``mc_*_<base>``) live in disjoint namespaces.
def f_rd(port: int) -> str:
    return f"mc_rd{port}"


def f_iv(base: str) -> str:
    return f"mc_iv_{base}"


def f_osgn(base: str, letter: str) -> str:
    return f"mc_{base}_{letter}s"


def f_ysgn(base: str, port: int) -> str:
    return f"mc_{base}_y{port}s"


def f_selc(port: int) -> str:
    return f"mc_selc{port}"


def f_cidx(port: int) -> str:
    return f"mc_cidx{port}"


def f_we(base: str, port: int) -> str:
    return f"mc_we_{base}_y{port}"


def f_wa(base: str, port: int) -> str:
    return f"mc_wa_{base}_y{port}"


def f_binv(base: str, port: int) -> str:
    return f"mc_binv_{base}_y{port}"


# Per-REGISTER constant-install fields: a phi-arm constant install is not an operator, so it has no (instance, port)
# write lane. Each destination register that receives a ucode-driven constant install carries a 1-bit write-enable; a
# wide register additionally carries a const-pool index selector when it installs more than one distinct constant, and
# a boolean register carries the 1-bit value. Names key on ``stable_label`` (``r<k>``/``b<k>``), disjoint from the
# operator-lane fields (``mc_*_<base>_y<port>``).
def f_cwe(dst: RegRef | BoolRegRef) -> str:
    return f"mc_cwe_{dst.stable_label}"


def f_ccidx(reg: RegRef) -> str:
    return f"mc_ccidx_{reg.stable_label}"


def f_cval(breg: BoolRegRef) -> str:
    return f"mc_cval_{breg.stable_label}"


def is_ucode_const_copy(copy: FloatCopy) -> bool:
    """
    Whether a wide phi-arm copy is a constant install the microcode can drive directly: a constant source with the
    identity sign, so its write data is a bare ``const_N`` net needing no register read port and no sign-conditioning
    wire. A register-source copy (needs a read port) or a signed constant install (``-CONST``, needs a sign wire in the
    write-data mux) stays pc-gated -- the deferred harder cases.
    """
    return copy.is_const and copy.source.sign == FloatSignControl()


def const_install_codebooks(lir: Lir) -> dict[int, list[int]]:
    """
    Wide register -> the sorted distinct const-pool indices its ucode-driven constant installs use -- the per-register
    write-data codebook the ``mc_ccidx`` selector indexes (the write-side analogue of ``port_const_map``). A register
    installing a single constant has a one-entry book (the selector is then a lifted-out constant, no ROM bits).
    """
    books: dict[int, list[int]] = {}
    for block in lir.blocks:
        for copy in block.copies:
            if is_ucode_const_copy(copy):
                assert isinstance(copy.source.source, FloatConstRef)
                book = books.setdefault(copy.dst.index, [])
                if copy.source.source.index not in book:
                    book.append(copy.source.source.index)
    for book in books.values():
        book.sort()
    return books


def const_install_bool_regs(lir: Lir) -> list[int]:
    """
    The boolean registers that receive a ucode-driven constant install (the bool analogue of
    :func:`const_install_codebooks`; a boolean constant carries no pool, only its 1-bit value).
    """
    return sorted({write.dst.index for block in lir.blocks for write in block.bool_writes if write.is_const})


def _op_expr(op: PooledScheduledOp) -> str:
    dsts = "/".join(write.conditioner.decorate(write.dst.stable_label) for write in op.writes)
    return f"{dsts}={op.inst.operator.render(*[operand.stable_label for operand in op.operands])}"


def cycle_summary(issues: list[PooledScheduledOp], commits: list[PooledScheduledOp]) -> str:
    parts: list[str] = []
    if issues:
        parts.append("issue " + ", ".join(_op_expr(op) for op in issues))
    if commits:
        parts.append("commit " + ", ".join("/".join(write.dst.stable_label for write in op.writes) for op in commits))
    return "; ".join(parts)


def read_ports(lir: Lir) -> dict[tuple[OperatorInstance, int], int]:
    """One dedicated read port per operator operand, numbered in instance/operand order (counts to ``nrd``)."""
    read_port: dict[tuple[OperatorInstance, int], int] = {}
    for inst in lir.instances:
        for pos in range(inst.operator.arity):
            read_port[(inst, pos)] = len(read_port)
    return read_port


def port_const_map(lir: Lir, read_port: dict[tuple[OperatorInstance, int], int]) -> dict[int, list[int]]:
    """Read port -> the distinct constant-pool indices it ever sources (drives the per-operand constant select)."""
    port_consts: dict[int, list[int]] = {}
    for op in lir.ops:
        for pos, operand in enumerate(op.operands):
            if isinstance(operand.source, FloatConstRef):
                port = read_port[(op.inst, pos)]
                port_consts.setdefault(port, [])
                if operand.source.index not in port_consts[port]:
                    port_consts[port].append(operand.source.index)
    return port_consts


def build_microcode(
    lir: Lir,
    read_port: dict[tuple[OperatorInstance, int], int],
    port_consts: dict[int, list[int]],
    write_lists: dict[tuple[OperatorInstance, int], list[int]],
) -> dict[str, Field]:
    """
    Build the per-step value table of every control field from the static schedule.

    ``in_valid`` and the write-enables are concrete every step (they gate operation), so they default to 0; every
    other field is a don't-care (``None``) except on the step its firing issues or commits, which maximises the
    constant columns that later get lifted out of the ROM.

    Control is placed on the step each bank's discipline requires: the read-address group is presented 1 step before
    the firing issues (so the latched wide operand arrives on the issue step); a lane's write-enable and write-address
    are presented ON the commit step (both banks write combinationally). A WIDE lane's sign conditioner rides the issue
    step (consumed inside the wrapper); a BOOLEAN lane's inversion rides the commit step with its write-enable. Placing
    the write word on the commit step -- not one later -- is exactly what gives a branch condition its one cycle of slack
    at the block boundary.
    """
    depth = lir.last_pc + 1  # one control word per fetch PC: blocks are laid out across 0..last_pc with NOP gaps
    fields: dict[str, Field] = {}
    defaults: dict[str, int | None] = {}

    def add(name: str, width: int, default: int | None) -> None:
        fields[name] = Field(name, width, [default] * depth)
        defaults[name] = default

    def put(name: str, step: int, value: int) -> None:
        # Single-writer rule: a field's step slot may be set once to a non-default value (or repeatedly to the same
        # value). Every write word stays inside its block (only the result LANDING spills into a successor frame -- see
        # pooled_write_word), so under per-block draining no two firings share a control word and this never fires.
        # Under cross-block overlap a successor's base PC drops so its head words can share an absolute fetch step with
        # the predecessor's tail words; this catches at build time any two firings' control words colliding on one slot
        # instead of silently clobbering.
        field = fields[name]
        current = field.values[step]
        assert (
            current == defaults[name] or current == value
        ), f"microcode single-writer violation on field {name!r} step {step}: holds {current!r}, cannot write {value!r}"
        field.values[step] = value

    # The read-address field selects within a port's read-set, not the whole register file: it carries the dense
    # read-set index (0..K-1), so its width is ceil(log2 K) and the emitter's read-mux case selects by it. A
    # single-reader or always-constant port keeps the constant value finalize_fields lifts out of the ROM.
    port_read_set = {read_port[key]: regs for key, regs in lir.read_set_per_port.items()}

    for inst in lir.instances:
        base = base_name(inst)
        add(f_iv(base), 1, 0)
        for pos in range(inst.operator.arity):
            add(f_osgn(base, PORT_LETTERS[pos]), 2, None)
            port = read_port[(inst, pos)]
            add(f_rd(port), code_width(len(port_read_set.get(port, []))), None)
            if port in port_consts:
                add(f_selc(port), 1, None)
                if len(port_consts[port]) > 1:
                    add(f_cidx(port), code_width(len(port_consts[port])), None)
    for (inst, port_index), targets in sorted(write_lists.items(), key=lambda kv: (base_name(kv[0][0]), kv[0][1])):
        base = base_name(inst)
        add(f_we(base, port_index), 1, 0)
        # The write-address field carries the dense write-target index (0..M-1), symmetric to the read-address field.
        add(f_wa(base, port_index), code_width(len(targets)), None)
        if is_wide_type(inst.operator.signature.result_types[port_index]):
            add(f_ysgn(base, port_index), 2, None)
        else:
            add(f_binv(base, port_index), 1, None)

    # Per-register constant-install fields. A wide register's selector is declared only when it installs more than one
    # distinct constant; otherwise the single index is a constant column finalize lifts out of the ROM. A boolean
    # register carries its 1-bit value. The write-enable defaults to 0 (concrete every step, so it always packs).
    const_books = const_install_codebooks(lir)
    for reg in sorted(const_books):
        add(f_cwe(RegRef(reg)), 1, 0)
        if len(const_books[reg]) > 1:
            add(f_ccidx(RegRef(reg)), code_width(len(const_books[reg])), None)
    for reg in const_install_bool_regs(lir):
        add(f_cwe(BoolRegRef(reg)), 1, 0)
        add(f_cval(BoolRegRef(reg)), 1, None)

    for op in lir.ops:
        base = base_name(op.inst)
        ci = op.issue_cycle  # in_valid and sign controls, consumed inside the wrapper on the issue step
        rci = op.issue_cycle - 1  # read-address group, presented early so the read latch delivers on issue
        assert 0 <= rci, f"microcode step out of range: rci={rci}, depth={depth}"
        put(f_iv(base), ci, 1)
        for pos, operand in enumerate(op.operands):
            port = read_port[(op.inst, pos)]
            assert isinstance(operand, FloatOperand), "pooled operators read only wide operands today (no read lane)"
            put(f_osgn(base, PORT_LETTERS[pos]), ci, operand.sign.encoded)
            if isinstance(operand.source, FloatConstRef):
                put(f_selc(port), rci, 1)
                if f_cidx(port) in fields:
                    put(f_cidx(port), rci, port_consts[port].index(operand.source.index))
            elif isinstance(operand.source, RegRef):
                if f_selc(port) in fields:
                    put(f_selc(port), rci, 0)
                put(f_rd(port), rci, port_read_set[port].index(operand.source.index))
        for write in op.writes:
            lane = (op.inst, write.port)
            wide = isinstance(write.dst, RegRef)
            # Both banks write combinationally, so the write-enable/address ride the commit step (NOT one later -- a +1
            # would land the result past a branch's boundary read). The same step the overlap layout uses to keep every
            # write word inside the block (see pooled_write_word).
            wcc = pooled_write_word(op.commit_cycle)
            assert wcc < depth, f"microcode step out of range: wcc={wcc}, depth={depth}"
            if wide:
                put(f_ysgn(base, write.port), ci, write.conditioner.encoded)  # sign rides the wrapper at issue
            else:
                put(f_binv(base, write.port), wcc, write.conditioner.encoded)  # inversion applied at the write
            put(f_we(base, write.port), wcc, 1)
            put(f_wa(base, write.port), wcc, write_lists[lane].index(write.dst.index))

    # Constant installs ride the microcode like operator writes: the write-enable (and the wide selector / boolean
    # value) are placed at ROM step ``block_base + issue_cycle`` == install_pc - FETCH_LAG, so the datapath write fires
    # on the very clock the former pc-gate fired -- schedule-neutral. A register-source or signed-const install is not
    # selected here and stays pc-gated.
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for copy in block.copies:
            if not is_ucode_const_copy(copy):
                continue
            assert isinstance(copy.source.source, FloatConstRef)
            step = base_pc + copy.issue_cycle
            assert 0 <= step < depth, f"const-install ROM step out of range: {step}"
            put(f_cwe(copy.dst), step, 1)
            if f_ccidx(copy.dst) in fields:
                put(f_ccidx(copy.dst), step, const_books[copy.dst.index].index(copy.source.source.index))
        for bwrite in block.bool_writes:
            if not bwrite.is_const:
                continue
            assert isinstance(bwrite.source.source, BoolConstRef)
            step = base_pc + bwrite.issue_cycle
            assert 0 <= step < depth, f"const-install ROM step out of range: {step}"
            put(f_cwe(bwrite.dst), step, 1)
            put(f_cval(bwrite.dst), step, int(bwrite.source.source.value))

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
