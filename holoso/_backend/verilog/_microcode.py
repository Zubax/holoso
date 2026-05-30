"""
The microcode model for the Verilog ZISC backend.

From a scheduled :class:`Lir` this derives the per-step VLIW control word: the dedicated read/write port assignment,
every control field and its value on each step (``None`` == don't-care), and the partition into constant fields
(driven by constant nets) versus varying fields (packed into the ROM word). It also owns the Verilog-safe naming the
emitter wires up. It emits no Verilog text -- that is the emitter's job; this module is pure data.
"""

from dataclasses import dataclass
from string import ascii_letters

from ..._lir import FloatConstRef, Lir, FloatOperatorInstance, FloatScheduledOp

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


def base_name(inst: FloatOperatorInstance) -> str:
    return f"{inst.operator.instance_stem}_{inst.index}"


# Microcode field names. Signal names (``s_<base>_*``) and field names (``mc_*_<base>``) live in disjoint namespaces.
def f_rd(port: int) -> str:
    return f"mc_rd{port}"


def f_iv(base: str) -> str:
    return f"mc_iv_{base}"


def f_osgn(base: str, letter: str) -> str:
    return f"mc_{base}_{letter}s"


def f_ysgn(base: str) -> str:
    return f"mc_{base}_ys"


def f_selc(port: int) -> str:
    return f"mc_selc{port}"


def f_cidx(port: int) -> str:
    return f"mc_cidx{port}"


def f_we(base: str) -> str:
    return f"mc_we_{base}"


def f_wa(base: str) -> str:
    return f"mc_wa_{base}"


def _op_expr(op: FloatScheduledOp) -> str:
    return f"r{op.dst.index}={op.inst.operator.render(*[o.stable_label for o in op.operands])}"


def cycle_summary(issues: list[FloatScheduledOp], commits: list[FloatScheduledOp]) -> str:
    parts: list[str] = []
    if issues:
        parts.append("issue " + ", ".join(_op_expr(op) for op in issues))
    if commits:
        parts.append("commit " + ", ".join(f"r{op.dst.index}" for op in commits))
    return "; ".join(parts)


def read_ports(lir: Lir) -> dict[tuple[FloatOperatorInstance, int], int]:
    """One dedicated read port per operator operand, numbered in instance/operand order (counts to ``nrd``)."""
    read_port: dict[tuple[FloatOperatorInstance, int], int] = {}
    for inst in lir.float_instances:
        for pos in range(inst.operator.arity):
            read_port[(inst, pos)] = len(read_port)
    return read_port


def port_const_map(lir: Lir, read_port: dict[tuple[FloatOperatorInstance, int], int]) -> dict[int, list[int]]:
    """Read port -> the distinct constant-pool indices it ever sources (drives the per-operand constant select)."""
    port_consts: dict[int, list[int]] = {}
    for op in lir.float_ops:
        for pos, operand in enumerate(op.operands):
            if isinstance(operand.source, FloatConstRef):
                port = read_port[(op.inst, pos)]
                port_consts.setdefault(port, [])
                if operand.source.index not in port_consts[port]:
                    port_consts[port].append(operand.source.index)
    return port_consts


def build_microcode(
    lir: Lir, read_port: dict[tuple[FloatOperatorInstance, int], int], port_consts: dict[int, list[int]], waddr: int
) -> dict[str, Field]:
    """
    Build the per-step value table of every control field from the static schedule.

    ``in_valid`` and ``write-enable`` are concrete every step (they gate operation), so they default to 0; every other
    field is a don't-care (``None``) except on the step its operator issues or commits, which maximises the constant
    columns that later get lifted out of the ROM.

    Read and write control are placed on shifted steps to line up with the register-file latches: the read-address group
    is presented 1 step before the operator issues (so the latched operand arrives on the issue step), and the
    write-enable/address are presented 1 step after the operator commits (so they line up with the writeback latch).
    """
    depth = lir.makespan + 3  # steps 0..present (present == makespan + WRITE_LATCH + 1)
    fields: dict[str, Field] = {}

    def add(name: str, width: int, default: int | None) -> None:
        fields[name] = Field(name, width, [default] * depth)

    # The read-address field selects within a port's read-set, not the whole register file: it carries the dense
    # read-set index (0..K-1), so its width is ceil(log2 K) and the emitter's gather + part-select indexes by it. A
    # single-reader or always-constant port keeps the constant value finalize_fields lifts out of the ROM.
    port_read_set = {read_port[key]: regs for key, regs in lir.read_set_per_port.items()}

    def rd_width(port: int) -> int:
        regs = port_read_set.get(port, [])
        return max(1, (len(regs) - 1).bit_length()) if len(regs) > 1 else 1

    for inst in lir.float_instances:
        base = base_name(inst)
        add(f_iv(base), 1, 0)
        add(f_we(base), 1, 0)
        add(f_wa(base), waddr, None)
        add(f_ysgn(base), 2, None)
        for pos in range(inst.operator.arity):
            add(f_osgn(base, PORT_LETTERS[pos]), 2, None)
            port = read_port[(inst, pos)]
            add(f_rd(port), rd_width(port), None)
            if port in port_consts:
                add(f_selc(port), 1, None)
                if len(port_consts[port]) > 1:
                    add(f_cidx(port), max(1, (len(port_consts[port]) - 1).bit_length()), None)

    for op in lir.float_ops:
        base = base_name(op.inst)
        ci = op.issue_cycle  # in_valid and sign controls, consumed inside the wrapper on the issue step
        rci = op.issue_cycle - 1  # read-address group, presented early so the read latch delivers on issue
        wcc = op.commit_cycle + 1  # write-enable/address, delayed to line up with the writeback latch
        assert 0 <= rci and wcc < depth, f"microcode step out of range: rci={rci}, wcc={wcc}, depth={depth}"
        fields[f_iv(base)].values[ci] = 1
        fields[f_ysgn(base)].values[ci] = op.result_sign.encoded
        for pos, operand in enumerate(op.operands):
            port = read_port[(op.inst, pos)]
            fields[f_osgn(base, PORT_LETTERS[pos])].values[ci] = operand.sign.encoded
            if isinstance(operand.source, FloatConstRef):
                fields[f_selc(port)].values[rci] = 1
                if f_cidx(port) in fields:
                    fields[f_cidx(port)].values[rci] = port_consts[port].index(operand.source.index)
            else:
                if f_selc(port) in fields:
                    fields[f_selc(port)].values[rci] = 0
                fields[f_rd(port)].values[rci] = port_read_set[port].index(operand.source.index)
        fields[f_we(base)].values[wcc] = 1
        fields[f_wa(base)].values[wcc] = op.dst.index

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
