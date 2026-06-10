"""
Render a scheduled :class:`Lir` into a synthesizable Verilog ZISC module, plus access to the shared ``holoso_support``
HDL that the generated module instantiates.

The controller is a microcode ROM (see :mod:`._microcode`): one pre-decoded VLIW control word per step, stored in a
(BRAM-inferable) ROM read through a 3-stage fetch (a PC latch, the array read, and the BRAM output register) so the
critical control cones are short register-to-register paths. The executing step therefore lags the fetch PC by
FETCH_LAG, which the sequencer accounts for: the PC counts up to LASTPC and out_valid is asserted there.

Storage is a sparse, schedule-specific register file emitted inline instead of a general-purpose multiport file.
The register array is a plain ``reg`` bank. Each operator operand has a dedicated read port whose mux spans only the
registers that operand ever reads across the schedule (a single-register operand needs no mux), followed by a read
latch. Each operator result passes through a writeback latch into a per-register write select that spans only the
instances that ever write that register (a single-writer register needs no address compare).

All sequential logic -- the fetch pipeline, the read/write register-file latches, the register writes, and the
reset-gated control state -- is emitted as a single ``always @(posedge clk)`` block; the only combinational
``always @*`` blocks are the next-PC sequencer and the shared comparator's operand mux. Control fields that are
constant across the whole program are driven by constant nets and omitted from the ROM, so synthesis prunes them.
"""

from dataclasses import dataclass
from importlib import resources
from textwrap import dedent

from ..._hir import RelationalOp
from ..._lir import *
from ..._operators import *
from ._microcode import *

_SUPPORT_FILES = {
    name: resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
    for name in ("holoso_support.v", "holoso_support.vh")
}


@dataclass(frozen=True, slots=True)
class VerilogOutput:
    """The Verilog backend's output: the generated module text and the shared support files it instantiates."""

    verilog: str
    support_files: dict[str, str]  # filename -> content

    def __str__(self) -> str:
        sup = ",".join(f"{name!r}:{len(text.encode())}" for name, text in self.support_files.items())
        return f"{type(self).__name__}(verilog_bytes={len(self.verilog.encode())}, support_bytes={{{sup}}})"


class _Writer:
    """Accumulates 4-space-indented lines; ``w(...)`` accepts single lines or dedented multiline blocks."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def __call__(self, *texts: str) -> None:
        for text in texts:
            if "\n" in text:
                block = dedent(text).removeprefix("\n").removesuffix("\n")
                for line in block.split("\n"):
                    self._append(line)
            else:
                self._append(text)

    def _append(self, text: str) -> None:
        self._lines.append(("    " * self._depth + text) if text else "")

    def push(self) -> None:
        self._depth += 1

    def pop(self) -> None:
        assert self._depth > 0
        self._depth -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def _sig(inst: FloatOperatorInstance) -> str:
    return f"s_{base_name(inst)}"


def _decl_range(width: int) -> str:
    return "" if width == 1 else f"[{width - 1:2}:0] "


def _lit(width: int, value: int) -> str:
    return f"{width}'d{value}"


def _cterm_expr(port: int, consts: list[int]) -> str:
    """The constant operand value for a read port: the single immediate, or the per-port constant-index selector."""
    return f"const_{consts[0]}" if len(consts) == 1 else f"cterm{port}"


def _source_net(source: FloatRegRef | FloatConstRef) -> str:
    """The net carrying a register-or-constant source value: the pooled constant immediate, or the register read."""
    return f"const_{source.index}" if isinstance(source, FloatConstRef) else f"regs[{source.index}]"


def _fsgnop(w: _Writer, raw: str, sign: FloatSignControl, dst: str, inst: str) -> None:
    """Emit a sign-conditioning wrapper instance applying ``sign`` to ``raw`` and driving ``dst``."""
    w(f"holoso_fsgnop #(.WFULL(W)) {inst} (.x({raw}), .op(2'd{sign.encoded}), .y({dst}));")


def _state_sign_wire(slot: FloatStateSlot) -> str | None:
    """
    The sign-conditioning wire name for a slot's writeback copy, or None when the copied tap needs no sign op. The stem
    is deliberately not ``state_*``: a public attribute is exposed as a ``state_<attr>`` port, so a ``state_<name>_d``
    net would collide with the port of an attribute literally named ``<name>_d``.
    """
    if slot.needs_copy and slot.tap.sign != FloatSignControl():
        return f"statesgn_{slot.name}"
    return None


def _state_copy_rhs(slot: FloatStateSlot) -> str:
    """The value latched into a non-coalesced slot register on its install step: sign-conditioned wire, or raw tap."""
    return _state_sign_wire(slot) or _source_net(slot.tap.source)


def _fcmp_reduce(relation: RelationalOp, gt: str, eq: str, lt: str) -> str:
    """Reduce the comparator's three one-hot order flags to the boolean the relation selects."""
    return {
        RelationalOp.LT: lt,
        RelationalOp.LE: f"{lt} | {eq}",
        RelationalOp.GT: gt,
        RelationalOp.GE: f"{gt} | {eq}",
        RelationalOp.EQ: eq,
        RelationalOp.NE: f"{gt} | {lt}",
    }[relation]


def _fcmp_label(block_index: int, position: int) -> str:
    return f"{block_index}_{position}"


def _block_comparisons(block: LirBlock) -> list[tuple[int, CombScheduledOp, FComparisonOperator]]:
    """
    The block's comparison operations with their per-block position index. Comparisons are the combinational ops that
    drive the shared ``holoso_fcmp``; other combinational ops (boolean logic, casts) emit differently and are skipped.
    """
    result: list[tuple[int, CombScheduledOp, FComparisonOperator]] = []
    for op in block.comb_ops:
        operator = op.operator
        if isinstance(operator, FComparisonOperator):
            result.append((len(result), op, operator))
    return result


def _fcmp_in_valid_pc(lir: Lir, block_index: int, op: CombScheduledOp) -> int:
    """The fetch PC at which a comparator's in_valid pulses: late enough that its float operands have landed."""
    return lir.block_base[block_index] + op.issue_cycle + FETCH_LAG


def _comb_writeback_pc(lir: Lir, block_index: int, op: CombScheduledOp) -> int:
    """The fetch PC at which a combinational op latches its result: in_valid pc plus the operator latency."""
    return _fcmp_in_valid_pc(lir, block_index, op) + op.latency


def _block_logic_ops(lir: Lir) -> list[tuple[int, CombScheduledOp]]:
    """Every boolean-logic op (AND/OR/NOT) with its block index; these emit as PC-gated inline ``& | ~`` writebacks."""
    return [
        (block.index, op) for block in lir.blocks for op in block.comb_ops if isinstance(op.operator, BoolLogicOperator)
    ]


def _block_ftobool_ops(lir: Lir) -> list[tuple[int, CombScheduledOp]]:
    """Every float->bool cast with its block index; each emits as a PC-gated ``holoso_ftobool`` writeback."""
    return [
        (block.index, op)
        for block in lir.blocks
        for op in block.comb_ops
        if isinstance(op.operator, FloatToBoolOperator)
    ]


def _block_ffrombool_ops(lir: Lir) -> list[tuple[int, CombScheduledOp]]:
    """Every bool->float cast with its block index; each writes a float register, so it is timed on the float frame."""
    return [
        (block.index, op)
        for block in lir.blocks
        for op in block.comb_ops
        if isinstance(op.operator, BoolToFloatOperator)
    ]


def _comb_float_writeback_pc(lir: Lir, block_index: int, op: CombScheduledOp) -> int:
    """
    The fetch PC at which a float-result combinational op latches its float register. A float result lands one cycle
    later than a boolean one (the write latch plus the read-first edge of the float register file), matching the
    microcode write-enable at ``commit + 1`` fetched FETCH_LAG ahead, so a downstream float operator reads it on time.
    """
    return lir.block_base[block_index] + op.commit_cycle + 1 + FETCH_LAG


def _ffrombool_rhs(op: CombScheduledOp) -> str:
    """The RHS of ``float(cond)``: the shared ``holoso_ffrombool`` cast applied to the boolean operand (ZKF 1.0/0.0)."""
    (operand,) = op.operands
    assert isinstance(operand, BoolOperand)  # the cast reads a boolean operand
    return f"holoso_ffrombool({_bool_operand_rhs(operand)})"


def _ftobool_rhs(op: CombScheduledOp) -> str:
    """The RHS of ``bool(x)``: the shared ``holoso_ftobool`` cast applied to the operand (its sign is irrelevant)."""
    (operand,) = op.operands
    assert isinstance(operand, FloatOperand)  # the cast reads a float operand (its sign is irrelevant to the exponent)
    return f"holoso_ftobool({_source_net(operand.source)})"


def _bool_logic_rhs(op: CombScheduledOp) -> str:
    """The combinational RHS of a boolean-logic op: a plain ``& | ~`` over its boolean register/constant operands."""
    operands: list[str] = []
    for operand in op.operands:
        assert isinstance(operand, BoolOperand)  # boolean-logic operands are boolean
        operands.append(_bool_operand_rhs(operand))
    match op.operator:
        case BoolAndOperator():
            a, b = operands
            return f"{a} & {b}"
        case BoolOrOperator():
            a, b = operands
            return f"{a} | {b}"
        case BoolNotOperator():
            (a,) = operands
            return f"~{a}"
        case _:
            raise AssertionError(f"not a boolean-logic operator: {op.operator!r}")


def _emit_fcmp_instance(w: _Writer, lir: Lir) -> None:
    """
    The single shared ``holoso_fcmp``, by the one-instance-per-operator convention. Every comparison drives it in
    turn: each issues at a distinct fetch PC -- comparisons live in mutually-exclusive blocks and execute sequentially,
    so the comparator pipeline never holds two at once -- with its float operands presented by a combinational mux
    keyed on the fetch PC and ``in_valid`` pulsed at that PC. The one-hot order flags are reduced per each comparison's
    relation; the clocked process latches each result into its boolean register at the comparison's writeback PC.
    """
    comparisons = [
        (block.index, position, op, cmp) for block in lir.blocks for position, op, cmp in _block_comparisons(block)
    ]
    if not comparisons:
        return
    sample = comparisons[0][2]  # one fcmp configuration serves every comparison: uniform params and latency
    params = "".join(f".{name}({value}), " for name, value in sample.operator.hdl_params().items())

    w("reg  [W-1:0] fcmp_a, fcmp_b;  // shared comparator operands, PC-muxed across every comparison")
    w("reg  [1:0]   fcmp_a_sgnop, fcmp_b_sgnop;")
    w("reg          fcmp_iv;")
    w("always @* begin  // present the active comparison's operands to the one shared comparator")
    w.push()
    w("fcmp_a = {W{1'b0}};  fcmp_b = {W{1'b0}};")
    w("fcmp_a_sgnop = 2'd0; fcmp_b_sgnop = 2'd0; fcmp_iv = 1'b0;")
    w("case (pc)")
    w.push()
    for block_index, _position, op, _cmp in comparisons:
        a, b = op.operands
        assert isinstance(a, FloatOperand) and isinstance(b, FloatOperand)  # comparison operands are float
        w(f"{_fcmp_in_valid_pc(lir, block_index, op)}: begin")
        w.push()
        w(f"fcmp_a = {_source_net(a.source)}; fcmp_a_sgnop = 2'd{a.sign.encoded};")
        w(f"fcmp_b = {_source_net(b.source)}; fcmp_b_sgnop = 2'd{b.sign.encoded};")
        w("fcmp_iv = 1'b1;")
        w.pop()
        w("end")
    w("default: ;")
    w.pop()
    w("endcase")
    w.pop()
    w("end")

    w("wire fcmp_gt, fcmp_eq, fcmp_lt;")
    w(f"holoso_fcmp #(.WEXP(WEXP), .WMAN(WMAN), {params}.LATENCY({sample.latency})) u_fcmp (")
    w.push()
    w(".clk(clk), .rst(rst), .in_valid(fcmp_iv),")
    w(".a_sgnop(fcmp_a_sgnop), .b_sgnop(fcmp_b_sgnop),")
    w(".a(fcmp_a), .b(fcmp_b),")
    w(".out_valid(), .a_gt_b(fcmp_gt), .a_eq_b(fcmp_eq), .a_lt_b(fcmp_lt)")
    w.pop()
    w(");")
    for block_index, position, _op, cmp in comparisons:
        label = _fcmp_label(block_index, position)
        reduction = _fcmp_reduce(cmp.relation, "fcmp_gt", "fcmp_eq", "fcmp_lt")
        w(f"wire fcmp_{label}_result = {reduction};  // {cmp.relation.value}")
    w("")


def generate(lir: Lir) -> VerilogOutput:
    # This emitter implements the mandatory v1 staging; the scheduler and microcode placement budget for exactly these.
    assert FETCH_STAGES == 3, "the Verilog emitter implements the 3-stage microcode fetch (may be configurable later)"
    w = _Writer()
    cycw = lir.cyc_width
    pcw = max(1, lir.initiation_interval.bit_length())

    # One dedicated read port per operator operand; the per-port read mux spans only the registers it actually reads.
    read_port = read_ports(lir)
    port_consts = port_const_map(lir, read_port)
    read_sets = lir.read_set_per_port
    write_sets = lir.write_set_per_register
    # Symmetric to the read-set index on the read side: the write-address field and the per-register write comparators
    # carry the dense write-target index (ceil(log2 M) over each instance's M write targets, not the whole file). The
    # codebook is shared between the microcode and the comparators so they cannot drift.
    write_lists = write_target_lists(lir)

    fields = build_microcode(lir, read_port, port_consts, write_lists)
    ucw = finalize_fields(fields)

    issues_by_cycle, commits_by_cycle = lir.group_by_cycle
    commits_by_step: dict[int, list[FloatScheduledOp]] = {}  # the writeback latch delays the commit step by one
    for commit_cycle, ops in commits_by_cycle.items():
        commits_by_step.setdefault(commit_cycle + 1, []).extend(ops)

    depth = lir.last_pc + 1  # one microcode word per fetch PC (0..last_pc); inter-block drains and the tail pack to NOP

    _emit_header(w, lir)
    _emit_localparams(w, lir, cycw, pcw, ucw)
    _emit_support_header(w, lir)
    _emit_declarations(w, lir)
    _emit_consts(w, lir)
    _emit_operators(w, lir)
    _emit_fcmp_instance(w, lir)
    _emit_microcode_rom(w, fields, ucw, depth, issues_by_cycle, commits_by_step)
    _emit_field_wires(w, fields)
    _emit_datapath_comb(w, lir, port_consts)
    _emit_state_next(w, lir)
    _emit_copy_sign_wires(w, lir)
    _emit_clocked(w, lir, read_port, port_consts, read_sets, write_sets, write_lists)
    _emit_outputs(w, lir)
    w("\nendmodule\n")
    return VerilogOutput(verilog=w.render(), support_files=_SUPPORT_FILES)


def _emit_header(w: _Writer, lir: Lir) -> None:
    from holoso import __url__, __version__

    # Generation time is not included for reproducibility.
    fmt = lir.float_regfile.fmt
    w(f"""
// Constructed by Holoso v{__version__} <{__url__}>. Do not edit.

`timescale 1ns/1ps

// Float format: exponent {fmt.wexp} bits, significand {fmt.wman} bits, total {fmt.width} bits.
module {lir.module_name} (
""")
    w.push()
    ports = lir.ports
    last = ports[-1]
    _emit_port_group(w, "CONTROL PORTS", "Clock/reset and ready/valid handshake for one scheduled invocation.")
    for control_port in [port for port in ports if isinstance(port, ControlPort) and port.name != "err_pc"]:
        _emit_port(w, control_port, control_port is not last)
    _emit_port_group(w, "INPUT PORTS", "Latched when in_valid && in_ready.")
    for input_port in [port for port in ports if isinstance(port, DataInputPort)]:
        _emit_port(w, input_port, input_port is not last)
    _emit_port_group(
        w,
        "OUTPUT/STATE PORTS",
        "Valid when out_valid is pulsed. Publicly visible states are included here.",
    )
    for output_port in [port for port in ports if isinstance(port, DataOutputPort)]:
        _emit_port(w, output_port, output_port is not last)
    _emit_port_group(w, "DIAGNOSTIC PORTS", "Runtime diagnostics available while the module is running.")
    for diagnostic_port in [port for port in ports if isinstance(port, ControlPort) and port.name == "err_pc"]:
        # err_pc: 0 = no error; otherwise the (last) step an error was detected. |err_pc answers "any error?".
        _emit_port(w, diagnostic_port, diagnostic_port is not last)
    w.pop()
    w(");", "")


def _emit_port(w: _Writer, port: Port, comma: bool) -> None:
    direction = "input " if port.direction == Direction.IN else "output"
    port_range = "" if port.width == 1 else f"[{port.width - 1}:0] "
    suffix = "," if comma else ""
    w(f"{direction} wire {port_range}{port.name}{suffix}")


def _emit_port_group(w: _Writer, title: str, comment: str) -> None:
    w(f"// {title}", f"// {comment}")


def _emit_localparams(w: _Writer, lir: Lir, cycw: int, pcw: int, ucw: int) -> None:
    fmt = lir.float_regfile.fmt
    nreg = max(1, lir.float_regfile.nreg)
    w(f"""
localparam           WEXP      ={fmt.wexp:3};  // Float exponent bits fixed by the static schedule
localparam           WMAN      ={fmt.wman:3};  // Float mantissa bits fixed by the static schedule
localparam           W         = WEXP + WMAN;
localparam           NREG      ={nreg:3};  // >= 1; the bank is unused when no value needs a register
localparam           CYCW      ={cycw:3};  // err_pc width: enough for any executing step (0..present)
localparam           PCW       ={pcw:3};  // fetch-PC width: counts to LASTPC (execution lags the fetch by FETCH_LAG)
localparam           FETCH_LAG ={FETCH_LAG:3};  // executing step = pc - FETCH_LAG ({FETCH_STAGES}-stage control fetch)
localparam [PCW-1:0] PRESENT   ={lir.present_step:3};  // executing step on which the outputs are valid in the array
localparam [PCW-1:0] LASTPC    ={lir.initiation_interval:3};  // = PRESENT + FETCH_LAG; out_valid asserts here
localparam           UCW       ={ucw:3};  // microcode word width after lifting out constant control fields
localparam           NBREG     ={max(1, lir.bool_regfile.nreg):3};  // 1-bit boolean register bank (branch conditions)
// pc: 0 = idle/accept, present at executing step PRESENT; out_valid at pc==LASTPC (fetch leads execution).
""")
    # Cross-check the ZKF +1.0 formula against the codec at build time. This is the contract holoso_ffrombool's
    # concatenation implements (bias exponent at the fraction MSb, zero sign/fraction); a format whose codec disagreed
    # with it would be caught here rather than miscompiling.
    assert (((1 << (fmt.wexp - 1)) - 1) << (fmt.wman - 1)) == fmt.encode(1.0)
    w("")


def _emit_support_header(w: _Writer, lir: Lir) -> None:
    """
    Include the shared support header unconditionally. Its functions (the float<->bool casts and the finiteness /
    saturation helpers) are the single place that assumes the ZKF bit layout; they reference the WEXP / WMAN / W
    localparams declared above, so the generated module's datapath invokes them by name and never open-codes the
    layout. Defining a function the kernel does not call is free, so the header always ships and is always included.
    """
    w('\n`include "holoso_support.vh"\n')


def _emit_declarations(w: _Writer, lir: Lir) -> None:
    w("""
        reg  [PCW-1:0]  pc;            // fetch program counter; the executing step lags it by FETCH_LAG
        reg  [PCW-1:0]  next_pc;       // combinational next-state presented to the ROM each cycle
        reg  [PCW-1:0]  ucode_addr_q;  // PC latch: splits pc -> next_pc -> ROM address from the array read
        reg  [CYCW-1:0] err_pc_q;
        wire            err;           // an operator error is detected on the current step

        reg  [W-1:0] regs  [0:NREG-1];   // the sparse register array (read-first: a write is visible the next step)
        reg          bregs [0:NBREG-1];  // 1-bit boolean register bank: branch conditions and boolean state

        """)
    for inst in lir.float_instances:
        sig = _sig(inst)
        w(f"wire         {sig}_iv;")
        for letter in PORT_LETTERS[: inst.operator.arity]:
            w(f"wire [1:0]   {sig}_{letter}s;")
            w(f"reg  [W-1:0] {sig}_{letter};")  # read-latched operand (the read mux output, registered)
        w(f"wire [1:0]   {sig}_ys;")
        w(f"wire [W-1:0] {sig}_y;")
        w(f"reg  [W-1:0] {sig}_y_q;")  # writeback latch between the operator output and the register write
        for port in inst.operator.error_ports:
            w(f"wire         {sig}_{port};")
            w(f"reg          {sig}_{port}_q;")  # error sideband rides the same writeback latch as the result
    w("")


def _emit_consts(w: _Writer, lir: Lir) -> None:
    fmt = lir.float_regfile.fmt
    width = fmt.width
    digits = (width + 3) // 4
    for index, value in enumerate(lir.float_consts):
        w(f"wire [W-1:0] const_{index} = {width}'h{fmt.encode(value):0{digits}x};  // {value!r}")
    if lir.float_consts:
        w("")


def _emit_operators(w: _Writer, lir: Lir) -> None:
    for inst in lir.float_instances:
        sig = _sig(inst)
        letters = PORT_LETTERS[: inst.operator.arity]
        # WEXP/WMAN frame the float format; hdl_params() lists K (ilog2) and every STAGE_* explicitly. LATENCY is
        # emitted separately from the scheduler model, so a model/RTL drift fails during wrapper elaboration.
        parts = [".WEXP(WEXP)", ".WMAN(WMAN)"] + [
            f".{param}({value})" for param, value in inst.operator.hdl_params().items()
        ]
        parts.append(f".LATENCY({inst.operator.latency})")
        params = ", ".join(parts)
        w(f"{inst.operator.module_name} #(", f"    {params}", f") u_{base_name(inst)} (")
        w.push()
        w(f".clk(clk), .rst(rst), .in_valid({sig}_iv),")
        for letter in letters:
            w(f".{letter}_sgnop({sig}_{letter}s),")
        w(f".y_sgnop({sig}_ys),")
        for letter in letters:
            w(f".{letter}({sig}_{letter}),")
        # out_valid is left unconnected: the static schedule already knows when each result is ready.
        w(".out_valid(),")
        w(f".y({sig}_y)" + ("," if inst.operator.error_ports else ""))
        for port in inst.operator.error_ports:
            w(f".{port}({sig}_{port})")
        w.pop()
        w(");", "")


def _emit_microcode_rom(
    w: _Writer,
    fields: dict[str, Field],
    ucw: int,
    depth: int,
    issues_by_cycle: dict[int, list[FloatScheduledOp]],
    commits_by_step: dict[int, list[FloatScheduledOp]],
) -> None:
    digits = (ucw + 3) // 4
    w("""
// Microcode VLIW ROM: one pre-decoded control word per fetch PC (0..LASTPC), registered on read (clocked block below).
// Steps with no scheduled event -- inter-block drains and the present/boundary tail -- pack to a NOP word; constant
// control fields are lifted out (below) and not stored here, enabling synthesis-time folding.
(* rom_style = "block", ram_style = "block", syn_romstyle = "EBR" *)
reg [UCW-1:0] ucode [0:LASTPC];
initial begin
    """)
    w.push()
    for step in range(depth):  # depth == LASTPC + 1: one word per fetch PC, NOP where no operator issues or commits
        summary = cycle_summary(issues_by_cycle.get(step, []), commits_by_step.get(step, []))
        comment = f"  // {summary}" if summary else ""
        w(f"ucode[{step: 5}] = {ucw}'h{pack(fields, step):0{digits}x};{comment}")
    w.pop()
    w("""
end

reg [UCW-1:0] ucode_q;     // 2nd fetch stage: control-store array-read register
reg [UCW-1:0] ucode_word;  // 3rd fetch stage: packs into the BRAM output register; drives this step

""")


def _emit_field_wires(w: _Writer, fields: dict[str, Field]) -> None:
    w("""
// Decoded control fields. A field that is constant across the whole program is driven by a constant net
// (so synthesis prunes the logic it feeds); a varying field is a slice of the instruction word.
""")
    for f in fields.values():
        if f.offset < 0:
            w(f"wire {_decl_range(f.width)}{f.name} = {_lit(f.width, f.const_value)};")
        elif f.width == 1:
            w(f"wire        {f.name} = ucode_word[{f.offset}];")
        else:
            w(f"wire {_decl_range(f.width)}{f.name} = ucode_word[{f.offset} +: {f.width}];")
    w("")


def _const_term_expr(port: int, consts: list[int]) -> str:
    expr = f"const_{consts[-1]}"
    for local in range(len(consts) - 2, -1, -1):
        expr = f"({f_cidx(port)} == {local}) ? const_{consts[local]} : {expr}"
    return expr


def _emit_datapath_comb(w: _Writer, lir: Lir, port_consts: dict[int, list[int]]) -> None:
    """Combinational datapath: constant terms, the input-load enable, operator control, the err flag, and next_pc."""
    for port in sorted(port_consts):
        if len(port_consts[port]) > 1:
            w(f"wire [W-1:0] cterm{port} = {_const_term_expr(port, port_consts[port])};")
    w("")

    w("// Operator control (in_valid and sign controls are consumed inside the wrapper on the issue step).")
    for inst in lir.float_instances:
        sig, base = _sig(inst), base_name(inst)
        w(f"assign {sig}_iv = {f_iv(base)};")
        w(f"assign {sig}_ys = {f_ysgn(base)};")
        for pos in range(inst.operator.arity):
            w(f"assign {sig}_{PORT_LETTERS[pos]}s = {f_osgn(base, PORT_LETTERS[pos])};")
    w("")

    # An error matters only on the step its operator commits, which is exactly that instance's write-enable; both the
    # write-enable and the error sideband are aligned to the writeback latch (commit + write latch).
    err_terms = [
        f"({f_we(base_name(inst))} & {_sig(inst)}_{port}_q)"
        for inst in lir.float_instances
        for port in inst.operator.error_ports
    ]
    err_rhs = " | ".join(err_terms) if err_terms else "1'b0"
    w(f"assign err = {err_rhs};", "")

    redirects = _terminator_redirects(lir)
    w("""
// Next-PC sequencer (combinational). The PC holds at the accept (pc==0) and present (pc==LASTPC) boundaries; bubble
// steps carry a NOP word and the PC keeps advancing. The executing step lags the fetch PC by FETCH_LAG. A block's
// terminator redirects the fetch PC at the block's boundary step (a branch reads its boolean register).
always @* begin
""")
    w.push()
    w("if (rst)            next_pc = 0;")
    w("else if (out_valid) next_pc = out_ready ? 0 : LASTPC;  // present: hold until the result is taken")
    w("else if (in_ready)  next_pc = in_valid ? 1 : 0;        // accept: hold until a transaction arrives")
    if not redirects:
        w("else                next_pc = pc + 1'b1;            // advance the fetch")
    else:
        w("else begin")
        w.push()
        w("case (pc)")
        w.push()
        for term_pc, expr in redirects:
            w(f"{term_pc}: next_pc = {expr};")
        w("default: next_pc = pc + 1'b1;")
        w.pop()
        w("endcase")
        w.pop()
        w("end")
    w.pop()
    w("end", "")


def _terminator_redirects(lir: Lir) -> list[tuple[int, str]]:
    """
    The non-fall-through fetch-PC redirects, one per block whose terminator is not a plain advance: a ``Jump`` to a
    non-adjacent block, or a ``Branch`` selecting a target by its boolean register. Each is keyed by the block's
    terminator fetch step (its boundary). A ``Ret`` block is the out_valid boundary and needs no redirect; a ``Jump``
    to the next-laid-out block falls through on ``pc + 1`` and needs no case arm.
    """
    redirects: list[tuple[int, str]] = []
    for block in lir.blocks:
        term_pc = lir.block_base[block.index] + boundary_step(block.block_makespan)
        match block.terminator:
            case Jump(target=target):
                target_pc = lir.block_base[target]
                if target_pc != term_pc + 1:
                    redirects.append((term_pc, str(target_pc)))
            case Branch(cond=cond, if_true=if_true, if_false=if_false):
                redirects.append(
                    (term_pc, f"bregs[{cond.index}] ? {lir.block_base[if_true]} : {lir.block_base[if_false]}")
                )
            case Ret():
                pass
    redirects.sort()
    return redirects


def _float_copy_pc(lir: Lir, block: LirBlock, copy: FloatCopy) -> int:
    """The fetch PC at which a phi-arm copy installs its value (its source has landed by this step)."""
    return lir.block_base[block.index] + copy_step_cycle(copy.issue_cycle)


def _bool_write_pc(lir: Lir, block: LirBlock, write: BoolWrite) -> int:
    return lir.block_base[block.index] + copy_step_cycle(write.issue_cycle)


def _copy_sign_wire(block_index: int, copy_index: int) -> str:
    return f"copysgn_{block_index}_{copy_index}"


def _float_copy_rhs(block_index: int, copy_index: int, copy: FloatCopy) -> str:
    """The net a float copy installs: the raw source for an identity sign, else a per-copy sign-conditioning wire."""
    if copy.source.sign == FloatSignControl():
        return _source_net(copy.source.source)
    return _copy_sign_wire(block_index, copy_index)


def _bool_operand_rhs(operand: BoolOperand) -> str:
    source = operand.source
    if isinstance(source, BoolConstRef):
        return "1'b1" if source.value else "1'b0"
    return f"bregs[{source.index}]"


def _bool_write_rhs(write: BoolWrite) -> str:
    return _bool_operand_rhs(write.source)


def _float_copies_grouped(lir: Lir) -> dict[int, list[tuple[int, str]]]:
    """Destination float register -> [(install PC, source net)], over every phi-arm copy in the program."""
    grouped: dict[int, list[tuple[int, str]]] = {}
    for block in lir.blocks:
        for copy_index, copy in enumerate(block.float_copies):
            grouped.setdefault(copy.dst.index, []).append(
                (_float_copy_pc(lir, block, copy), _float_copy_rhs(block.index, copy_index, copy))
            )
    return grouped


def _bool_writes_grouped(lir: Lir) -> dict[int, list[tuple[int, str]]]:
    """Destination boolean register -> [(install PC, source expression)], over every boolean phi-arm write."""
    grouped: dict[int, list[tuple[int, str]]] = {}
    for block in lir.blocks:
        for write in block.bool_writes:
            grouped.setdefault(write.dst.index, []).append((_bool_write_pc(lir, block, write), _bool_write_rhs(write)))
    return grouped


def _emit_pc_writes(w: _Writer, lhs: str, regs: set[int], grouped: dict[int, list[tuple[int, str]]]) -> None:
    """Emit, for each destination register in ``regs``, a one-hot priority chain of pc-gated installs (sorted by PC)."""
    for reg in sorted(regs):
        entries = sorted(grouped.get(reg, []))
        clause = "if"
        for install_pc, rhs in entries:
            w(f"{clause} (pc == {install_pc}) {lhs}[{reg}] <= {rhs};")
            clause = "else if"


def _emit_copy_sign_wires(w: _Writer, lir: Lir) -> None:
    """Emit a sign-conditioning wire for each float copy whose installed source carries a folded sign control."""
    emitted = False
    for block in lir.blocks:
        for copy_index, copy in enumerate(block.float_copies):
            if copy.source.sign != FloatSignControl():
                wire = _copy_sign_wire(block.index, copy_index)
                w(f"wire [W-1:0] {wire};")
                _fsgnop(w, _source_net(copy.source.source), copy.source.sign, wire, f"u_{wire}")
                emitted = True
    if emitted:
        w("")


def _emit_state_next(w: _Writer, lir: Lir) -> None:
    """Sign-condition the persisted next value of any non-coalesced slot whose copied source carries a folded sign."""
    emitted = False
    for slot in lir.float_state_slots:
        wire = _state_sign_wire(slot)
        if wire is None:
            continue
        w(f"wire [W-1:0] {wire};")
        _fsgnop(w, _source_net(slot.tap.source), slot.tap.sign, wire, f"u_statesgn_{slot.name}")
        emitted = True
    if emitted:
        w("")


def _read_latch_stmts(
    w: _Writer, target: str, port: int, read_set: list[int], port_consts: dict[int, list[int]]
) -> None:
    """
    Emit the read-mux + read-latch update for one operand, inside the clocked block.

    The mux spans only ``read_set`` (the registers this port ever reads): just the immediate when the operand is
    always a constant, a direct register read for a single register, and otherwise a ``case`` over the dense read-set
    index (the read-address field) selecting ``regs[...]`` directly. A const-select picks the immediate when the
    operand is sometimes a constant. On idle steps the latch captures a don't-care value the operator ignores (its
    in_valid is low). The read mux carries no indexed part-select, so there is no offset multiply for synthesis to
    (mis)infer as a DSP -- which is why the read-set index addresses a case rather than a packed gather bus.
    """
    consts = port_consts.get(port)
    cterm = _cterm_expr(port, consts) if consts else None
    if not read_set:  # the operand is always a constant immediate
        w(f"{target} <= {cterm};")
        return
    if len(read_set) == 1:
        reg_expr = f"regs[{read_set[0]}]"
        w(f"{target} <= {f_selc(port)} ? {cterm} : {reg_expr};" if cterm else f"{target} <= {reg_expr};")
        return
    # Multi-register operand: a case over the dense read-set index. The last entry is the default arm so the case is
    # full (no inferred latch); the unused high codes fall there too and are don't-cares on idle steps.
    if cterm:
        w(f"if ({f_selc(port)}) {target} <= {cterm};", "else begin")
        w.push()
    w(f"case ({f_rd(port)})")
    w.push()
    for index, reg in enumerate(read_set):
        label = "default" if index == len(read_set) - 1 else _lit(code_width(len(read_set)), index)
        w(f"{label}: {target} <= regs[{reg}];")
    w.pop()
    w("endcase")
    if cterm:
        w.pop()
        w("end")


def _reg_write_stmts(
    w: _Writer, reg: int, writers: list[FloatOperatorInstance], write_lists: dict[FloatOperatorInstance, list[int]]
) -> None:
    """
    Emit the write select for one register: a select spanning only that register's writer instances.

    Each register's flop gets one clock-enabled write (the input load is grouped separately, above), so the write
    logic stays one register per flop rather than scattering a register's drivers across per-instance blocks. A
    single-target instance needs no address compare; otherwise the guard compares the instance's write-address field
    against the dense write-target index this register occupies in the instance's codebook (the microcode value).
    """
    clause = "if"
    for inst in writers:
        sig, base = _sig(inst), base_name(inst)
        targets = write_lists[inst]
        if len(targets) == 1:
            cond = f_we(base)
        else:
            cond = f"{f_we(base)} && ({f_wa(base)} == {_lit(code_width(len(targets)), targets.index(reg))})"
        w(f"{clause} ({cond}) regs[{reg}] <= {sig}_y_q;")
        clause = "else if"


def _emit_clocked(
    w: _Writer,
    lir: Lir,
    read_port: dict[tuple[FloatOperatorInstance, int], int],
    port_consts: dict[int, list[int]],
    read_sets: dict[tuple[FloatOperatorInstance, int], list[int]],
    write_sets: dict[int, list[FloatOperatorInstance]],
    write_lists: dict[FloatOperatorInstance, list[int]],
) -> None:
    """Emit every sequential element in one always @(posedge clk): fetch, latches, writes, and control state."""
    nreg = max(1, lir.float_regfile.nreg)
    float_slot_regs = {slot.reg.index for slot in lir.float_state_slots}
    bool_slot_regs = {slot.reg.index for slot in lir.bool_state_slots}
    float_copies = _float_copies_grouped(lir)  # phi-arm register installs, pc-gated, grouped by destination register
    bool_writes = _bool_writes_grouped(lir)
    nonslot_copy_regs = {reg for reg in float_copies if reg not in float_slot_regs}
    nonslot_bwrite_regs = {reg for reg in bool_writes if reg not in bool_slot_regs}
    is_cfg = lir.is_control_flow  # the CFG path installs slots at the Ret boundary
    w("""
// All sequential logic in one clocked process. Reset gates only the control state (pc, err_pc_q), never data.
// We never multi-assign any register by explicitly segregating competing assignments into different condition branches.
always @(posedge clk) begin
""")
    w.push()

    w("// Microcode fetch: PC latch -> control-store array read -> BRAM output register.")
    w("ucode_addr_q <= next_pc;")
    w("ucode_q      <= ucode[ucode_addr_q];")
    w("ucode_word   <= ucode_q;")
    w("")

    w("// Operand read latches: a sparse mux over each operand's read-set, registered before the wrapper.")
    for inst in lir.float_instances:
        sig = _sig(inst)
        for pos in range(inst.operator.arity):
            port = read_port[(inst, pos)]
            _read_latch_stmts(w, f"{sig}_{PORT_LETTERS[pos]}", port, read_sets.get((inst, pos), []), port_consts)
    w("")

    w("// Writeback latches: the operator result (and any error sideband) registered before the register write.")
    for inst in lir.float_instances:
        sig = _sig(inst)
        w(f"{sig}_y_q <= {sig}_y;")
        for err_port in inst.operator.error_ports:
            w(f"{sig}_{err_port}_q <= {sig}_{err_port};")
    w("")

    # We MUST ensure that we DO NOT MULTI-ASSIGN any register in the same step; this is ensured by always placing each
    # assignment to the same register into different branches of the same condition.
    # Slot registers are updated in the reset-gated state block below (their snapshot and update share one rst
    # condition); the input load and operator writeback here stay reset-unconditional and cover only non-slot registers.
    state_regs = {slot.reg.index for slot in lir.float_state_slots}

    def emit_writeback() -> None:
        for reg in range(nreg):
            writers = write_sets.get(reg, [])
            if writers and reg not in state_regs:
                _reg_write_stmts(w, reg, writers, write_lists)

    has_writes = any(write_sets.get(reg) and reg not in state_regs for reg in range(nreg))
    if lir.float_inputs:
        # The accept-step input load and operator writeback are mutually exclusive in time (the schedule never commits
        # a result on the load step), but the tools cannot see that. Making writeback the else of the load encodes the
        # exclusivity structurally -- load wins, writes are gated behind it.
        w("// Register update: input load on the accept step, else the per-register writeback select.")
        w("if (in_ready && in_valid) begin")
        w.push()
        for load in sorted(lir.float_inputs, key=lambda load: load.dst.index):
            w(f"regs[{load.dst.index}] <= in_{load.name};")
        w.pop()
        if has_writes:
            w("end else begin")
            w.push()
            emit_writeback()
            w.pop()
        w("end", "")
    elif has_writes:
        w("// Register writes: a select spanning only each register's writers.")
        emit_writeback()
        w("")

    # Phi-arm installs into non-slot merge registers (pure datapath, unreset): a one-hot pc-gated copy/write per
    # register. These registers are written only by their installs, so they never collide with the writeback above.
    if nonslot_copy_regs or nonslot_bwrite_regs:
        w("// Phi-arm installs into non-slot merge registers (pc-gated, one per register).")
        _emit_pc_writes(w, "regs", nonslot_copy_regs, float_copies)
        _emit_pc_writes(w, "bregs", nonslot_bwrite_regs, bool_writes)
        w("")

    # Comparator results land in their (fresh, never reset) boolean register one latency after in_valid, in time for
    # the terminating branch; each is written only here, so it cannot collide with any other boolean write.
    if any(_block_comparisons(block) for block in lir.blocks):
        w("// Comparator boolean writebacks (pc-gated, pure datapath).")
        for block in lir.blocks:
            for position, op, _cmp in _block_comparisons(block):
                writeback_pc = _fcmp_in_valid_pc(lir, block.index, op) + op.latency
                result = f"fcmp_{_fcmp_label(block.index, position)}_result"
                w(f"if (pc == {writeback_pc}) bregs[{op.dst.index}] <= {result};")
        w("")

    # Boolean-logic writebacks: a pc-gated inline & | ~ over boolean registers, latched into a fresh boolean register
    # (each written only here, so no collision). The result reg is read by a later op or the branch, one drain later.
    logic_ops = _block_logic_ops(lir)
    if logic_ops:
        w("// Boolean logic writebacks (pc-gated, combinational).")
        for block_index, op in logic_ops:
            writeback_pc = _comb_writeback_pc(lir, block_index, op)
            w(f"if (pc == {writeback_pc}) bregs[{op.dst.index}] <= {_bool_logic_rhs(op)};")
        w("")

    # Float->bool cast writebacks: a pc-gated ``holoso_ftobool`` call latched into a fresh boolean register.
    ftobool_ops = _block_ftobool_ops(lir)
    if ftobool_ops:
        w("// Float-to-bool cast writebacks (pc-gated, combinational).")
        for block_index, op in ftobool_ops:
            writeback_pc = _comb_writeback_pc(lir, block_index, op)
            assert isinstance(op.dst, BoolRegRef)
            w(f"if (pc == {writeback_pc}) bregs[{op.dst.index}] <= {_ftobool_rhs(op)};")
        w("")

    # Bool->float cast writebacks: a pc-gated ``holoso_ffrombool`` call latched into a fresh float register, timed on
    # the float frame so a downstream float operator reads it on time. The register is written only here (no collision).
    ffrombool_ops = _block_ffrombool_ops(lir)
    if ffrombool_ops:
        w("// Bool-to-float cast writebacks (pc-gated, combinational).")
        for block_index, op in ffrombool_ops:
            writeback_pc = _comb_float_writeback_pc(lir, block_index, op)
            assert isinstance(op.dst, FloatRegRef)
            w(f"if (pc == {writeback_pc}) regs[{op.dst.index}] <= {_ffrombool_rhs(op)};")
        w("")

    # Control and persistent state are the reset-gated registers. Each slot register's reset snapshot (under rst) and
    # its update (a coalesced operator's writeback, or a writeback copy on its install step) are the two arms of this
    # single rst condition, segregating those assignments for the synthesizer; the pure datapath above stays unreset.
    fmt = lir.float_regfile.fmt
    digits = (fmt.width + 3) // 4
    copies = [slot for slot in lir.float_state_slots if slot.needs_copy]
    w("// Control and persistent state: the reset-gated registers.")
    w("if (rst) begin")
    w.push()
    w("pc       <= 0;")
    w("err_pc_q <= 0;")
    for slot in lir.float_state_slots:
        bits = f"{fmt.width}'h{fmt.encode(slot.reset_value):0{digits}x}"
        w(f"regs[{slot.reg.index}] <= {bits};  // {slot.name} reset snapshot")
    for bslot in lir.bool_state_slots:
        w(f"bregs[{bslot.reg.index}] <= 1'b{int(bslot.reset_value)};  // {bslot.name} reset snapshot")
    w.pop()
    w("end else begin")
    w.push()
    w("pc <= next_pc;")
    w("if (in_ready && in_valid) err_pc_q <= 0;  // clear the diagnostic when a new transaction is accepted")
    w("if (err) err_pc_q <= pc - FETCH_LAG;      // execution lags the fetch PC by FETCH_LAG, so the step is pc-lag")
    for slot in lir.float_state_slots:  # coalesced slots: their producing operator's writeback, moved under rst
        if not slot.needs_copy and write_sets.get(slot.reg.index):
            _reg_write_stmts(w, slot.reg.index, write_sets[slot.reg.index], write_lists)
    if is_cfg:
        # The slot register is read-only in the body (its phi/live-out lives in a fresh register); install its live-out
        # at the single Ret boundary (LASTPC), read-first and out_ready-gated. Read-first: the combinational output net
        # samples the OLD slot value at LASTPC (so `return old` sees the live-in) while this latches the new value for
        # the next transaction; out_ready makes it fire exactly once, never re-copying while the boundary is held.
        for slot in lir.float_state_slots:
            if slot.needs_copy:
                w(f"if (pc == LASTPC && out_ready) regs[{slot.reg.index}] <= {_state_copy_rhs(slot)};  // {slot.name}")
        for bslot in lir.bool_state_slots:
            if bslot.needs_copy:
                rhs = _bool_operand_rhs(bslot.live_out)
                w(f"if (pc == LASTPC && out_ready) bregs[{bslot.reg.index}] <= {rhs};  // {bslot.name}")
    elif copies:
        # Single-block: persist each non-coalesced slot on the step its writeback installs (state_copy_step; LASTPC for
        # a boundary copy). An early install step is traversed exactly once per accepted transaction, so it self-gates;
        # the LASTPC boundary step is held under back-pressure, so there the copy also waits for out_ready -- both fold
        # into one guard so state advances exactly once and a held boundary never re-copies (else a delay overruns).
        pcw = max(1, lir.initiation_interval.bit_length())
        for slot in copies:
            cond = f"pc == {_lit(pcw, lir.state_copy_step(slot))} && (pc != LASTPC || out_ready)"
            w(f"if ({cond}) regs[{slot.reg.index}] <= {_state_copy_rhs(slot)};  // {slot.name}")
    w.pop()
    w("end")

    w.pop()
    w("end", "")


def _emit_outputs(w: _Writer, lir: Lir) -> None:
    w("""
assign in_ready  = (pc == 0);
assign out_valid = (pc == LASTPC);  // the result is valid in the array on PRESENT; execution lags the fetch by FETCH_LAG
assign err_pc    = err_pc_q;
""")
    float_index = 0
    for wire in lir.outputs:
        match wire:
            case BoolOutputWire():
                w(f"assign {wire.name} = {_bool_operand_rhs(wire.tap)};")
            case FloatOutputWire():
                raw = _source_net(wire.tap.source)
                if wire.tap.sign == FloatSignControl():
                    w(f"assign {wire.name} = {raw};")
                else:
                    _fsgnop(w, raw, wire.tap.sign, wire.name, f"u_outsgn_{float_index}")
                float_index += 1
    w("")
