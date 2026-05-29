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

The RF read and write latches are mandatory in this version; the dependency scheduler and the microcode field placement
budget for them. Control fields that are constant across the whole program are driven by constant nets and omitted
from the ROM, so synthesis prunes the logic they feed.
"""

from dataclasses import dataclass
from importlib import resources
from textwrap import dedent

from ..._lir import (
    ControlPort,
    DataInputPort,
    DataOutputPort,
    Direction,
    FETCH_LAG,
    FETCH_STAGES,
    FloatConstRef,
    Lir,
    FloatOperatorInstance,
    FloatScheduledOp,
    Port,
    group_by_cycle,
)
from ..._operators import FloatSignControl
from ._microcode import (
    PORT_LETTERS,
    Field,
    base_name,
    build_microcode,
    cycle_summary,
    f_cidx,
    f_iv,
    f_osgn,
    f_rd,
    f_selc,
    f_we,
    f_wa,
    f_ysgn,
    finalize_fields,
    pack,
    port_const_map,
    read_ports,
)

_SUPPORT_FILES = {
    name: resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
    for name in ("holoso_support.v", "holoso_support.vh")
}


@dataclass(frozen=True, slots=True)
class VerilogOutput:
    """The Verilog backend's output: the generated module text and the shared support files it instantiates."""

    verilog: str
    support_files: dict[str, str]  # filename -> content


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


def generate(lir: Lir) -> VerilogOutput:
    # This emitter implements the mandatory v1 staging; the scheduler and microcode placement budget for exactly these.
    assert FETCH_STAGES == 3, "the Verilog emitter implements the 3-stage microcode fetch (may be configurable later)"

    w = _Writer()
    rf = lir.float_regfile
    waddr = max(1, (rf.nreg - 1).bit_length())
    cycw = lir.cyc_width
    pcw = max(1, lir.initiation_interval.bit_length())

    # One dedicated read port per operator operand; the per-port read mux spans only the registers it actually reads.
    read_port = read_ports(lir)
    port_consts = port_const_map(lir, read_port)
    read_sets = lir.read_set_per_port
    write_sets = lir.write_set_per_register
    inst_targets: dict[FloatOperatorInstance, set[int]] = {}
    for op in lir.float_ops:
        inst_targets.setdefault(op.inst, set()).add(op.dst.index)

    fields = build_microcode(lir, read_port, port_consts, waddr)
    ucw = finalize_fields(fields)

    issues_by_cycle, commits_by_cycle = group_by_cycle(lir)
    commits_by_step: dict[int, list[FloatScheduledOp]] = {}  # the writeback latch delays the commit step by write latch
    for commit_cycle, ops in commits_by_cycle.items():
        commits_by_step.setdefault(commit_cycle + 1, []).extend(ops)

    depth = lir.makespan + 3  # microcode table steps 0..present
    last_pc = lir.initiation_interval  # = present_step + FETCH_LAG; the ROM is padded with NOPs up to here

    _emit_header(w, lir)
    _emit_localparams(w, lir, waddr, cycw, pcw, ucw)
    _emit_declarations(w, lir)
    _emit_consts(w, lir)
    _emit_operators(w, lir)
    _emit_microcode_rom(w, fields, ucw, depth, last_pc, issues_by_cycle, commits_by_step)
    _emit_field_wires(w, fields)
    _emit_sequencer(w)
    _emit_datapath(w, lir, read_port, port_consts, read_sets, write_sets, inst_targets, waddr)
    _emit_outputs(w, lir)
    w("\nendmodule\n")
    return VerilogOutput(verilog=w.render(), support_files=_SUPPORT_FILES)


def _emit_header(w: _Writer, lir: Lir) -> None:
    fmt = lir.float_regfile.fmt
    w(f"""
`include "holoso_support.vh"
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
    _emit_port_group(w, "OUTPUT PORTS", "Valid when out_valid is pulsed.")
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


def _emit_localparams(w: _Writer, lir: Lir, waddr: int, cycw: int, pcw: int, ucw: int) -> None:
    fmt = lir.float_regfile.fmt
    compute = f"1..{lir.makespan} = compute, " if lir.makespan else ""
    nreg = max(1, lir.float_regfile.nreg)
    w(f"""
localparam           WEXP  = {fmt.wexp};  // Float exponent bits fixed by the static schedule
localparam           WMAN  = {fmt.wman};  // Float mantissa bits fixed by the static schedule
localparam           W     = WEXP + WMAN;
localparam           NREG  = {nreg};  // >= 1; the bank is unused when no value needs a register
localparam           WADDR = {waddr:2};  // register-index width used by the read/write selectors
localparam           CYCW  = {cycw:2};  // err_pc width: enough for any executing step (0..present)
localparam           PCW   = {pcw:2};  // fetch-PC width: counts to LASTPC (execution lags the fetch by FETCH_LAG)
localparam           FETCH_LAG = {FETCH_LAG};  // executing step = pc - FETCH_LAG ({FETCH_STAGES}-stage control fetch)
localparam [PCW-1:0] PRESENT   = {lir.present_step};  // executing step on which the outputs are valid in the array
localparam [PCW-1:0] LASTPC    = {lir.initiation_interval};  // = PRESENT + FETCH_LAG; out_valid asserts here
localparam           UCW   = {ucw};  // microcode word width after lifting out constant control fields
// pc: 0 = idle/accept, {compute}present at executing step PRESENT; out_valid at pc==LASTPC (fetch leads execution).

""")


def _emit_declarations(w: _Writer, lir: Lir) -> None:
    w("""
        reg  [PCW-1:0]  pc;       // fetch program counter; the executing step lags it by FETCH_LAG
        reg  [PCW-1:0]  next_pc;  // combinational next-state presented to the ROM each cycle
        reg  [CYCW-1:0] err_pc_q;
        wire            err;      // an operator error is detected on the current step

        reg  [W-1:0] regs [0:NREG-1];  // the sparse register array (read-first: a write is visible the next step)

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
        # WEXP/WMAN frame the float format; hdl_params() lists K (ilog2) and every STAGE_* explicitly (including zeros),
        # so the instantiation is self-describing and a param-name mismatch with the wrapper fails loudly at elaboration.
        parts = [".WEXP(WEXP)", ".WMAN(WMAN)"] + [
            f".{param}({value})" for param, value in inst.operator.hdl_params().items()
        ]
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
    last_pc: int,
    issues_by_cycle: dict[int, list[FloatScheduledOp]],
    commits_by_step: dict[int, list[FloatScheduledOp]],
) -> None:
    digits = (ucw + 3) // 4
    w("""
// Microcode VLIW ROM: one pre-decoded control word per step, registered on read (in the sequencer below).
// Constant control fields are lifted out (below) and not stored here, enabling synthesis-time folding.
(* rom_style = "block", ram_style = "block", syn_romstyle = "EBR" *)
reg [UCW-1:0] ucode [0:LASTPC];  // steps 0..PRESENT carry the program; PRESENT+1..LASTPC are NOP fetch padding
initial begin
    """)
    w.push()
    for step in range(depth):
        summary = cycle_summary(issues_by_cycle.get(step, []), commits_by_step.get(step, []))
        comment = f"  // {summary}" if summary else ""
        w(f"ucode[{step: 5}] = {ucw}'h{pack(fields, step):0{digits}x};{comment}")
    for step in range(depth, last_pc + 1):
        w(f"ucode[{step: 5}] = {ucw}'h{0:0{digits}x};  // NOP fetch padding")
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


def _emit_sequencer(w: _Writer) -> None:
    w("""
// Sequencer.
//
// The control store is read through a 3-stage fetch: a PC latch (ucode_addr_q) splits the pc -> next_pc -> ROM-address
// cone from the array read, then ucode_q is the array-read register and ucode_word a second register cascaded directly
// after it (no logic between them) so the tool packs it into the BRAM's dedicated output register. The executing step
// therefore lags the fetch PC by FETCH_LAG: pc runs 0..LASTPC and out_valid is asserted at LASTPC.
//
// Reset covers only control state (pc, err_pc_q); the fetch registers are reset-unconditional (so they can pack into
// the BRAM output register) and settle to ucode[0] under reset.
//
// FUTURE TUNING KNOB: the fetch depth (and the read/write latches) are fixed here; flows that register the control
// store in fabric, or that close timing without a latch, could drop a stage. That is a deferred per-target knob.
always @* begin
    """)
    w.push()
    w("""
if (rst)              next_pc = 0;
else if (pc == LASTPC) next_pc = out_ready ? 0 : LASTPC;  // present: hold until the result is taken
else if (pc == 0)      next_pc = in_valid ? 1 : 0;        // accept: hold until a transaction arrives
else                   next_pc = pc + 1'b1;               // advance the fetch
""")
    w.pop()
    w("""
end

reg [PCW-1:0] ucode_addr_q;  // PC latch: splits pc -> next_pc -> ROM address cone from the BRAM read

always @(posedge clk) begin
""")
    w.push()
    w("""
ucode_addr_q <= next_pc;                // 1st stage: PC latch (route-split helper)
ucode_q      <= ucode[ucode_addr_q];    // 2nd stage: control-store array read
ucode_word   <= ucode_q;                // 3rd stage: BRAM output register (fast clock-to-out)
if (rst) begin
""")
    w.push()
    w("""
pc     <= 0;
err_pc_q <= 0;
""")
    w.pop()
    w("end else begin")
    w.push()
    w("""
pc <= next_pc;
if ((pc == 0) && in_valid) err_pc_q <= 0;  // clear the diagnostic when a new transaction is accepted
if (err) err_pc_q <= pc - FETCH_LAG;        // execution lags the fetch PC by FETCH_LAG, so the step is pc-FETCH_LAG
""")
    w.pop()
    w("end")
    w.pop()
    w("""
end

""")


def _const_term_expr(port: int, consts: list[int]) -> str:
    expr = f"const_{consts[-1]}"
    for local in range(len(consts) - 2, -1, -1):
        expr = f"({f_cidx(port)} == {local}) ? const_{consts[local]} : {expr}"
    return expr


def _emit_read_latch(
    w: _Writer, target: str, port: int, read_set: list[int], port_consts: dict[int, list[int]], waddr: int
) -> None:
    """
    Emit the read mux + read latch for one operand: ``target`` is registered from the value selected this step.

    The mux spans only ``read_set`` (the registers this port ever reads): no register at all when the operand is
    always a constant, a direct wire for a single register, a case over the read-set otherwise. A const-select picks
    the immediate when the operand is sometimes a constant. On idle steps the latch captures a don't-care value that
    the operator ignores (its in_valid is low).
    """
    consts = port_consts.get(port)
    cterm = _cterm_expr(port, consts) if consts else None
    if not read_set:  # the operand is always a constant immediate
        w(f"always @(posedge clk) {target} <= {cterm};")
        return
    if len(read_set) == 1:
        reg_expr = f"regs[{read_set[0]}]"
        rhs = f"{f_selc(port)} ? {cterm} : {reg_expr}" if cterm else reg_expr
        w(f"always @(posedge clk) {target} <= {rhs};")
        return
    w("always @(posedge clk) begin")
    w.push()
    if cterm:
        w(f"if ({f_selc(port)}) {target} <= {cterm};")
        w(f"else case ({f_rd(port)})")
    else:
        w(f"case ({f_rd(port)})")
    w.push()
    for reg in read_set:
        w(f"{_lit(waddr, reg)}: {target} <= regs[{reg}];")
    w(f"default: {target} <= regs[{read_set[0]}];")
    w.pop()
    w("endcase")
    w.pop()
    w("end")


def _emit_datapath(
    w: _Writer,
    lir: Lir,
    read_port: dict[tuple[FloatOperatorInstance, int], int],
    port_consts: dict[int, list[int]],
    read_sets: dict[tuple[FloatOperatorInstance, int], list[int]],
    write_sets: dict[int, list[FloatOperatorInstance]],
    inst_targets: dict[FloatOperatorInstance, set[int]],
    waddr: int,
) -> None:
    nreg = max(1, lir.float_regfile.nreg)

    for port in sorted(port_consts):
        if len(port_consts[port]) > 1:
            w(f"wire [W-1:0] cterm{port} = {_const_term_expr(port, port_consts[port])};")
    if lir.float_inputs:
        w("wire load_en = (pc == 0) && in_valid;  // accept the input transaction into the load lanes")
    w("")

    w("// Operator control (in_valid and sign controls are consumed inside the wrapper on the issue step).")
    for inst in lir.float_instances:
        sig, base = _sig(inst), base_name(inst)
        w(f"assign {sig}_iv = {f_iv(base)};")
        w(f"assign {sig}_ys = {f_ysgn(base)};")
        for pos in range(inst.operator.arity):
            w(f"assign {sig}_{PORT_LETTERS[pos]}s = {f_osgn(base, PORT_LETTERS[pos])};")
    w("")

    w("// Operand read ports: a sparse mux over each operand's read-set, registered by the read latch.")
    for inst in lir.float_instances:
        sig = _sig(inst)
        for pos in range(inst.operator.arity):
            letter = PORT_LETTERS[pos]
            port = read_port[(inst, pos)]
            _emit_read_latch(w, f"{sig}_{letter}", port, read_sets.get((inst, pos), []), port_consts, waddr)
    w("")

    w("// Writeback latches: the operator result (and any error sideband) is registered before the register write.")
    for inst in lir.float_instances:
        sig = _sig(inst)
        w(f"always @(posedge clk) {sig}_y_q <= {sig}_y;")
        for err_port in inst.operator.error_ports:
            w(f"always @(posedge clk) {sig}_{err_port}_q <= {sig}_{err_port};")
    w("")

    w("// Register writes: a synchronous select spanning only each register's writers (plus the input load).")
    covered = {load.dst.index: load.name for load in lir.float_inputs}
    for reg in range(nreg):
        writers = write_sets.get(reg, [])
        is_load = reg in covered
        if not is_load and not writers:
            continue  # an unused register (only the NREG>=1 floor with no values); leave it undriven
        w("always @(posedge clk) begin")
        w.push()
        clause = "if"
        if is_load:
            w(f"if (load_en) regs[{reg}] <= in_{covered[reg]};")
            clause = "else if"
        for inst in writers:
            sig, base = _sig(inst), base_name(inst)
            # A register written by a single instance that only ever writes here needs no address compare.
            if inst_targets.get(inst) == {reg}:
                cond = f_we(base)
            else:
                cond = f"{f_we(base)} && ({f_wa(base)} == {_lit(waddr, reg)})"
            w(f"{clause} ({cond}) regs[{reg}] <= {sig}_y_q;")
            clause = "else if"
        w.pop()
        w("end")
    w("")

    # An error matters only on the step its operator commits, which is exactly that instance's write-enable; both the
    # write-enable and the error sideband are aligned to the writeback latch (commit + WRITE_LATCH).
    err_terms = [
        f"({f_we(base_name(inst))} & {_sig(inst)}_{port}_q)"
        for inst in lir.float_instances
        for port in inst.operator.error_ports
    ]
    err_rhs = " | ".join(err_terms) if err_terms else "1'b0"
    w(f"assign err = {err_rhs};")
    w("")


def _emit_outputs(w: _Writer, lir: Lir) -> None:
    w("""
assign in_ready  = (pc == 0);
assign out_valid = (pc == LASTPC);  // the result is valid in the array on PRESENT; execution lags the fetch by FETCH_LAG
assign err_pc    = err_pc_q;
""")
    for index, wire in enumerate(lir.float_outputs):
        if isinstance(wire.source, FloatConstRef):
            raw = f"const_{wire.source.index}"
        else:
            raw = f"regs[{wire.source.index}]"
        if wire.sign == FloatSignControl():
            w(f"assign {wire.name} = {raw};")
        else:
            w(f"holoso_fsgnop #(.WFULL(W)) u_outsgn_{index} (.x({raw}), .op(2'd{wire.sign.encoded}), .y({wire.name}));")
    w("")
