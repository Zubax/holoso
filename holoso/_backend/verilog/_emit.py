"""
Render a scheduled :class:`Lir` into a synthesizable Verilog ZISC module, plus access to the shared ``holoso_support``
HDL that the generated module instantiates.

The controller is a microcode ROM (see :mod:`._microcode`): one pre-decoded VLIW control word per step, stored in a
(BRAM-inferable) ROM and registered on read. Addressing the ROM with the *next* program counter and registering its
output keeps the word for step N available exactly during step N (no added latency), while splitting the old
combinational ``case(cyc)`` cone into two short register-to-register paths -- ``pc -> ROM -> word`` and
``word -> datapath``. Each operator operand has a dedicated register-file read port (the word carries only its address,
no operand crossbar), and each operator instance has a dedicated write port (its result wires straight in). Control
fields that are constant across the whole program (very common for sign controls) are driven by constant nets and
omitted from the ROM, so synthesis prunes the logic they feed.
"""

from dataclasses import dataclass
from importlib import resources

from ..._lir import FloatConstRef, Lir, FloatOperatorInstance, FloatScheduledOp
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
    group_by_cycle,
    pack,
    port_const_map,
    read_ports,
    write_ports,
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
    """Accumulates 4-space-indented lines. Use :meth:`line` for content and :meth:`push`/:meth:`pop` for nesting."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def line(self, text: str = "") -> None:
        self._lines.append(("    " * self._depth + text) if text else "")

    def lines(self, *texts: str) -> None:
        for text in texts:
            self.line(text)

    def push(self) -> None:
        self._depth += 1

    def pop(self) -> None:
        assert self._depth > 0
        self._depth -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def _sig(inst: FloatOperatorInstance) -> str:
    return f"s_{base_name(inst)}"


def _rf_data(port: int) -> str:
    return f"`REGF_DATA({port})"


def _rf_addr(port: int) -> str:
    return f"`REGF_ADDR({port})"


def _rf_view(reg: int) -> str:
    return f"`REGF_VIEW({reg})"


def _decl_range(width: int) -> str:
    return "" if width == 1 else f"[{width - 1}:0] "


def _lit(width: int, value: int) -> str:
    return f"{width}'d{value}"


def generate(lir: Lir) -> VerilogOutput:
    w = _Writer()
    rf = lir.float_regfile
    waddr = max(1, (rf.nreg - 1).bit_length())
    cycw = lir.cyc_width

    # Dedicated ports: one read port per operator operand, one write port per operator instance (matching
    # FloatRegFileLayout.nrd/nwr), so the control word carries only addresses and there is no operand/write crossbar.
    read_port = read_ports(lir)
    write_port = write_ports(lir)
    port_consts = port_const_map(lir, read_port)
    fields = build_microcode(lir, read_port, port_consts, waddr)
    ucw = finalize_fields(fields)
    issues_by_cycle, commits_by_cycle = group_by_cycle(lir)

    _emit_header(w, lir, cycw)
    _emit_localparams(w, lir, waddr, cycw, ucw)
    _emit_declarations(w, lir)
    _emit_consts(w, lir)
    _emit_regfile(w, lir)
    _emit_operators(w, lir)
    _emit_microcode_rom(w, fields, ucw, lir.makespan + 2, issues_by_cycle, commits_by_cycle)
    _emit_field_wires(w, fields)
    _emit_sequencer(w)
    _emit_datapath(w, lir, read_port, write_port, port_consts)
    _emit_outputs(w, lir)
    w.lines("endmodule", "", "`undef REGF_DATA", "`undef REGF_ADDR", "`undef REGF_VIEW")
    return VerilogOutput(verilog=w.render(), support_files=_SUPPORT_FILES)


def _emit_header(w: _Writer, lir: Lir, cycw: int) -> None:
    fmt = lir.float_regfile.fmt
    w.lines('`include "holoso_support.vh"', "`timescale 1ns/1ps", "")
    w.line(f"// Float format: exponent {fmt.wexp} bits, significand {fmt.wman} bits, total {fmt.width} bits.")
    w.line(f"module {lir.module_name} (")
    w.push()
    _emit_port_group(w, "CONTROL PORTS", "Clock/reset and ready/valid handshake for one scheduled invocation.")
    ports = [
        "input  wire clk,",
        "input  wire rst,",
        "input  wire in_valid,",
        "output wire in_ready,",
        "output wire out_valid,",
        "input  wire out_ready,",
    ]
    for line in ports:
        w.line(line)
    _emit_port_group(w, "INPUT PORTS", "Latched when in_valid && in_ready.")
    for load in lir.float_inputs:
        w.line(f"input  wire [{fmt.width - 1}:0] in_{load.name},")
    _emit_port_group(w, "OUTPUT PORTS", "Valid when out_valid is pulsed.")
    for wire in lir.float_outputs:
        w.line(f"output wire [{fmt.width - 1}:0] {wire.name},")
    _emit_port_group(w, "DIAGNOSTIC PORTS", "Runtime diagnostics available while the module is running.")
    # err_pc: 0 = no error; otherwise the (last) step an error was detected. |err_pc answers "any error?".
    w.line(f"output reg  [{cycw - 1}:0] err_pc")
    w.pop()
    w.lines(");", "")


def _emit_port_group(w: _Writer, title: str, comment: str) -> None:
    w.lines(f"// {title}", f"// {comment}")


def _emit_localparams(w: _Writer, lir: Lir, waddr: int, cycw: int, ucw: int) -> None:
    fmt = lir.float_regfile.fmt
    w.line(f"localparam WEXP  = {fmt.wexp};  // Float exponent bits fixed by the static schedule")
    w.line(f"localparam WMAN  = {fmt.wman};  // Float mantissa bits fixed by the static schedule")
    w.line("localparam W     = WEXP + WMAN;")
    w.line(
        f"localparam NREG  = {max(1, lir.float_regfile.nreg)};  // >= 1; the bank is unused when no value needs a register"
    )
    w.line(f"localparam WADDR = {waddr};")
    w.line(f"localparam NRD   = {lir.float_regfile.nrd};  // dedicated read ports: one per operator operand")
    w.line(f"localparam NWR   = {lir.float_regfile.nwr};  // dedicated write ports: one per operator instance")
    w.line(f"localparam NLOAD = {lir.float_regfile.nload};")
    w.line(f"localparam CYCW  = {cycw};")
    w.line(f"localparam [CYCW-1:0] LAST = {lir.makespan + 1};")
    w.line(f"localparam UCW   = {ucw};  // microcode word width after lifting out constant control fields")
    compute = f"1..{lir.makespan} = pipelined compute, " if lir.makespan else ""
    w.line(f"// pc: 0 = idle/accept, {compute}LAST = present outputs")
    w.lines("", "`define REGF_DATA(PORT) `HOLOSO_REGFILE_LANE(W, PORT)")
    w.line("`define REGF_ADDR(PORT) `HOLOSO_REGFILE_LANE(WADDR, PORT)")
    w.line("`define REGF_VIEW(REG)  `HOLOSO_REGFILE_LANE(W, REG)")
    w.line("")


def _emit_declarations(w: _Writer, lir: Lir) -> None:
    w.lines(
        "reg  [CYCW-1:0] pc;       // program counter: the current step",
        "reg  [CYCW-1:0] next_pc;  // combinational next-state presented to the ROM each cycle",
        "wire err;  // an operator error is detected on the current step",
        "",
    )
    w.lines(
        "wire [NWR-1:0]       rf_wr_en;",
        "wire [NWR*WADDR-1:0] rf_wr_addr;",
        "wire [NWR*W-1:0]     rf_wr_data;",
        "wire [NRD*WADDR-1:0] rf_rd_addr;",
        "wire [NRD*W-1:0]     rf_rd_data;",
        "wire [NREG*W-1:0]    rf_view;",
        "",
    )
    if lir.float_regfile.nload:
        w.lines(
            "wire                 rf_load_en;",
            "wire [NLOAD*W-1:0]   rf_load_data;",
            "",
        )
    for inst in lir.float_instances:
        sig = _sig(inst)
        w.line(f"wire         {sig}_iv;")
        for letter in PORT_LETTERS[: inst.operator.arity]:
            w.line(f"wire [1:0]   {sig}_{letter}s;")
            w.line(f"wire [W-1:0] {sig}_{letter};")
        w.line(f"wire [1:0]   {sig}_ys;")
        w.line(f"wire [W-1:0] {sig}_y;")
        for port in inst.operator.error_ports:
            w.line(f"wire         {sig}_{port};")
    w.line("")


def _emit_consts(w: _Writer, lir: Lir) -> None:
    fmt = lir.float_regfile.fmt
    width = fmt.width
    digits = (width + 3) // 4
    for index, value in enumerate(lir.float_consts):
        w.line(f"wire [W-1:0] const_{index} = {width}'h{fmt.encode(value):0{digits}x};  // {value!r}")
    if lir.float_consts:
        w.line("")


def _emit_regfile(w: _Writer, lir: Lir) -> None:
    w.line("// Read-first register file (RWPASS=0): a value written on a step is readable only on the next step.")
    w.line("// The scheduler's +1 dependency latency and the allocator's last_use<=def register sharing rely on this.")
    w.line(
        "holoso_regfile #(.W(W), .WADDR(WADDR), .NRD(NRD), .NWR(NWR), .NLOAD(NLOAD), .NREG(NREG), .RWPASS(0)) u_rf ("
    )
    w.push()
    w.line(".clk(clk),")
    if lir.float_regfile.nload:
        w.line(".load_en(rf_load_en), .load_data(rf_load_data),")
    else:
        w.line(".load_en(1'b0), .load_data(1'b0),  // no inputs: load port disabled (NLOAD=0)")
    w.lines(
        ".wr_en(rf_wr_en), .wr_addr(rf_wr_addr), .wr_data(rf_wr_data),",
        ".rd_addr(rf_rd_addr), .rd_data(rf_rd_data),",
        ".view(rf_view)",
    )
    w.pop()
    w.lines(");", "")


def _emit_operators(w: _Writer, lir: Lir) -> None:
    for inst in lir.float_instances:
        sig = _sig(inst)
        letters = PORT_LETTERS[: inst.operator.arity]
        # WEXP/WMAN frame the float format; hdl_params() lists K (ilog2) and every STAGE_* explicitly (including zeros),
        # so the instantiation is self-describing and a param-name mismatch with the wrapper fails loudly at elaboration.
        parts = [".WEXP(WEXP)", ".WMAN(WMAN)"] + [
            f".{param}({value})" for param, value in inst.operator.hdl_params().items()
        ]
        params = "#(" + ", ".join(parts) + ")"
        w.line(f"{inst.operator.module_name} {params} u_{base_name(inst)} (")
        w.push()
        w.line(f".clk(clk), .rst(rst), .in_valid({sig}_iv),")
        sgn_ports = ", ".join(f".{letter}_sgnop({sig}_{letter}s)" for letter in letters)
        w.line(f"{sgn_ports}, .y_sgnop({sig}_ys),")
        data_ports = ", ".join(f".{letter}({sig}_{letter})" for letter in letters)
        w.line(f"{data_ports},")
        # out_valid is left unconnected: the static schedule already knows when each result is ready.
        tail = f".out_valid(), .y({sig}_y)"
        for port in inst.operator.error_ports:
            tail += f", .{port}({sig}_{port})"
        w.line(tail)
        w.pop()
        w.lines(");", "")


def _emit_microcode_rom(
    w: _Writer,
    fields: dict[str, Field],
    ucw: int,
    depth: int,
    issues_by_cycle: dict[int, list[FloatScheduledOp]],
    commits_by_cycle: dict[int, list[FloatScheduledOp]],
) -> None:
    digits = (ucw + 3) // 4
    w.line("// Microcode VLIW ROM: one pre-decoded control word per step, registered on read (in the sequencer below).")
    w.line("// Constant control fields are lifted out (below) and not stored here, enabling synthesis-time folding.")
    w.line('(* rom_style = "block", ram_style = "block", syn_romstyle = "EBR" *)')
    w.line("reg [UCW-1:0] ucode [0:LAST];")
    w.line("initial begin")
    w.push()
    for step in range(depth):
        summary = cycle_summary(issues_by_cycle.get(step, []), commits_by_cycle.get(step, []))
        comment = f"  // {summary}" if summary else ""
        w.line(f"ucode[{step: 5}] = {ucw}'h{pack(fields, step):0{digits}x};{comment}")
    w.pop()
    w.lines("end", "")
    w.line("reg [UCW-1:0] ucode_word;  // the current step's control word (registered ROM read; see the sequencer)")
    w.line("")


def _emit_field_wires(w: _Writer, fields: dict[str, Field]) -> None:
    w.line("// Decoded control fields. A field that is constant across the whole program is driven by a constant net")
    w.line("// (so synthesis prunes the logic it feeds); a varying field is a slice of the instruction word.")
    for f in fields.values():
        if f.offset < 0:
            w.line(f"wire {_decl_range(f.width)}{f.name} = {_lit(f.width, f.const_value)};")
        elif f.width == 1:
            w.line(f"wire {f.name} = ucode_word[{f.offset}];")
        else:
            w.line(f"wire {_decl_range(f.width)}{f.name} = ucode_word[{f.offset} +: {f.width}];")
    w.line("")


def _emit_sequencer(w: _Writer) -> None:
    w.line("// Sequencer. The ROM is addressed with next_pc, so the registered word for step N is ready exactly during")
    w.line("// step N (no added latency). Reset covers only the control state (pc, err_pc); the ROM word register is")
    w.line("// reset-unconditional and settles to ucode[0] before the first active step (next_pc is 0 under reset).")
    w.line("always @* begin")
    w.push()
    w.line("if (rst)             next_pc = 0;")
    w.line("else if (pc == LAST) next_pc = out_ready ? 0 : LAST;  // present: hold until the result is taken")
    w.line("else if (pc == 0)    next_pc = in_valid ? 1 : 0;      // accept: hold until a transaction arrives")
    w.line("else                 next_pc = pc + 1'b1;             // compute: replay the schedule")
    w.pop()
    w.lines("end", "")
    w.line("always @(posedge clk) begin")
    w.push()
    w.line("ucode_word <= ucode[next_pc];  // registered ROM read (BRAM-inferable); reset-unconditional, aligned by pc")
    w.line("if (rst) begin")
    w.push()
    w.lines("pc     <= 0;", "err_pc <= 0;")
    w.pop()
    w.line("end else begin")
    w.push()
    w.line("pc <= next_pc;")
    w.line("if ((pc == 0) && in_valid) err_pc <= 0;  // clear the diagnostic when a new transaction is accepted")
    w.line("if (err) err_pc <= pc;                   // latch the step on which an error was detected")
    w.pop()
    w.line("end")
    w.pop()
    w.lines("end", "")


def _operand_expr(port: int, port_consts: dict[int, list[int]]) -> str:
    # Constant operands keep using the const_<i> immediate wires through a small select. Alternatives, if this ever
    # becomes a constraint: (1) fold constants into the register file -- preload them like inputs so every operand is
    # a uniform register read (cleaner datapath, but moves special-casing into the allocator and grows NREG/NLOAD);
    # (2) emit explicit constant-load micro-instructions that move a constant into a free register just before use
    # (uniform operand path, better register pressure than (1), but adds scheduling/allocation complexity).
    rd = f"rf_rd_data[{_rf_data(port)}]"
    if port not in port_consts:
        return rd
    consts = port_consts[port]
    cterm = f"const_{consts[0]}" if len(consts) == 1 else f"cterm{port}"
    return f"{f_selc(port)} ? {cterm} : {rd}"


def _const_term_expr(port: int, consts: list[int]) -> str:
    expr = f"const_{consts[-1]}"
    for local in range(len(consts) - 2, -1, -1):
        expr = f"({f_cidx(port)} == {local}) ? const_{consts[local]} : {expr}"
    return expr


def _emit_datapath(
    w: _Writer,
    lir: Lir,
    read_port: dict[tuple[FloatOperatorInstance, int], int],
    write_port: dict[FloatOperatorInstance, int],
    port_consts: dict[int, list[int]],
) -> None:
    rf = lir.float_regfile
    owned_rd = set(read_port.values())
    owned_wr = set(write_port.values())

    for port in sorted(port_consts):
        if len(port_consts[port]) > 1:
            w.line(f"wire [W-1:0] cterm{port} = {_const_term_expr(port, port_consts[port])};")

    w.line("// Register-file read addresses: each dedicated port is wired to the step's address for its operand.")
    for port in range(rf.nrd):
        rhs = f_rd(port) if port in owned_rd else "{WADDR{1'b0}}"
        w.line(f"assign rf_rd_addr[{_rf_addr(port)}] = {rhs};")
    w.line("")

    w.line("// Operator control and operand data (operand data is fixed wiring from its dedicated read port).")
    for inst in lir.float_instances:
        sig, base = _sig(inst), base_name(inst)
        w.line(f"assign {sig}_iv = {f_iv(base)};")
        w.line(f"assign {sig}_ys = {f_ysgn(base)};")
        for pos in range(inst.operator.arity):
            letter = PORT_LETTERS[pos]
            w.line(f"assign {sig}_{letter}s = {f_osgn(base, letter)};")
            w.line(f"assign {sig}_{letter} = {_operand_expr(read_port[(inst, pos)], port_consts)};")
    w.line("")

    w.line("// Register-file write ports: each instance's result wires straight into its own dedicated port.")
    for inst in lir.float_instances:
        sig, base, port = _sig(inst), base_name(inst), write_port[inst]
        w.line(f"assign rf_wr_en[{port}] = {f_we(base)};")
        w.line(f"assign rf_wr_addr[{_rf_addr(port)}] = {f_wa(base)};")
        w.line(f"assign rf_wr_data[{_rf_data(port)}] = {sig}_y;")
    for port in range(rf.nwr):
        if port not in owned_wr:
            w.line(f"assign rf_wr_en[{port}] = 1'b0;")
            w.line(f"assign rf_wr_addr[{_rf_addr(port)}] = {{WADDR{{1'b0}}}};")
            w.line(f"assign rf_wr_data[{_rf_data(port)}] = {{W{{1'b0}}}};")
    w.line("")

    # An error matters only on the step its operator commits, which is exactly that instance's write-enable.
    err_terms = [
        f"({f_we(base_name(inst))} & {_sig(inst)}_{port})"
        for inst in lir.float_instances
        for port in inst.operator.error_ports
    ]
    err_rhs = " | ".join(err_terms) if err_terms else "1'b0"
    w.line(f"assign err = {err_rhs};")

    if rf.nload:
        w.line("")
        w.line("// Input parallel-load through the regfile load port, taken on the accept handshake.")
        w.line("assign rf_load_en = (pc == 0) && in_valid;")
        covered = {load.dst.index: load.name for load in lir.float_inputs}
        for lane in range(rf.nload):
            rhs = f"in_{covered[lane]}" if lane in covered else "{W{1'b0}}"
            w.line(f"assign rf_load_data[{_rf_view(lane)}] = {rhs};")
    w.line("")


def _emit_outputs(w: _Writer, lir: Lir) -> None:
    w.line("assign in_ready  = (pc == 0);")
    w.line("assign out_valid = (pc == LAST);")
    for index, wire in enumerate(lir.float_outputs):
        if isinstance(wire.source, FloatConstRef):
            raw = f"const_{wire.source.index}"
        else:
            raw = f"rf_view[{_rf_view(wire.source.index)}]"
        if wire.sign == FloatSignControl():
            w.line(f"assign {wire.name} = {raw};")
        else:
            w.line(
                f"holoso_fsgnop #(.WFULL(W)) u_outsgn_{index} "
                f"(.x({raw}), .op(2'd{wire.sign.encoded}), .y({wire.name}));"
            )
    w.line("")
