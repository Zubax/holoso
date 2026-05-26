"""
Render a :class:`Lir` into a synthesizable Verilog ZISC module, plus access to the shared ``holoso_support`` HDL that
the generated module instantiates.
"""

from dataclasses import dataclass
from importlib import resources
from string import ascii_letters

from ..._lir import ConstRef, Lir, OperatorInstance, Operand, RegRef, ScheduledOp
from ..._operators import ALL_OP_CLASSES, Sgnop

_CLASS_ORDER = {cls: index for index, cls in enumerate(ALL_OP_CLASSES)}
_PORT_LETTERS = ascii_letters  # operand position -> wrapper port letter (a, b, ...)

_SUPPORT_FILES = {
    name: resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
    for name in ("holoso_support.v", "holoso_support.vh")
}


@dataclass(frozen=True, slots=True)
class VerilogOutput:
    """The Verilog backend's output: the generated module text and the shared support files it instantiates."""

    verilog: str
    support_files: dict[str, str]  # filename -> content


class VerilogWriter:
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


def _base(inst: OperatorInstance) -> str:
    return f"{inst.op.mnemonic}_{inst.index}"


def _sig(inst: OperatorInstance) -> str:
    return f"s_{_base(inst)}"


def _rf_data(port: int) -> str:
    return f"`REGF_DATA({port})"


def _rf_addr(port: int) -> str:
    return f"`REGF_ADDR({port})"


def _rf_view(reg: int) -> str:
    return f"`REGF_VIEW({reg})"


def _operand_value(operand: Operand, lanes: dict[int, int]) -> str:
    if isinstance(operand.source, ConstRef):
        return f"const_{operand.source.index}"
    return f"rf_rd_data[{_rf_data(lanes[operand.source.index])}]"


def _operand_name(operand: Operand) -> str:
    base = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return operand.sgnop.decorate(base)


def _op_expr(op: ScheduledOp) -> str:
    return f"r{op.dst.index}={op.inst.op.render(*[_operand_name(o) for o in op.operands])}"


def _cycle_summary(issues: list[ScheduledOp], commits: list[ScheduledOp]) -> str:
    parts: list[str] = []
    if issues:
        parts.append("issue " + ", ".join(_op_expr(op) for op in issues))
    if commits:
        parts.append("commit " + ", ".join(f"r{op.dst.index}" for op in commits))
    return "; ".join(parts)


def _group_by_cycle(lir: Lir) -> tuple[dict[int, list[ScheduledOp]], dict[int, list[ScheduledOp]]]:
    """Group the schedule into per-cycle issues (by issue_cycle) and commits (by commit_cycle), canonically ordered."""
    issues: dict[int, list[ScheduledOp]] = {}
    commits: dict[int, list[ScheduledOp]] = {}
    for op in lir.ops:
        issues.setdefault(op.issue_cycle, []).append(op)
        commits.setdefault(op.commit_cycle, []).append(op)
    for group in (issues, commits):
        for ops in group.values():
            ops.sort(key=lambda op: (_CLASS_ORDER[type(op.inst.op)], op.inst.index))
    return issues, commits


def _read_lanes_for(issues: list[ScheduledOp]) -> dict[int, int]:
    """Assign a read-port lane to each distinct register operand read by a cycle's issues (shared reads dedup)."""
    lanes: dict[int, int] = {}
    for op in issues:
        for operand in op.operands:
            if isinstance(operand.source, RegRef) and operand.source.index not in lanes:
                lanes[operand.source.index] = len(lanes)
    return lanes


def generate(lir: Lir) -> VerilogOutput:
    w = VerilogWriter()
    waddr = max(1, (lir.regfile.nreg - 1).bit_length())
    cycw = lir.cyc_width
    issues_by_cycle, commits_by_cycle = _group_by_cycle(lir)
    read_lanes = {cycle: _read_lanes_for(issues) for cycle, issues in issues_by_cycle.items()}

    _emit_header(w, lir, cycw)
    _emit_localparams(w, lir, waddr, cycw)
    _emit_declarations(w, lir)
    _emit_consts(w, lir)
    _emit_regfile(w, lir)
    _emit_operators(w, lir)
    _emit_datapath(w, lir, issues_by_cycle, commits_by_cycle, read_lanes)
    _emit_fsm(w)
    _emit_outputs(w, lir)
    w.lines("endmodule", "", "`undef REGF_DATA", "`undef REGF_ADDR", "`undef REGF_VIEW")
    return VerilogOutput(verilog=w.render(), support_files=_SUPPORT_FILES)


def _emit_header(w: VerilogWriter, lir: Lir, cycw: int) -> None:
    w.lines('`include "holoso_support.vh"', "`timescale 1ns/1ps", "")
    w.line(
        f"// Float format: exponent {lir.fmt.wexp} bits, significand {lir.fmt.wman} bits, total {lir.fmt.width} bits."
    )
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
    for load in lir.inputs:
        w.line(f"input  wire [{lir.fmt.width - 1}:0] in_{load.name},")
    _emit_port_group(w, "OUTPUT PORTS", "Valid when out_valid is pulsed.")
    for wire in lir.outputs:
        w.line(f"output wire [{lir.fmt.width - 1}:0] {wire.name},")
    _emit_port_group(w, "DIAGNOSTIC PORTS", "Runtime diagnostics available while the module is running.")
    # err_cyc: 0 = no error; otherwise the (last) cycle an error was detected. |err_cyc answers "any error?".
    w.line(f"output reg  [{cycw - 1}:0] err_cyc")
    w.pop()
    w.lines(");", "")


def _emit_port_group(w: VerilogWriter, title: str, comment: str) -> None:
    w.lines(f"// {title}", f"// {comment}")


def _emit_localparams(w: VerilogWriter, lir: Lir, waddr: int, cycw: int) -> None:
    w.line(f"localparam WEXP  = {lir.fmt.wexp};  // Float exponent bits fixed by the static schedule")
    w.line(f"localparam WMAN  = {lir.fmt.wman};  // Float mantissa bits fixed by the static schedule")
    w.line("localparam W     = WEXP + WMAN;")
    w.line(
        f"localparam NREG  = {max(1, lir.regfile.nreg)};  // >= 1; the bank is unused when no value needs a register"
    )
    w.line(f"localparam WADDR = {waddr};")
    w.line(f"localparam NRD   = {lir.regfile.nrd};")
    w.line(f"localparam NWR   = {lir.regfile.nwr};")
    w.line(f"localparam NLOAD = {lir.regfile.nload};")
    w.line(f"localparam CYCW  = {cycw};")
    w.line(f"localparam [CYCW-1:0] LAST = {lir.makespan + 1};")
    compute = f"1..{lir.makespan} = pipelined compute, " if lir.makespan else ""
    w.line(f"// cyc: 0 = idle/accept, {compute}LAST = present outputs")
    w.lines("", "`define REGF_DATA(PORT) `HOLOSO_REGFILE_LANE(W, PORT)")
    w.line("`define REGF_ADDR(PORT) `HOLOSO_REGFILE_LANE(WADDR, PORT)")
    w.line("`define REGF_VIEW(REG)  `HOLOSO_REGFILE_LANE(W, REG)")
    w.line("")


def _emit_declarations(w: VerilogWriter, lir: Lir) -> None:
    w.lines("reg [CYCW-1:0] cyc;", "reg err;  // combinational: an error is detected on the current cycle", "")
    w.lines(
        "reg  [NWR-1:0]       rf_wr_en;",
        "reg  [NWR*WADDR-1:0] rf_wr_addr;",
        "reg  [NWR*W-1:0]     rf_wr_data;",
        "reg  [NRD*WADDR-1:0] rf_rd_addr;",
        "wire [NRD*W-1:0]     rf_rd_data;",
        "wire [NREG*W-1:0]    rf_view;",
        "",
    )
    if lir.regfile.nload:
        w.lines(
            "reg                  rf_load_en;",
            "reg  [NLOAD*W-1:0]   rf_load_data;",
            "",
        )
    for inst in lir.instances:
        sig = _sig(inst)
        w.line(f"reg          {sig}_iv;")
        for letter in _PORT_LETTERS[: inst.op.arity]:
            w.line(f"reg  [1:0]   {sig}_{letter}s;")
            w.line(f"reg  [W-1:0] {sig}_{letter};")
        w.line(f"reg  [1:0]   {sig}_ys;")
        w.line(f"wire [W-1:0] {sig}_y;")
        for port in inst.op.error_ports:
            w.line(f"wire         {sig}_{port};")
    w.line("")


def _emit_consts(w: VerilogWriter, lir: Lir) -> None:
    width = lir.fmt.width
    digits = (width + 3) // 4
    for index, value in enumerate(lir.consts):
        w.line(f"wire [W-1:0] const_{index} = {width}'h{lir.fmt.encode(value):0{digits}x};  // {value!r}")
    if lir.consts:
        w.line("")


def _emit_regfile(w: VerilogWriter, lir: Lir) -> None:
    w.line("// Read-first register file (RWPASS=0): a value written on a cycle is readable only on the next cycle.")
    w.line("// The scheduler's +1 dependency latency and the allocator's last_use<=def register sharing both rely")
    w.line("// on this; do NOT switch to write-through (RWPASS=1) without revisiting holoso/regalloc.py.")
    w.line(
        "holoso_regfile #(.W(W), .WADDR(WADDR), .NRD(NRD), .NWR(NWR), .NLOAD(NLOAD), .NREG(NREG), .RWPASS(0)) u_rf ("
    )
    w.push()
    w.line(".clk(clk),")
    if lir.regfile.nload:
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


def _emit_operators(w: VerilogWriter, lir: Lir) -> None:
    for inst in lir.instances:
        sig = _sig(inst)
        letters = _PORT_LETTERS[: inst.op.arity]
        # WEXP/WMAN frame the float format; hdl_params() adds K (ilog2) and any enabled STAGE_* (defaults omitted),
        # so the schedule's op.latency and the emitted instantiation params always describe the same module.
        parts = [".WEXP(WEXP)", ".WMAN(WMAN)"] + [f".{param}({value})" for param, value in inst.op.hdl_params().items()]
        params = "#(" + ", ".join(parts) + ")"
        w.line(f"{inst.op.module_name} {params} u_{_base(inst)} (")
        w.push()
        w.line(f".clk(clk), .rst(rst), .in_valid({sig}_iv),")
        sgn_ports = ", ".join(f".{letter}_sgnop({sig}_{letter}s)" for letter in letters)
        w.line(f"{sgn_ports}, .y_sgnop({sig}_ys),")
        data_ports = ", ".join(f".{letter}({sig}_{letter})" for letter in letters)
        w.line(f"{data_ports},")
        # out_valid is left unconnected: the static schedule already knows when each result is ready.
        tail = f".out_valid(), .y({sig}_y)"
        for port in inst.op.error_ports:
            tail += f", .{port}({sig}_{port})"
        w.line(tail)
        w.pop()
        w.lines(");", "")


def _emit_datapath(
    w: VerilogWriter,
    lir: Lir,
    issues_by_cycle: dict[int, list[ScheduledOp]],
    commits_by_cycle: dict[int, list[ScheduledOp]],
    read_lanes: dict[int, dict[int, int]],
) -> None:
    # One combinational block: per cycle, set the operand reads + in_valid for issuing operators and the write ports
    # for committing operators. `always @*` over `reg` targets is pure combinational logic; the only flip-flops are
    # the regfile and the control block below.
    w.line("always @* begin")
    w.push()
    for inst in lir.instances:
        sig = _sig(inst)
        w.line(f"{sig}_iv = 1'b0; {sig}_ys = 2'd0;")
        for letter in _PORT_LETTERS[: inst.op.arity]:
            w.line(f"{sig}_{letter} = {{W{{1'b0}}}}; {sig}_{letter}s = 2'd0;")
    w.lines(
        "rf_rd_addr = {(NRD*WADDR){1'b0}};",
        "rf_wr_en   = {NWR{1'b0}};",
        "rf_wr_addr = {(NWR*WADDR){1'b0}};",
        "rf_wr_data = {(NWR*W){1'b0}};",
        "err        = 1'b0;",
    )
    if lir.regfile.nload:
        w.lines(
            "rf_load_en = 1'b0;",
            "rf_load_data  = {(NLOAD*W){1'b0}};",
        )
    w.line("case (cyc)")
    w.push()
    if lir.regfile.nload:
        w.line("0: if (in_valid) begin  // parallel-load the input ports into registers 0..NLOAD-1 in one cycle")
        w.push()
        w.line("rf_load_en = 1'b1;")
        for load in lir.inputs:
            w.line(f"rf_load_data[{_rf_view(load.dst.index)}] = in_{load.name};")
        w.pop()
        w.line("end")
    for cycle in sorted(set(issues_by_cycle) | set(commits_by_cycle)):
        issues = issues_by_cycle.get(cycle, [])
        commits = commits_by_cycle.get(cycle, [])
        lanes = read_lanes.get(cycle, {})
        w.line(f"{cycle}: begin  // {_cycle_summary(issues, commits)}")
        w.push()
        for reg, lane in lanes.items():
            w.line(f"rf_rd_addr[{_rf_addr(lane)}] = {reg};")
        for op in issues:
            sig = _sig(op.inst)
            w.line(f"{sig}_iv = 1'b1;")
            for letter, operand in zip(_PORT_LETTERS, op.operands):
                w.line(f"{sig}_{letter} = {_operand_value(operand, lanes)}; {sig}_{letter}s = 2'd{int(operand.sgnop)};")
            w.line(f"{sig}_ys = 2'd{int(op.y_sgnop)};")
        for lane, op in enumerate(commits):
            sig = _sig(op.inst)
            w.line(f"rf_wr_en[{lane}] = 1'b1;")
            w.line(f"rf_wr_addr[{_rf_addr(lane)}] = {op.dst.index};")
            w.line(f"rf_wr_data[{_rf_data(lane)}] = {sig}_y;")
        err_terms = [f"{_sig(op.inst)}_{port}" for op in commits for port in op.inst.op.error_ports]
        if err_terms:
            w.line(f"err = {' | '.join(err_terms)};  // error(s) detected as these operators commit")
        w.pop()
        w.line("end")
    w.line("default: ;")
    w.pop()
    w.lines("endcase", "")
    w.pop()
    w.lines("end", "")


def _emit_fsm(w: VerilogWriter) -> None:
    # Sequential control: a plain up-counter replaying the static schedule, plus the generic error latch. All
    # cycle-indexed logic -- including which operator's error matters when -- lives in the combinational block above;
    # here we only advance the counter and, every cycle, latch err_cyc <= cyc whenever the combinational `err` is set.
    w.line("always @(posedge clk) begin")
    w.push()
    w.line("if (rst) begin")
    w.push()
    w.lines("cyc     <= 0;", "err_cyc <= 0;")
    w.pop()
    w.line("end else begin")
    w.push()
    w.line("if (cyc == 0) begin")
    w.push()
    w.line("if (in_valid) begin")
    w.push()
    w.line("cyc     <= 1;")
    w.line("err_cyc <= 0;")
    w.pop()
    w.line("end")
    w.pop()
    w.line("end else if (cyc == LAST) begin")
    w.push()
    w.line("if (out_ready) cyc <= 0;")
    w.pop()
    w.line("end else begin")
    w.push()
    w.line("cyc <= cyc + 1'b1;")
    w.pop()
    w.line("end")
    w.line("if (err) err_cyc <= cyc;")
    w.pop()
    w.line("end")
    w.pop()
    w.lines("end", "")


def _emit_outputs(w: VerilogWriter, lir: Lir) -> None:
    w.line("assign in_ready  = (cyc == 0);")
    w.line("assign out_valid = (cyc == LAST);")
    for index, wire in enumerate(lir.outputs):
        if isinstance(wire.source, ConstRef):
            raw = f"const_{wire.source.index}"
        else:
            raw = f"rf_view[{_rf_view(wire.source.index)}]"
        if wire.sgnop is Sgnop.NONE:
            w.line(f"assign {wire.name} = {raw};")
        else:
            w.line(
                f"holoso_fsgnop #(.WFULL(W)) u_outsgn_{index} (.x({raw}), .op(2'd{int(wire.sgnop)}), .y({wire.name}));"
            )
    w.line("")
