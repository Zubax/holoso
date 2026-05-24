"""Render a :class:`Lir` into a synthesizable Verilog ZISC module.

Datapath: a ``holoso_regfile`` flip-flop bank, one operator-wrapper instance per :class:`OperatorInstance`, and one
``holoso_fconst`` per pooled constant. Controller: a cycle counter ``cyc`` driving a ``case(cyc)`` microprogram that
replays the static software-pipelined schedule. ``cyc==0`` is idle/accept (inputs are written when ``in_valid``),
``cyc`` then advances every clock through the compute cycles ``1..makespan``, and ``cyc==LAST`` (``makespan+1``)
presents the outputs and asserts ``out_valid``. On each compute cycle the microprogram asserts ``in_valid`` to the
operators issued that cycle (driving their operand reads) and writes back the operators that commit that cycle (whose
result lands at the next edge). Because operator latencies are static the controller needs no scoreboard; the only
data-dependent signal is ``div0``, latched at each ``fdiv``'s commit cycle. The register file is read-first
(``RWPASS=0``). Reset covers only the control registers (``cyc``, ``diag_q``).
"""

from __future__ import annotations

from .emit import VerilogWriter
from .lir import ConstRef, Lir, OperatorInstance, Operand, RegRef, ScheduledOp
from .operators import MODULE_NAMES, OpKind, Sgnop, arity, has_div0

_KIND_ORDER = {kind: index for index, kind in enumerate(OpKind)}


def _base(inst: OperatorInstance) -> str:
    return f"{inst.kind.value}_{inst.index}"


def _sig(inst: OperatorInstance) -> str:
    return f"s_{_base(inst)}"


def _is_binary(inst: OperatorInstance) -> bool:
    return arity(inst.kind) == 2


def _operand_value(operand: Operand, lanes: dict[int, int]) -> str:
    if isinstance(operand.source, ConstRef):
        return f"const_{operand.source.index}"
    return f"rf_rd_data[`HOLOSO_REGFILE_LANE(W, {lanes[operand.source.index]})]"


def _operand_name(operand: Operand) -> str:
    base = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return operand.sgnop.decorate(base)


def _op_expr(op: ScheduledOp) -> str:
    dst = f"r{op.dst.index}"
    if op.inst.kind is OpKind.FMUL_ILOG2:
        return f"{dst}={_operand_name(op.a)}*2^{op.k}"
    symbol = {OpKind.FADD: "+", OpKind.FMUL: "*", OpKind.FDIV: "/"}[op.inst.kind]
    assert op.b is not None
    return f"{dst}={_operand_name(op.a)}{symbol}{_operand_name(op.b)}"


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
            ops.sort(key=lambda op: (_KIND_ORDER[op.inst.kind], op.inst.index))
    return issues, commits


def _read_lanes_for(issues: list[ScheduledOp]) -> dict[int, int]:
    """Assign a read-port lane to each distinct register operand read by a cycle's issues (shared reads dedup)."""
    lanes: dict[int, int] = {}
    for op in issues:
        for operand in (op.a, op.b):
            if operand is not None and isinstance(operand.source, RegRef) and operand.source.index not in lanes:
                lanes[operand.source.index] = len(lanes)
    return lanes


def _output_lanes(lir: Lir) -> dict[int, int]:
    lanes: dict[int, int] = {}
    for wire in lir.outputs:
        if isinstance(wire.source, RegRef) and wire.source.index not in lanes:
            lanes[wire.source.index] = len(lanes)
    return lanes


def generate(lir: Lir) -> str:
    w = VerilogWriter()
    waddr = max(1, (lir.regfile.nreg - 1).bit_length())
    cycw = max(1, (lir.makespan + 1).bit_length())
    issues_by_cycle, commits_by_cycle = _group_by_cycle(lir)
    read_lanes = {cycle: _read_lanes_for(issues) for cycle, issues in issues_by_cycle.items()}
    out_lanes = _output_lanes(lir)

    _emit_header(w, lir)
    _emit_localparams(w, lir, waddr, cycw)
    _emit_declarations(w, lir)
    _emit_consts(w, lir)
    _emit_regfile(w)
    _emit_operators(w, lir)
    _emit_datapath(w, lir, issues_by_cycle, commits_by_cycle, read_lanes, out_lanes)
    _emit_fsm(w, lir, commits_by_cycle)
    _emit_outputs(w, lir, out_lanes)
    w.line("endmodule")
    return w.render()


def _emit_header(w: VerilogWriter, lir: Lir) -> None:
    w.lines('`include "holoso_support.vh"', "`timescale 1ns/1ps", "")
    w.line(f"module {lir.module_name} #(")
    w.push()
    w.lines(f"parameter WEXP = {lir.fmt.wexp},", f"parameter WMAN = {lir.fmt.wman}")
    w.pop()
    w.line(") (")
    w.push()
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
    for load in lir.inputs:
        w.line(f"input  wire [WEXP+WMAN-1:0] in_{load.name},")
    for wire in lir.outputs:
        w.line(f"output wire [WEXP+WMAN-1:0] {wire.name},")
    w.line("output wire diag_error")
    w.pop()
    w.lines(");", "")


def _emit_localparams(w: VerilogWriter, lir: Lir, waddr: int, cycw: int) -> None:
    w.line("localparam W     = WEXP + WMAN;")
    w.line(f"localparam NREG  = {lir.regfile.nreg};")
    w.line(f"localparam WADDR = {waddr};")
    w.line(f"localparam NRD   = {lir.regfile.nrd};")
    w.line(f"localparam NWR   = {lir.regfile.nwr};")
    w.line(f"localparam CYCW  = {cycw};")
    w.line(f"localparam [CYCW-1:0] LAST = {lir.makespan + 1};")
    w.line(f"// cyc: 0 = idle/accept, 1..{lir.makespan} = pipelined compute, LAST = present outputs")
    w.line("")


def _emit_declarations(w: VerilogWriter, lir: Lir) -> None:
    w.lines("reg [CYCW-1:0] cyc;", "reg diag_q;", "")
    w.lines(
        "reg  [NWR-1:0]       rf_wr_en;",
        "reg  [NWR*WADDR-1:0] rf_wr_addr;",
        "reg  [NWR*W-1:0]     rf_wr_data;",
        "reg  [NRD*WADDR-1:0] rf_rd_addr;",
        "wire [NRD*W-1:0]     rf_rd_data;",
        "",
    )
    for value in range(len(lir.consts)):
        w.line(f"wire [W-1:0] const_{value};")
    if lir.consts:
        w.line("")
    for inst in lir.instances:
        sig = _sig(inst)
        w.line(f"reg          {sig}_iv;")
        w.line(f"reg  [1:0]   {sig}_as;")
        w.line(f"reg  [1:0]   {sig}_ys;")
        w.line(f"reg  [W-1:0] {sig}_a;")
        if _is_binary(inst):
            w.line(f"reg  [1:0]   {sig}_bs;")
            w.line(f"reg  [W-1:0] {sig}_b;")
        w.line(f"wire         {sig}_ov;")  # unused at runtime (the schedule is static); kept for the operator port
        w.line(f"wire [W-1:0] {sig}_y;")
        if has_div0(inst.kind):
            w.line(f"wire         {sig}_div0;")
    w.line("")


def _emit_consts(w: VerilogWriter, lir: Lir) -> None:
    for index, value in enumerate(lir.consts):
        w.line(
            f"holoso_fconst #(.WEXP(WEXP), .WMAN(WMAN), .VALUE({value!r}), .INF(0)) u_const_{index} "
            f"(.y(const_{index}));"
        )
    if lir.consts:
        w.line("")


def _emit_regfile(w: VerilogWriter) -> None:
    w.line("// Read-first register file (RWPASS=0): a value written on a cycle is readable only on the next cycle.")
    w.line("// The scheduler's +1 dependency latency and the allocator's last_use<=def register sharing both rely")
    w.line("// on this; do NOT switch to write-through (RWPASS=1) without revisiting holoso/regalloc.py.")
    w.line("holoso_regfile #(.W(W), .WADDR(WADDR), .NRD(NRD), .NWR(NWR), .NREG(NREG), .RWPASS(0)) u_rf (")
    w.push()
    w.lines(
        ".clk(clk),",
        ".wr_en(rf_wr_en), .wr_addr(rf_wr_addr), .wr_data(rf_wr_data),",
        ".rd_addr(rf_rd_addr), .rd_data(rf_rd_data)",
    )
    w.pop()
    w.lines(");", "")


def _emit_operators(w: VerilogWriter, lir: Lir) -> None:
    for inst in lir.instances:
        sig = _sig(inst)
        module = MODULE_NAMES[inst.kind]
        params = "#(.WEXP(WEXP), .WMAN(WMAN))"
        if inst.kind is OpKind.FMUL_ILOG2:
            params = f"#(.WEXP(WEXP), .WMAN(WMAN), .K({inst.k}))"
        w.line(f"{module} {params} u_{_base(inst)} (")
        w.push()
        w.line(".clk(clk), .rst(rst), .in_valid(" + sig + "_iv),")
        if _is_binary(inst):
            w.line(f".a_sgnop({sig}_as), .b_sgnop({sig}_bs), .y_sgnop({sig}_ys),")
            w.line(f".a({sig}_a), .b({sig}_b),")
        else:
            w.line(f".a_sgnop({sig}_as), .y_sgnop({sig}_ys),")
            w.line(f".a({sig}_a),")
        tail = f".out_valid({sig}_ov), .y({sig}_y)"
        if has_div0(inst.kind):
            tail += f", .div0({sig}_div0)"
        w.line(tail)
        w.pop()
        w.lines(");", "")


def _emit_datapath(
    w: VerilogWriter,
    lir: Lir,
    issues_by_cycle: dict[int, list[ScheduledOp]],
    commits_by_cycle: dict[int, list[ScheduledOp]],
    read_lanes: dict[int, dict[int, int]],
    out_lanes: dict[int, int],
) -> None:
    # One combinational block: per cycle, set the operand reads + in_valid for issuing operators and the write ports
    # for committing operators. `always @*` over `reg` targets is pure combinational logic; the only flip-flops are
    # the regfile and the control block below.
    w.line("always @* begin")
    w.push()
    for inst in lir.instances:
        sig = _sig(inst)
        w.line(f"{sig}_iv = 1'b0; {sig}_a = {{W{{1'b0}}}}; {sig}_as = 2'd0; {sig}_ys = 2'd0;")
        if _is_binary(inst):
            w.line(f"{sig}_b = {{W{{1'b0}}}}; {sig}_bs = 2'd0;")
    w.lines(
        "rf_rd_addr = {(NRD*WADDR){1'b0}};",
        "rf_wr_en   = {NWR{1'b0}};",
        "rf_wr_addr = {(NWR*WADDR){1'b0}};",
        "rf_wr_data = {(NWR*W){1'b0}};",
    )
    w.line("case (cyc)")
    w.push()
    w.line("0: if (in_valid) begin  // sample input ports into their registers")
    w.push()
    for port, load in enumerate(lir.inputs):
        w.line(f"rf_wr_en[{port}] = 1'b1;")
        w.line(f"rf_wr_addr[`HOLOSO_REGFILE_LANE(WADDR, {port})] = {load.dst.index};")
        w.line(f"rf_wr_data[`HOLOSO_REGFILE_LANE(W, {port})] = in_{load.name};")
    w.pop()
    w.line("end")
    for cycle in sorted(set(issues_by_cycle) | set(commits_by_cycle)):
        issues = issues_by_cycle.get(cycle, [])
        commits = commits_by_cycle.get(cycle, [])
        lanes = read_lanes.get(cycle, {})
        w.line(f"{cycle}: begin  // {_cycle_summary(issues, commits)}")
        w.push()
        for reg, lane in lanes.items():
            w.line(f"rf_rd_addr[`HOLOSO_REGFILE_LANE(WADDR, {lane})] = {reg};")
        for op in issues:
            sig = _sig(op.inst)
            w.line(f"{sig}_iv = 1'b1;")
            w.line(f"{sig}_a = {_operand_value(op.a, lanes)}; {sig}_as = 2'd{int(op.a.sgnop)};")
            if op.b is not None:
                w.line(f"{sig}_b = {_operand_value(op.b, lanes)}; {sig}_bs = 2'd{int(op.b.sgnop)};")
            w.line(f"{sig}_ys = 2'd{int(op.y_sgnop)};")
        for lane, op in enumerate(commits):
            sig = _sig(op.inst)
            w.line(f"rf_wr_en[{lane}] = 1'b1;")
            w.line(f"rf_wr_addr[`HOLOSO_REGFILE_LANE(WADDR, {lane})] = {op.dst.index};")
            w.line(f"rf_wr_data[`HOLOSO_REGFILE_LANE(W, {lane})] = {sig}_y;")
        w.pop()
        w.line("end")
    if out_lanes:
        w.line("LAST: begin  // present output registers on the read ports")
        w.push()
        for reg, lane in out_lanes.items():
            w.line(f"rf_rd_addr[`HOLOSO_REGFILE_LANE(WADDR, {lane})] = {reg};")
        w.pop()
        w.line("end")
    w.line("default: ;")
    w.pop()
    w.lines("endcase", "")
    w.pop()
    w.lines("end", "")


def _emit_fsm(w: VerilogWriter, lir: Lir, commits_by_cycle: dict[int, list[ScheduledOp]]) -> None:
    div0_by_cycle: dict[int, list[OperatorInstance]] = {}
    for cycle, ops in commits_by_cycle.items():
        insts = [op.inst for op in ops if has_div0(op.inst.kind)]
        if insts:
            div0_by_cycle[cycle] = insts
    # Sequential control: a plain up-counter replaying the static schedule; only control registers are reset.
    w.line("always @(posedge clk) begin")
    w.push()
    w.line("if (rst) begin")
    w.push()
    w.lines("cyc    <= 0;", "diag_q <= 1'b0;")
    w.pop()
    w.line("end else begin")
    w.push()
    w.line("if (cyc == 0) begin")
    w.push()
    w.line("if (in_valid) begin")
    w.push()
    w.lines("cyc    <= 1;", "diag_q <= 1'b0;")
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
    if div0_by_cycle:
        w.line("case (cyc)  // latch divide-by-zero at each fdiv's commit cycle")
        w.push()
        for cycle in sorted(div0_by_cycle):
            terms = " | ".join(f"{_sig(inst)}_div0" for inst in div0_by_cycle[cycle])
            w.line(f"{cycle}: if ({terms}) diag_q <= 1'b1;")
        w.line("default: ;")
        w.pop()
        w.line("endcase")
    w.pop()
    w.line("end")
    w.pop()
    w.lines("end", "")


def _emit_outputs(w: VerilogWriter, lir: Lir, out_lanes: dict[int, int]) -> None:
    w.line("assign in_ready  = (cyc == 0);")
    w.line("assign out_valid = (cyc == LAST);")
    w.line("assign diag_error = diag_q;")
    for index, wire in enumerate(lir.outputs):
        if isinstance(wire.source, ConstRef):
            raw = f"const_{wire.source.index}"
        else:
            raw = f"rf_rd_data[`HOLOSO_REGFILE_LANE(W, {out_lanes[wire.source.index]})]"
        if wire.sgnop is Sgnop.NONE:
            w.line(f"assign {wire.name} = {raw};")
        else:
            w.line(
                f"holoso_fsgnop #(.WFULL(W)) u_outsgn_{index} (.x({raw}), .op(2'd{int(wire.sgnop)}), .y({wire.name}));"
            )
    w.line("")
