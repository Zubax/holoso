"""Render a :class:`Lir` into a synthesizable Verilog ZISC module.

Datapath: a ``holoso_regfile`` flip-flop bank, one operator-wrapper instance per :class:`OperatorInstance`, and one
``holoso_fconst`` per pooled constant. Controller: a ``case(state)`` FSM. Inputs are written into the register file in
one cycle when ``in_valid`` is accepted. Each compute step asserts ``in_valid`` to its operators for one cycle, then
writes each operator's result into its destination register the cycle that operator's ``out_valid`` fires; the step
advances once every issued operator has fired. Reset covers only control registers.
"""

from __future__ import annotations

from .emit import VerilogWriter
from .lir import ConstRef, Issue, Lir, OperatorInstance, Operand, RegRef
from .operators import MODULE_NAMES, OpKind, Sgnop, arity, has_div0


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


def _read_lanes(lir: Lir) -> list[dict[int, int]]:
    per_step: list[dict[int, int]] = []
    for step in lir.steps:
        lanes: dict[int, int] = {}
        for issue in step.issues:
            for operand in (issue.a, issue.b):
                if operand is not None and isinstance(operand.source, RegRef) and operand.source.index not in lanes:
                    lanes[operand.source.index] = len(lanes)
        per_step.append(lanes)
    return per_step


def _output_lanes(lir: Lir) -> dict[int, int]:
    lanes: dict[int, int] = {}
    for wire in lir.outputs:
        if isinstance(wire.source, RegRef) and wire.source.index not in lanes:
            lanes[wire.source.index] = len(lanes)
    return lanes


def generate(lir: Lir) -> str:
    w = VerilogWriter()
    steps = lir.steps
    k = len(steps)
    sw = max(1, (k + 1).bit_length())
    waddr = max(1, (lir.regfile.nreg - 1).bit_length())
    read_lanes = _read_lanes(lir)
    out_lanes = _output_lanes(lir)
    st_done = k + 1

    _emit_header(w, lir)
    _emit_localparams(w, lir, sw, waddr, st_done)
    _emit_declarations(w, lir)
    _emit_consts(w, lir)
    _emit_regfile(w)
    _emit_operators(w, lir)
    _emit_datapath(w, lir, read_lanes, out_lanes)
    _emit_fsm(w, lir)
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


def _emit_localparams(w: VerilogWriter, lir: Lir, sw: int, waddr: int, st_done: int) -> None:
    w.line("localparam W     = WEXP + WMAN;")
    w.line(f"localparam NREG  = {lir.regfile.nreg};")
    w.line(f"localparam WADDR = {waddr};")
    w.line(f"localparam NRD   = {lir.regfile.nrd};")
    w.line(f"localparam NWR   = {lir.regfile.nwr};")
    w.line(f"localparam SW    = {sw};")
    w.line("localparam [SW-1:0] ST_IDLE = 0;")
    w.line(f"localparam [SW-1:0] ST_DONE = {st_done};")
    if lir.steps:
        w.line(f"// compute states are the bare step numbers 1..{len(lir.steps)} (IDLE=0, DONE={st_done})")
    w.line("")


def _emit_declarations(w: VerilogWriter, lir: Lir) -> None:
    w.lines("reg [SW-1:0] state;", "reg started;", "reg diag_q;", "reg step_done;", "")
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
        w.line(f"wire         {sig}_ov;")
        w.line(f"wire [W-1:0] {sig}_y;")
        if has_div0(inst.kind):
            w.line(f"wire         {sig}_div0;")
        w.line(f"reg          done_{_base(inst)};")
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


def _operand_name(operand: Operand) -> str:
    base = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return operand.sgnop.decorate(base)


def _issue_summary(issue: Issue) -> str:
    """A short human-readable description of an issue, used as a per-state comment in the generated controller."""
    dst = f"r{issue.dst.index}"
    if issue.inst.kind is OpKind.FMUL_ILOG2:
        return f"{dst}={_operand_name(issue.a)}*2^{issue.k}"
    symbol = {OpKind.FADD: "+", OpKind.FMUL: "*", OpKind.FDIV: "/"}[issue.inst.kind]
    assert issue.b is not None
    return f"{dst}={_operand_name(issue.a)}{symbol}{_operand_name(issue.b)}"


def _emit_datapath(w: VerilogWriter, lir: Lir, read_lanes: list[dict[int, int]], out_lanes: dict[int, int]) -> None:
    # A single combinational block: each FSM state sets all of its operand connections, register-file read/write
    # ports, and the step-done condition together. `always @*` over `reg` targets is purely combinational logic
    # (no flip-flops); the only sequential element is the clocked block below.
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
        "step_done  = 1'b1;",
    )
    w.line("case (state)")
    w.push()
    w.line("ST_IDLE: if (in_valid) begin  // sample input ports into their registers")
    w.push()
    for port, load in enumerate(lir.inputs):
        w.line(f"rf_wr_en[{port}] = 1'b1;")
        w.line(f"rf_wr_addr[`HOLOSO_REGFILE_LANE(WADDR, {port})] = {load.dst.index};")
        w.line(f"rf_wr_data[`HOLOSO_REGFILE_LANE(W, {port})] = in_{load.name};")
    w.pop()
    w.line("end")
    for index, step in enumerate(lir.steps):
        lanes = read_lanes[index]
        summary = "; ".join(_issue_summary(issue) for issue in step.issues)
        w.line(f"{index + 1}: begin  // {summary}")
        w.push()
        for reg, lane in lanes.items():
            w.line(f"rf_rd_addr[`HOLOSO_REGFILE_LANE(WADDR, {lane})] = {reg};")
        for port, issue in enumerate(step.issues):
            sig = _sig(issue.inst)
            base = _base(issue.inst)
            w.line(f"{sig}_iv = ~started;")
            w.line(f"{sig}_a = {_operand_value(issue.a, lanes)}; {sig}_as = 2'd{int(issue.a.sgnop)};")
            if issue.b is not None:
                w.line(f"{sig}_b = {_operand_value(issue.b, lanes)}; {sig}_bs = 2'd{int(issue.b.sgnop)};")
            w.line(f"{sig}_ys = 2'd{int(issue.y_sgnop)};")
            w.line(f"rf_wr_en[{port}] = {sig}_ov & ~done_{base};")
            w.line(f"rf_wr_addr[`HOLOSO_REGFILE_LANE(WADDR, {port})] = {issue.dst.index};")
            w.line(f"rf_wr_data[`HOLOSO_REGFILE_LANE(W, {port})] = {sig}_y;")
        terms = " & ".join(f"(done_{_base(issue.inst)} | {_sig(issue.inst)}_ov)" for issue in step.issues)
        w.line(f"step_done = {terms};")
        w.pop()
        w.line("end")
    if out_lanes:
        w.line("ST_DONE: begin  // present output registers on the read ports")
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


def _emit_fsm(w: VerilogWriter, lir: Lir) -> None:
    first_state = "1" if lir.steps else "ST_DONE"
    # Sequential control: nonblocking assignments; only control registers are reset (the datapath regfile is not).
    w.line("always @(posedge clk) begin")
    w.push()
    w.line("if (rst) begin")
    w.push()
    w.lines("state <= ST_IDLE;", "started <= 1'b0;", "diag_q <= 1'b0;")
    for inst in lir.instances:
        w.line(f"done_{_base(inst)} <= 1'b1;")
    w.pop()
    w.line("end else begin")
    w.push()
    w.line("case (state)")
    w.push()
    w.line("ST_IDLE: if (in_valid) begin")
    w.push()
    w.lines("diag_q <= 1'b0;", "started <= 1'b0;", f"state <= {first_state};")
    w.pop()
    w.line("end")
    for index, step in enumerate(lir.steps):
        next_state = str(index + 2) if index + 1 < len(lir.steps) else "ST_DONE"
        issued = [issue.inst for issue in step.issues]
        w.line(f"{index + 1}: begin")
        w.push()
        w.line("if (!started) begin")
        w.push()
        w.line("started <= 1'b1;")
        for inst in issued:
            w.line(f"done_{_base(inst)} <= 1'b0;")
        w.pop()
        w.line("end else begin")
        w.push()
        for inst in issued:
            sig = _sig(inst)
            w.line(f"if ({sig}_ov) done_{_base(inst)} <= 1'b1;")
            if has_div0(inst.kind):
                w.line(f"if ({sig}_ov & {sig}_div0) diag_q <= 1'b1;")
        w.line("if (step_done) begin")
        w.push()
        w.lines(f"state <= {next_state};", "started <= 1'b0;")
        w.pop()
        w.line("end")
        w.pop()
        w.line("end")
        w.pop()
        w.line("end")
    w.line("ST_DONE: if (out_ready) state <= ST_IDLE;")
    w.line("default: state <= ST_IDLE;")
    w.pop()
    w.line("endcase")
    w.pop()
    w.line("end")
    w.pop()
    w.lines("end", "")


def _emit_outputs(w: VerilogWriter, lir: Lir, out_lanes: dict[int, int]) -> None:
    w.line("assign in_ready  = (state == ST_IDLE);")
    w.line("assign out_valid = (state == ST_DONE);")
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
