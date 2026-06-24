"""
Render a scheduled :class:`Lir` into a synthesizable Verilog ZISC module that instantiates the shared support library
(assembled by :mod:`._support`).

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
``always @*`` block is the next-PC sequencer. Control fields that are constant across the whole program are driven
by constant nets and omitted from the ROM, so synthesis prunes them. Pooled instances -- the comparator included --
are driven uniformly through microcode lanes: read-address/latch lanes per operand, a write lane per tapped output
port (wide lanes through the writeback latch, latch-free boolean lanes written directly at their commit step with a
fabric-XOR inversion conditioner). Inline operators (boolean logic, the float<->bool casts) are single PC-gated
statements rendered by the operator's own ``verilog_expr``.
"""

from dataclasses import dataclass
from textwrap import dedent

from ..._lir import *
from ..._operators import *
from ..._type import is_wide_type
from ._microcode import *
from ._support import support_files


@dataclass(frozen=True, slots=True)
class VerilogOutput:
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


def _sig(inst: OperatorInstance) -> str:
    return f"s_{base_name(inst)}"


def _decl_range(width: int) -> str:
    return "" if width == 1 else f"[{width - 1:2}:0] "


def _lit(width: int, value: int) -> str:
    return f"{width}'d{value}"


def _cterm_expr(port: int, consts: list[int]) -> str:
    return f"const_{consts[0]}" if len(consts) == 1 else f"cterm{port}"


def _source_net(source: RegRef | FloatConstRef) -> str:
    return f"const_{source.index}" if isinstance(source, FloatConstRef) else f"regs[{source.index}]"


def _fsgnop(w: _Writer, raw: str, sign: FloatSignControl, dst: str, inst: str) -> None:
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
    return _state_sign_wire(slot) or _source_net(slot.tap.source)


def _inline_fire_pc(lir: Lir, block_index: int, op: InlineScheduledOp) -> int:
    """
    The fetch PC at which an inline firing's single PC-gated statement executes: its combinational fire step, one
    ``FETCH_LAG`` after the commit on either bank. An inline op drives its destination's write data combinationally, so
    -- unlike a pooled lane -- it carries no writeback latch.
    """
    return lir.block_base[block_index] + inline_fire_cycle(op.commit_cycle)


def _inline_sign_wire(block_index: int, op_index: int, pos: int) -> str:
    return f"inlsgn_{block_index}_{op_index}_{pos}"


def _inline_rhs(block_index: int, op_index: int, op: InlineScheduledOp) -> str:
    """
    The RHS of one inline firing: the operator's own combinational expression over its operand nets (a float operand
    routes through its sign-conditioning wire when its folded sign is non-identity), with the result conditioner
    applied -- an inversion folds into the expression; sign-conditioned wide inline results have no producer yet.
    """
    nets: list[str] = []
    for pos, operand in enumerate(op.operands):
        if isinstance(operand, FloatOperand):
            if operand.sign != FloatSignControl():
                nets.append(_inline_sign_wire(block_index, op_index, pos))
            else:
                nets.append(_source_net(operand.source))
        else:
            nets.append(_bool_operand_rhs(operand))
    expr = op.operator.verilog_expr(*nets)
    conditioner = op.write.conditioner
    if isinstance(conditioner, BoolInversion):
        return conditioner.decorate(f"({expr})") if conditioner.invert else expr
    assert conditioner == FloatSignControl(), "no pass produces sign-conditioned wide inline results yet"
    return expr


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
    # Commit annotations land on each firing's write step (the same step the microcode places its write-enable): a
    # bool-only firing writes ON the commit step (latch-free bank), one with a wide lane one step later (writeback
    # latch). A pooled firing's taps are single-bank, so the firing's write step is pooled_writeback_word of that bank.
    commits_by_step: dict[int, list[PooledScheduledOp]] = {}
    for commit_cycle, ops in commits_by_cycle.items():
        for op in ops:
            wide = any(isinstance(write.dst, RegRef) for write in op.writes)
            commits_by_step.setdefault(pooled_writeback_word(commit_cycle, wide), []).append(op)

    depth = lir.last_pc + 1  # one microcode word per fetch PC (0..last_pc); inter-block drains and the tail pack to NOP

    _emit_header(w, lir)
    _emit_localparams(w, lir, cycw, pcw, ucw)
    _emit_support_header(w, lir)
    _emit_declarations(w, lir, write_lists)
    _emit_consts(w, lir)
    _emit_operators(w, lir, write_lists)
    _emit_microcode_rom(w, fields, ucw, depth, issues_by_cycle, commits_by_step)
    _emit_field_wires(w, fields)
    _emit_datapath_comb(w, lir, port_consts, write_lists)
    _emit_state_next(w, lir)
    _emit_copy_sign_wires(w, lir)
    _emit_inline_sign_wires(w, lir)
    _emit_clocked(w, lir, read_port, port_consts, read_sets, write_sets, write_lists)
    _emit_outputs(w, lir)
    w("\nendmodule\n")
    return VerilogOutput(verilog=w.render(), support_files=support_files())


def _emit_header(w: _Writer, lir: Lir) -> None:
    from holoso import __url__, __version__

    # Generation time is not included for reproducibility.
    fmt = lir.float_format
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
    fmt = lir.float_format
    nreg = max(1, lir.regfile.nreg)
    w(f"""
localparam           WEXP      ={fmt.wexp:3};  // Float exponent bits fixed by the static schedule
localparam           WMAN      ={fmt.wman:3};  // Float mantissa bits fixed by the static schedule
localparam           W         = WEXP + WMAN;
localparam           NREG      ={nreg:3};  // >= 1; the wide bank is unused when no value needs a register
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


def _emit_declarations(w: _Writer, lir: Lir, write_lists: dict[tuple[OperatorInstance, int], list[int]]) -> None:
    w("""
        reg  [PCW-1:0]  pc;            // fetch program counter; the executing step lags it by FETCH_LAG
        reg  [PCW-1:0]  next_pc;       // combinational next-state presented to the ROM each cycle
        reg  [PCW-1:0]  ucode_addr_q;  // PC latch: splits pc -> next_pc -> ROM address from the array read
        reg  [CYCW-1:0] err_pc_q;
        wire            err;           // an operator error is detected on the current step

        reg  [W-1:0] regs  [0:NREG-1];   // the sparse register array (read-first: a write is visible the next step)
        reg          bregs [0:NBREG-1];  // 1-bit boolean register bank: branch conditions and boolean state

        """)
    for inst in lir.instances:
        sig = _sig(inst)
        w(f"wire         {sig}_iv;")
        for pos, operand_type in enumerate(inst.operator.signature.operand_types):
            assert is_wide_type(operand_type), "pooled operators read only wide operands today"
            letter = PORT_LETTERS[pos]
            w(f"wire [1:0]   {sig}_{letter}s;")
            w(f"reg  [W-1:0] {sig}_{letter};")  # read-latched operand (the read mux output, registered)
        # One lane per TAPPED output port: a wide lane gets its conditioner wire and a writeback latch; a latch-free
        # boolean lane is the raw 1-bit module output, written into bregs directly at its commit step.
        for q, result_type in enumerate(inst.operator.signature.result_types):
            if (inst, q) not in write_lists:
                continue  # a never-tapped output port: no nets, the module port is left unconnected
            if is_wide_type(result_type):
                w(f"wire [1:0]   {sig}_y{q}s;")
                w(f"wire [W-1:0] {sig}_y{q};")
                w(f"reg  [W-1:0] {sig}_y{q}_q;")  # writeback latch between the operator output and the register write
            else:
                w(f"wire         {sig}_y{q};")
        for port in inst.operator.error_ports:
            w(f"wire         {sig}_{port};")
            w(f"reg          {sig}_{port}_q;")  # error sideband rides the same writeback latch as the result
    w("")


def _emit_consts(w: _Writer, lir: Lir) -> None:
    fmt = lir.float_format
    width = fmt.width
    digits = (width + 3) // 4
    for index, value in enumerate(lir.float_consts):
        w(f"wire [W-1:0] const_{index} = {width}'h{fmt.encode(value):0{digits}x};  // {value!r}")
    if lir.float_consts:
        w("")


def _emit_operators(w: _Writer, lir: Lir, write_lists: dict[tuple[OperatorInstance, int], list[int]]) -> None:
    for inst in lir.instances:
        sig = _sig(inst)
        operator = inst.operator
        letters = PORT_LETTERS[: operator.arity]
        # WEXP/WMAN frame the float format; hdl_params() lists K (ilog2) and every STAGE_* explicitly. LATENCY is
        # emitted separately from the scheduler model, so a model/RTL drift fails during wrapper elaboration.
        parts = [".WEXP(WEXP)", ".WMAN(WMAN)"] + [
            f".{param}({value})" for param, value in operator.hdl_params().items()
        ]
        parts.append(f".LATENCY({operator.latency})")
        params = ", ".join(parts)
        w(f"{operator.module_name} #(", f"    {params}", f") u_{base_name(inst)} (")
        w.push()
        w(f".clk(clk), .rst(rst), .in_valid({sig}_iv),")
        for letter in letters:
            w(f".{letter}_sgnop({sig}_{letter}s),")
        # A float output port carries a hardware sign conditioner (piped inside the wrapper); an untapped one is tied
        # to the identity. Boolean output ports have none -- their inversion conditioner is fabric-side at the write.
        for q, result_type in enumerate(operator.signature.result_types):
            if is_wide_type(result_type):
                conditioner = f"{sig}_y{q}s" if (inst, q) in write_lists else "2'd0"
                w(f".{operator.output_hdl_ports[q]}_sgnop({conditioner}),")
        for letter in letters:
            w(f".{letter}({sig}_{letter}),")
        # out_valid is left unconnected: the static schedule already knows when each result is ready.
        w(".out_valid(),")
        outputs = [
            f".{operator.output_hdl_ports[q]}({f'{sig}_y{q}' if (inst, q) in write_lists else ''})"
            for q in range(len(operator.signature.result_types))
        ]
        for line_index, line in enumerate(outputs):
            last = line_index == len(outputs) - 1 and not operator.error_ports
            w(line + ("" if last else ","))
        for port_index, port in enumerate(operator.error_ports):
            w(f".{port}({sig}_{port})" + ("," if port_index < len(operator.error_ports) - 1 else ""))
        w.pop()
        w(");", "")


def _emit_microcode_rom(
    w: _Writer,
    fields: dict[str, Field],
    ucw: int,
    depth: int,
    issues_by_cycle: dict[int, list[PooledScheduledOp]],
    commits_by_step: dict[int, list[PooledScheduledOp]],
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


def _const_pool_mux(selector: str, consts: list[int]) -> str:
    """
    A const-pool read expression: the lone ``const_N`` net, or a ``selector``-indexed ternary mux over ``consts``. Used
    on the read side (an operand's per-port const select, ``mc_cidx``) and the write side (a register's ucode-driven
    constant install, ``mc_ccidx``) alike.
    """
    expr = f"const_{consts[-1]}"
    for local in range(len(consts) - 2, -1, -1):
        expr = f"({selector} == {local}) ? const_{consts[local]} : {expr}"
    return expr


def _emit_datapath_comb(
    w: _Writer, lir: Lir, port_consts: dict[int, list[int]], write_lists: dict[tuple[OperatorInstance, int], list[int]]
) -> None:
    for port in sorted(port_consts):
        if len(port_consts[port]) > 1:
            w(f"wire [W-1:0] cterm{port} = {_const_pool_mux(f_cidx(port), port_consts[port])};")
    w("")

    w("// Operator control (in_valid and sign controls are consumed inside the wrapper on the issue step).")
    for inst in lir.instances:
        sig, base = _sig(inst), base_name(inst)
        w(f"assign {sig}_iv = {f_iv(base)};")
        for q, result_type in enumerate(inst.operator.signature.result_types):
            if is_wide_type(result_type) and (inst, q) in write_lists:
                w(f"assign {sig}_y{q}s = {f_ysgn(base, q)};")
        for pos in range(inst.operator.arity):
            w(f"assign {sig}_{PORT_LETTERS[pos]}s = {f_osgn(base, PORT_LETTERS[pos])};")
    w("")

    # An error matters only on the step its operator commits, which is exactly its wide lanes' write-enable window;
    # both the write-enables and the error sideband are aligned to the writeback latch (commit + write latch).
    err_terms: list[str] = []
    for inst in lir.instances:
        if not inst.operator.error_ports:
            continue
        lane_wes = [
            f_we(base_name(inst), q)
            for q, result_type in enumerate(inst.operator.signature.result_types)
            if is_wide_type(result_type) and (inst, q) in write_lists
        ]
        assert lane_wes, "an error-bearing operator must have a tapped wide lane to align its sideband with"
        gate = lane_wes[0] if len(lane_wes) == 1 else "(" + " | ".join(lane_wes) + ")"
        for err_port in inst.operator.error_ports:
            err_terms.append(f"({gate} & {_sig(inst)}_{err_port}_q)")
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
        term_pc = lir.term_pc(block)
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
    """The fetch PC at which a phi-arm copy installs its value (a const fires inline-class, a reg copy one step later)."""
    return lir.block_base[block.index] + copy.fire_step


def _bool_write_pc(lir: Lir, block: LirBlock, write: BoolWrite) -> int:
    return lir.block_base[block.index] + write.fire_step


def _copy_sign_wire(block_index: int, copy_index: int) -> str:
    return f"copysgn_{block_index}_{copy_index}"


def _float_copy_rhs(block_index: int, copy_index: int, copy: FloatCopy) -> str:
    if copy.source.sign == FloatSignControl():
        return _source_net(copy.source.source)
    return _copy_sign_wire(block_index, copy_index)


def _bool_operand_rhs(operand: BoolOperand) -> str:
    source = operand.source
    if isinstance(source, BoolConstRef):
        return "1'b1" if source.value else "1'b0"  # an inverted immediate folded at construction
    net = f"bregs[{source.index}]"
    return f"~{net}" if operand.inversion.invert else net


def _bool_write_rhs(write: BoolWrite) -> str:
    return _bool_operand_rhs(write.source)


def _copies_grouped(lir: Lir) -> dict[int, list[tuple[int, str]]]:
    """
    Destination wide register -> [(install PC, source net)], over the phi-arm copies that remain pc-gated: register-
    source copies and signed-constant installs. Identity-sign constant installs are ucode-driven (see
    ``_wide_writer_entries``), not pc-gated.
    """
    grouped: dict[int, list[tuple[int, str]]] = {}
    for block in lir.blocks:
        for copy_index, copy in enumerate(block.copies):
            if is_ucode_const_copy(copy):
                continue
            grouped.setdefault(copy.dst.index, []).append(
                (_float_copy_pc(lir, block, copy), _float_copy_rhs(block.index, copy_index, copy))
            )
    return grouped


def _bool_writes_grouped(lir: Lir) -> dict[int, list[tuple[int, str]]]:
    """
    Destination boolean register -> [(install PC, source expression)], over the boolean phi-arm writes that remain
    pc-gated: register-source writes. Constant boolean installs are ucode-driven (see ``_bool_writer_entries``).
    """
    grouped: dict[int, list[tuple[int, str]]] = {}
    for block in lir.blocks:
        for write in block.bool_writes:
            if write.is_const:
                continue
            grouped.setdefault(write.dst.index, []).append((_bool_write_pc(lir, block, write), _bool_write_rhs(write)))
    return grouped


def _emit_copy_sign_wires(w: _Writer, lir: Lir) -> None:
    emitted = False
    for block in lir.blocks:
        for copy_index, copy in enumerate(block.copies):
            if copy.source.sign != FloatSignControl():
                wire = _copy_sign_wire(block.index, copy_index)
                w(f"wire [W-1:0] {wire};")
                _fsgnop(w, _source_net(copy.source.source), copy.source.sign, wire, f"u_{wire}")
                emitted = True
    if emitted:
        w("")


def _emit_inline_sign_wires(w: _Writer, lir: Lir) -> None:
    emitted = False
    for block in lir.blocks:
        for op_index, op in enumerate(block.inline_ops):
            for pos, operand in enumerate(op.operands):
                if isinstance(operand, FloatOperand) and operand.sign != FloatSignControl():
                    wire = _inline_sign_wire(block.index, op_index, pos)
                    w(f"wire [W-1:0] {wire};")
                    _fsgnop(w, _source_net(operand.source), operand.sign, wire, f"u_{wire}")
                    emitted = True
    if emitted:
        w("")


def _emit_state_next(w: _Writer, lir: Lir) -> None:
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


def _writeback_cond(
    inst: OperatorInstance, port: int, reg: int, write_lists: dict[tuple[OperatorInstance, int], list[int]]
) -> str:
    """
    The guard under which lane ``(inst, port)`` writes ``reg``: its write-enable, plus a write-address compare when
    the lane targets more than one register (the dense write-target index this register occupies in its codebook).
    """
    base = base_name(inst)
    targets = write_lists[(inst, port)]
    if len(targets) == 1:
        return f_we(base, port)
    return f"{f_we(base, port)} && ({f_wa(base, port)} == {_lit(code_width(len(targets)), targets.index(reg))})"


def _wide_writer_entries(
    lir: Lir,
    write_sets: dict[int, list[tuple[OperatorInstance, int]]],
    write_lists: dict[tuple[OperatorInstance, int], list[int]],
) -> dict[int, list[tuple[str, str]]]:
    """
    Per wide register, the ordered ``(condition, rhs)`` of every driver other than its state-slot install: the
    accept-step input load (highest priority), each pooled lane's writeback, each pc-gated wide-result inline firing
    (the bool->float cast), and each pc-gated phi-arm copy. The conditions are pairwise mutually exclusive (the load
    step, the per-lane write-enable steps, and the distinct inline/copy PCs never coincide for one register -- the
    schedule and the allocator's interference guarantee it), so the emitter folds them into one priority chain and a
    register is driven by exactly one statement, however many sources reuse or coalesce onto it.
    """
    entries: dict[int, list[tuple[str, str]]] = {}
    for fload in sorted(lir.float_inputs, key=lambda load: load.dst.index):
        entries.setdefault(fload.dst.index, []).append(("in_ready && in_valid", f"in_{fload.name}"))
    for reg in sorted(write_sets):
        for inst, port in write_sets[reg]:
            entries.setdefault(reg, []).append(
                (_writeback_cond(inst, port, reg, write_lists), f"{_sig(inst)}_y{port}_q")
            )
    # Ucode-driven constant installs: a microcode write-enable arm (like an operator lane), reusing the const-pool nets.
    const_books = const_install_codebooks(lir)
    for reg in sorted(const_books):
        entries.setdefault(reg, []).append(
            (f_cwe(RegRef(reg)), _const_pool_mux(f_ccidx(RegRef(reg)), const_books[reg]))
        )
    for block in lir.blocks:
        for op_index, inline_op in enumerate(block.inline_ops):
            if isinstance(inline_op.write.dst, RegRef):
                entries.setdefault(inline_op.write.dst.index, []).append(
                    (
                        f"pc == {_inline_fire_pc(lir, block.index, inline_op)}",
                        _inline_rhs(block.index, op_index, inline_op),
                    )
                )
    for reg, items in _copies_grouped(lir).items():
        for install_pc, rhs in sorted(items):
            entries.setdefault(reg, []).append((f"pc == {install_pc}", rhs))
    return entries


def _bool_writer_entries(
    lir: Lir, write_lists: dict[tuple[OperatorInstance, int], list[int]]
) -> dict[int, list[tuple[str, str]]]:
    """
    Per boolean register, the ordered ``(condition, rhs)`` of every driver other than its state-slot install: the
    input load, each pooled boolean lane (microcode-gated, latch-free, with its fabric-XOR inversion conditioner),
    each pc-gated bool-result inline firing, and each pc-gated phi-arm write.
    """
    entries: dict[int, list[tuple[str, str]]] = {}
    for bload in sorted(lir.bool_inputs, key=lambda load: load.dst.index):
        entries.setdefault(bload.dst.index, []).append(("in_ready && in_valid", f"in_{bload.name}"))
    bool_write_sets = lir.bool_write_set_per_register
    for reg in sorted(bool_write_sets):
        for inst, port in bool_write_sets[reg]:
            rhs = f"{_sig(inst)}_y{port} ^ {f_binv(base_name(inst), port)}"
            entries.setdefault(reg, []).append((_writeback_cond(inst, port, reg, write_lists), rhs))
    # Ucode-driven constant installs: a microcode write-enable arm latching the 1-bit value (latch-free, no XOR -- the
    # inversion is already folded into the value at construction).
    for reg in const_install_bool_regs(lir):
        entries.setdefault(reg, []).append((f_cwe(BoolRegRef(reg)), f_cval(BoolRegRef(reg))))
    for block in lir.blocks:
        for op_index, inline_op in enumerate(block.inline_ops):
            if isinstance(inline_op.write.dst, BoolRegRef):
                entries.setdefault(inline_op.write.dst.index, []).append(
                    (
                        f"pc == {_inline_fire_pc(lir, block.index, inline_op)}",
                        _inline_rhs(block.index, op_index, inline_op),
                    )
                )
    for reg, items in _bool_writes_grouped(lir).items():
        for install_pc, rhs in sorted(items):
            entries.setdefault(reg, []).append((f"pc == {install_pc}", rhs))
    return entries


def _wide_state_install_entry(lir: Lir, slot: FloatStateSlot) -> list[tuple[str, str]]:
    """
    The slot register's live-out install, appended under the reset-else as the lowest-priority arm of its chain. A
    coalesced slot has no install (its producing operator already writes the slot register); a non-coalesced one is
    installed read-first on its writeback step (``state_copy_step``, the absolute install PC -- which reduces to
    ``LASTPC`` for a boundary install), out_ready-gated so a held boundary copies exactly once.
    """
    if not slot.needs_copy:
        return []
    pcw = max(1, lir.initiation_interval.bit_length())
    cond = f"pc == {_lit(pcw, lir.state_copy_step(slot))} && (pc != LASTPC || out_ready)"
    return [(cond, _state_copy_rhs(slot))]


def _emit_chain(w: _Writer, lhs: str, entries: list[tuple[str, str]]) -> None:
    """One priority chain per register, so the register has exactly one driver (the multi-assign rule)."""
    clause = "if"
    for cond, rhs in entries:
        w(f"{clause} ({cond}) {lhs} <= {rhs};")
        clause = "else if"


def _emit_clocked(
    w: _Writer,
    lir: Lir,
    read_port: dict[tuple[OperatorInstance, int], int],
    port_consts: dict[int, list[int]],
    read_sets: dict[tuple[OperatorInstance, int], list[int]],
    write_sets: dict[int, list[tuple[OperatorInstance, int]]],
    write_lists: dict[tuple[OperatorInstance, int], list[int]],
) -> None:
    """Emit every sequential element in one always @(posedge clk): fetch, latches, writes, and control state."""
    # We MUST ensure that we DO NOT MULTI-ASSIGN any register in the same step; this is ensured by always placing each
    # assignment to the same register into different branches of the same condition.
    nreg = max(1, lir.regfile.nreg)
    nbreg = max(1, lir.bool_regfile.nreg)
    float_slots = {slot.reg.index: slot for slot in lir.float_state_slots}
    bool_slots = {slot.reg.index: slot for slot in lir.bool_state_slots}
    # Each register's whole write is one priority chain over all its drivers (input load, operator writebacks, casts,
    # phi copies, comparator/logic results, and -- for a slot -- its boundary install), so a register is never assigned
    # by two separate statements. The conditions within a chain are pairwise mutually exclusive by construction.
    wide = _wide_writer_entries(lir, write_sets, write_lists)
    boolw = _bool_writer_entries(lir, write_lists)
    w("""
// All sequential logic in one clocked process. Reset gates only the control state (pc, err_pc_q) and the persistent
// state registers; every other register is reset-unconditional. Each register is driven by exactly one write chain.
always @(posedge clk) begin
""")
    w.push()

    w("// Microcode fetch: PC latch -> control-store array read -> BRAM output register.")
    w("ucode_addr_q <= next_pc;")
    w("ucode_q      <= ucode[ucode_addr_q];")
    w("ucode_word   <= ucode_q;")
    w("")

    w("// Operand read latches: a sparse mux over each operand's read-set, registered before the wrapper.")
    for inst in lir.instances:
        sig = _sig(inst)
        for pos in range(inst.operator.arity):
            port = read_port[(inst, pos)]
            _read_latch_stmts(w, f"{sig}_{PORT_LETTERS[pos]}", port, read_sets.get((inst, pos), []), port_consts)
    w("")

    w("// Writeback latches: each WIDE lane's result (and any error sideband) registered before the register")
    w("// write; a boolean lane is latch-free and written directly at its commit step.")
    for inst in lir.instances:
        sig = _sig(inst)
        for q, result_type in enumerate(inst.operator.signature.result_types):
            if is_wide_type(result_type) and (inst, q) in write_lists:
                w(f"{sig}_y{q}_q <= {sig}_y{q};")
        for err_port in inst.operator.error_ports:
            w(f"{sig}_{err_port}_q <= {sig}_{err_port};")
    w("")

    # Non-slot registers: one reset-unconditional write chain each driving regs[]/bregs[]. Datapath payload carries no
    # reset, keeping the high-fanout reset net off the wide cone (only control/valid state is reset); the contents are
    # don't-care until a valid write lands. A register with no drivers is simply omitted.
    nonslot_wide = [reg for reg in range(nreg) if reg not in float_slots and wide.get(reg)]
    nonslot_bool = [reg for reg in range(nbreg) if reg not in bool_slots and boolw.get(reg)]
    if nonslot_wide or nonslot_bool:
        w("// Register write chains (reset-unconditional): one priority chain per register over all its drivers.")
        for reg in nonslot_wide:
            _emit_chain(w, f"regs[{reg}]", wide[reg])
        for reg in nonslot_bool:
            _emit_chain(w, f"bregs[{reg}]", boolw[reg])
        w("")

    # Control and persistent state are the reset-gated registers: the slot snapshot (under rst) and the slot's update
    # chain (its coalesced operator writebacks and/or its boundary install, under the else) are the two arms of one rst
    # condition, segregating those assignments for the synthesizer.
    fmt = lir.float_format
    digits = (fmt.width + 3) // 4
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
    for reg, slot in sorted(float_slots.items()):
        chain = wide.get(reg, []) + _wide_state_install_entry(lir, slot)
        if chain:
            _emit_chain(w, f"regs[{reg}]", chain)
    for reg, bslot in sorted(bool_slots.items()):
        install = [("pc == LASTPC && out_ready", _bool_operand_rhs(bslot.live_out))] if bslot.needs_copy else []
        chain = boolw.get(reg, []) + install
        if chain:
            _emit_chain(w, f"bregs[{reg}]", chain)
    w.pop()
    w("end")

    w.pop()
    w("end", "")


def _emit_outputs(w: _Writer, lir: Lir) -> None:
    w("""
assign in_ready  = (pc == 0);
assign out_valid = (pc == LASTPC);  // result valid on PRESENT; execution lags the fetch by FETCH_LAG
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
