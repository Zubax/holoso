"""
Render a scheduled :class:`Lir` into a synthesizable Verilog ZISC module that instantiates the shared support library
(assembled by :mod:`._support`).

The controller is a microcode ROM (see :mod:`._microcode`): one pre-decoded VLIW control word per step, stored in a
(BRAM-inferable) ROM read through a 3-stage fetch (a PC latch, the array read, and the BRAM output register) so the
critical control cones are short register-to-register paths. The executing step therefore lags the fetch PC by
FETCH_LAG, which the sequencer accounts for: the PC counts up to LASTPC and out_valid is asserted there.

Storage is a sparse, schedule-specific register file emitted inline instead of a general-purpose multiport file. Value
routing is uniform: each operand port's read mux is a ``case`` over that port's read codebook (its registers and the
constants it reads), and each register's write is a ``case`` over that register's write codebook selected by a tiny
per-register opcode (code 0 == NOP hold). PC drives only the sequencer; it never gates a datapath read or write.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from textwrap import dedent
from typing import assert_never

from ..._lir import *
from ..._operators import *
from ..._type import is_wide_type
from ..._legal import output_header
from ._microcode import *
from ._support import inline_support, support_files


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


def _lit(width: int, value: int) -> str:
    return f"{width}'d{value}"


def _wire(width: int) -> str:
    """Aligned ``wire`` declaration prefix so field names line up regardless of bus width."""
    return f"wire [{width - 1:2}:0] " if width > 1 else "wire        "


def _source_net(source: RegRef | FloatConstRef) -> str:
    return f"const_{source.index}" if isinstance(source, FloatConstRef) else f"regs[{source.index}]"


def _signed_source_net(source: RegRef | FloatConstRef, sign: FloatSignControl) -> str:
    """A source net with its folded sign applied inline via ``holoso_fsgnop``, or bare when the sign is identity."""
    raw = _source_net(source)
    return raw if sign == FloatSignControl() else f"holoso_fsgnop({raw}, 2'd{sign.encoded})"


def _bool_operand_rhs(operand: BoolOperand) -> str:
    source = operand.source
    if isinstance(source, BoolConstRef):
        return "1'b1" if source.value else "1'b0"  # an inverted immediate folded at construction
    net = f"bregs[{source.index}]"
    return f"~{net}" if operand.inversion.invert else net


def _render_inline(
    operator: InlineHardwareOperator, operands: tuple[FloatOperand | BoolOperand, ...], conditioner: PortConditioner
) -> str:
    """
    An inline firing's combinational RHS: the operator's own expression over its operand nets (a float operand's folded
    sign applies inline via ``holoso_fsgnop``), with the result conditioner applied -- an inversion folds into the
    expression; sign-conditioned wide inline results have no producer yet.
    """
    nets: list[str] = []
    for operand in operands:
        if isinstance(operand, FloatOperand):
            nets.append(_signed_source_net(operand.source, operand.sign))
        else:
            nets.append(_bool_operand_rhs(operand))
    expr = operator.verilog_expr(*nets)
    if isinstance(conditioner, BoolInversion):
        return conditioner.decorate(f"({expr})") if conditioner.invert else expr
    assert conditioner == FloatSignControl(), "no pass produces sign-conditioned wide inline results yet"
    return expr


def _state_copy_rhs(slot: FloatStateSlot) -> str:
    return _signed_source_net(slot.tap.source, slot.tap.sign)


def _write_source_rhs(source: WriteSource) -> str:
    """Render one write-codebook source to its Verilog RHS -- the dual of the microcode's structured source key."""
    match source:
        case OpWriteSource(inst=inst, port=port, invert=invert):
            net = f"{_sig(inst)}_y{port}"  # wide: sign rode the wrapper; bool: fabric inversion folds here
            return f"~{net}" if invert else net
        case InlineWriteSource(operator=operator, operands=operands, conditioner=conditioner):
            return _render_inline(operator, operands, conditioner)
        case FloatMoveWriteSource(operand=operand):
            return _signed_source_net(operand.source, operand.sign)
        case BoolMoveWriteSource(operand=operand):
            return _bool_operand_rhs(operand)
        case _:
            assert_never(source)


def generate(lir: Lir) -> VerilogOutput:
    assert lir.fetch_lag == 2, "only the 2-lag (3-stage) fetch RTL is implemented; 1-lag awaits the latch-removal mode"
    w = _Writer()
    cycw = lir.cyc_width
    pcw = max(1, lir.initiation_interval.bit_length())

    # The two dual codebooks, built once and threaded to both the microcode packer and the emitters so the
    # code<->source mapping cannot drift: per operand port (read) and per register (write). The write side derives from
    # a single ``write_events`` traversal, shared by the codebook, the packer, and the ROM-comment landings.
    read_port = read_ports(lir)
    read_books = read_codebook(lir, read_port)
    events = write_events(lir)
    write_books = write_codebook(events)
    tapped = tapped_lanes(lir)

    fields = build_microcode(lir, read_port, read_books, write_books, events, tapped)
    ucw = finalize_fields(fields)

    issues_by_cycle, commits_by_cycle = lir.group_by_cycle
    landings = landings_by_step(events)

    depth = lir.last_pc + 1  # one microcode word per fetch PC (0..last_pc); inter-block drains and the tail pack to NOP

    _emit_header(w, lir)
    _emit_localparams(w, lir, cycw, pcw, ucw)
    _emit_inline_support(w)
    _emit_declarations(w, lir, tapped)
    _emit_consts(w, lir)
    _emit_microcode_rom(w, fields, ucw, depth, issues_by_cycle, commits_by_cycle, landings)
    _emit_field_wires(w, fields)
    _emit_operators(w, lir, tapped)
    _emit_datapath_comb(w, lir, write_books)
    _emit_read_muxes(w, lir, read_port, read_books)
    _emit_clocked(w, lir, write_books)
    _emit_outputs(w, lir)
    w("\nendmodule\n")
    return VerilogOutput(verilog=w.render(), support_files=support_files())


def _emit_header(w: _Writer, lir: Lir) -> None:
    # Generation time is not included for reproducibility.
    fmt = lir.float_format
    w(f"""
{output_header("// ")}

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
    nreg, nbreg = lir.regfile.nreg, lir.bool_regfile.nreg
    fetch_lag = lir.fetch_lag
    fetch_stages = fetch_lag + 1  # the control-fetch pipeline depth, shown in the localparam comment
    # An unused bank emits no localparam and no register array (a zero-length reg array is illegal Verilog).
    nreg_line = f"\nlocalparam           NREG      ={nreg:4};" if nreg else ""
    nbreg_line = f"\nlocalparam           NBREG     ={nbreg:4};" if nbreg else ""
    w(f"""
localparam           WEXP      ={fmt.wexp:4};  // Float exponent bits fixed by the static schedule
localparam           WMAN      ={fmt.wman:4};  // Float mantissa bits fixed by the static schedule
localparam           W         = WEXP + WMAN;{nreg_line}
localparam           CYCW      ={cycw:4};  // err_pc width: enough for any executing step (0..present)
localparam           PCW       ={pcw:4};  // fetch-PC width: counts to LASTPC (execution lags the fetch by FETCH_LAG)
localparam           FETCH_LAG ={fetch_lag:4};  // executing step = pc - FETCH_LAG ({fetch_stages}-stage control fetch)
localparam [PCW-1:0] PRESENT   ={lir.present_step:4};  // executing step on which the outputs are valid in the array
localparam [PCW-1:0] LASTPC    ={lir.initiation_interval:4};  // = PRESENT + FETCH_LAG; out_valid asserts here
localparam           UCW       ={ucw:4};  // microcode word width after lifting out constant control fields{nbreg_line}
// pc: 0 = idle/accept, present at executing step PRESENT; out_valid at pc==LASTPC (fetch leads execution).
""")
    # Cross-check the ZKF +1.0 formula against the codec at build time. This is the contract holoso_ffrombool's
    # concatenation implements (bias exponent at the fraction MSb, zero sign/fraction); a format whose codec disagreed
    # with it would be caught here rather than miscompiling.
    assert (((1 << (fmt.wexp - 1)) - 1) << (fmt.wman - 1)) == fmt.encode(1.0)
    w("")


def _emit_inline_support(w: _Writer) -> None:
    w(inline_support())
    w("")


def _emit_declarations(w: _Writer, lir: Lir, tapped: set[tuple[OperatorInstance, int]]) -> None:
    assert lir.fetch_lag >= 1, "transacting_q is [FETCH_LAG-1:0]; FETCH_LAG==0 needs an explicit leading NOP"
    w("""
    reg  [PCW-1:0]  pc;            // fetch program counter; the executing step lags it by FETCH_LAG
    reg  [PCW-1:0]  next_pc;       // combinational next-state presented to the ROM each cycle
    reg  [PCW-1:0]  ucode_addr_q;  // PC latch: splits pc -> next_pc -> ROM address from the array read
    reg  [CYCW-1:0] err_pc_q;
    wire            err;           // an operator error is detected on the current step

    reg                 transacting_in;                           // per-branch tag: this pc's word is a live step
    reg [FETCH_LAG-1:0] transacting_q;                            // delays the tag FETCH_LAG onto executing word
    wire                transacting = transacting_q[FETCH_LAG-1]; // gates the executing word's effects
""")
    w("")
    if lir.regfile.nreg:
        w("reg  [W-1:0] regs  [0:NREG-1];   // read-first: a write is visible next step")
    if lir.bool_regfile.nreg:
        w("reg          bregs [0:NBREG-1];")
    w("")
    for inst in lir.instances:
        sig = _sig(inst)
        for pos, operand_type in enumerate(inst.operator.signature.operand_types):
            assert is_wide_type(operand_type), "pooled operators read only wide operands today"
            letter = PORT_LETTERS[pos]
            w(f"reg  [W-1:0] {sig}_{letter};")  # combinational read-mux output (driven in the read-mux always @*)
        # One net per TAPPED output port -- the raw operator output (wide W-bit or boolean 1-bit). The in_valid and
        # sign-control ports bind directly to the decoded uc_* fields, so no s_* control net is declared for them.
        for q, result_type in enumerate(inst.operator.signature.result_types):
            if (inst, q) not in tapped:
                continue  # a never-tapped output port: no nets, the module port is left unconnected
            if is_wide_type(result_type):
                w(f"wire [W-1:0] {sig}_y{q};")
            else:
                w(f"wire         {sig}_y{q};")
        for port in inst.operator.error_ports:
            w(f"wire         {sig}_{port};")
    w("")


def _emit_consts(w: _Writer, lir: Lir) -> None:
    fmt = lir.float_format
    width = fmt.width
    digits = (width + 3) // 4
    for index, value in enumerate(lir.float_consts):
        w(f"wire [W-1:0] const_{index} = {width}'h{fmt.encode(value):0{digits}x};  // {value!r}")
    if lir.float_consts:
        w("")


def _emit_operators(w: _Writer, lir: Lir, tapped: set[tuple[OperatorInstance, int]]) -> None:
    for inst in lir.instances:
        sig, base = _sig(inst), base_name(inst)
        operator = inst.operator
        letters = PORT_LETTERS[: operator.arity]
        # WEXP/WMAN frame the float format; hdl_params() lists K (ilog2) and every STAGE_* explicitly. LATENCY is
        # emitted separately from the scheduler model, so a model/RTL drift fails during wrapper elaboration.
        parts = [".WEXP(WEXP)", ".WMAN(WMAN)"] + [
            f".{param}({value})" for param, value in operator.hdl_params().items()
        ]
        parts.append(f".LATENCY({operator.latency})")
        params = ", ".join(parts)
        w(f"{operator.module_name} #(", f"    {params}", f") u_{base} (")
        w.push()
        w(f".clk(clk), .rst(rst), .in_valid({f_issue(base)}),")
        for imm in operator.immediate_ports:
            w(f".{imm.name}({f_imm(base, imm.name)}),")
        for letter in letters:
            w(f".{letter}_sgnop({f_osgn(base, letter)}),")
        # A float output port carries a hardware sign conditioner (piped inside the wrapper); an untapped one is tied
        # to the identity. Boolean output ports have none -- their inversion conditioner is fabric-side at the write.
        for q, result_type in enumerate(operator.signature.result_types):
            if is_wide_type(result_type):
                conditioner = f_ysgn(base, q) if (inst, q) in tapped else "2'd0"
                w(f".{operator.output_hdl_ports[q]}_sgnop({conditioner}),")
        for letter in letters:
            w(f".{letter}({sig}_{letter}),")
        # out_valid is left unconnected: the static schedule already knows when each result is ready.
        w(".out_valid(),")
        outputs = [
            f".{operator.output_hdl_ports[q]}({f'{sig}_y{q}' if (inst, q) in tapped else ''})"
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
    commits_by_cycle: dict[int, list[PooledScheduledOp]],
    landings: dict[int, list[str]],
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
    for step in range(depth):  # depth == LASTPC + 1: one word per fetch PC, NOP where nothing issues, commits, or lands
        summary = cycle_summary(issues_by_cycle.get(step, []), commits_by_cycle.get(step, []), landings.get(step, []))
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
// Decoded control fields. A field constant across the whole program is driven by a constant net (so synthesis prunes
// the logic it feeds); a varying field is a slice of the instruction word. The effect-trigger fields -- the ``gated``
// ones (operator issue strobes and per-register write opcodes) -- are ANDed with `transacting` HERE, so a held
// ucode[0] dwell, a fill bubble, or a stale pre-reset word decodes to 0 (NOP): no issue and every register holds.
// The AND wraps the constant branch too, so the gate stays unconditional even if a trigger ever folds to a constant.
""")
    for f in fields.values():
        rhs = _lit(f.width, f.const_value) if f.offset < 0 else f"ucode_word[{f.offset} +: {f.width}]"
        if f.gated:
            mask = "transacting" if f.width == 1 else f"{{{f.width}{{transacting}}}}"
            rhs = f"{mask} & {rhs}"
        w(f"{_wire(f.width)}{f.name} = {rhs};")
    w("")


def _emit_datapath_comb(w: _Writer, lir: Lir, write_books: dict[RegRef | BoolRegRef, WriteCodebook]) -> None:
    # An error matters only on the step its operator commits -- the step some destination register's write opcode
    # selects this operator's output lane. So the error output is gated by the OR, over the op's write-codebook entries,
    # of (uc_op_<reg> == that entry's code); the opcode is transacting-masked, so the gate is live exactly on that
    # commit step.
    op_gates: dict[OperatorInstance, list[str]] = {}
    for dst, book in write_books.items():
        for code, source in book.arms():
            if isinstance(source, OpWriteSource):
                op_gates.setdefault(source.inst, []).append(f"({f_op(dst)} == {_lit(book.opcode_width, code)})")
    err_terms: list[str] = []
    for inst in lir.instances:
        if not inst.operator.error_ports:
            continue
        gates = op_gates.get(inst, [])
        assert gates, "an error-bearing operator must have a tapped lane to align its sideband with"
        gate = gates[0] if len(gates) == 1 else "(" + " | ".join(gates) + ")"
        for err_port in inst.operator.error_ports:
            err_terms.append(f"({gate} & {_sig(inst)}_{err_port})")
    err_rhs = " | ".join(err_terms) if err_terms else "1'b0"
    w(f"assign err = {err_rhs};", "")

    redirects = _terminator_redirects(lir)
    w("""
// Next-PC sequencer (combinational). The PC holds at the accept (pc==0) and present (pc==LASTPC) boundaries; bubble
// steps carry a NOP word and the PC keeps advancing. The executing step lags the fetch PC by FETCH_LAG. A block's
// terminator redirects the fetch PC at the block's boundary step (a branch reads its boolean register). Each branch
// also sets transacting_in -- 1 for a live accept/body word, 0 at the boundaries -- so each branch tags its own word.
always @* begin
""")
    w.push()
    w("if (rst) begin")
    w.push()
    w("next_pc        = 0;")
    w("transacting_in = 1'b0;")
    w.pop()
    w("end else if (out_valid) begin  // present: hold until the result is taken")
    w.push()
    w("next_pc        = out_ready ? 0 : LASTPC;")
    w("transacting_in = 1'b0;")
    w.pop()
    w("end else if (in_ready) begin   // accept: hold until a transaction arrives")
    w.push()
    w("next_pc        = in_valid ? 1 : 0;")
    w("transacting_in = in_valid;")
    w.pop()
    w("end else begin                 // advance the fetch: the body of a live transaction")
    w.push()
    w("transacting_in = 1'b1;")
    if not redirects:
        w("next_pc        = pc + 1'b1;")
    else:
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


def _emit_read_case(w: _Writer, target: str, port: int, book: ReadCodebook) -> None:
    """
    Emit one operand's combinational read mux: a direct assign for a single source, else a ``case`` over the port's
    read opcode selecting a register or a constant directly. The last entry is the ``default`` arm so the case is full
    (no inferred latch on this combinational path); unused high codes fall there too and are don't-cares on idle steps.
    The mux carries no indexed part-select, so there is no offset multiply for synthesis to (mis)infer as a DSP.
    """
    assert book.sources, "an operand port always reads at least one source"
    if len(book.sources) == 1:
        w(f"{target} = {_source_net(book.sources[0])};")
        return
    arms = book.arms()
    w(f"case ({f_rd(port)})")
    w.push()
    for index, (code, source) in enumerate(arms):
        label = "default" if index == len(arms) - 1 else _lit(book.opcode_width, code)
        w(f"{label}: {target} = {_source_net(source)};")
    w.pop()
    w("endcase")


def _emit_read_muxes(
    w: _Writer,
    lir: Lir,
    read_port: dict[tuple[OperatorInstance, int], int],
    read_books: dict[int, ReadCodebook],
) -> None:
    """
    Emit the combinational operand read muxes driving each wrapper directly, so regfile-read -> operator is
    combinational and the operand is sampled FETCH_LAG after its read-opcode word. A pure-inline kernel has no pooled
    instances, so this would be empty -- skip it rather than emit a bare ``always @* begin end``.
    """
    if not lir.instances:
        return
    w("// Operand read muxes: a per-port case over the read codebook (registers and constants), driving the wrapper.")
    w("always @* begin")
    w.push()
    for inst in lir.instances:
        sig = _sig(inst)
        for pos in range(inst.operator.arity):
            port = read_port[(inst, pos)]
            _emit_read_case(w, f"{sig}_{PORT_LETTERS[pos]}", port, read_books[port])
    w.pop()
    w("end")
    w("")


def _emit_reg_write(
    w: _Writer,
    lhs: str,
    dst: RegRef | BoolRegRef,
    book: WriteCodebook | None,
    special_arms: list[tuple[str, str]],
) -> None:
    """
    One segregated write statement per register (the multi-assign rule): the handshake-gated special arms first (an
    input load, a boundary state install), then the opcode ``case`` over the register's write codebook as the final
    ``else``. Code 0 is the NOP hold -- an unlisted code in this clocked ``case`` retains the flop -- so the
    write-enable is folded into the opcode with no extra logic level. A single-source register degenerates to
    ``if (opcode)``.
    """
    clause = "if"
    for cond, rhs in special_arms:
        w(f"{clause} ({cond}) {lhs} <= {rhs};")
        clause = "else if"
    if book is None or not book.sources:
        return
    prefix = "else " if special_arms else ""
    opcode = f_op(dst)
    if len(book.sources) == 1:
        w(f"{prefix}if ({opcode}) {lhs} <= {_write_source_rhs(book.sources[0])};")
        return
    w(f"{prefix}case ({opcode})")
    w.push()
    for code, source in book.arms():
        w(f"{_lit(book.opcode_width, code)}: {lhs} <= {_write_source_rhs(source)};")
    w.pop()
    w("endcase")


def _emit_clocked(w: _Writer, lir: Lir, write_books: dict[RegRef | BoolRegRef, WriteCodebook]) -> None:
    """Emit every sequential element in one always @(posedge clk): fetch, register writes, and control state."""
    nreg, nbreg = lir.regfile.nreg, lir.bool_regfile.nreg
    float_slots = {slot.reg.index: slot for slot in lir.float_state_slots}
    bool_slots = {slot.reg.index: slot for slot in lir.bool_state_slots}
    float_loads = {load.dst.index: load for load in lir.float_inputs}
    bool_loads = {load.dst.index: load for load in lir.bool_inputs}

    def load_arm(loads: Mapping[int, FloatInputLoad | BoolInputLoad], reg: int) -> list[tuple[str, str]]:
        load = loads.get(reg)
        return [("in_ready && in_valid", f"in_{load.name}")] if load else []

    w("""
// All sequential logic in one clocked process. Reset gates only the control state (pc, err_pc_q, transacting_q) and the
// persistent state registers; every other register is reset-unconditional. Each is driven by exactly one statement.
always @(posedge clk) begin
""")
    w.push()

    w("// Microcode fetch: PC latch -> control-store array read -> BRAM output register.")
    w("ucode_addr_q <= next_pc;")
    w("ucode_q      <= ucode[ucode_addr_q];")
    w("ucode_word   <= ucode_q;")
    w("")

    # Non-slot registers: one reset-unconditional statement each. Datapath payload carries no reset, keeping the
    # high-fanout reset net off the wide cone (only control/valid state is reset); contents are don't-care until a
    # valid write lands. A register with neither an input load nor any opcode source is simply omitted.
    nonslot_wide = [
        reg for reg in range(nreg) if reg not in float_slots and (RegRef(reg) in write_books or reg in float_loads)
    ]
    nonslot_bool = [
        reg for reg in range(nbreg) if reg not in bool_slots and (BoolRegRef(reg) in write_books or reg in bool_loads)
    ]
    if nonslot_wide or nonslot_bool:
        w("// Register writes (reset-unconditional): one opcode-selected statement per register.")
        for reg in nonslot_wide:
            _emit_reg_write(w, f"regs[{reg}]", RegRef(reg), write_books.get(RegRef(reg)), load_arm(float_loads, reg))
        for reg in nonslot_bool:
            _emit_reg_write(
                w, f"bregs[{reg}]", BoolRegRef(reg), write_books.get(BoolRegRef(reg)), load_arm(bool_loads, reg)
            )
        w("")

    # Control and persistent state are the reset-gated registers: the slot snapshot (under rst) and the slot's update
    # statement (its opcode-selected writes plus a boundary install arm, under the else) are the two arms of one rst
    # condition, segregating those assignments for the synthesizer.
    fmt = lir.float_format
    digits = (fmt.width + 3) // 4
    w("// Control and persistent state: the reset-gated registers.")
    w("if (rst) begin")
    w.push()
    w("pc            <= 0;")
    w("err_pc_q      <= 0;")
    w("transacting_q <= 0;")
    for slot in lir.float_state_slots:
        bits = f"{fmt.width}'h{fmt.encode(slot.reset_value):0{digits}x}"
        w(f"regs[{slot.reg.index}] <= {bits};  // {slot.name} reset snapshot")
    for bslot in lir.bool_state_slots:
        w(f"bregs[{bslot.reg.index}] <= 1'b{int(bslot.reset_value)};  // {bslot.name} reset snapshot")
    w.pop()
    w("end else begin")
    w.push()
    w("pc <= next_pc;")
    w("transacting_q <= (transacting_q << 1) | transacting_in;")
    w("if (err) err_pc_q <= pc - FETCH_LAG;  // err wins; execution lags the fetch PC by FETCH_LAG, so step is pc-lag")
    w("else if (in_ready && in_valid) err_pc_q <= 0;  // clear the diagnostic when a new transaction is accepted")
    # A non-coalesced slot installs its live-out read-first at the accepted-output boundary (out_valid && out_ready, so
    # a held boundary copies exactly once), a lower-priority arm of the same statement; an early install is an ordinary
    # opcode source (see write_events). Boolean state installs are boundary-only.
    for reg, slot in sorted(float_slots.items()):
        arms = load_arm(float_loads, reg)
        if slot.needs_copy and lir.float_state_install_is_boundary(slot):
            # The boundary install is a higher-priority arm than the opcode case, so it must not shadow any opcode
            # write. It never can: a non-coalesced boundary slot holds its live-in until the read-first boundary read,
            # so the allocator lands no intermediate write on it -- pinned here so a future scheduler cannot regress it.
            assert RegRef(reg) not in write_books, "a boundary-install slot must carry no opcode write sources"
            arms.append(("out_valid && out_ready", _state_copy_rhs(slot)))
        _emit_reg_write(w, f"regs[{reg}]", RegRef(reg), write_books.get(RegRef(reg)), arms)
    for reg, bslot in sorted(bool_slots.items()):
        arms = load_arm(bool_loads, reg)
        if bslot.needs_copy:
            assert BoolRegRef(reg) not in write_books, "a boundary-install bool slot must carry no opcode write sources"
            arms.append(("out_valid && out_ready", _bool_operand_rhs(bslot.live_out)))
        _emit_reg_write(w, f"bregs[{reg}]", BoolRegRef(reg), write_books.get(BoolRegRef(reg)), arms)
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
    for wire in lir.outputs:
        match wire:
            case BoolOutputWire():
                w(f"assign {wire.name} = {_bool_operand_rhs(wire.tap)};")
            case FloatOutputWire():
                w(f"assign {wire.name} = {_signed_source_net(wire.tap.source, wire.tap.sign)};")
    w("")
