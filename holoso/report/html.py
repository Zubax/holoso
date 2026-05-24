"""Render a self-contained, light-themed single-page HTML report for a synthesized module.

The stylesheet and the interactive layer live alongside this module as ``html.css`` and ``html.js`` (declared as
package data in ``pyproject.toml``); they are inlined into the self-contained report so it has no external dependency
beyond the web font.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

from ..format import FloatFormat
from ..lir import Lir, Operand, OperatorInstance, RegRef, ScheduledOp
from ..operators import OpKind, Sgnop, latency_of
from ..result import ModuleInterface, SynthesisMetrics

_GITHUB_URL = "https://github.com/Zubax/holoso"

_CSS = resources.files(__package__).joinpath("html.css").read_text(encoding="utf-8")
# Interactive layer; ``__DATA__`` is replaced by the per-module payload in ``_sched_script``.
_SCHED_JS = resources.files(__package__).joinpath("html.js").read_text(encoding="utf-8")

# Reserved, high-contrast operator colors (white text legible on each) for a light background.
_KIND_COLOR: dict[OpKind, str] = {
    OpKind.FADD: "#2456a6",
    OpKind.FMUL: "#1f7a3d",
    OpKind.FDIV: "#b3261e",
    OpKind.FMUL_ILOG2: "#6d4c9f",
}
_KIND_LABEL: dict[OpKind, str] = {
    OpKind.FADD: "+",
    OpKind.FMUL: "*",
    OpKind.FDIV: "/",
    OpKind.FMUL_ILOG2: "<<",
}
_MODULE_HEADER_RE = re.compile(r"(?ms)^module\b.*?^\);")
_VERILOG_TOKEN_RE = re.compile(r"(?P<space>\s+)|(?P<ident>[A-Za-z_]\w*)|(?P<number>\d+)|(?P<other>.)")
_VERILOG_KEYWORDS = frozenset({"module", "parameter", "input", "output", "wire", "reg"})


def _esc(text: str) -> str:
    return html.escape(text)


def _operand(operand: Operand) -> str:
    name = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return operand.sgnop.decorate(name)


def _op_text(op: ScheduledOp) -> str:
    # Spaceless (``r1=r1+r3``): many operations can commit on one clock now, so the operations column packs tightly.
    if op.inst.kind is OpKind.FMUL_ILOG2:
        body = f"{_operand(op.a)}*2^{op.k}"
    else:
        assert op.b is not None
        body = f"{_operand(op.a)}{_KIND_LABEL[op.inst.kind]}{_operand(op.b)}"
    return op.y_sgnop.decorate(f"r{op.dst.index}={body}")


def build_report_html(lir: Lir, interface: ModuleInterface, metrics: SynthesisMetrics, module_verilog: str) -> str:
    fmt = lir.fmt
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    link = f"<a href='{_GITHUB_URL}'>Holoso</a>"
    out: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Module {_esc(lir.module_name)} - Holoso</title><style>{_CSS}</style></head><body>",
        f"<header><h1>Module {_esc(lir.module_name)}</h1>"
        f"<div class='sub'>Synthesized by {link} at {generated}</div></header><main>",
    ]
    # The compact summary sections share one wrapping row (metrics, then the narrow constants and interface) so they
    # do not waste page height; the wide register-grid schedule follows below.
    out.append("<div class='toprow'>")
    out.append(f"<div class='sec'>{_metrics(interface, metrics, fmt)}</div>")
    constants = _constants(lir)
    if constants:
        out.append(f"<div class='sec'>{constants}</div>")
    out.append(f"<div class='sec'>{_interface(interface)}</div>")
    out.append(f"<div class='sec modhdrsec'>{_module_header(module_verilog)}</div>")
    out.append("</div>")
    if interface.ii.cycles != metrics.ii_cycles:
        raise RuntimeError(
            f"report timing mismatch: interface II is {interface.ii.cycles} cycles, "
            f"metrics II is {metrics.ii_cycles} cycles"
        )
    out.append(_schedule(lir, fmt, interface.ii.cycles))
    out.append("</main></body></html>")
    return "".join(out)


def _metrics(interface: ModuleInterface, metrics: SynthesisMetrics, fmt: FloatFormat) -> str:
    instances = " ".join(f"{count}×{kind}" for kind, count in metrics.operator_instances.items())
    rows: list[tuple[str, object]] = [
        ("ZKF format", f"e{fmt.wexp}+m{fmt.wman} = {fmt.width}-bit"),
        ("operator instances", instances or "-"),
        ("float registers", metrics.n_float_regs),
        ("regfile R/W ports", f"{metrics.read_ports} / {metrics.write_ports}"),
        ("schedule makespan", metrics.makespan),
        ("operations", metrics.op_count),
        ("II (cycles)", metrics.ii_cycles),
        ("longest op chain", metrics.max_chain_len),
    ]
    body = "".join(f"<tr><th>{_esc(label)}</th><td>{_esc(str(value))}</td></tr>" for label, value in rows)
    note = f"Initiation interval = in_valid&rarr;out_valid latency: {_esc(interface.ii.formula)}."
    return f"<h2>Metrics</h2><table class='metrics'>{body}</table><p class='note'>{note}</p>"


def _interface(interface: ModuleInterface) -> str:
    out = ["<h2>Interface</h2><div class='ifaces'>"]
    ctrl = interface.control_ports
    out.append(f"<div class='iface'><h3>ctrl ({len(ctrl)})</h3><table><tr><th>port</th><th>dir</th><th>bits</th></tr>")
    for port in ctrl:
        out.append(f"<tr><td>{_esc(port.name)}</td><td>{port.direction.value}</td><td>{port.width}</td></tr>")
    out.append("</table></div>")
    for title, ports in (("in", interface.input_ports), ("out", interface.output_ports)):
        out.append(f"<div class='iface'><h3>{title} ({len(ports)})</h3><table><tr><th>port</th><th>bits</th></tr>")
        for port in ports:
            out.append(f"<tr><td>{_esc(port.name)}</td><td>{port.width}</td></tr>")
        out.append("</table></div>")
    out.append("</div>")
    return "".join(out)


def _module_header(module_verilog: str) -> str:
    header = _extract_module_header(module_verilog)
    return f"<h2>Module Header</h2><pre class='modhdr'><code>{_highlight_verilog(header)}</code></pre>"


def _extract_module_header(module_verilog: str) -> str:
    match = _MODULE_HEADER_RE.search(module_verilog)
    if match is None:
        raise RuntimeError("cannot find generated Verilog module header")
    return match.group(0)


def _highlight_verilog(text: str) -> str:
    return "\n".join(_highlight_verilog_line(line) for line in text.splitlines())


def _highlight_verilog_line(line: str) -> str:
    code, sep, comment = line.partition("//")
    highlighted = _highlight_verilog_code(code)
    if sep:
        highlighted += f"<span class='vh-comment'>{_esc(sep + comment)}</span>"
    return highlighted


def _highlight_verilog_code(code: str) -> str:
    out: list[str] = []
    for match in _VERILOG_TOKEN_RE.finditer(code):
        token = match.group(0)
        if match.lastgroup == "ident" and token in _VERILOG_KEYWORDS:
            out.append(f"<span class='vh-keyword'>{_esc(token)}</span>")
        elif match.lastgroup == "number":
            out.append(f"<span class='vh-number'>{token}</span>")
        else:
            out.append(_esc(token))
    return "".join(out)


def _constants(lir: Lir) -> str:
    if not lir.consts:
        return ""
    chips = "".join(
        f"<span class='const'>c{index} = {_esc(repr(value))}</span>" for index, value in enumerate(lir.consts)
    )
    return f"<h2>Constants</h2><div>{chips}</div>"


_NEUTRAL = "#6b7280"  # input/output bookend chips (port boundary, not an operator)
_LIVE_BG = "#edf2fb"  # legend swatch for the liveness tint; the grid cells use the ``.live`` CSS class (same color)
ColKey = tuple[str, int]  # ("r", reg index) or ("c", constant index)


def _is_live(col: ColKey, row_id: int, live: dict[int, set[int]]) -> bool:
    """Whether register column ``col`` holds a live value on grid row ``row_id`` (constants are never tinted)."""
    return col[0] == "r" and col[1] in live and row_id in live[col[1]]


def _write_label(index: int, tip: str) -> str:
    """The result marker on the operator's commit cell: its instance index, white text on the filled result cell.

    Operands are no longer chips; the dataflow edges (drawn by the overlay) connect this cell to its operand cells, so
    the cell only needs to identify the operator instance. The tooltip carries the full expression with operand signs.
    """
    return f"<span class='wl' title='{tip}'>{index}</span>"


def _operand_col(operand: Operand) -> ColKey:
    return ("r", operand.source.index) if isinstance(operand.source, RegRef) else ("c", operand.source.index)


@dataclass(frozen=True, slots=True)
class _Dividers:
    """Right-border seams between grid columns, kept on the *left* cell's right edge (under ``border-collapse`` the left
    cell wins an equal-width conflict, so a left border on the right column would not show). A ``thick`` 2px seam marks
    the two block boundaries (constants | operator-pipeline and operator-pipeline | OPERATIONS); a ``thin`` 1px seam
    marks the lighter ones (registers | constants and between operator groups). Data and stage columns index separately.
    """

    data_thin: set[int]
    data_thick: set[int]
    stage_thin: set[int]
    stage_thick: set[int]


def _border_suffix(idx: int, thin: set[int], thick: set[int]) -> str:
    if idx in thick:
        return " rbk2"
    if idx in thin:
        return " rbk"
    return ""


def _gc_class(ordinal: int, dv: _Dividers) -> str:
    """Class for a register/constant data cell in column ``ordinal`` (with its right-border divider, if any)."""
    return "gc" + _border_suffix(ordinal, dv.data_thin, dv.data_thick)


def _oc_class(sidx: int, dv: _Dividers) -> str:
    """Class for an operator-stage cell in stage column ``sidx`` (with its right-border divider, if any)."""
    return "oc" + _border_suffix(sidx, dv.stage_thin, dv.stage_thick)


def _stage_columns(lir: Lir, fmt: FloatFormat) -> list[tuple[OperatorInstance, int]]:
    """The operator-stage columns, in instance order: ``(instance, stage)`` for each pipeline stage of each operator.

    One column per stage, so an L-cycle operator contributes L columns labeled ``s0..s(L-1)``. As an operation flows
    through its operator, stage ``k`` is occupied on cycle ``issue + k``; the result commits to its register on cycle
    ``issue + L``. This makes the pipeline's advance directly visible and exposes any structural hazard once several
    operations are in flight at once.
    """
    cols: list[tuple[OperatorInstance, int]] = []
    for inst in lir.instances:
        cols.extend((inst, k) for k in range(latency_of(inst.kind, fmt, lir.stages)))
    return cols


def _bookend_row(
    label: str,
    cells: dict[ColKey, str],
    columns: list[ColKey],
    live: dict[int, set[int]],
    row_id: int,
    n_stage: int,
    dv: _Dividers,
) -> str:
    """The input-load row (cycle 0). No operator is in flight yet, so the operator-stage cells and ops cell are empty."""
    out = [f"<tr><td class='clk'>{label}</td>"]
    for ordinal, col in enumerate(columns):
        extra = " live" if _is_live(col, row_id, live) else ""
        out.append(f"<td class='{_gc_class(ordinal, dv)}{extra}'>{cells.get(col, '')}</td>")
    for sidx in range(n_stage):
        out.append(f"<td class='{_oc_class(sidx, dv)}'></td>")
    out.append("<td class='opcell'></td></tr>")
    return "".join(out)


def _liveness(lir: Lir) -> dict[int, set[int]]:
    """Map each register to the clock cycles on which it holds a live value.

    A value is written on its definition cycle -- the accept cycle 0 for an input, the operator's commit cycle
    (``issue + latency``) for a result -- and read on each consumer's issue cycle, or on the output-present cycle
    ``makespan + 1`` if it drives an output. The report displays cycles 0..makespan; retaining the present cycle in the
    intervals keeps the final visible row live for output values. Liveness is per value, so a register reused for
    several values yields several disjoint residence intervals with dead gaps between them.
    """
    present = lir.makespan + 1
    defs: dict[int, list[int]] = {}
    uses: dict[int, list[int]] = {}
    for load in lir.inputs:
        defs.setdefault(load.dst.index, []).append(0)
    for op in lir.ops:
        defs.setdefault(op.dst.index, []).append(op.commit_cycle)
        for operand in (op.a, op.b):
            if operand is not None and isinstance(operand.source, RegRef):
                uses.setdefault(operand.source.index, []).append(op.issue_cycle)
    for wire in lir.outputs:
        if isinstance(wire.source, RegRef):
            uses.setdefault(wire.source.index, []).append(present)
    live: dict[int, set[int]] = {}
    for reg in defs.keys() | uses.keys():
        writes = sorted(defs.get(reg, []))
        reads = sorted(uses.get(reg, []))
        rows: set[int] = set()
        for i, start in enumerate(writes):
            nxt = writes[i + 1] if i + 1 < len(writes) else present + 1  # this value persists until the next overwrite
            last = max((u for u in reads if start <= u < nxt), default=start)
            rows.update(range(start, last + 1))
        live[reg] = rows
    return live


def _live_intervals(rows: set[int]) -> list[list[int]]:
    """Collapse a register's set of live rows into sorted ``[start, end]`` intervals for compact hand-off to the hover
    script (so it can answer alive/dead for any cycle without shipping a per-cell flag)."""
    if not rows:
        return []
    ordered = sorted(rows)
    intervals = [[ordered[0], ordered[0]]]
    for value in ordered[1:]:
        if value == intervals[-1][1] + 1:
            intervals[-1][1] = value
        else:
            intervals.append([value, value])
    return intervals


def _cell_style(
    col: ColKey, cyc: int, live: dict[int, set[int]], fills: dict[tuple[int, ColKey], str]
) -> tuple[str, str]:
    """Background for a register/constant cell, as ``(extra_class, inline_style)``. The single cycle on which a result
    commits is filled solid with its operator color via an inline style (this takes precedence, and being inline it
    survives the row-hover tint); otherwise a live register gets the faint residence tint via the ``live`` class, so a
    row-hover can override it. The operators' cycle-by-cycle occupancy lives in the separate operator-stage block.
    """
    color = fills.get((cyc, col))
    if color is not None:
        return "", f" style='background:{color}'"
    if _is_live(col, cyc, live):
        return " live", ""
    return "", ""


def _schedule(lir: Lir, fmt: FloatFormat, expected_ii: int) -> str:
    nreg, nconst = lir.regfile.nreg, len(lir.consts)
    columns = _columns_of(lir)
    col_ord = {col: ordinal for ordinal, col in enumerate(columns)}
    compute_cycles = list(range(1, lir.makespan + 1))
    displayed_cycles = 1 + len(compute_cycles)  # input-load cycle 0 plus compute/writeback cycles 1..makespan
    if displayed_cycles != expected_ii:
        raise RuntimeError(
            f"schedule grid displays {displayed_cycles} cycle rows, but computed II is {expected_ii} cycles "
            f"(makespan={lir.makespan})"
        )

    # The operator-stage block: one square column per pipeline stage of each operator, in instance order.
    stage_cols = _stage_columns(lir, fmt)
    n_stage = len(stage_cols)
    stage_base: dict[OperatorInstance, int] = {}
    for sidx, (inst, _k) in enumerate(stage_cols):
        stage_base.setdefault(inst, sidx)
    group_ends = {
        stage_base[inst] + latency_of(inst.kind, fmt, lir.stages) - 1 for inst in lir.instances
    }  # last stage per operator

    # Column seams: 2px at the two block boundaries (constants | pipeline and pipeline | OPERATIONS); 1px at the lighter
    # registers | constants seam and between operator groups.
    dv = _Dividers(
        data_thin={nreg - 1} if nreg and nconst else set(),
        data_thick={len(columns) - 1} if columns else set(),
        stage_thin=group_ends - {n_stage - 1},
        stage_thick={n_stage - 1} if n_stage else set(),
    )

    live = _liveness(lir)
    edges: list[tuple[str, str, str, int]] = []  # (commit id, operand id, color, operation group) for the overlay
    # Operator pipeline occupancy: instance ``inst`` is in stage ``k`` on cycle ``issue + k``. Keyed to the operation
    # group so a hover lights the whole pipeline trail together with the result cell, its chip and its edges.
    # ``conflicts`` flags any cell two operations claim at once -- the alarm for a scheduling bug (a structural hazard
    # the pipelined scheduler should never emit).
    stage_fill: dict[tuple[int, int], tuple[str, int]] = {}  # (stage column, cycle) -> (operator color, group)
    stage_tip: dict[tuple[int, int], str] = {}
    conflicts: set[tuple[int, int]] = set()
    # Result-commit cells, keyed by absolute (cycle, column): each operation lands its result on its commit cycle.
    fills: dict[tuple[int, ColKey], str] = {}
    writes_at: dict[tuple[int, ColKey], str] = {}
    endpoints: set[tuple[int, int]] = set()  # (column ordinal, cycle) cells that anchor a dataflow edge
    cell_group: dict[tuple[int, int], int] = {}  # (column ordinal, cycle) -> operation group, for commit cells
    chips_at: dict[int, list[str]] = {}  # cycle -> chips for the operations committing on that row

    group = 0  # global per-operation id, linking a commit cell with its edges/chip for the hover-focus behavior
    for op in lir.ops:
        tip = _esc(_op_text(op))
        color = _KIND_COLOR[op.inst.kind]
        issue, commit = op.issue_cycle, op.commit_cycle
        dcol: ColKey = ("r", op.dst.index)
        dord = col_ord[dcol]
        writes_at[(commit, dcol)] = writes_at.get((commit, dcol), "") + _write_label(op.inst.index, tip)
        fills[(commit, dcol)] = color
        endpoints.add((dord, commit))
        cell_group[(dord, commit)] = group
        for operand in (op.a, op.b):
            if operand is not None:
                oord = col_ord[_operand_col(operand)]
                endpoints.add((oord, issue))  # operands are read on the issue cycle
                edges.append((f"g{dord}_{commit}", f"g{oord}_{issue}", color, group))
        chips_at.setdefault(commit, []).append(
            f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>"
        )
        base = stage_base[op.inst]
        for k in range(op.latency):  # stamp the pipeline trail: stage k of this operator is busy on cycle issue + k
            key = (base + k, issue + k)
            if key in stage_fill and stage_fill[key][1] != group:
                conflicts.add(key)
            stage_fill[key] = (color, group)
            stage_tip[key] = f"{op.inst.kind.value}_{op.inst.index} s{k}: {tip}"
        group += 1

    out = [_schedule_key(lir), "<div id='schedwrap'><table class='grid'>"]
    # Header row 0: group bands over the register, constant and operator-pipeline blocks. clk/operations span all rows.
    out.append("<tr><th class='gh clkh' rowspan='3'><span>clk</span></th>")
    if nreg:
        reg_seam = _border_suffix(nreg - 1, dv.data_thin, dv.data_thick)
        out.append(f"<th class='gband{reg_seam}' colspan='{nreg}'><span>registers</span></th>")
    if nconst:
        const_seam = _border_suffix(nreg + nconst - 1, dv.data_thin, dv.data_thick)
        out.append(f"<th class='gband{const_seam}' colspan='{nconst}'><span>constants</span></th>")
    if n_stage:
        seam = _border_suffix(n_stage - 1, dv.stage_thin, dv.stage_thick)
        out.append(f"<th class='gband{seam}' colspan='{n_stage}'><span>operator pipelines</span></th>")
    out.append("<th class='oph' rowspan='3'>operations</th></tr>")
    # Header row 1: register and constant column labels, and one group cell per operator spanning its stages.
    out.append("<tr>")
    for index in range(nreg):
        cls = "gh" + _border_suffix(index, dv.data_thin, dv.data_thick)
        out.append(f"<th class='{cls}' rowspan='2'><span>r{index}</span></th>")
    for index in range(nconst):
        cls = "gh k" + _border_suffix(nreg + index, dv.data_thin, dv.data_thick)
        out.append(f"<th class='{cls}' rowspan='2'><span>c{index}</span></th>")
    for inst in lir.instances:
        lat = latency_of(inst.kind, fmt, lir.stages)
        name = f"{inst.kind.value}_{inst.index}"  # full name, set vertically so a 1-stage operator does not widen
        seam = _border_suffix(stage_base[inst] + lat - 1, dv.stage_thin, dv.stage_thick)
        out.append(
            f"<th class='ohgrp{seam}' colspan='{lat}' style='color:{_KIND_COLOR[inst.kind]}'>"
            f"<span>{_esc(name)}</span></th>"
        )
    out.append("</tr>")
    # Header row 2: the per-stage labels (s0, s1, ...) under each operator group.
    out.append("<tr>")
    for sidx, (_inst, k) in enumerate(stage_cols):
        cls = "gh" + _border_suffix(sidx, dv.stage_thin, dv.stage_thick)
        out.append(f"<th class='{cls}'><span>s{k}</span></th>")
    out.append("</tr>")

    # The grid has exactly one displayed row per II cycle: the input-load cycle (0), then the compute/writeback cycles
    # 1..makespan. The output-present boundary is not an extra cycle row; out_valid rises after these II cycles.
    in_cells = {("r", load.dst.index): _input_chip(f"in_{load.name}") for load in lir.inputs}
    out.append(_bookend_row("in", in_cells, columns, live, 0, n_stage, dv))

    for cyc in compute_cycles:  # one row per compute cycle; idle cycles show only pipeline advance
        out.append("<tr>")
        out.append(f"<td class='clk'>{cyc}</td>")
        for ordinal, col in enumerate(columns):
            content = writes_at.get((cyc, col), "")
            extra, style = _cell_style(col, cyc, live, fills)
            attrs = f" id='g{ordinal}_{cyc}'" if (ordinal, cyc) in endpoints else ""
            if (ordinal, cyc) in cell_group:
                attrs += f" data-op='{cell_group[(ordinal, cyc)]}'"
            out.append(f"<td class='{_gc_class(ordinal, dv)}{extra}'{attrs}{style}>{content}</td>")
        for sidx in range(n_stage):
            out.append(_stage_cell(sidx, cyc, dv, stage_fill, stage_tip, conflicts))
        out.append(f"<td class='opcell'>{''.join(chips_at.get(cyc, []))}</td>")
        out.append("</tr>")
    out.append("</table><svg class='edges'></svg></div>")
    out.append(_sched_script(lir, edges, live))
    return "".join(out)


def _stage_cell(
    sidx: int,
    cyc: int,
    dv: _Dividers,
    stage_fill: dict[tuple[int, int], tuple[str, int]],
    stage_tip: dict[tuple[int, int], str],
    conflicts: set[tuple[int, int]],
) -> str:
    """One operator-stage cell: empty unless an operation occupies this stage on this cycle, in which case it is filled
    with the operator color and tagged with the operation group (so a hover lights the whole pipeline trail)."""
    cls = _oc_class(sidx, dv)
    occ = stage_fill.get((sidx, cyc))
    if occ is None:
        return f"<td class='{cls}'></td>"
    color, group = occ
    if (sidx, cyc) in conflicts:
        cls += " conflict"
    return f"<td class='{cls}' data-op='{group}' title='{stage_tip[(sidx, cyc)]}' style='background:{color}'></td>"


def _sched_script(lir: Lir, edges: list[tuple[str, str, str, int]], live: dict[int, set[int]]) -> str:
    """Build the interactive layer: substitute the per-module data into the readable script template (:data:`_SCHED_JS`).

    The data is the edge list, the column labels, the constant values, and the per-register live-row intervals. This is
    enough for the script to draw the dataflow overlay and synthesize hover tooltips on demand without a per-cell
    attribute. Without JS the grid still renders fully; only these behaviors are absent.
    """
    cols = [f"{kind}{index}" for kind, index in _columns_of(lir)]
    data = {
        "edges": edges,
        "columns": cols,
        "constants": {f"c{i}": repr(value) for i, value in enumerate(lir.consts)},
        "liveness": {str(reg): _live_intervals(rows) for reg, rows in live.items()},
    }
    return "<script>\n" + _SCHED_JS.replace("__DATA__", json.dumps(data)) + "\n</script>"


def _columns_of(lir: Lir) -> list[ColKey]:
    """The grid columns: one per float register, then one per constant (matches the order rendered in the table)."""
    return [("r", i) for i in range(lir.regfile.nreg)] + [("c", i) for i in range(len(lir.consts))]


def _input_chip(tip: str) -> str:
    """A neutral input-write marker for the cycle-0 latch row."""
    return f"<span class='wr' style='background:{_NEUTRAL}' title='{tip}'>&#9662;</span>"


def _schedule_key(lir: Lir) -> str:
    """A small legend above the grid: operator-kind colors plus the read/write chip shapes."""
    seen: dict[OpKind, None] = {}  # operator kinds present, in instance order, de-duplicated
    for inst in lir.instances:
        seen.setdefault(inst.kind, None)
    kinds = [
        f"<span class='wr' style='background:{_KIND_COLOR[kind]}'>{kind.value} {_esc(_KIND_LABEL[kind])}</span>"
        for kind in seen
    ]
    return (
        "<h2>Schedule</h2><div class='gridkey'>"
        + " ".join(kinds)
        + "<span><span class='sw' style='background:#374151'></span> filled cell = result committed (operator n)</span>"
        + "<span><svg class='lk' width='24' height='12'><line x1='2' y1='3' x2='21' y2='10' stroke='#374151' "
        "stroke-width='1'/><circle cx='21' cy='10' r='1.8' fill='#374151'/></svg> edge: result &rarr; its operands</span>"
        + "<span><span class='sw' style='background:#374151'></span> operator-stage block: s0..sN occupancy as the "
        "pipeline advances</span>"
        + f"<span><span class='sw' style='background:{_LIVE_BG}'></span> register holds a live value</span>"
        + f"<span><span class='wr' style='background:{_NEUTRAL}'>&#9662;</span> module input latch</span>"
        + "</div>"
    )
