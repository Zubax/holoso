"""Render a self-contained, light-themed single-page HTML report for a synthesized module.

The stylesheet and the interactive layer live alongside this module as ``html.css`` and ``html.js`` (declared as
package data in ``pyproject.toml``); they are inlined into the self-contained report so it has no external dependency
beyond the web font.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

from ..format import FloatFormat
from ..lir import Issue, Lir, Operand, OperatorInstance, RegRef
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


def _esc(text: str) -> str:
    return html.escape(text)


def _operand(operand: Operand) -> str:
    name = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return operand.sgnop.decorate(name)


def _issue_text(issue: Issue) -> str:
    # Spaceless (``r1=r1+r3``): many operations can complete on one clock now, so the operations column packs tightly.
    if issue.inst.kind is OpKind.FMUL_ILOG2:
        body = f"{_operand(issue.a)}*2^{issue.k}"
    else:
        assert issue.b is not None
        body = f"{_operand(issue.a)}{_KIND_LABEL[issue.inst.kind]}{_operand(issue.b)}"
    return issue.y_sgnop.decorate(f"r{issue.dst.index}={body}")


def build_report_html(lir: Lir, interface: ModuleInterface, metrics: SynthesisMetrics) -> str:
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
    out.append("</div>")
    out.append(_schedule(lir, fmt))
    out.append("</main></body></html>")
    return "".join(out)


def _metrics(interface: ModuleInterface, metrics: SynthesisMetrics, fmt: FloatFormat) -> str:
    instances = " ".join(f"{count}×{kind}" for kind, count in metrics.operator_instances.items())
    rows: list[tuple[str, object]] = [
        ("ZKF format", f"e{fmt.wexp}+m{fmt.wman} = {fmt.width}-bit"),
        ("operator instances", instances or "-"),
        ("float registers", metrics.n_float_regs),
        ("regfile R/W ports", f"{metrics.read_ports} / {metrics.write_ports}"),
        ("FSM steps", metrics.step_count),
        ("operations", metrics.op_count),
        ("II (cycles)", metrics.ii_estimate),
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


def _sgn_prefix(sgnop: Sgnop) -> str:
    """Compact operand/result sign-op marker for a chip: ``-`` negate, ``|`` abs (``-|`` for both); ``""`` for none."""
    prefix = "-" if Sgnop.NEG in sgnop else ""
    return prefix + ("|" if Sgnop.ABS in sgnop else "")


def _write_label(index: int, tip: str) -> str:
    """The result marker on the operator's completion cell: its instance index, white text on the filled result cell.

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
    through its operator, stage ``k`` is occupied on cycle ``launch + k``; the result lands at the last stage. This
    makes the pipeline's advance directly visible -- and, once multiple-issue lands, exposes any structural hazard.
    """
    cols: list[tuple[OperatorInstance, int]] = []
    for inst in lir.instances:
        cols.extend((inst, k) for k in range(latency_of(inst.kind, fmt)))
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
    """A grid row outside the FSM steps: the ``in`` input-load row or the ``out`` output-read row. No operator is in
    flight at the I/O boundary, so the operator-stage cells and the ops cell are empty."""
    out = [f"<tr><td class='clk'>{label}</td><td class='stepn'></td>"]
    for ordinal, col in enumerate(columns):
        extra = " live" if _is_live(col, row_id, live) else ""
        out.append(f"<td class='{_gc_class(ordinal, dv)}{extra}'>{cells.get(col, '')}</td>")
    for sidx in range(n_stage):
        out.append(f"<td class='{_oc_class(sidx, dv)}'></td>")
    out.append("<td class='opcell'></td></tr>")
    return "".join(out)


def _liveness(lir: Lir, fmt: FloatFormat) -> dict[int, set[int]]:
    """Map each register to the grid rows on which it holds a live value.

    Rows are cycle numbers, with ``-1`` for the ``in`` bookend and ``total`` for the ``out`` bookend. A value lives from
    its write (an operator's completion cycle, or the input load) down to its last read (the consuming step's launch
    cycle, or the output read). Liveness is computed per value, so a register reused for several values yields several
    disjoint residence intervals with dead gaps between them.
    """
    bases: list[int] = []
    cycle = 0
    for step in lir.steps:
        bases.append(cycle)
        cycle += step.latency
    total = cycle
    defs: dict[int, list[int]] = {}
    uses: dict[int, list[int]] = {}
    for load in lir.inputs:
        defs.setdefault(load.dst.index, []).append(-1)
    for base, step in zip(bases, lir.steps):
        for issue in step.issues:
            busy = min(latency_of(issue.inst.kind, fmt), step.latency)
            defs.setdefault(issue.dst.index, []).append(base + busy - 1)
            for operand in (issue.a, issue.b):
                if operand is not None and isinstance(operand.source, RegRef):
                    uses.setdefault(operand.source.index, []).append(base)
    for wire in lir.outputs:
        if isinstance(wire.source, RegRef):
            uses.setdefault(wire.source.index, []).append(total)
    live: dict[int, set[int]] = {}
    for reg in defs.keys() | uses.keys():
        writes = sorted(defs.get(reg, []))
        reads = sorted(uses.get(reg, []))
        rows: set[int] = set()
        for i, start in enumerate(writes):
            nxt = writes[i + 1] if i + 1 < len(writes) else total + 1  # this value persists until the next overwrite
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
    col: ColKey, row_id: int, offset: int, live: dict[int, set[int]], fills: dict[tuple[int, ColKey], str]
) -> tuple[str, str]:
    """Background for a register/constant cell, as ``(extra_class, inline_style)``. The single cycle on which a result
    lands is filled solid with its operator color via an inline style (this takes precedence, and being inline it
    survives the row-hover tint); otherwise a live register gets the faint residence tint via the ``live`` class, so a
    row-hover can override it. The operators' cycle-by-cycle occupancy lives in the separate operator-stage block.
    """
    color = fills.get((offset, col))
    if color is not None:
        return "", f" style='background:{color}'"
    if _is_live(col, row_id, live):
        return " live", ""
    return "", ""


def _schedule(lir: Lir, fmt: FloatFormat) -> str:
    if not lir.steps:
        return ""
    nreg, nconst = lir.regfile.nreg, len(lir.consts)
    columns = _columns_of(lir)
    col_ord = {col: ordinal for ordinal, col in enumerate(columns)}

    # The operator-stage block: one square column per pipeline stage of each operator, in instance order.
    stage_cols = _stage_columns(lir, fmt)
    n_stage = len(stage_cols)
    stage_base: dict[OperatorInstance, int] = {}
    for sidx, (inst, _k) in enumerate(stage_cols):
        stage_base.setdefault(inst, sidx)
    group_ends = {
        stage_base[inst] + latency_of(inst.kind, fmt) - 1 for inst in lir.instances
    }  # last stage per operator

    # Column seams: 2px at the two block boundaries (constants | pipeline and pipeline | OPERATIONS); 1px at the lighter
    # registers | constants seam and between operator groups.
    dv = _Dividers(
        data_thin={nreg - 1} if nconst else set(),
        data_thick={len(columns) - 1},
        stage_thin=group_ends - {n_stage - 1},
        stage_thick={n_stage - 1} if n_stage else set(),
    )

    live = _liveness(lir, fmt)
    total = sum(step.latency for step in lir.steps)
    edges: list[tuple[str, str, str, int]] = []  # (write id, operand id, color, operation group) for the overlay
    # Operator pipeline occupancy, accumulated across all steps (a trail can cross step boundaries once issues overlap):
    # instance ``inst`` is in stage ``k`` on cycle ``launch + k``. Keyed to the operation group so a hover lights the
    # whole pipeline trail together with the result cell, its chip and its edges. ``conflicts`` flags any cell two
    # operations claim at once -- impossible under today's barrier schedule, the alarm for a future multiple-issue bug.
    stage_fill: dict[tuple[int, int], tuple[str, int]] = {}  # (stage column, cycle) -> (operator color, group)
    stage_tip: dict[tuple[int, int], str] = {}
    conflicts: set[tuple[int, int]] = set()

    out = [_schedule_key(lir), "<div id='schedwrap'><table class='grid'>"]
    # Header row 0: group bands over the register, constant and operator-pipeline blocks (each band clips to its span,
    # so a single constant just reads "C"). clk/step/operations span all three header rows.
    out.append("<tr><th class='gh clkh' rowspan='3'><span>clk</span></th>")
    out.append("<th class='gh steph' rowspan='3'><span>step</span></th>")
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
        lat = latency_of(inst.kind, fmt)
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

    # Bookend the FSM with the I/O boundary: inputs are defined (written) into registers at accept; outputs are read
    # from registers at completion. Neutral chips distinguish the module boundary from operator activity.
    in_cells = {("r", load.dst.index): _bookend_chip(False, f"in_{load.name}") for load in lir.inputs}
    out_cells: dict[ColKey, str] = {}
    for wire in lir.outputs:
        wcol: ColKey = ("r", wire.source.index) if isinstance(wire.source, RegRef) else ("c", wire.source.index)
        out_cells[wcol] = out_cells.get(wcol, "") + _bookend_chip(True, f"out_{wire.name}", wire.sgnop)
    out.append(_bookend_row("in", in_cells, columns, live, -1, n_stage, dv))

    group = 0  # global per-operation id, used to cross-link a result cell with its edges for the hover-focus behavior
    cycle = 0
    for step in lir.steps:
        # Only the destination cell on a result's completion cycle is highlighted with the operator color; the operand
        # cells are linked by the overlay's edges, and the operation's chip is placed on the row where its result
        # completes, so the operations column reads in lock-step with the highlighted cells (many chips may share a row).
        # Each issue also stamps its operator's pipeline trail into the stage block above (handled in stage_fill).
        writes_at: dict[tuple[int, ColKey], str] = {}
        fills: dict[tuple[int, ColKey], str] = {}  # (offset within step, column) -> operator color, completion cells
        endpoints: set[tuple[int, int]] = set()
        cell_group: dict[tuple[int, int], int] = {}  # (column ordinal, cycle) -> operation group, for result cells
        chips_at: dict[int, list[str]] = {}  # offset within step -> chips for the operations completing on that row
        for issue in step.issues:
            tip = _esc(_issue_text(issue))
            color = _KIND_COLOR[issue.inst.kind]
            lat = latency_of(issue.inst.kind, fmt)
            busy = min(lat, step.latency)
            completion = cycle + busy - 1
            dcol: ColKey = ("r", issue.dst.index)
            dord = col_ord[dcol]
            writes_at[(busy - 1, dcol)] = writes_at.get((busy - 1, dcol), "") + _write_label(issue.inst.index, tip)
            fills[(busy - 1, dcol)] = color
            endpoints.add((dord, completion))
            cell_group[(dord, completion)] = group
            for operand in (issue.a, issue.b):
                if operand is not None:
                    oord = col_ord[_operand_col(operand)]
                    endpoints.add((oord, cycle))
                    edges.append((f"g{dord}_{completion}", f"g{oord}_{cycle}", color, group))
            chips_at.setdefault(busy - 1, []).append(
                f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>"
            )
            base = stage_base[issue.inst]
            for k in range(lat):  # stamp the pipeline trail: stage k of this operator is busy on cycle launch + k
                key = (base + k, cycle + k)
                if key in stage_fill and stage_fill[key][1] != group:
                    conflicts.add(key)
                stage_fill[key] = (color, group)
                stage_tip[key] = f"{issue.inst.kind.value}_{issue.inst.index} s{k}: {tip}"
            group += 1
        for offset in range(step.latency):  # one row per clock cycle of the step
            cyc = cycle + offset
            out.append("<tr>")
            out.append(f"<td class='clk'>{cyc}</td>")
            out.append(f"<td class='stepn'>{'S' + str(step.index) if offset == 0 else ''}</td>")
            for ordinal, col in enumerate(columns):
                content = writes_at.get((offset, col), "")
                extra, style = _cell_style(col, cyc, offset, live, fills)
                attrs = f" id='g{ordinal}_{cyc}'" if (ordinal, cyc) in endpoints else ""
                if (ordinal, cyc) in cell_group:
                    attrs += f" data-op='{cell_group[(ordinal, cyc)]}'"
                out.append(f"<td class='{_gc_class(ordinal, dv)}{extra}'{attrs}{style}>{content}</td>")
            for sidx in range(n_stage):
                out.append(_stage_cell(sidx, cyc, dv, stage_fill, stage_tip, conflicts))
            out.append(f"<td class='opcell'>{''.join(chips_at.get(offset, []))}</td>")
            out.append("</tr>")
        cycle += step.latency
    out.append(_bookend_row("out", out_cells, columns, live, total, n_stage, dv))
    out.append("</table><svg class='edges'></svg></div>")
    out.append(_sched_script(lir, edges, live, total))
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


def _sched_script(lir: Lir, edges: list[tuple[str, str, str, int]], live: dict[int, set[int]], total: int) -> str:
    """Build the interactive layer: substitute the per-module data into the readable script template (:data:`_SCHED_JS`).

    The data is the edge list, the column labels, the constant values, the per-register live-row intervals, and the
    ``out`` row id -- enough for the script to draw the dataflow overlay and synthesize hover tooltips on demand without
    a per-cell attribute. Without JS the grid still renders fully; only these behaviors are absent.
    """
    cols = [f"{kind}{index}" for kind, index in _columns_of(lir)]
    data = {
        "edges": edges,
        "columns": cols,
        "constants": {f"c{i}": repr(value) for i, value in enumerate(lir.consts)},
        "liveness": {str(reg): _live_intervals(rows) for reg, rows in live.items()},
        "lastRow": total,
    }
    return "<script>\n" + _SCHED_JS.replace("__DATA__", json.dumps(data)) + "\n</script>"


def _columns_of(lir: Lir) -> list[ColKey]:
    """The grid columns: one per float register, then one per constant (matches the order rendered in the table)."""
    return [("r", i) for i in range(lir.regfile.nreg)] + [("c", i) for i in range(len(lir.consts))]


def _bookend_chip(read: bool, tip: str, sgnop: Sgnop = Sgnop.NONE) -> str:
    """A neutral input (write) or output (read) marker for the I/O boundary rows."""
    if read:
        return f"<span class='rd' style='border-color:{_NEUTRAL};color:{_NEUTRAL}' title='{tip}'>{_sgn_prefix(sgnop)}&#9652;</span>"
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
        + "<span><span class='sw' style='background:#374151'></span> filled cell = result available (operator n)</span>"
        + "<span><svg class='lk' width='24' height='12'><line x1='2' y1='3' x2='21' y2='10' stroke='#374151' "
        "stroke-width='1'/><circle cx='21' cy='10' r='1.8' fill='#374151'/></svg> edge: result &rarr; its operands</span>"
        + "<span><span class='sw' style='background:#374151'></span> operator-stage block: s0..sN occupancy as the "
        "pipeline advances</span>"
        + f"<span><span class='sw' style='background:{_LIVE_BG}'></span> register holds a live value</span>"
        + f"<span><span class='wr' style='background:{_NEUTRAL}'>&#9662;</span>"
        + f"<span class='rd' style='border-color:{_NEUTRAL};color:{_NEUTRAL}'>&#9652;</span> module in / out</span>"
        + "</div>"
    )
