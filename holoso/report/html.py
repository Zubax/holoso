"""Render a self-contained, light-themed single-page HTML report for a synthesized module."""

from __future__ import annotations

import html
import json
from datetime import datetime

from ..format import FloatFormat
from ..lir import ConstRef, InputLoad, Issue, Lir, Operand, OutputWire, RegRef
from ..operators import OpKind, Sgnop, latency_of
from ..result import ModuleInterface, SynthesisMetrics

_GITHUB_URL = "https://github.com/Zubax/holoso"

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

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Ubuntu+Mono:wght@400;700&display=swap');
* { box-sizing: border-box; }
body { font-family: "Ubuntu Mono", ui-monospace, monospace; margin: 0; background: #ffffff; color: #111827; font-size: 14px; }
header { padding: 16px 28px; background: #111827; color: #f9fafb; }
header h1 { margin: 0; font-size: 22px; }
header .sub { color: #cbd5e1; font-size: 13px; margin-top: 3px; }
header .sub a { color: #93c5fd; text-decoration: underline; }
main { padding: 14px 28px 70px; }
/* Compact summary sections share one wrapping row; the utilization and register-grid schedule are full-width below. */
.toprow { display: flex; flex-wrap: wrap; gap: 0 40px; align-items: flex-start; }
.toprow .sec { min-width: 0; }
.scrollx { overflow-x: auto; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.09em; color: #111827; font-weight: 700; margin: 26px 0 8px; border-bottom: 1px solid #000; padding-bottom: 4px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; }
.card { border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px 13px; min-width: 110px; }
.card .v { font-size: 18px; font-weight: 700; }
.card .l { font-size: 11px; color: #4b5563; }
.note { color: #4b5563; font-size: 12px; margin: 6px 0 0; }
.ifaces { display: flex; gap: 30px; flex-wrap: wrap; }
.iface h3 { font-size: 12px; color: #111827; margin: 0 0 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.iface table { border-collapse: collapse; font-size: 12px; }
.iface th, .iface td { text-align: left; padding: 2px 12px 2px 0; border-bottom: 1px solid #c2c7cf; white-space: nowrap; }
.iface th { color: #4b5563; font-weight: 700; }
.const { display: inline-block; background: #f3f4f6; color: #92400e; border: 1px solid #cbd5e1; margin: 2px; padding: 1px 7px; border-radius: 3px; font-size: 12px; }
/* Schedule: a register grid beside a flowing operation list. Rows = clock cycles (height proportional to step
   latency); columns = registers then constants (black divider between). A result column is filled with its operator
   color over its compute cycles; an SVG overlay links each result to its operand cells. Bold rules separate steps. */
#schedwrap { position: relative; display: inline-block; }
.edges { position: absolute; top: 0; left: 0; pointer-events: none; overflow: visible; }
.grid { border-collapse: collapse; table-layout: fixed; }
.grid th.gh { width: 14px; font-size: 9px; color: #4b5563; font-weight: 700; padding: 0 1px; border-bottom: 1px solid #000; vertical-align: bottom; }
.grid th.clkh { width: 42px; border-right: 1px solid #000; }
.grid th.steph { width: 52px; border-right: 1px solid #000; }
.grid th.gh span { writing-mode: vertical-rl; }
.grid th.gh.k span { color: #92400e; }
.grid td { height: 13px; font-size: 11px; line-height: 1; white-space: nowrap; border-top: 1px solid #d7dbe0; }
.grid tr.sstart td { border-top: 1px solid #000; }
.grid tr.band td.gc { background: #fafbfc; }
.grid td.gc { width: 13px; padding: 0; text-align: center; border-right: 1px solid #eef0f2; overflow: hidden; }
/* The black structural dividers: same width/style as the faint inter-cell line, so they must out-specify .grid td.gc
   (equal specificity, declared later) and sit on the left cell's right edge -- the side border-collapse keeps. */
.grid td.rbk, .grid th.rbk { border-right: 1px solid #000; }
.clk { color: #9aa1ab; text-align: right; font-size: 10px; padding: 0 5px; border-right: 1px solid #000; }
.stepn { font-weight: 700; color: #111827; padding: 0 6px; border-right: 1px solid #000; white-space: nowrap; }
.rd { display: inline-block; font-size: 9px; line-height: 11px; border: 1px solid; border-radius: 2px; padding: 0 1px; }
.wr { display: inline-block; font-size: 9px; line-height: 11px; border-radius: 2px; padding: 0 1px; color: #fff; font-weight: 700; }
.wl { color: #fff; font-size: 9px; font-weight: 700; line-height: 11px; }
/* Operations column: one cell per step (rowspan over its cycles), so the list is step-aligned and grows to hold all
   operator chips no matter how many issue in a step -- the contingency for many parallel operators. */
.grid th.oph { width: 250px; text-align: left; vertical-align: bottom; padding: 0 6px 2px 8px; border-bottom: 1px solid #000; font-size: 10px; color: #4b5563; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
.grid td.opcell { vertical-align: top; padding: 1px 4px 2px 8px; white-space: normal; }
.opf { display: inline-block; background: #374151; color: #fff; font-size: 10px; font-weight: 700; border-radius: 3px; padding: 0 4px; margin: 1px 3px 1px 0; white-space: nowrap; }
/* Hover-focus: only the hovered operation's own elements get the .hl class (its edges, result cells and ops chip),
   so no per-hover sweep of the rest of the grid. They turn black with a soft glow to stand out among the colors. */
.edges line.hl { stroke: #000; stroke-width: 2px; stroke-opacity: 1; }
.edges circle.hl { fill: #000; }
.grid td.gc.hl { background: #111 !important; }
.opf.hl { background: #111 !important; box-shadow: 0 0 0 2px rgba(17, 17, 17, 0.35); }
.gridkey { display: flex; flex-wrap: wrap; gap: 18px; font-size: 12px; color: #374151; margin: 0 0 8px; align-items: center; }
.gridkey .rd, .gridkey .wr { font-size: 11px; }
.gridkey .sw { display: inline-block; width: 14px; height: 13px; border: 1px solid #cbd5e1; vertical-align: middle; margin-right: 5px; }
.gridkey .lk { vertical-align: middle; margin-right: 5px; }
.util { border-collapse: collapse; margin-top: 4px; }
.util td, .util th { border: 1px solid #c2c7cf; padding: 0; text-align: center; }
.util .name { padding: 1px 8px; text-align: left; color: #111827; border: none; font-size: 11px; }
.util .st { writing-mode: vertical-rl; font-size: 9px; color: #6b7280; font-weight: 400; padding: 3px 0; }
.cell { width: 11px; height: 15px; background: #fff; position: relative; }
.fill { position: absolute; bottom: 0; left: 0; right: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 22px; font-size: 12px; color: #374151; margin-top: 8px; align-items: center; }
.legend .box { display: inline-block; width: 12px; height: 15px; border: 1px solid #94a3b8; vertical-align: middle; margin-right: 5px; position: relative; }
"""

# Interactive layer for the schedule grid. ``__DATA__`` is replaced by the per-module payload in ``_sched_script``.
# Kept deliberately readable (the report is a tool, not a minified asset); without JS the grid still renders fully.
_SCHED_JS = """
(function () {
    "use strict";
    var data = __DATA__;
    var edges = data.edges;          // [writeCellId, operandCellId, color, operationGroup]
    var columns = data.columns;      // label per grid column, indexed by (cell index - 2)
    var constants = data.constants;  // { "c0": "1.0", ... }
    var liveness = data.liveness;    // { "<registerIndex>": [[start, end], ...] } live-row intervals
    var lastRow = data.lastRow;      // grid row id of the "out" bookend

    var wrap = document.getElementById("schedwrap");
    if (!wrap) {
        return;
    }
    var svg = wrap.querySelector("svg.edges");
    var SVG_NS = "http://www.w3.org/2000/svg";

    // Elements grouped by operation, so a hover can light up just one operation's nodes. The result cells and the ops
    // chip are static (tagged with data-op); the edges are rebuilt on every redraw and tracked separately.
    var nodesByGroup = {};
    wrap.querySelectorAll("[data-op]").forEach(function (element) {
        var group = element.dataset.op;
        (nodesByGroup[group] = nodesByGroup[group] || []).push(element);
    });
    var edgesByGroup = {};

    // Draw each dataflow edge (result cell -> operand cell) by measuring the rendered cell centres, so the overlay
    // stays aligned no matter the exact cell metrics or a late web-font swap.
    function drawEdges() {
        edgesByGroup = {};
        while (svg.firstChild) {
            svg.removeChild(svg.firstChild);
        }
        var origin = wrap.getBoundingClientRect();
        svg.setAttribute("width", wrap.scrollWidth);
        svg.setAttribute("height", wrap.scrollHeight);
        edges.forEach(function (edge) {
            var fromCell = document.getElementById(edge[0]);
            var toCell = document.getElementById(edge[1]);
            if (!fromCell || !toCell || fromCell === toCell) {
                return;
            }
            var from = fromCell.getBoundingClientRect();
            var to = toCell.getBoundingClientRect();
            var x1 = from.left - origin.left + from.width / 2;
            var y1 = from.top - origin.top + from.height / 2;
            var x2 = to.left - origin.left + to.width / 2;
            var y2 = to.top - origin.top + to.height / 2;
            var color = edge[2];
            var group = edge[3];

            var line = document.createElementNS(SVG_NS, "line");
            line.setAttribute("x1", x1);
            line.setAttribute("y1", y1);
            line.setAttribute("x2", x2);
            line.setAttribute("y2", y2);
            line.setAttribute("stroke", color);
            line.setAttribute("stroke-width", "1");
            line.setAttribute("stroke-opacity", "0.85");
            svg.appendChild(line);

            var dot = document.createElementNS(SVG_NS, "circle");
            dot.setAttribute("cx", x2);
            dot.setAttribute("cy", y2);
            dot.setAttribute("r", "1.7");
            dot.setAttribute("fill", color);
            svg.appendChild(dot);

            (edgesByGroup[group] = edgesByGroup[group] || []).push(line, dot);
        });
    }

    drawEdges();
    window.addEventListener("resize", drawEdges);
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(drawEdges);
    }

    // Hovering an operation (its result column or its ops chip) makes only that one operation stand out: we toggle a
    // class on its own handful of elements rather than restyling every other operation, so there is no per-hover
    // sweep of the grid. The .hl class blackens its edges, result cells and chip.
    var focused = null;  // currently focused group (a "data-op" string), or null

    function setHighlighted(group, on) {
        [nodesByGroup[group], edgesByGroup[group]].forEach(function (list) {
            if (list) {
                list.forEach(function (element) {
                    element.classList.toggle("hl", on);
                });
            }
        });
    }

    function focus(group) {
        if (group === focused) {
            return;
        }
        if (focused !== null) {
            setHighlighted(focused, false);
        }
        focused = group;
        if (group !== null) {
            setHighlighted(group, true);
        }
    }

    // Whether register `label` (e.g. "r41") holds a live value on `cycle`, from its residence intervals.
    function isAlive(label, cycle) {
        var intervals = liveness[label.slice(1)];
        if (!intervals) {
            return false;
        }
        for (var i = 0; i < intervals.length; i++) {
            if (cycle >= intervals[i][0] && cycle <= intervals[i][1]) {
                return true;
            }
        }
        return false;
    }

    wrap.addEventListener("mouseover", function (event) {
        var owner = event.target.closest("[data-op]");  // a result cell or an ops chip
        focus(owner ? owner.dataset.op : null);

        var cell = event.target.closest("td");
        if (!cell || !cell.classList.contains("gc")) {
            return;
        }
        var label = columns[cell.cellIndex - 2];
        if (label === undefined) {
            return;
        }
        var clk = cell.parentNode.cells[0].textContent.trim();
        if (label.charAt(0) === "c") {
            cell.title = label + " = " + constants[label];
        } else {
            var cycle = clk === "in" ? -1 : (clk === "out" ? lastRow : parseInt(clk, 10));
            cell.title = label + "@" + clk + " " + (isAlive(label, cycle) ? "alive" : "dead");
        }
    });
    wrap.addEventListener("mouseout", function (event) {
        if (!wrap.contains(event.relatedTarget)) {
            focus(null);
        }
    });
})();
"""


def _esc(text: str) -> str:
    return html.escape(text)


def _operand(operand: Operand) -> str:
    name = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return operand.sgnop.decorate(name)


def _issue_text(issue: Issue) -> str:
    if issue.inst.kind is OpKind.FMUL_ILOG2:
        body = f"{_operand(issue.a)}*2^{issue.k}"
    else:
        assert issue.b is not None
        body = f"{_operand(issue.a)} {_KIND_LABEL[issue.inst.kind]} {_operand(issue.b)}"
    return issue.y_sgnop.decorate(f"r{issue.dst.index} = {body}")


def _card(value: object, label: str) -> str:
    return f'<div class="card"><div class="v">{_esc(str(value))}</div><div class="l">{_esc(label)}</div></div>'


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
    # do not waste page height; the full-width utilization and the wide register-grid schedule follow below.
    out.append("<div class='toprow'>")
    out.append(f"<div class='sec'>{_metrics(interface, metrics, fmt)}</div>")
    constants = _constants(lir)
    if constants:
        out.append(f"<div class='sec'>{constants}</div>")
    out.append(f"<div class='sec'>{_interface(interface)}</div>")
    out.append("</div>")
    out.append(_utilization(lir, fmt))
    out.append(_schedule(lir, fmt))
    out.append("</main></body></html>")
    return "".join(out)


def _metrics(interface: ModuleInterface, metrics: SynthesisMetrics, fmt: FloatFormat) -> str:
    instances = " ".join(f"{count}×{kind}" for kind, count in metrics.operator_instances.items())
    cards = [
        _card(f"e{fmt.wexp}+m{fmt.wman} = {fmt.width}-bit", f"ZKF floating point format"),
        _card(instances or "-", "operator instances"),
        _card(metrics.n_float_regs, "float registers"),
        _card(f"{metrics.read_ports} / {metrics.write_ports}", "regfile R/W ports"),
        _card(metrics.step_count, "FSM steps"),
        _card(metrics.op_count, "operations"),
        _card(metrics.ii_estimate, "II (cycles)"),
        _card(metrics.max_chain_len, "longest op chain"),
    ]
    note = (
        f"Initiation interval = in_valid&rarr;out_valid latency, data-independent and verified cycle-exact in "
        f"simulation: {_esc(interface.ii.formula)}."
    )
    return f"<h2>Metrics</h2><div class='cards'>{''.join(cards)}</div><p class='note'>{note}</p>"


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
_LIVE_BG = "#edf2fb"  # faint tint over the cycles a register holds a live value (its write down to its last read)
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


def _gc_class(ordinal: int, dividers: set[int]) -> str:
    """Grid-cell class for column ``ordinal``. A column in ``dividers`` carries a black divider as its *right* border:
    under ``border-collapse`` the left cell of a pair wins an equal-width conflict, so the divider must live on the left
    column's right edge (e.g. the last register before the constants) to actually show -- hence ``rbk`` here, not a left
    border on the constant column."""
    return "gc rbk" if ordinal in dividers else "gc"


def _bookend_row(
    cls: str,
    label: str,
    cells: dict[ColKey, str],
    columns: list[ColKey],
    live: dict[int, set[int]],
    row_id: int,
    dividers: set[int],
) -> str:
    """A grid row outside the FSM steps: the ``in`` input-load row or the ``out`` output-read row (empty ops cell)."""
    out = [f"<tr class='{cls}'><td class='clk'>{label}</td><td class='stepn'></td>"]
    for ordinal, col in enumerate(columns):
        bg = f" style='background:{_LIVE_BG}'" if _is_live(col, row_id, live) else ""
        out.append(f"<td class='{_gc_class(ordinal, dividers)}'{bg}>{cells.get(col, '')}</td>")
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
    col: ColKey, row_id: int, offset: int, live: dict[int, set[int]], bars: dict[ColKey, tuple[str, int]]
) -> str:
    """Inline background for a grid cell. A result column in flight is filled solid with its operator color for the
    compute cycles (this takes precedence); otherwise a live register gets the faint residence tint."""
    bar = bars.get(col)
    if bar is not None and offset < bar[1]:  # the result column is "in flight" for the operator's compute cycles
        return f" style='background:{bar[0]}'"
    if _is_live(col, row_id, live):
        return f" style='background:{_LIVE_BG}'"
    return ""


def _schedule(lir: Lir, fmt: FloatFormat) -> str:
    if not lir.steps:
        return ""
    nreg, nconst = lir.regfile.nreg, len(lir.consts)
    columns: list[ColKey] = [("r", i) for i in range(nreg)] + [("c", i) for i in range(nconst)]
    col_ord = {col: ordinal for ordinal, col in enumerate(columns)}
    dividers = {len(columns) - 1}  # black right-border columns: last data column | operations ...
    if nconst:
        dividers.add(nreg - 1)  # ... and registers | constants
    live = _liveness(lir, fmt)
    total = sum(step.latency for step in lir.steps)
    edges: list[tuple[str, str, str, int]] = []  # (write id, operand id, color, operation group) for the overlay
    out = [_schedule_key(lir), "<div id='schedwrap'><table class='grid'>"]
    out.append("<tr><th class='gh clkh'><span>clk</span></th><th class='gh steph'><span>step</span></th>")
    for ordinal, (kind, index) in enumerate(columns):
        cls = "gh k" if kind == "c" else "gh"
        if ordinal in dividers:
            cls += " rbk"
        out.append(f"<th class='{cls}'><span>{kind}{index}</span></th>")
    out.append("<th class='oph'>operations</th></tr>")

    # Bookend the FSM with the I/O boundary: inputs are defined (written) into registers at accept; outputs are read
    # from registers at completion. Neutral chips distinguish the module boundary from operator activity.
    in_cells = {("r", load.dst.index): _bookend_chip(False, f"in_{load.name}") for load in lir.inputs}
    out_cells: dict[ColKey, str] = {}
    for wire in lir.outputs:
        wcol: ColKey = ("r", wire.source.index) if isinstance(wire.source, RegRef) else ("c", wire.source.index)
        out_cells[wcol] = out_cells.get(wcol, "") + _bookend_chip(True, f"out_{wire.name}", wire.sgnop)
    out.append(_bookend_row("sstart", "in", in_cells, columns, live, -1, dividers))

    group = 0  # global per-operation id, used to cross-link a result block with its edges for the hover-focus behavior
    cycle = 0
    for step in lir.steps:
        # The result column is filled solid with the operator color over its compute cycles, with the instance label on
        # the completion cycle (offset = its latency - 1) -- so the filled height shows the operator latency, and the
        # blank gap from the completion to the bold step rule is the barrier idle. Operands are not drawn as chips: the
        # overlay draws an operator-colored edge from each write cell up to its operand cells on the step's first cycle.
        # We tag edge endpoints with ids and every in-flight result cell with its operation group (data-op).
        writes_at: dict[tuple[int, ColKey], str] = {}
        bars: dict[ColKey, tuple[str, int]] = {}
        endpoints: set[tuple[int, int]] = set()
        cell_group: dict[tuple[int, int], int] = {}  # (column ordinal, cycle) -> operation group, for result cells
        chips: list[str] = []  # filled operator chips for this step's operations column (rowspan over the step)
        for issue in step.issues:
            tip = _esc(_issue_text(issue))
            color = _KIND_COLOR[issue.inst.kind]
            busy = min(latency_of(issue.inst.kind, fmt), step.latency)
            completion = cycle + busy - 1
            dcol: ColKey = ("r", issue.dst.index)
            dord = col_ord[dcol]
            writes_at[(busy - 1, dcol)] = writes_at.get((busy - 1, dcol), "") + _write_label(issue.inst.index, tip)
            bars[dcol] = (color, busy)
            endpoints.add((dord, completion))
            for off in range(busy):
                cell_group[(dord, cycle + off)] = group
            for operand in (issue.a, issue.b):
                if operand is not None:
                    oord = col_ord[_operand_col(operand)]
                    endpoints.add((oord, cycle))
                    edges.append((f"g{dord}_{completion}", f"g{oord}_{cycle}", color, group))
            chips.append(f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>")
            group += 1
        band = " band" if step.index % 2 else ""
        for offset in range(step.latency):  # one row per clock cycle of the step
            cyc = cycle + offset
            cls = "sstart" if offset == 0 else ""
            out.append(f"<tr class='{(cls + band).strip()}'>")
            out.append(f"<td class='clk'>{cyc}</td>")
            out.append(f"<td class='stepn'>{'S' + str(step.index) if offset == 0 else ''}</td>")
            for ordinal, col in enumerate(columns):
                content = writes_at.get((offset, col), "")
                style = _cell_style(col, cyc, offset, live, bars)
                attrs = f" id='g{ordinal}_{cyc}'" if (ordinal, cyc) in endpoints else ""
                if (ordinal, cyc) in cell_group:
                    attrs += f" data-op='{cell_group[(ordinal, cyc)]}'"
                out.append(f"<td class='{_gc_class(ordinal, dividers)}'{attrs}{style}>{content}</td>")
            if offset == 0:  # one ops cell per step, spanning its cycles, so the list stays step-aligned
                out.append(f"<td class='opcell' rowspan='{step.latency}'>{''.join(chips)}</td>")
            out.append("</tr>")
        cycle += step.latency
    out.append(_bookend_row("sstart", "out", out_cells, columns, live, total, dividers))
    out.append("</table><svg class='edges'></svg></div>")
    out.append(_sched_script(lir, edges, live, total))
    return "".join(out)


def _sched_script(lir: Lir, edges: list[tuple[str, str, str, int]], live: dict[int, set[int]], total: int) -> str:
    """Build the interactive layer: substitute the per-module data into the readable script template (:data:`_SCHED_JS`).

    The data is the edge list, the column labels, the constant values, the per-register live-row intervals, and the
    ``out`` row id -- enough for the script to draw the dataflow overlay and synthesize hover tooltips on demand without
    a per-cell attribute. Without JS the grid still renders fully; only these behaviors are absent.
    """
    cols = [f"{kind}{index}" for kind, index in columns_of(lir)]
    data = {
        "edges": edges,
        "columns": cols,
        "constants": {f"c{i}": repr(value) for i, value in enumerate(lir.consts)},
        "liveness": {str(reg): _live_intervals(rows) for reg, rows in live.items()},
        "lastRow": total,
    }
    return "<script>\n" + _SCHED_JS.replace("__DATA__", json.dumps(data)) + "\n</script>"


def columns_of(lir: Lir) -> list[ColKey]:
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
        f"<span class='wr' style='background:{_KIND_COLOR[kind]}'>{kind.value} {_KIND_LABEL[kind]}</span>"
        for kind in seen
    ]
    return (
        "<h2>Schedule</h2><div class='gridkey'>"
        + " ".join(kinds)
        + "<span><span class='sw' style='background:#374151'></span> filled column = operator n result (height = latency)</span>"
        + "<span><svg class='lk' width='24' height='12'><line x1='2' y1='3' x2='21' y2='10' stroke='#374151' "
        "stroke-width='1'/><circle cx='21' cy='10' r='1.8' fill='#374151'/></svg> edge: result &rarr; its operands</span>"
        + f"<span><span class='sw' style='background:{_LIVE_BG}'></span> register holds a live value</span>"
        + f"<span><span class='wr' style='background:{_NEUTRAL}'>&#9662;</span>"
        + f"<span class='rd' style='border-color:{_NEUTRAL};color:{_NEUTRAL}'>&#9652;</span> module in / out</span>"
        + "</div>"
    )


def _utilization(lir: Lir, fmt: FloatFormat) -> str:
    if not lir.steps:
        return ""
    util: dict[tuple[OpKind, int], dict[int, float]] = {}
    for step in lir.steps:
        for issue in step.issues:
            fraction = min(1.0, latency_of(issue.inst.kind, fmt) / max(step.latency, 1))
            util.setdefault((issue.inst.kind, issue.inst.index), {})[step.index] = fraction
    out = [
        "<h2>Operator utilization</h2><div class='scrollx'><table class='util'><tr><th class='name'>operator \\ state</th>"
    ]
    for step in lir.steps:
        out.append(f"<th class='st'>{step.index + 1}</th>")  # the generated Verilog state number
    out.append("</tr>")
    for inst in lir.instances:
        color = _KIND_COLOR[inst.kind]
        out.append(f"<tr><td class='name'>{inst.kind.value}_{inst.index}</td>")
        cells = util.get((inst.kind, inst.index), {})
        for step in lir.steps:
            frac = cells.get(step.index)
            if frac is None:
                out.append("<td><div class='cell'></div></td>")
            else:
                height = max(2, round(frac * 15))
                out.append(
                    f"<td><div class='cell'><div class='fill' style='height:{height}px;background:{color}'></div></div></td>"
                )
        out.append("</tr>")
    out.append("</table></div>")
    out.append(_legend())
    return "".join(out)


def _legend() -> str:
    full = "<span class='box' style='background:#374151'></span>"
    partial = "<span class='box'><span class='fill' style='height:6px;background:#374151'></span></span>"
    empty = "<span class='box'></span>"
    return (
        "<div class='legend'>"
        f"<span>{empty}idle &mdash; operator not issued this state</span>"
        f"<span>{full}full &mdash; active and on the step's critical path (busy the whole step)</span>"
        f"<span>{partial}partial &mdash; active but finished early and idled at the barrier "
        "(fill = its latency &divide; step latency)</span>"
        "</div>"
    )
