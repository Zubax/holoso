"""Render the cycle-accurate schedule grid for the HTML report."""

import colorsys
import html
import json
from dataclasses import dataclass
from importlib import resources

from ..._lir import FETCH_LAG, Lir, FloatConstRef, FloatOperand, FloatOperatorInstance, FloatRegRef, FloatScheduledOp
from ..._operators import HardwareOperator

# Interactive layer; ``__DATA__`` is replaced by the per-module payload in ``_sched_script``.
_SCHED_JS = resources.files(__package__).joinpath("html.js").read_text(encoding="utf-8")
_EDGE_KEY_MARKER = (
    "<svg class='lk' width='24' height='12'>"
    "<line x1='2' y1='3' x2='21' y2='10' stroke='currentColor' stroke-width='1'/>"
    "<circle cx='21' cy='10' r='1.8' fill='currentColor'/>"
    "</svg>"
)


def render_schedule(lir: Lir) -> str:
    nreg, nconst = lir.float_regfile.nreg, len(lir.float_consts)
    columns = _columns_of(lir)
    col_ord = {col: ordinal for ordinal, col in enumerate(columns)}
    operator_colors = _operator_colors(lir)
    # Cycle-accurate clock cycles 1..II (cycle 0 is the accept/input-load bookend row): the grid reflects what the
    # register array physically holds each cycle, including the read/write-latch and microcode-fetch staging.
    compute_cycles = list(range(1, lir.initiation_interval + 1))

    # The operator-stage block: one square column per pipeline stage of each operator, in instance order.
    stage_cols = _stage_columns(lir)
    n_stage = len(stage_cols)
    stage_base: dict[FloatOperatorInstance, int] = {}
    for sidx, (inst, _k) in enumerate(stage_cols):
        stage_base.setdefault(inst, sidx)
    group_ends = {stage_base[inst] + inst.operator.latency - 1 for inst in lir.float_instances}

    # Column seams: 2px at the two block boundaries (constants | pipeline and pipeline | OPERATIONS); 1px at the lighter
    # registers | constants seam and between operator groups.
    dv = _Dividers(
        data_thin={nreg - 1} if nreg and nconst else set(),
        data_thick={len(columns) - 1} if columns else set(),
        stage_thin=group_ends - {n_stage - 1},
        stage_thick={n_stage - 1} if n_stage else set(),
    )

    live = lir.float_liveness
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
    state_cells: set[tuple[int, ColKey]] = set()  # non-coalesced state writebacks, filled by CSS class
    writes_at: dict[tuple[int, ColKey], str] = {}
    endpoints: set[tuple[int, int]] = set()  # (column ordinal, cycle) cells that anchor a dataflow edge
    cell_group: dict[tuple[int, int], int] = {}  # (column ordinal, cycle) -> operation group, for commit cells
    chips_at: dict[int, list[str]] = {}  # cycle -> chips for the operations committing on that row

    group = 0  # global per-operation id, linking a commit cell with its edges/chip for the hover-focus behavior
    for op in lir.float_ops:
        tip = _esc(_op_text(op))
        color = operator_colors[type(op.inst.operator)]
        # Physical clock cycles (cycle-accurate), from the single Lir definitions shared with float_liveness.
        read_cyc = lir.operand_read_cycle(op)
        write_cyc = lir.result_landing_cycle(op)
        dcol: ColKey = op.dst
        dord = col_ord[dcol]
        writes_at[(write_cyc, dcol)] = writes_at.get((write_cyc, dcol), "") + _write_label(op.inst.index, tip)
        fills[(write_cyc, dcol)] = color
        endpoints.add((dord, write_cyc))
        cell_group[(dord, write_cyc)] = group
        for operand in op.operands:
            oord = col_ord[_operand_col(operand)]
            endpoints.add((oord, read_cyc))  # operands are read a read-latch cycle before the operator issues
            edges.append((f"g{dord}_{write_cyc}", f"g{oord}_{read_cyc}", color, group))
        chips_at.setdefault(write_cyc, []).append(
            f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>"
        )
        base = stage_base[op.inst]
        for k in range(op.latency):  # stamp the pipeline trail: stage k of this operator is busy on issue + k + lag
            key = (base + k, op.issue_cycle + k + FETCH_LAG)
            if key in stage_fill and stage_fill[key][1] != group:
                conflicts.add(key)
            stage_fill[key] = (color, group)
            stage_tip[key] = f"{op.inst.operator.mnemonic}_{op.inst.index} s{k}: {tip}"
        group += 1

    # State updates as first-class writes: a non-coalesced slot latches its tap into its register on its install step (a
    # coalesced slot is already drawn as its operator's commit above). Render it like a commit -- a filled cell, a write
    # marker, a chip, and a dataflow edge from the tap, in the state color -- so the schedule shows the update rather
    # than an invisible side effect. The tap is read on the same step (read-first), reg or constant.
    for slot in lir.float_state_slots:
        if not slot.needs_copy:
            continue
        step = lir.state_copy_step(slot)
        dord = col_ord[slot.reg]
        tip = _esc(f"{slot.name} <= {slot.tap.stable_label}")
        writes_at[(step, slot.reg)] = (
            writes_at.get((step, slot.reg), "") + f"<span class='wl' title='{tip}'>&#9662;</span>"
        )
        state_cells.add((step, slot.reg))
        endpoints.add((dord, step))
        cell_group[(dord, step)] = group
        oord = col_ord[slot.tap.source]
        endpoints.add((oord, step))
        edges.append((f"g{dord}_{step}", f"g{oord}_{step}", "state", group))  # JS resolves to --c-state
        chips_at.setdefault(step, []).append(f"<span class='opf state' data-op='{group}'>{tip}</span>")
        group += 1

    out = [_schedule_key(operator_colors, lir.has_state), "<div id='schedwrap'><table class='grid'>"]
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
    out.append("<th class='oph' rowspan='3'>operations</th>")
    out.append("<th class='oph pch' rowspan='3'>pc</th></tr>")
    # Header row 1: register and constant column labels, and one group cell per operator spanning its stages. A register
    # that latches an input or retains state is labeled with that name (color-coded) beside its r-index.
    reg_names = _register_names(lir)
    out.append("<tr>")
    for index in range(nreg):
        cls = "gh" + _border_suffix(index, dv.data_thin, dv.data_thick)
        named = reg_names.get(index)
        label = f"r{index}" if named is None else f"<span class='rn {named[1]}'>{_esc(named[0])}</span> r{index}"
        out.append(f"<th class='{cls}' rowspan='2'><span>{label}</span></th>")
    for index in range(nconst):
        cls = "gh k" + _border_suffix(nreg + index, dv.data_thin, dv.data_thick)
        out.append(f"<th class='{cls}' rowspan='2'><span>c{index}</span></th>")
    for inst in lir.float_instances:
        lat = inst.operator.latency
        name = f"{inst.operator.instance_stem}_{inst.index}"  # full name, set vertically so a 1-stage operator does not widen
        seam = _border_suffix(stage_base[inst] + lat - 1, dv.stage_thin, dv.stage_thick)
        out.append(
            f"<th class='ohgrp{seam}' colspan='{lat}' style='color:{operator_colors[type(inst.operator)]}'>"
            f"<span>{_esc(name)}</span></th>"
        )
    out.append("</tr>")
    # Header row 2: the per-stage labels (s0, s1, ...) under each operator group.
    out.append("<tr>")
    for sidx, (_inst, k) in enumerate(stage_cols):
        cls = "gh" + _border_suffix(sidx, dv.stage_thin, dv.stage_thick)
        out.append(f"<th class='{cls}'><span>s{k}</span></th>")
    out.append("</tr>")

    # One displayed row per clock cycle, cycle-accurate to the hardware: the accept/input-load cycle (0), then the
    # compute, latch, and fetch-staging cycles 1..II. out_valid rises on the last row (the present cycle == II).
    in_cells: dict[ColKey, str] = {load.dst: _input_chip(f"in_{load.name}") for load in lir.float_inputs}
    for slot in lir.float_state_slots:  # at cycle 0 the persistent registers hold their reset snapshot, not an input
        in_cells[slot.reg] = _state_chip(f"{slot.name} = {slot.reset_value!r}")
    out.append(_bookend_row("in", in_cells, columns, live, 0, n_stage, dv))

    for cyc in compute_cycles:  # one row per compute cycle; idle cycles show only pipeline advance
        out.append("<tr>")
        out.append(f"<td class='clk'>{cyc}</td>")
        for ordinal, col in enumerate(columns):
            content = writes_at.get((cyc, col), "")
            extra, style = _cell_style(col, cyc, live, fills, state_cells)
            attrs = f" id='g{ordinal}_{cyc}'" if (ordinal, cyc) in endpoints else ""
            if (ordinal, cyc) in cell_group:
                attrs += f" data-op='{cell_group[(ordinal, cyc)]}'"
            out.append(f"<td class='{_gc_class(ordinal, dv)}{extra}'{attrs}{style}>{content}</td>")
        for sidx in range(n_stage):
            out.append(_stage_cell(sidx, cyc, dv, stage_fill, stage_tip, conflicts))
        out.append(f"<td class='opcell'>{''.join(chips_at.get(cyc, []))}</td>")
        out.append(_pc_cell(cyc))
        out.append("</tr>")
    out.append("</table><svg class='edges'></svg></div>")
    out.append(_sched_script(lir, edges, live))
    return "".join(out)


type ColKey = FloatRegRef | FloatConstRef


def _esc(text: str) -> str:
    return html.escape(text)


def _op_text(op: FloatScheduledOp) -> str:
    body = op.inst.operator.render(*[o.stable_label for o in op.operands])
    return op.result_sign.decorate(f"{op.dst.stable_label}={body}")


def _is_live(col: ColKey, row_id: int, live: dict[FloatRegRef, set[int]]) -> bool:
    """Whether register column ``col`` holds a live value on grid row ``row_id`` (constants are never tinted)."""
    return isinstance(col, FloatRegRef) and col in live and row_id in live[col]


def _write_label(index: int, tip: str) -> str:
    """
    The result marker on the operator's commit cell: its instance index, white text on the filled result cell.

    Operands are no longer chips; the dataflow edges (drawn by the overlay) connect this cell to its operand cells, so
    the cell only needs to identify the operator instance. The tooltip carries the full expression with operand signs.
    """
    return f"<span class='wl' title='{tip}'>{index}</span>"


def _operand_col(operand: FloatOperand) -> ColKey:
    return operand.source


@dataclass(frozen=True, slots=True)
class _Dividers:
    """
    Right-border seams between grid columns, kept on the *left* cell's right edge (under ``border-collapse`` the left
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


def _stage_columns(lir: Lir) -> list[tuple[FloatOperatorInstance, int]]:
    """
    The operator-stage columns, in instance order: ``(instance, stage)`` for each pipeline stage of each operator.

    One column per stage, so an L-cycle operator contributes L columns labeled ``s0..s(L-1)``. As an operation flows
    through its operator, stage ``k`` is occupied on cycle ``issue + k``; the result commits to its register on cycle
    ``issue + L``. This makes the pipeline's advance directly visible and exposes any structural hazard once several
    operations are in flight at once.
    """
    cols: list[tuple[FloatOperatorInstance, int]] = []
    for inst in lir.float_instances:
        cols.extend((inst, k) for k in range(inst.operator.latency))
    return cols


def _pc_cell(cyc: int) -> str:
    """
    The executing microcode step for grid row ``cyc`` (``clk - FETCH_LAG``): the ROM address whose control word drives
    this cycle's datapath, and exactly what ``err_pc`` latches. Blank during the fetch warmup, where it is negative.
    """
    step = cyc - FETCH_LAG
    return f"<td class='pc'>{step if step >= 0 else ''}</td>"


def _bookend_row(
    label: str,
    cells: dict[ColKey, str],
    columns: list[ColKey],
    live: dict[FloatRegRef, set[int]],
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
    out.append("<td class='opcell'></td>")
    out.append(_pc_cell(row_id))
    out.append("</tr>")
    return "".join(out)


def _live_intervals(rows: set[int]) -> list[list[int]]:
    """
    Collapse a register's set of live rows into sorted ``[start, end]`` intervals for compact hand-off to the hover
    script (so it can answer alive/dead for any cycle without shipping a per-cell flag).
    """
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
    col: ColKey,
    cyc: int,
    live: dict[FloatRegRef, set[int]],
    fills: dict[tuple[int, ColKey], str],
    state_cells: set[tuple[int, ColKey]],
) -> tuple[str, str]:
    """
    Background for a register/constant cell, as ``(extra_class, inline_style)``. The single cycle on which a result
    commits is filled solid with its operator color via an inline style (this takes precedence, and being inline it
    survives the row-hover tint); a non-coalesced state writeback is filled by the ``stw`` class (its color lives
    in CSS); otherwise a live register gets the faint residence tint via the ``live`` class, so a row-hover can override
    it. The operators' cycle-by-cycle occupancy lives in the separate operator-stage block.
    """
    color = fills.get((cyc, col))
    if color is not None:
        return "", f" style='background:{color}'"
    if (cyc, col) in state_cells:
        return " stw", ""
    if _is_live(col, cyc, live):
        return " live", ""
    return "", ""


def _stage_cell(
    sidx: int,
    cyc: int,
    dv: _Dividers,
    stage_fill: dict[tuple[int, int], tuple[str, int]],
    stage_tip: dict[tuple[int, int], str],
    conflicts: set[tuple[int, int]],
) -> str:
    """
    One operator-stage cell: empty unless an operation occupies this stage on this cycle, in which case it is filled
    with the operator color and tagged with the operation group (so a hover lights the whole pipeline trail).
    """
    cls = _oc_class(sidx, dv)
    occ = stage_fill.get((sidx, cyc))
    if occ is None:
        return f"<td class='{cls}'></td>"
    color, group = occ
    if (sidx, cyc) in conflicts:
        cls += " conflict"
    return f"<td class='{cls}' data-op='{group}' title='{stage_tip[(sidx, cyc)]}' style='background:{color}'></td>"


def _sched_script(lir: Lir, edges: list[tuple[str, str, str, int]], live: dict[FloatRegRef, set[int]]) -> str:
    """
    Build the interactive layer: substitute the per-module data into the readable script template (:data:`_SCHED_JS`).

    The data is the edge list, the column labels, the constant values, and the per-register live-row intervals. This is
    enough for the script to draw the dataflow overlay and synthesize hover tooltips on demand without a per-cell
    attribute. Without JS the grid still renders fully; only these behaviors are absent.
    """
    cols = [col.stable_label for col in _columns_of(lir)]
    data = {
        "edges": edges,
        "columns": cols,
        "constants": {f"c{i}": repr(value) for i, value in enumerate(lir.float_consts)},
        "liveness": {str(reg.index): _live_intervals(rows) for reg, rows in live.items()},
    }
    return "<script>\n" + _SCHED_JS.replace("__DATA__", json.dumps(data)) + "\n</script>"


def _columns_of(lir: Lir) -> list[ColKey]:
    """The grid columns: one per float register, then one per constant (matches the order rendered in the table)."""
    return [FloatRegRef(i) for i in range(lir.float_regfile.nreg)] + [
        FloatConstRef(i) for i in range(len(lir.float_consts))
    ]


def _input_chip(tip: str) -> str:
    """A neutral input-write marker for the cycle-0 latch row (``.wr.input`` carries its color)."""
    return f"<span class='wr input' title='{tip}'>&#9662;</span>"


def _state_chip(tip: str) -> str:
    """A retained-state latch marker for the cycle-0 row: a persistent slot holding its reset snapshot (``.wr.state``)."""
    return f"<span class='wr state' title='{tip}'>&#9662;</span>"


def _register_names(lir: Lir) -> dict[int, tuple[str, str]]:
    """
    Map each register that has a stable role to ``(label, kind)``: the input lanes to the port they latch at accept and
    the state slots to the attribute they retain. Other registers are anonymous scratch; both kinds may still be reused
    by operations later, but their cycle-0 role is what the label names.
    """
    names: dict[int, tuple[str, str]] = {}
    for load in lir.float_inputs:
        names[load.dst.index] = (f"in_{load.name}", "input")
    for slot in lir.float_state_slots:
        names[slot.reg.index] = (slot.name, "state")
    return names


def _schedule_key(operator_colors: dict[type[HardwareOperator], str], has_state: bool) -> str:
    """A small legend above the grid: operator-kind colors plus the read/write chip shapes."""
    items = [
        f"<span class='wr' style='background:{color}'>{_esc(cls.mnemonic)}</span>"
        for cls, color in operator_colors.items()
    ]
    items.extend(
        [
            _key_item("<span class='sw ink'></span>", "filled cell = result committed (operator n)"),
            _key_item(_EDGE_KEY_MARKER, "edge: result &rarr; its operands"),
            _key_item(
                "<span class='sw ink'></span>",
                "operator-stage block: s0..sN occupancy as the pipeline advances",
            ),
            _key_item("<span class='sw live'></span>", "register holds a live value"),
            _key_item("<span class='wr input'>&#9662;</span>", "module input latch"),
        ]
    )
    if has_state:
        items.extend(
            [
                _key_item("<span class='wr state'>&#9662;</span>", "persistent state: reset snapshot (cycle 0)"),
                _key_item("<span class='sw state'></span>", "state update latched on its install step"),
            ]
        )
    items.append(f"<span>pc = microcode step executing this cycle (clk&minus;{FETCH_LAG} fetch lag)</span>")
    return "<h2>Schedule</h2><div class='gridkey'>" + " ".join(items) + "</div>"


def _key_item(marker: str, text: str) -> str:
    return f"<span>{marker} {text}</span>"


def _operator_colors(lir: Lir) -> dict[type[HardwareOperator], str]:
    kinds = sorted(
        {type(inst.operator) for inst in lir.float_instances},
        key=lambda kind: (kind.mnemonic, kind.__module__, kind.__qualname__),
    )
    return dict(zip(kinds, _html_palette(len(kinds)), strict=True))


def _html_palette(n: int, lightness: float = 0.2, saturation: float = 1.0, hue_start: float = 0.0) -> list[str]:
    """
    n: number of colors equidistant on the hue wheel, starting at hue_start
    lightness: 0..1, target WCAG relative luminance
    saturation: 0..1, color intensity
    hue_start: 0..1, rotates the palette around the hue wheel
    """
    colors = []
    for i in range(n):
        hue = (hue_start + i / n) % 1.0
        color = "#{:02X}{:02X}{:02X}".format(*[round(x * 255) for x in _hls_for_luminance(hue, lightness, saturation)])
        colors.append(color)
    return colors


def _hls_for_luminance(hue: float, luminance: float, saturation: float) -> tuple[float, float, float]:
    low, high = 0.0, 1.0
    for _ in range(24):
        lightness = (low + high) / 2
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        if _relative_luminance(rgb) < luminance:
            low = lightness
        else:
            high = lightness
    return colorsys.hls_to_rgb(hue, (low + high) / 2, saturation)


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    red, green, blue = (_linear_srgb(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _linear_srgb(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return float(((channel + 0.055) / 1.055) ** 2.4)
