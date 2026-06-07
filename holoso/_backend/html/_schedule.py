"""Render the cycle-accurate schedule grid for the HTML report."""

import colorsys
import html
import json
from dataclasses import dataclass
from importlib import resources

from ..._lir import (
    FETCH_LAG,
    BoolRegRef,
    BoolScheduledOp,
    BoolSource,
    Branch,
    boundary_step,
    copy_step_cycle,
    FloatConstRef,
    FloatOperand,
    FloatRegRef,
    FloatScheduledOp,
    Jump,
    landing_cycle,
    Lir,
    OperatorInstance,
    read_latch_cycle,
)
from ..._operators import HardwareOperator

# Interactive layer; ``__DATA__`` is replaced by the per-module payload in ``_sched_script``.
_SCHED_JS = resources.files(__package__).joinpath("html.js").read_text(encoding="utf-8")
_EDGE_KEY_MARKER = (
    "<svg class='lk' width='24' height='12'>"
    "<line x1='2' y1='3' x2='21' y2='10' stroke='currentColor' stroke-width='1'/>"
    "<circle cx='21' cy='10' r='1.8' fill='currentColor'/>"
    "</svg>"
)
# A miniature of a margin control-transfer arrow (bracket out to a channel and back, with a left-pointing head).
_ARROW_KEY_MARKER = (
    "<svg class='lk' width='24' height='12'>"
    "<polyline points='3,3 20,3 20,9 3,9' fill='none' stroke='var(--c-arrow)' stroke-width='1.25'/>"
    "<polygon points='3,9 8,6 8,12' fill='var(--c-arrow)'/>"
    "</svg>"
)


def render_schedule(lir: Lir) -> str:
    nreg, nbreg, nconst = lir.float_regfile.nreg, lir.bool_regfile.nreg, len(lir.float_consts)
    columns = _columns_of(lir)
    col_ord = {col: ordinal for ordinal, col in enumerate(columns)}
    operator_colors = _operator_colors(lir)
    # The microcode PC timeline, cycles 1..span (cycle 0 is the accept/input-load bookend row): the grid reflects what
    # the register array holds at each PC, including the read/write-latch and microcode-fetch staging. For a control-
    # flow kernel this lays out every block's PC range; one transaction follows a single path through it, so the grid
    # is the static program, not one transaction's cycle-accurate trace.
    compute_cycles = list(range(1, lir.initiation_interval + 1))

    # The operator-stage block: one square column per pipeline stage of each operator, in instance order.
    stage_cols = _stage_columns(lir)
    n_stage = len(stage_cols)
    stage_base: dict[OperatorInstance, int] = {}
    for sidx, (inst, _k) in enumerate(stage_cols):
        stage_base.setdefault(inst, sidx)
    group_ends = {stage_base[inst] + inst.operator.latency - 1 for inst in _all_instances(lir)}

    # Column seams: a 2px black seam marks every register-bank boundary (float|bool, bool|constants) and the two block
    # boundaries (the last data column | pipeline, and pipeline | OPERATIONS); a 1px seam marks the legacy float|const
    # seam of a kernel without a boolean bank and the seams between operator groups.
    data_thin, data_thick = _data_seams(nreg, nbreg, nconst)
    if columns:
        data_thick.add(len(columns) - 1)  # the data block | operator-pipeline boundary
    dv = _Dividers(
        data_thin=data_thin,
        data_thick=data_thick,
        stage_thin=group_ends - {n_stage - 1},
        stage_thick={n_stage - 1} if n_stage else set(),
    )

    # Residence tint for both register banks: the float liveness comes from the LIR; the boolean liveness is derived
    # here (the LIR has no equivalent) by the same def/use-interval method, then merged into one column-keyed map.
    live: dict[ColKey, set[int]] = {}
    for freg, frows in lir.float_liveness.items():
        live[freg] = frows
    for breg, brows in _bool_liveness(lir).items():
        live[breg] = brows
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

    # Comparators (boolean operations on the pooled fcmp): a comparison commits a boolean register -- now a grid column
    # -- so it renders like a float op. A solid commit cell lands in its bX column on the result-landing cycle, carries
    # the instance-index write label, sources dataflow edges from its float operands (read a read-latch cycle before the
    # operator issues), an ops chip, and the operator-pipeline trail. All share one group so a hover lights them
    # together. Block-relative cycles are rebased to the absolute PC timeline, as the flattened float ops are.
    comparator_inst = {inst.operator: inst for inst in _comparator_instances(lir)}
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for bop in block.bool_ops:
            inst = comparator_inst[bop.operator]
            color = operator_colors[type(bop.operator)]
            issue = base_pc + bop.issue_cycle
            read_cyc = read_latch_cycle(issue)
            write_cyc = landing_cycle(base_pc + bop.commit_cycle)
            tip = _esc(_bool_op_text(bop))
            dcol = bop.dst
            dord = col_ord[dcol]
            writes_at[(write_cyc, dcol)] = writes_at.get((write_cyc, dcol), "") + _write_label(inst.index, tip)
            fills[(write_cyc, dcol)] = color
            endpoints.add((dord, write_cyc))
            cell_group[(dord, write_cyc)] = group
            for operand in bop.operands:
                oord = col_ord[operand.source]
                endpoints.add((oord, read_cyc))
                edges.append((f"g{dord}_{write_cyc}", f"g{oord}_{read_cyc}", color, group))
            chips_at.setdefault(write_cyc, []).append(
                f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>"
            )
            base = stage_base[inst]
            for k in range(bop.latency):
                key = (base + k, issue + k + FETCH_LAG)
                if key in stage_fill and stage_fill[key][1] != group:
                    conflicts.add(key)
                stage_fill[key] = (color, group)
                stage_tip[key] = f"{bop.operator.mnemonic}_{inst.index} s{k}: {tip}"
            group += 1

    # Boolean phi/state installs: a block writes a boolean register (a constant or another boolean register) at its
    # install step, exactly like a non-coalesced float-slot copy. Render it as a state-style write -- a filled cell, a
    # write marker, a chip, and a dataflow edge from the source if it is a register -- in its bX column.
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for bwrite in block.bool_writes:
            step = base_pc + copy_step_cycle(bwrite.issue_cycle)
            dcol = bwrite.dst
            dord = col_ord[dcol]
            tip = _esc(f"{bwrite.dst.stable_label} <= {_bool_source_label(bwrite.source.source)}")
            writes_at[(step, dcol)] = writes_at.get((step, dcol), "") + f"<span class='wl' title='{tip}'>&#9662;</span>"
            state_cells.add((step, dcol))
            endpoints.add((dord, step))
            cell_group[(dord, step)] = group
            if isinstance(bwrite.source.source, BoolRegRef):  # a bool constant has no source cell, so no edge
                oord = col_ord[bwrite.source.source]
                endpoints.add((oord, step))
                edges.append((f"g{dord}_{step}", f"g{oord}_{step}", "state", group))
            chips_at.setdefault(step, []).append(f"<span class='opf state' data-op='{group}'>{tip}</span>")
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
        tip = _esc(f"{slot.name} <= {_operand_label(slot.tap)}")
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

    # Control-transfer arrows: anchor each conditional arrow's root to the boolean register it tests by making that
    # register's cell at the source row a dataflow endpoint, so the overlay can draw the dotted feed to it -- the
    # register's residence then visibly ends at the branch rather than in nothingness.
    arrows = _control_arrows(lir)
    for arrow in arrows:
        if arrow.cond is not None:
            endpoints.add((col_ord[arrow.cond], arrow.src_cyc))

    out = [
        _schedule_key(operator_colors, lir.has_state, bool(arrows)),
        "<div id='schedwrap'><table class='grid'>",
    ]
    # Header row 0: group bands over the register banks (float, bool), the constant block and the operator pipelines.
    # clk/operations span all rows. Each band carries the seam of its rightmost data column so the bank edges align.
    out.append("<tr><th class='gh clkh' rowspan='3'><span>clk</span></th>")
    if nreg:
        out.append(
            f"<th class='gband{_border_suffix(nreg - 1, dv.data_thin, dv.data_thick)}' colspan='{nreg}'>"
            "<span>float</span></th>"
        )
    if nbreg:
        out.append(
            f"<th class='gband{_border_suffix(nreg + nbreg - 1, dv.data_thin, dv.data_thick)}' "
            f"colspan='{nbreg}'><span>bool</span></th>"
        )
    if nconst:
        const_seam = _border_suffix(nreg + nbreg + nconst - 1, dv.data_thin, dv.data_thick)
        out.append(f"<th class='gband{const_seam}' colspan='{nconst}'><span>constants</span></th>")
    if n_stage:
        seam = _border_suffix(n_stage - 1, dv.stage_thin, dv.stage_thick)
        out.append(f"<th class='gband{seam}' colspan='{n_stage}'><span>operator pipelines</span></th>")
    out.append("<th class='oph' rowspan='3'>operations</th>")
    out.append("<th class='oph pch' rowspan='3'>pc</th></tr>")
    # Header row 1: register and constant column labels, and one group cell per operator spanning its stages. A register
    # that latches an input or retains state is labeled with that name (color-coded) beside its bank index.
    reg_names = _register_names(lir)
    out.append("<tr>")
    for index in range(nreg):
        cls = "gh" + _border_suffix(index, dv.data_thin, dv.data_thick)
        out.append(
            f"<th class='{cls}' rowspan='2'><span>{_named_label('f', index, reg_names.get(('f', index)))}"
            "</span></th>"
        )
    for index in range(nbreg):
        cls = "gh" + _border_suffix(nreg + index, dv.data_thin, dv.data_thick)
        out.append(
            f"<th class='{cls}' rowspan='2'><span>{_named_label('b', index, reg_names.get(('b', index)))}"
            "</span></th>"
        )
    for index in range(nconst):
        cls = "gh k" + _border_suffix(nreg + nbreg + index, dv.data_thin, dv.data_thick)
        out.append(f"<th class='{cls}' rowspan='2'><span>c{index}</span></th>")
    for inst in _all_instances(lir):
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
    for bslot in lir.bool_state_slots:  # boolean persistent slots likewise show their reset snapshot in their bX column
        in_cells[bslot.reg] = _state_chip(f"{bslot.name} = {bslot.reset_value!r}")
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
    out.append(_sched_script(lir, edges, live, arrows, col_ord))
    return "".join(out)


type ColKey = FloatRegRef | BoolRegRef | FloatConstRef


def _esc(text: str) -> str:
    return html.escape(text)


def _col_label(col: ColKey) -> str:
    """
    The report's display label for a grid column: ``fX`` for a float register, ``bX`` for a boolean register, ``cX`` for
    a float constant. This is the single labeling authority for the schedule -- the headers, tooltips, dataflow operand
    expressions, arrow conditions, and the JS payload all route through it, so the float bank reads ``fX`` everywhere
    even though the LIR's own ``stable_label`` (which the report must not change) still names a float register ``rX``.
    """
    return f"f{col.index}" if isinstance(col, FloatRegRef) else col.stable_label


def _operand_label(operand: FloatOperand) -> str:
    """A float operand's display label: its source column relabeled by :func:`_col_label`, with the folded sign."""
    return operand.sign.decorate(_col_label(operand.source))


def _op_text(op: FloatScheduledOp) -> str:
    body = op.inst.operator.render(*[_operand_label(o) for o in op.operands])
    return op.result_sign.decorate(f"{_col_label(op.dst)}={body}")


def _is_live(col: ColKey, row_id: int, live: dict[ColKey, set[int]]) -> bool:
    """Whether register column ``col`` holds a live value on grid row ``row_id`` (constants are never tinted)."""
    return isinstance(col, (FloatRegRef, BoolRegRef)) and col in live and row_id in live[col]


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
    every register-bank boundary (float|bool, bool|constants) and the two block boundaries (the data block |
    operator-pipeline and operator-pipeline | OPERATIONS); a ``thin`` 1px seam marks the legacy float|constants seam of a
    boolean-free kernel and the seams between operator groups. Data and stage columns index separately.
    """

    data_thin: set[int]
    data_thick: set[int]
    stage_thin: set[int]
    stage_thick: set[int]


def _data_seams(nreg: int, nbreg: int, nconst: int) -> tuple[set[int], set[int]]:
    """
    Right-border seams within the register/constant block, as ``(thin, thick)`` left-cell column-index sets.

    A thick 2px black seam marks each register-bank boundary: float|bool and bool|constants. With no boolean bank the
    float bank abuts the constants directly; that legacy seam stays the lighter 1px so a float-only kernel renders
    exactly as before. Empty banks contribute no seam. The constants|pipeline block boundary is added by the caller.
    """
    thin: set[int] = set()
    thick: set[int] = set()
    if nbreg:
        if nreg:
            thick.add(nreg - 1)  # float | bool
        if nconst:
            thick.add(nreg + nbreg - 1)  # bool | constants
    elif nreg and nconst:
        thin.add(nreg - 1)  # legacy float | constants (no boolean bank)
    return thin, thick


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


def _all_instances(lir: Lir) -> list[OperatorInstance]:
    """Every operator instance shown in the schedule: the float operator instances then the pooled comparator(s)."""
    return [*lir.float_instances, *_comparator_instances(lir)]


def _comparator_instances(lir: Lir) -> list[OperatorInstance]:
    """
    The comparator instances: one per distinct ``fcmp`` configuration across every block's boolean operations. The
    comparator is pooled (one instance per configuration), so the comparisons in different blocks share one instance.
    """
    instances: dict[HardwareOperator, OperatorInstance] = {}
    for block in lir.blocks:
        for op in block.bool_ops:
            instances.setdefault(op.operator, OperatorInstance(op.operator, len(instances)))
    return list(instances.values())


_RELATION_SYMBOL = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}


def _bool_source_label(source: BoolSource) -> str:
    """The label of a boolean source for a tooltip: the register's ``bX`` label or the inline ``True``/``False``."""
    return source.stable_label if isinstance(source, BoolRegRef) else repr(source.value)


def _comparison_expr(op: BoolScheduledOp) -> str:
    """The bare relational expression ``<a> <relation> <b>`` (with operand sign decoration), no destination prefix."""
    a, b = (_operand_label(operand) for operand in op.operands)
    return f"{a} {_RELATION_SYMBOL.get(op.relation.value, op.relation.value)} {b}"


def _bool_op_text(op: BoolScheduledOp) -> str:
    """The comparison's expression for its chip/tooltip: ``b<dst> = <a> <relation> <b>`` with operand sign decoration."""
    return f"{op.dst.stable_label} = {_comparison_expr(op)}"


def _stage_columns(lir: Lir) -> list[tuple[OperatorInstance, int]]:
    """
    The operator-stage columns, in instance order: ``(instance, stage)`` for each pipeline stage of each operator.

    One column per stage, so an L-cycle operator contributes L columns labeled ``s0..s(L-1)``. As an operation flows
    through its operator, stage ``k`` is occupied on cycle ``issue + k``; the result commits to its register on cycle
    ``issue + L``. This makes the pipeline's advance directly visible and exposes any structural hazard once several
    operations are in flight at once. Comparator instances follow the float instances, on the same per-stage grid.
    """
    cols: list[tuple[OperatorInstance, int]] = []
    for inst in _all_instances(lir):
        cols.extend((inst, k) for k in range(inst.operator.latency))
    return cols


def _pc_cell(cyc: int) -> str:
    """
    The executing microcode step for grid row ``cyc`` (``clk - FETCH_LAG``): the ROM address whose control word drives
    this cycle's datapath, and exactly what ``err_pc`` latches. Blank during the fetch warmup, where it is negative. The
    ``pc_<cyc>`` id lets the overlay measure this row's y-centre to route the control-transfer arrows in the margin.
    """
    step = cyc - FETCH_LAG
    return f"<td class='pc' id='pc_{cyc}'>{step if step >= 0 else ''}</td>"


def _bookend_row(
    label: str,
    cells: dict[ColKey, str],
    columns: list[ColKey],
    live: dict[ColKey, set[int]],
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


def _bool_liveness(lir: Lir) -> dict[BoolRegRef, set[int]]:
    """
    The residence tint for the boolean register bank, derived here because the LIR has no ``bool_liveness`` and the
    constraint is to add no field. It mirrors :attr:`Lir.float_liveness` in the executing-step frame: a boolean register
    is defined when a comparison commits its result (``landing_cycle`` of the rebased commit), when a boolean phi/state
    install lands (the rebased copy step), and -- for a persistent slot -- at the live-in resident from cycle 1; it is
    used where a branch reads its condition (the terminator's boundary row) and at the boundary, where a slot's live-out
    must reside for the next initiation (and its source is read, read-first). Each def-interval extends to its last use
    before the next def, exactly as the float liveness does.
    """
    present = lir.initiation_interval
    defs: dict[BoolRegRef, list[int]] = {}
    uses: dict[BoolRegRef, list[int]] = {}
    for slot in lir.bool_state_slots:
        defs.setdefault(slot.reg, []).append(1)  # the live-in is resident in the slot register from the start
        uses.setdefault(slot.reg, []).append(present)  # ...and the live-out must reside there through the boundary
        if slot.needs_copy:
            defs.setdefault(slot.reg, []).append(present)  # the new live-out lands at the boundary (read-first)
            if isinstance(slot.live_out.source, BoolRegRef):
                uses.setdefault(slot.live_out.source, []).append(present)  # its source is read at the boundary
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for bop in block.bool_ops:
            defs.setdefault(bop.dst, []).append(landing_cycle(base_pc + bop.commit_cycle))
        for bwrite in block.bool_writes:
            defs.setdefault(bwrite.dst, []).append(base_pc + copy_step_cycle(bwrite.issue_cycle))
            if isinstance(bwrite.source.source, BoolRegRef):
                uses.setdefault(bwrite.source.source, []).append(base_pc + copy_step_cycle(bwrite.issue_cycle))
        if isinstance(block.terminator, Branch):  # the branch reads its condition at the block's boundary row
            read = base_pc + boundary_step(block.block_makespan) + FETCH_LAG
            uses.setdefault(block.terminator.cond, []).append(read)
    return {reg: _residence_rows(defs.get(reg, []), uses.get(reg, []), present) for reg in defs.keys() | uses.keys()}


def _residence_rows(defs: list[int], uses: list[int], present: int) -> set[int]:
    """
    Collapse a register's defs/uses into the set of live rows by the read-first rule shared with ``Lir.float_liveness``:
    each def's value resides from its landing through the last use that still returns it (the last use no later than the
    next def, since a read on the next def's cycle is read-first and still sees the old value), the boundary at latest.
    """
    writes = sorted(defs)
    reads = sorted(uses)
    rows: set[int] = set()
    for i, start in enumerate(writes):
        nxt = writes[i + 1] if i + 1 < len(writes) else present + 1
        last = max((use for use in reads if start <= use <= nxt), default=start)
        rows.update(range(start, last + 1))
    return rows


@dataclass(frozen=True, slots=True)
class _Arrow:
    """
    One control-transfer arrow in the right margin: a non-fall-through jump from grid row ``src_cyc`` into row
    ``dst_cyc``. ``tip`` is its hover label -- the branch arm's condition (the boolean register the branch evaluates),
    or ``jump`` if unconditional -- as raw text (rendered via the SVG ``<title>``'s textContent, not HTML-escaped;
    json.dumps makes it JS-safe). ``cond`` is the boolean register the branch reads, or ``None`` for an unconditional
    jump; the overlay draws a dotted feed from that register's cell at the source row to the arrow's root, so the
    register's residence visibly ends at the branch rather than in nothingness.
    """

    src_cyc: int
    dst_cyc: int
    tip: str
    cond: BoolRegRef | None


def _control_arrows(lir: Lir) -> list[_Arrow]:
    """
    The control transfers that are not the fall-through to the physically next ROM step, one arrow each: a ``Jump`` to a
    non-adjacent block, and each ``Branch`` arm whose target is not the fall-through (usually one arm falls through and
    the other jumps). The source row is the block's boundary PC, the target row is the destination block's base PC; each
    is mapped to its grid row by the FETCH_LAG offset, and an arrow is skipped if either row falls outside the grid. A
    branch arm carries the condition register it reads (for the tooltip and the dotted feed); a jump carries ``None``.
    """
    present = lir.initiation_interval
    arrows: list[_Arrow] = []

    def emit(term_pc: int, target: int, tip: str, cond: BoolRegRef | None) -> None:
        src_cyc, dst_cyc = term_pc + FETCH_LAG, lir.block_base[target] + FETCH_LAG
        if 1 <= src_cyc <= present and 1 <= dst_cyc <= present:
            arrows.append(_Arrow(src_cyc, dst_cyc, tip, cond))

    for block in lir.blocks:
        term_pc = lir.block_base[block.index] + boundary_step(block.block_makespan)
        fall_pc = term_pc + 1  # the physically next ROM step; a target landing here is the fall-through, drawn as none
        match block.terminator:
            case Jump(target=target):
                if lir.block_base[target] != fall_pc:
                    emit(term_pc, target, "jump", None)
            case Branch(cond=cond, if_true=if_true, if_false=if_false):
                if lir.block_base[if_true] != fall_pc:
                    emit(term_pc, if_true, _branch_arm_text(cond, taken=True), cond)
                if lir.block_base[if_false] != fall_pc:
                    emit(term_pc, if_false, _branch_arm_text(cond, taken=False), cond)
    return arrows


def _branch_arm_text(cond: BoolRegRef, taken: bool) -> str:
    """
    The condition label for a branch arm, naming the boolean register the branch evaluates (not the comparison that
    populated it): ``if b<i>`` for the taken arm, ``if not b<i>`` for the not-taken arm.
    """
    label = _col_label(cond)
    return f"if {label}" if taken else f"if not {label}"


def _cell_style(
    col: ColKey,
    cyc: int,
    live: dict[ColKey, set[int]],
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


def _sched_script(
    lir: Lir,
    edges: list[tuple[str, str, str, int]],
    live: dict[ColKey, set[int]],
    arrows: list[_Arrow],
    col_ord: dict[ColKey, int],
) -> str:
    """
    Build the interactive layer: substitute the per-module data into the readable script template (:data:`_SCHED_JS`).

    The data is the edge list, the column labels, the constant values, the per-register live-row intervals (keyed by the
    full column label so the float and boolean banks never collide on a shared index), and the control-transfer arrows.
    Each arrow carries ``cond``: the cell id of the boolean register it reads (so the overlay draws a dotted feed from
    that register to the arrow's root), or ``None`` for an unconditional jump. This is enough for the script to draw the
    dataflow and arrow overlays and synthesize hover tooltips on demand; without JS the grid still renders fully.
    """
    cols = [_col_label(col) for col in _columns_of(lir)]
    data = {
        "edges": edges,
        "columns": cols,
        "constants": {f"c{i}": repr(value) for i, value in enumerate(lir.float_consts)},
        "liveness": {_col_label(col): _live_intervals(rows) for col, rows in live.items()},
        "arrows": [
            {
                "from": a.src_cyc,
                "to": a.dst_cyc,
                "tip": a.tip,
                "cond": (f"g{col_ord[a.cond]}_{a.src_cyc}" if a.cond is not None else None),
            }
            for a in arrows
        ],
    }
    return "<script>\n" + _SCHED_JS.replace("__DATA__", json.dumps(data)) + "\n</script>"


def _columns_of(lir: Lir) -> list[ColKey]:
    """
    The grid columns in bank order: float registers (``fX``), then boolean registers (``bX``), then float constants
    (``cX``). This is exactly the order the table renders, so a column ordinal indexes straight into this list.
    """
    cols: list[ColKey] = [FloatRegRef(i) for i in range(lir.float_regfile.nreg)]
    cols += [BoolRegRef(i) for i in range(lir.bool_regfile.nreg)]
    cols += [FloatConstRef(i) for i in range(len(lir.float_consts))]
    return cols


def _input_chip(tip: str) -> str:
    """A neutral input-write marker for the cycle-0 latch row (``.wr.input`` carries its color)."""
    return f"<span class='wr input' title='{tip}'>&#9662;</span>"


def _state_chip(tip: str) -> str:
    """A retained-state latch marker for the cycle-0 row: a persistent slot holding its reset snapshot (``.wr.state``)."""
    return f"<span class='wr state' title='{tip}'>&#9662;</span>"


def _register_names(lir: Lir) -> dict[tuple[str, int], tuple[str, str]]:
    """
    Map each register that has a stable role to ``(label, kind)``, keyed by ``(bank, index)`` where bank is ``"f"`` for
    a float register or ``"b"`` for a boolean one: the float input lanes to the port they latch at accept, the float
    state slots to the attribute they retain, and the boolean state slots to the boolean attribute they retain. Other
    registers are anonymous scratch; both kinds may still be reused later, but their cycle-0 role is what the label names.
    """
    names: dict[tuple[str, int], tuple[str, str]] = {}
    for load in lir.float_inputs:
        names[("f", load.dst.index)] = (f"in_{load.name}", "input")
    for slot in lir.float_state_slots:
        names[("f", slot.reg.index)] = (slot.name, "state")
    for bslot in lir.bool_state_slots:
        names[("b", bslot.reg.index)] = (bslot.name, "state")
    return names


def _named_label(bank: str, index: int, named: tuple[str, str] | None) -> str:
    """A column header label ``<bank><index>`` (e.g. ``f3``, ``b0``), prefixed with its color-coded role name if any."""
    base = f"{bank}{index}"
    if named is None:
        return base
    return f"<span class='rn {named[1]}'>{_esc(named[0])}</span> {base}"


def _schedule_key(operator_colors: dict[type[HardwareOperator], str], has_state: bool, has_arrows: bool) -> str:
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
    if has_arrows:
        items.append(_key_item(_ARROW_KEY_MARKER, "control transfer: branch / jump (hover for the condition)"))
    items.append(f"<span>pc = microcode step executing this cycle (clk&minus;{FETCH_LAG} fetch lag)</span>")
    return "<h2>Schedule</h2><div class='gridkey'>" + " ".join(items) + "</div>"


def _key_item(marker: str, text: str) -> str:
    return f"<span>{marker} {text}</span>"


def _operator_colors(lir: Lir) -> dict[type[HardwareOperator], str]:
    kinds = sorted(
        {type(inst.operator) for inst in _all_instances(lir)},
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
