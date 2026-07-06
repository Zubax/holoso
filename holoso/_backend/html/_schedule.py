"""Render the cycle-accurate schedule grid for the HTML report."""

import colorsys
import html
import json
from dataclasses import dataclass, replace
from importlib import resources

from ..._lir import *
from ..._operators import HardwareOperator

# Interactive layer; ``__DATA__`` is replaced by the per-module payload in ``_sched_script``.
_SCHED_JS = resources.files(__package__).joinpath("html.js").read_text(encoding="utf-8")
_EDGE_KEY_MARKER = (
    "<svg class='lk' width='24' height='12'>"
    "<line x1='2' y1='3' x2='21' y2='10' stroke='currentColor' stroke-width='1'/>"
    "<circle cx='21' cy='10' r='1.8' fill='currentColor'/>"
    "</svg>"
)
# Miniature of a margin control-transfer arrow (bracket out to a channel and back, with a left-pointing head).
_ARROW_KEY_MARKER = (
    "<svg class='lk' width='24' height='12'>"
    "<path d='M3,3 H20 V9 H3' fill='none' stroke='currentColor' stroke-width='1.25' "
    "stroke-linejoin='round' stroke-linecap='round'/>"
    "<polygon points='3,9 8,6 8,12' fill='currentColor'/>"
    "</svg>"
)


def render_schedule(lir: Lir) -> str:
    nreg, nbreg, nconst = lir.regfile.nreg, lir.bool_regfile.nreg, len(lir.float_consts)
    bool_consts = _bool_consts(lir)
    nbbool = len(bool_consts)
    columns = _columns_of(lir)
    col_ord = {col: ordinal for ordinal, col in enumerate(columns)}
    operator_colors = _operator_colors(lir)
    # The microcode PC timeline, cycles 1..span (cycle 0 is the accept/input-load bookend row): the grid reflects what
    # the register array holds at each PC, including the microcode-fetch staging.
    # For a control-flow kernel this lays out every block's PC range; one transaction follows a single path through it,
    # so the grid is the static program, not one transaction's cycle-accurate trace.
    compute_cycles = list(range(1, lir.initiation_interval + 1))
    # Block boundaries: the grid row axis is the model fetch PC, and blocks tile it contiguously in layout order, so a
    # non-Ret block ends at its terminator PC and the next block begins on the following row. A thick horizontal seam
    # below each such row makes the block structure visible. The single Ret block ends at the grid's last row (the
    # out_valid boundary), so it needs no seam. A straight-line kernel has one block and no seams.
    boundary_rows = {lir.term_pc(block) for block in lir.blocks if not isinstance(block.terminator, Ret)}

    # The operator-stage block: one square column per pipeline stage of each operator, in instance order.
    stage_cols = _stage_columns(lir)
    n_stage = len(stage_cols)
    stage_base: dict[OperatorInstance, int] = {}
    for sidx, (inst, _k) in enumerate(stage_cols):
        stage_base.setdefault(inst, sidx)
    group_ends = {stage_base[inst] + inst.operator.latency - 1 for inst in lir.instances}

    # Column seams: a 2px black seam marks every register-bank boundary (wide|bool, bool|constants) and the two block
    # boundaries (the last data column | pipeline, and pipeline | OPERATIONS); a 1px seam marks the legacy wide|const
    # seam of a kernel without a boolean bank, the float|bool divide within the constants block, and the seams between
    # operator groups.
    data_thin, data_thick = _data_seams(nreg, nbreg, nconst, nbbool)
    if columns:
        data_thick.add(len(columns) - 1)  # the data block | operator-pipeline boundary
    dv = _Dividers(
        data_thin=data_thin,
        data_thick=data_thick,
        stage_thin=group_ends - {n_stage - 1},
        stage_thick={n_stage - 1} if n_stage else set(),
    )

    # Residence tint for both register banks, from the LIR's complete liveness (each accounts for the combinational
    # ops -- comparisons, boolean logic, and the float<->bool casts -- as well as arithmetic, state, and branches),
    # merged into one column-keyed map.
    live: dict[ColKey, set[int]] = {}
    for freg, frows in lir.reg_liveness.items():
        live[freg] = frows
    for breg, brows in lir.bool_liveness.items():
        live[breg] = brows
    edges: list[tuple[str, str, str, int]] = []  # (commit id, operand id, color, operation group) for the overlay
    # Operator pipeline occupancy: instance ``inst`` is in stage ``k`` on cycle ``issue + k + fetch_lag``. Keyed to the
    # operation group so a hover lights the whole pipeline trail together with the result cell, its chip and its edges.
    # ``conflicts`` flags any cell two operations claim at once -- the alarm for a scheduling bug (a structural hazard
    # the pipelined scheduler should never emit).
    stage_fill: dict[tuple[int, int], tuple[str, int]] = {}  # (stage column, cycle) -> (operator color, group)
    stage_tip: dict[tuple[int, int], str] = {}
    conflicts: set[tuple[int, int]] = set()
    # Result-commit cells, keyed by absolute (cycle, column). A result lands on its commit's landing cycle; under
    # cross-block overlap a result spilling past the shrunk terminator lands in EACH successor arm's frame, so one
    # firing may stamp a commit cell on more than one row (``lir.write_landing_pcs``, exactly the model's landing PCs).
    fills: dict[tuple[int, ColKey], str] = {}
    state_cells: set[tuple[int, ColKey]] = set()  # non-coalesced state writebacks, filled by CSS class
    writes_at: dict[tuple[int, ColKey], str] = {}
    endpoints: set[tuple[int, int]] = set()  # (column ordinal, cycle) cells that anchor a dataflow edge
    cell_group: dict[tuple[int, int], int] = {}  # (column ordinal, cycle) -> operation group, for commit cells
    chips_at: dict[int, list[str]] = {}  # cycle -> chips for the operations committing on that row

    group = 0  # global per-operation id, linking a commit cell with its edges/chip for the hover-focus behavior
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for op in block.ops:
            color = operator_colors[type(op.inst.operator)]
            # Physical clock cycles (cycle-accurate), from the single Lir definitions shared with reg_liveness. One
            # firing renders one cell per tapped output port (all in one hover group) on EACH arm its writeback reaches,
            # edges from its shared operands, and one pipeline trail. Block-local cycles are rebased to absolute here.
            issue_pc = base_pc + op.issue_cycle
            read_cyc = operand_read_cycle(op.inst.operator, issue_pc, lir.fetch_lag)
            operand_labels = [_operand_label(operand) for operand in op.operands]
            firing_tip = _esc(_op_text(op))
            landing_pcs = lir.write_landing_pcs(block, op)  # op-wide (not per-write); hoisted out of the lane loop
            for write in op.writes:
                tip = _esc(
                    f"{_col_label(write.dst)} 🠄 "
                    + op.inst.operator.render_output(
                        write.port, write.conditioner, *operand_labels, immediates=op.immediates
                    )
                )
                dcol: ColKey = write.dst
                dord = col_ord[dcol]
                # One commit cell PER successor arm the writeback reaches, each with its OWN dataflow edges and
                # ops chip on its landing row -- the report is path-exact, so a spilled result reads the same on every
                # arm it lands in, not just the first. A single-landing (drained) write draws exactly one cell, edge
                # set, and chip, identical to a non-overlapping schedule.
                for write_cyc in landing_pcs:
                    writes_at[(write_cyc, dcol)] = writes_at.get((write_cyc, dcol), "") + _write_label(
                        op.inst.index, tip
                    )
                    fills[(write_cyc, dcol)] = color
                    endpoints.add((dord, write_cyc))
                    cell_group[(dord, write_cyc)] = group
                    for operand in op.operands:
                        oord = col_ord[operand.source]  # the operand's source column; read combinationally at read_cyc
                        endpoints.add((oord, read_cyc))
                        edges.append((f"g{dord}_{write_cyc}", f"g{oord}_{read_cyc}", color, group))
                    chips_at.setdefault(write_cyc, []).append(
                        f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>"
                    )
            base = stage_base[op.inst]
            # A throughput-1 pipeline advances one stage per cycle: stage k is busy only on row issue + k + lag (a
            # diagonal). A non-pipelined FSM core (initiation_interval > 1) holds one transaction and keeps every stage
            # busy across the whole in-flight window (a rectangle over the same rows).
            pipelined = op.inst.operator.initiation_interval == 1
            rows = [issue_pc + i + lir.fetch_lag for i in range(op.latency)]
            for k in range(op.latency):
                for cyc in ([rows[k]] if pipelined else rows):
                    key = (base + k, cyc)
                    if key in stage_fill and stage_fill[key][1] != group:
                        conflicts.add(key)
                    stage_fill[key] = (color, group)
                    stage_tip[key] = f"{op.inst.operator.mnemonic}_{op.inst.index} s{k}: {firing_tip}"
            group += 1

    # Inline firings (boolean logic and the float<->bool casts): single PC-gated statements with no pooled instance,
    # so no pipeline-stage trail. Each renders a colored result cell on its bank's landing cycle, a write marker,
    # dataflow edges from its operands (all read on the op's single fire step per operand_read_cycle; a constant operand
    # -- float or boolean -- anchors its const-pool column), and an ops chip, all in one hover group.
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for bop in block.inline_ops:
            color = operator_colors[type(bop.operator)]
            read_cyc = operand_read_cycle(bop.operator, base_pc + bop.issue_cycle, lir.fetch_lag)
            landing_pcs = lir.write_landing_pcs(block, bop)
            tip = _esc(_inline_op_text(bop))
            dcol = bop.write.dst
            dord = col_ord[dcol]
            # One result cell per successor arm the writeback reaches (overlap spill), each with its own edges and chip
            # on its landing row -- path-exact, exactly as for pooled firings; a single-landing write is byte-identical.
            for write_cyc in landing_pcs:
                writes_at[(write_cyc, dcol)] = (
                    writes_at.get((write_cyc, dcol), "") + f"<span class='wl' title='{tip}'>&#9656;</span>"
                )
                fills[(write_cyc, dcol)] = color
                endpoints.add((dord, write_cyc))
                cell_group[(dord, write_cyc)] = group
                for inline_operand in bop.operands:
                    oord = col_ord[inline_operand.source]
                    endpoints.add((oord, read_cyc))
                    edges.append((f"g{dord}_{write_cyc}", f"g{oord}_{read_cyc}", color, group))
                chips_at.setdefault(write_cyc, []).append(
                    f"<span class='opf' data-op='{group}' style='background:{color}'>{tip}</span>"
                )
            group += 1

    # Installs -- wide phi-arm copies, boolean phi/state writes, and non-coalesced state writebacks -- are first-class
    # write events the schedule must show, not invisible side effects. Each samples its source on its FIRE step and
    # latches its destination on its LANDING row; the caller passes both, computed per kind exactly as the numerical
    # model's decode does, so the marker row matches the residence tint and the model commit. Render each like a state
    # commit -- a filled cell, a write marker, and a chip at the landing row, plus a dataflow edge from the landing to
    # the source's const-pool/register cell on the fire row, so the one-cycle-early write span is visible (a boolean
    # constant source now anchors a ``T``/``F`` column, exactly as a float constant does).
    def install_event(
        dst: ColKey, label: str, source: FloatOperand | BoolOperand, fire_step: int, landing: int
    ) -> None:
        nonlocal group
        dord = col_ord[dst]
        tip = _esc(f"{label} 🠄 {_operand_label(source)}")
        writes_at[(landing, dst)] = writes_at.get((landing, dst), "") + f"<span class='wl' title='{tip}'>&#9662;</span>"
        state_cells.add((landing, dst))
        endpoints.add((dord, landing))
        cell_group[(dord, landing)] = group
        oord = col_ord[source.source]
        endpoints.add((oord, fire_step))
        edges.append((f"g{dord}_{landing}", f"g{oord}_{fire_step}", "state", group))  # JS resolves to --c-state
        chips_at.setdefault(landing, []).append(f"<span class='opf state' data-op='{group}'>{tip}</span>")
        group += 1

    # The landing per install kind mirrors the model's decode: a pc-gated install (a phi copy, a boolean write) always
    # lands at ``install_landing`` (fire + 1); a float slot lands there only if it fires before the boundary, else it is
    # a read-first boundary install at ``last_pc``; a boolean slot always installs read-first at ``last_pc``.
    for block in lir.blocks:
        base_pc = lir.block_base[block.index]
        for copy in block.copies:  # a non-coalesced wide phi-arm merge copy
            fire = base_pc + copy.fire_step(lir.fetch_lag)
            install_event(copy.dst, copy.dst.stable_label, copy.source, fire, install_landing(fire))
        for bwrite in block.bool_writes:  # a boolean phi/state install (a constant or another boolean register)
            fire = base_pc + bwrite.fire_step(lir.fetch_lag)
            install_event(bwrite.dst, bwrite.dst.stable_label, bwrite.source, fire, install_landing(fire))
    for slot in lir.float_state_slots:  # a non-coalesced float slot latches its tap (early, or read-first at boundary)
        if slot.needs_copy:
            fire = lir.state_copy_step(slot)
            landing = fire if lir.float_state_install_is_boundary(slot) else install_landing(fire)
            install_event(slot.reg, slot.name, slot.tap, fire, landing)
    for bslot in lir.bool_state_slots:  # a non-coalesced boolean slot installs its live-out read-first at the boundary
        if bslot.needs_copy:
            install_event(bslot.reg, bslot.name, bslot.live_out, lir.last_pc, lir.last_pc)

    # Control-transfer arrows: anchor each conditional arrow's root to the boolean register it tests by making that
    # register's cell at the source row a dataflow endpoint, so the overlay can draw the dotted feed to it -- the
    # register's residence then visibly ends at the branch rather than in nothingness.
    arrows = _control_arrows(lir)
    for arrow in arrows:
        if arrow.cond is not None:
            endpoints.add((col_ord[arrow.cond], arrow.src_cyc))

    out = [
        _schedule_key(
            operator_colors,
            bool(lir.float_state_slots or lir.bool_state_slots),
            bool(arrows),
            bool(boundary_rows),
            lir.fetch_lag,
        ),
        "<div id='schedwrap'><table class='grid'>",
    ]
    # Header row 0: group bands over the register banks (wide, bool), the constant block and the operator pipelines.
    # clk/operations span all rows. Each band carries the seam of its rightmost data column so the bank edges align.
    out.append("<tr><th class='gh clkh' rowspan='3'><span>clk</span></th>")
    if nreg:
        out.append(
            f"<th class='gband{_border_suffix(nreg - 1, dv.data_thin, dv.data_thick)}' colspan='{nreg}'>"
            "<span title='registers'>registers</span></th>"
        )
    if nbreg:
        out.append(
            f"<th class='gband{_border_suffix(nreg + nbreg - 1, dv.data_thin, dv.data_thick)}' "
            f"colspan='{nbreg}'><span title='bool registers'>bool registers</span></th>"
        )
    if nconst + nbbool:
        const_seam = _border_suffix(nreg + nbreg + nconst + nbbool - 1, dv.data_thin, dv.data_thick)
        out.append(
            f"<th class='gband{const_seam}' colspan='{nconst + nbbool}'>"
            "<span title='constants'>constants</span></th>"
        )
    if n_stage:
        seam = _border_suffix(n_stage - 1, dv.stage_thin, dv.stage_thick)
        out.append(
            f"<th class='gband{seam}' colspan='{n_stage}'>"
            f"<span title='operator pipelines'>operator pipelines</span></th>"
        )
    out.append("<th class='oph' rowspan='3'>operations</th>")
    out.append("<th class='oph pch' rowspan='3'>pc</th></tr>")
    # Header row 1: register and constant column labels, and one group cell per operator spanning its stages. A register
    # that latches an input or retains state is labeled with that name (color-coded) beside its bank index.
    reg_names = _register_names(lir)
    out.append("<tr>")
    for index in range(nreg):
        cls = "gh" + _border_suffix(index, dv.data_thin, dv.data_thick)
        out.append(
            f"<th class='{cls}' rowspan='2'><span>{_named_label('r', index, reg_names.get(('r', index)))}"
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
    for index, value in enumerate(bool_consts):
        cls = "gh k" + _border_suffix(nreg + nbreg + nconst + index, dv.data_thin, dv.data_thick)
        out.append(f"<th class='{cls}' rowspan='2'><span>{'T' if value else 'F'}</span></th>")
    for inst in lir.instances:
        lat = inst.operator.latency
        # full name, set vertically so a 1-stage operator does not widen
        name = f"{inst.operator.instance_stem}_{inst.index}"
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
    # compute, latch, and fetch-staging cycles 1..II. The row axis is the fetch PC, so out_valid (pc == LASTPC == II)
    # rises on the last row; the executing PRESENT step shown there lags the fetch by the fetch lag (present==II-lag).
    in_cells: dict[ColKey, str] = {load.dst: _input_chip(f"in_{load.name}") for load in lir.inputs}
    for slot in lir.float_state_slots:  # at cycle 0 the persistent registers hold their reset snapshot, not an input
        in_cells[slot.reg] = _state_chip(f"{slot.name} = {slot.reset_value!r}")
    for bslot in lir.bool_state_slots:  # boolean persistent slots likewise show their reset snapshot in their bX column
        in_cells[bslot.reg] = _state_chip(f"{bslot.name} = {bslot.reset_value!r}")
    out.append(_bookend_row("in", in_cells, columns, live, 0, n_stage, dv, lir.fetch_lag))

    for cyc in compute_cycles:  # one row per compute cycle; idle cycles show only pipeline advance
        out.append("<tr class='bbk'>" if cyc in boundary_rows else "<tr>")  # thick seam below a block's last row
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
        out.append(_pc_cell(cyc, lir.fetch_lag))
        out.append("</tr>")
    out.append("</table><svg class='edges'></svg></div>")
    out.append(_sched_script(lir, edges, live, arrows, col_ord))
    return "".join(out)


type ColKey = RegRef | BoolRegRef | FloatConstRef | BoolConstRef


def _esc(text: str) -> str:
    return html.escape(text)


def _col_label(col: ColKey) -> str:
    """
    The report's display label for a grid column: ``rX`` for a wide register, ``bX`` for a boolean register, ``cX`` for
    a float constant, and ``T``/``F`` for a boolean constant (which carries no pool index). This is the single labeling
    authority for the schedule: the headers, tooltips, dataflow operand expressions, arrow conditions, and the JS
    payload all route through it.
    """
    if isinstance(col, BoolConstRef):
        return "T" if col.value else "F"
    return col.stable_label


def _op_text(op: PooledScheduledOp) -> str:
    body = op.inst.operator.render(*[_operand_label(o) for o in op.operands], immediates=op.immediates)
    dsts = "/".join(write.conditioner.decorate(_col_label(write.dst)) for write in op.writes)
    return f"{dsts}={body}"


def _is_live(col: ColKey, row_id: int, live: dict[ColKey, set[int]]) -> bool:
    """Whether register column ``col`` holds a live value on grid row ``row_id`` (constants are never tinted)."""
    return isinstance(col, (RegRef, BoolRegRef)) and col in live and row_id in live[col]


def _write_label(index: int, tip: str) -> str:
    """
    The commit-cell marker is just the operator instance index: the overlay's dataflow edges already connect this cell
    to its operand cells, so the cell need not name them; the tooltip carries the full signed expression.
    """
    return f"<span class='wl' title='{tip}'>{index}</span>"


@dataclass(frozen=True, slots=True)
class _Dividers:
    """
    Right-border seams between grid columns, kept on the *left* cell's right edge (under ``border-collapse`` the left
    cell wins an equal-width conflict, so a left border on the right column would not show). A ``thick`` 2px seam marks
    every register-bank boundary (wide|bool, bool|constants) and the two block boundaries (the data block |
    operator-pipeline and operator-pipeline | OPERATIONS); a ``thin`` 1px seam marks the legacy wide|constants seam of a
    boolean-free kernel, the float|bool divide within the constants block, and the seams between operator groups. Data
    and stage columns index separately.
    """

    data_thin: set[int]
    data_thick: set[int]
    stage_thin: set[int]
    stage_thick: set[int]


def _data_seams(nreg: int, nbreg: int, nconst: int, nbbool: int) -> tuple[set[int], set[int]]:
    """
    Right-border seams within the register/constant block, as ``(thin, thick)`` left-cell column-index sets. The
    constant block is the float constants followed by the boolean constants.

    A thick 2px black seam marks each register-bank boundary: wide|bool and bool|constants. With no boolean bank the
    wide bank abuts the constants directly; that legacy seam stays the lighter 1px so a float-only kernel renders
    exactly as before. A thin 1px seam separates the float constants from the boolean constants within the constant
    block. Empty banks contribute no seam. The constants|pipeline block boundary is added by the caller.
    """
    thin: set[int] = set()
    thick: set[int] = set()
    if nbreg:
        if nreg:
            thick.add(nreg - 1)  # wide | bool
        if nconst or nbbool:
            thick.add(nreg + nbreg - 1)  # bool | constants
    elif nreg and (nconst or nbbool):
        thin.add(nreg - 1)  # legacy wide | constants (no boolean bank)
    if nconst and nbbool:
        thin.add(nreg + nbreg + nconst - 1)  # float constants | boolean constants
    return thin, thick


def _border_suffix(idx: int, thin: set[int], thick: set[int]) -> str:
    if idx in thick:
        return " rbk2"
    if idx in thin:
        return " rbk"
    return ""


def _gc_class(ordinal: int, dv: _Dividers) -> str:
    return "gc" + _border_suffix(ordinal, dv.data_thin, dv.data_thick)


def _oc_class(sidx: int, dv: _Dividers) -> str:
    return "oc" + _border_suffix(sidx, dv.stage_thin, dv.stage_thick)


def _operand_label(operand: FloatOperand | BoolOperand) -> str:
    if isinstance(operand, FloatOperand):
        return operand.sign.decorate(_col_label(operand.source))
    source = operand.source
    if isinstance(source, BoolRegRef):
        return operand.inversion.decorate(_col_label(source))
    return repr(source.value)


def _inline_op_text(op: InlineScheduledOp) -> str:
    operands = [_operand_label(o) for o in op.operands]
    body = op.operator.render_output(op.write.port, op.write.conditioner, *operands, immediates=op.immediates)
    return f"{_col_label(op.write.dst)} 🠄 {body}"


def _stage_columns(lir: Lir) -> list[tuple[OperatorInstance, int]]:
    """
    The operator-stage columns, in instance order: ``(instance, stage)`` for each pipeline stage of each operator.

    One column per stage, so an L-cycle operator contributes L columns labeled ``s0..s(L-1)``. As an operation flows
    through its operator, stage ``k`` is occupied on cycle ``issue + k``; the result commits to its register on cycle
    ``issue + L``. This makes the pipeline's advance directly visible and exposes any structural hazard once several
    operations are in flight at once.
    """
    cols: list[tuple[OperatorInstance, int]] = []
    for inst in lir.instances:
        cols.extend((inst, k) for k in range(inst.operator.latency))
    return cols


def _pc_cell(cyc: int, fetch_lag: int) -> str:
    """
    The executing microcode step for grid row ``cyc`` (``clk - fetch_lag``): the ROM address whose control word drives
    this cycle's datapath, and exactly what ``err_pc`` latches. Blank during the fetch warmup, where it is negative. The
    ``pc_<cyc>`` id lets the overlay measure this row's y-centre to route the control-transfer arrows in the margin.
    """
    step = cyc - fetch_lag
    return f"<td class='pc' id='pc_{cyc}'>{step if step >= 0 else ''}</td>"


def _bookend_row(
    label: str,
    cells: dict[ColKey, str],
    columns: list[ColKey],
    live: dict[ColKey, set[int]],
    row_id: int,
    n_stage: int,
    dv: _Dividers,
    fetch_lag: int,
) -> str:
    """Cycle 0: no operator is in flight yet, so the operator-stage cells and ops cell are empty."""
    out = [f"<tr><td class='clk'>{label}</td>"]
    for ordinal, col in enumerate(columns):
        extra = " live" if _is_live(col, row_id, live) else ""
        out.append(f"<td class='{_gc_class(ordinal, dv)}{extra}'>{cells.get(col, '')}</td>")
    for sidx in range(n_stage):
        out.append(f"<td class='{_oc_class(sidx, dv)}'></td>")
    out.append("<td class='opcell'></td>")
    out.append(_pc_cell(row_id, fetch_lag))
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


@dataclass(frozen=True, slots=True)
class _Arrow:
    """
    One control-transfer arrow in the right margin: a non-fall-through jump from grid row ``src_cyc`` into row
    ``dst_cyc``. ``lane`` is the packed right-margin channel. ``tip`` is its hover label -- the branch arm's condition
    (the boolean register the branch evaluates), or ``jump`` if unconditional -- as raw text (rendered via the SVG
    ``<title>``'s textContent, not HTML-escaped; json.dumps makes it JS-safe). ``cond`` is the boolean register the
    branch reads, or ``None`` for an unconditional jump; the overlay draws a dotted feed from that register's cell at
    the source row to the arrow's root, so the register's residence visibly ends at the branch rather than in
    nothingness.
    """

    src_cyc: int
    dst_cyc: int
    tip: str
    cond: BoolRegRef | None
    lane: int = 0


def _control_arrows(lir: Lir) -> list[_Arrow]:
    """
    The control transfers that are not the fall-through to the physically next ROM step, one arrow each: a ``Jump`` to a
    non-adjacent block, and each ``Branch`` arm whose target is not the fall-through (usually one arm falls through and
    the other jumps). The grid row axis is the fetch PC, so the source row is the terminator PC (where the redirect
    mux reads the condition register and the residence of that register ends) and the target row is the destination
    block's base PC (where the model lands after the redirect) -- no offset. An arrow is skipped if either row falls
    outside the grid. A branch arm carries the condition register it reads (for the tooltip and the dotted feed); a jump
    carries ``None``.
    """
    present = lir.initiation_interval
    arrows: list[_Arrow] = []

    def emit(term_pc: int, target: int, tip: str, cond: BoolRegRef | None) -> None:
        src_cyc, dst_cyc = term_pc, lir.block_base[target]
        if 1 <= src_cyc <= present and 1 <= dst_cyc <= present:
            arrows.append(_Arrow(src_cyc, dst_cyc, tip, cond))

    for block in lir.blocks:
        term_pc = lir.term_pc(block)
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
    return _pack_arrow_lanes(arrows)


def _pack_arrow_lanes(arrows: list[_Arrow]) -> list[_Arrow]:
    """Interval-color the arrow row spans; shared endpoints stay separate because both arrows touch that row."""
    lane_ends: list[int] = []
    lanes = [0] * len(arrows)

    def interval(arrow: _Arrow) -> tuple[int, int]:
        return (min(arrow.src_cyc, arrow.dst_cyc), max(arrow.src_cyc, arrow.dst_cyc))

    for index, arrow in sorted(enumerate(arrows), key=lambda item: (*interval(item[1]), item[0])):
        start, end = interval(arrow)
        for lane, busy_until in enumerate(lane_ends):
            if busy_until < start:
                lane_ends[lane] = end
                break
        else:
            lane = len(lane_ends)
            lane_ends.append(end)
        lanes[index] = lane

    return [replace(arrow, lane=lane) for arrow, lane in zip(arrows, lanes, strict=True)]


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
    full column label so the wide and boolean banks never collide on a shared index), and the control-transfer arrows.
    Each arrow carries its packed right-margin ``lane`` and ``cond``: the cell id of the boolean register it reads (so
    the overlay draws a dotted feed from that register to the arrow's root), or ``None`` for an unconditional jump. This
    is enough for the script to draw the dataflow and arrow overlays and synthesize hover tooltips on demand; without JS
    the grid still renders fully.
    """
    cols = [_col_label(col) for col in _columns_of(lir)]
    data = {
        "edges": edges,
        "columns": cols,
        "constants": {f"c{i}": repr(value) for i, value in enumerate(lir.float_consts)}
        | {_col_label(BoolConstRef(value)): repr(value) for value in _bool_consts(lir)},
        "liveness": {_col_label(col): _live_intervals(rows) for col, rows in live.items()},
        "arrows": [
            {
                "from": a.src_cyc,
                "to": a.dst_cyc,
                "lane": a.lane,
                "tip": a.tip,
                "cond": (f"g{col_ord[a.cond]}_{a.src_cyc}" if a.cond is not None else None),
            }
            for a in arrows
        ],
    }
    return "<script>\n" + _SCHED_JS.replace("__DATA__", json.dumps(data)) + "\n</script>"


def _columns_of(lir: Lir) -> list[ColKey]:
    """
    The grid columns in bank order: wide registers (``rX``), then boolean registers (``bX``), then float constants
    (``cX``), then the boolean constants used (``T``/``F``). This is exactly the order the table renders, so a column
    ordinal indexes straight into this list.
    """
    cols: list[ColKey] = [RegRef(i) for i in range(lir.regfile.nreg)]
    cols += [BoolRegRef(i) for i in range(lir.bool_regfile.nreg)]
    cols += [FloatConstRef(i) for i in range(len(lir.float_consts))]
    cols += [BoolConstRef(value) for value in _bool_consts(lir)]
    return cols


def _bool_consts(lir: Lir) -> list[bool]:
    """
    The boolean constant values that appear as an operand or install source, rendered as const-pool columns (``False``
    before ``True``). Booleans have no interned pool (only two values), so the columns are presentation-only -- their
    purpose is to anchor an install edge from the constant into its landing, exactly as the float const pool does, so a
    boolean constant install reads as the one-cycle-early write it is rather than appearing to land late.
    """
    used: set[bool] = set()
    for block in lir.blocks:
        for bop in block.inline_ops:
            for operand in bop.operands:
                if isinstance(operand.source, BoolConstRef):
                    used.add(operand.source.value)
        for bwrite in block.bool_writes:
            if isinstance(bwrite.source.source, BoolConstRef):
                used.add(bwrite.source.source.value)
    for bslot in lir.bool_state_slots:
        if bslot.needs_copy and isinstance(bslot.live_out.source, BoolConstRef):
            used.add(bslot.live_out.source.value)
    return sorted(used)


def _input_chip(tip: str) -> str:
    """A neutral input-write marker for the cycle-0 latch row (``.wr.input`` carries its color)."""
    return f"<span class='wr input' title='{tip}'>&#9662;</span>"


def _state_chip(tip: str) -> str:
    """
    A retained-state latch marker for the cycle-0 row: a persistent slot holding its reset snapshot (``.wr.state``).
    """
    return f"<span class='wr state' title='{tip}'>&#9662;</span>"


def _register_names(lir: Lir) -> dict[tuple[str, int], tuple[str, str]]:
    """
    Map each register that has a stable role to ``(label, kind)``, keyed by ``(bank, index)`` where bank is ``"r"`` for
    a wide register or ``"b"`` for a boolean one: the float input lanes to the port they latch at accept, the float
    state slots to the attribute they retain, and the boolean state slots to the boolean attribute they retain. Other
    registers are anonymous scratch; both kinds may still be reused later, but their cycle-0 role is what the label
    names.
    """
    names: dict[tuple[str, int], tuple[str, str]] = {}
    for fload in lir.float_inputs:
        names[("r", fload.dst.index)] = (f"in_{fload.name}", "input")
    for bload in lir.bool_inputs:
        names[("b", bload.dst.index)] = (f"in_{bload.name}", "input")
    for slot in lir.float_state_slots:
        names[("r", slot.reg.index)] = (slot.name, "state")
    for bslot in lir.bool_state_slots:
        names[("b", bslot.reg.index)] = (bslot.name, "state")
    return names


def _named_label(bank: str, index: int, named: tuple[str, str] | None) -> str:
    base = f"{bank}{index}"
    if named is None:
        return base
    return f"<span class='rn {named[1]}'>{_esc(named[0])}</span> {base}"


def _schedule_key(
    operator_colors: dict[type[HardwareOperator], str],
    has_state: bool,
    has_arrows: bool,
    has_blocks: bool,
    fetch_lag: int,
) -> str:
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
                _key_item(
                    "<span class='sw state'></span>",
                    "state update: live-out lands the step after its copy fires (boundary copy: read-first at LASTPC)",
                ),
            ]
        )
    if has_arrows:
        items.append(
            _key_item(
                f"<span style='color:var(--c-branch-arrow)'>{_ARROW_KEY_MARKER}</span>",
                "conditional control transfer (hover for the condition)",
            )
        )
        items.append(
            _key_item(
                f"<span style='color:var(--c-jump-arrow)'>{_ARROW_KEY_MARKER}</span>",
                "unconditional control transfer",
            )
        )
    if has_blocks:
        items.append(_key_item("<span class='sw bbkey'></span>", "basic-block boundary (terminator row)"))
    items.append(f"<span>pc = microcode step executing this cycle (clk&minus;{fetch_lag} fetch lag)</span>")
    return "<h2>Schedule</h2><div class='gridkey'>" + " ".join(items) + "</div>"


def _key_item(marker: str, text: str) -> str:
    return f"<span>{marker} {text}</span>"


def _operator_colors(lir: Lir) -> dict[type[HardwareOperator], str]:
    # Pooled operators (one color per concrete class) plus the inline operators -- boolean logic and the float<->bool
    # casts -- so every operator the schedule renders has a color.
    kinds: set[type[HardwareOperator]] = {type(inst.operator) for inst in lir.instances}
    for block in lir.blocks:
        for bop in block.inline_ops:
            kinds.add(type(bop.operator))
    ordered = sorted(kinds, key=lambda kind: (kind.mnemonic, kind.__module__, kind.__qualname__))
    return dict(zip(ordered, _html_palette(len(ordered)), strict=True))


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
