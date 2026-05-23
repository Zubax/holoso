"""Render a self-contained, colorful single-page HTML report for a synthesized module."""

from __future__ import annotations

import html

from ..format import FloatFormat
from ..lir import ConstRef, Issue, Lir, Operand, RegRef
from ..operators import OpKind, latency_of
from ..result import ModuleInterface, SynthesisMetrics

_KIND_COLOR: dict[OpKind, str] = {
    OpKind.FADD: "#2563eb",
    OpKind.FMUL: "#16a34a",
    OpKind.FDIV: "#dc2626",
    OpKind.FMUL_ILOG2: "#9333ea",
}
_KIND_LABEL: dict[OpKind, str] = {
    OpKind.FADD: "+",
    OpKind.FMUL: "×",
    OpKind.FDIV: "÷",
    OpKind.FMUL_ILOG2: "≪",
}

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
header { padding: 24px 32px; background: linear-gradient(120deg, #6366f1, #ec4899); color: white; }
header h1 { margin: 0 0 4px; font-size: 26px; }
header .sub { opacity: 0.92; font-size: 14px; }
main { padding: 24px 32px; max-width: 1400px; }
h2 { font-size: 16px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin: 28px 0 10px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; }
.card { background: #1e293b; border-radius: 10px; padding: 12px 16px; min-width: 130px; border: 1px solid #334155; }
.card .v { font-size: 22px; font-weight: 700; }
.card .l { font-size: 12px; color: #94a3b8; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid #1e293b; }
th { color: #94a3b8; font-weight: 600; }
.chip { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; color: white; }
.dir-in { background: #16a34a; } .dir-out { background: #2563eb; } .dir-ctrl { background: #64748b; }
.const { background: #334155; color: #fbbf24; margin: 2px; }
.steps { display: flex; flex-direction: column; gap: 3px; }
.step { display: flex; align-items: center; gap: 10px; }
.step .idx { width: 52px; color: #64748b; font-variant-numeric: tabular-nums; font-size: 12px; }
.op { padding: 2px 9px; border-radius: 6px; color: white; font-size: 12px; font-weight: 600; margin-right: 6px; }
.lanes td { border: none; padding: 1px; }
.cell { width: 8px; height: 14px; border-radius: 2px; background: #1e293b; }
.lane-name { font-size: 12px; color: #cbd5e1; padding-right: 10px !important; white-space: nowrap; }
"""


def _esc(text: str) -> str:
    return html.escape(text)


def _sgnop(text: str, sgnop: int) -> str:
    if sgnop == 1:
        return f"-{text}"
    if sgnop == 2:
        return f"|{text}|"
    if sgnop == 3:
        return f"-|{text}|"
    return text


def _operand(operand: Operand) -> str:
    name = f"r{operand.source.index}" if isinstance(operand.source, RegRef) else f"c{operand.source.index}"
    return _sgnop(name, operand.sgnop)


def _issue_text(issue: Issue) -> str:
    if issue.inst.kind is OpKind.FMUL_ILOG2:
        body = f"{_operand(issue.a)}·2^{issue.k}"
    else:
        assert issue.b is not None
        body = f"{_operand(issue.a)} {_KIND_LABEL[issue.inst.kind]} {_operand(issue.b)}"
    return _sgnop(f"r{issue.dst.index} = {body}", issue.y_sgnop)


def _card(value: object, label: str) -> str:
    return f'<div class="card"><div class="v">{_esc(str(value))}</div><div class="l">{_esc(label)}</div></div>'


def build_report_html(lir: Lir, interface: ModuleInterface, metrics: SynthesisMetrics) -> str:
    fmt = lir.fmt
    parts: list[str] = ["<!DOCTYPE html><html><head><meta charset='utf-8'>", f"<style>{_CSS}</style>"]
    parts.append(f"<title>{_esc(lir.module_name)} — Holoso</title></head><body>")
    parts.append(
        f"<header><h1>{_esc(lir.module_name)}</h1>"
        f"<div class='sub'>Zubax Kulibin float WEXP={fmt.wexp} WMAN={fmt.wman} "
        f"(width {fmt.width}) · synthesized by Holoso</div></header><main>"
    )

    parts.append("<h2>Metrics</h2><div class='cards'>")
    insts = " ".join(f"{count}×{kind}" for kind, count in metrics.operator_instances.items())
    parts.append(_card(insts or "—", "operator instances"))
    parts.append(_card(metrics.n_float_regs, "float registers"))
    parts.append(_card(metrics.step_count, "FSM steps"))
    parts.append(_card(f"~{metrics.ii_estimate}", "initiation interval (cycles)"))
    parts.append(_card(metrics.op_count, "operations"))
    parts.append(_card(metrics.max_chain_len, "longest op chain"))
    parts.append("</div>")
    parts.append(f"<p class='l'>II model: {_esc(interface.ii.formula)}</p>")

    parts.append("<h2>Interface</h2><table><tr><th>Port</th><th>Dir</th><th>Width</th></tr>")
    for port in interface.ports:
        parts.append(
            f"<tr><td>{_esc(port.name)}</td>"
            f"<td><span class='chip dir-{port.direction}'>{port.direction}</span></td>"
            f"<td>{port.width}</td></tr>"
        )
    parts.append("</table>")

    if lir.consts:
        parts.append("<h2>Constants</h2><div>")
        for index, value in enumerate(lir.consts):
            parts.append(f"<span class='chip const'>c{index} = {_esc(repr(value))}</span>")
        parts.append("</div>")

    parts.append("<h2>Schedule</h2><div class='steps'>")
    for step in lir.steps:
        parts.append(f"<div class='step'><span class='idx'>S{step.index} ({step.latency}c)</span><span>")
        for issue in step.issues:
            color = _KIND_COLOR[issue.inst.kind]
            parts.append(f"<span class='op' style='background:{color}'>{_esc(_issue_text(issue))}</span>")
        parts.append("</span></div>")
    parts.append("</div>")

    parts.append(_swimlanes(lir, fmt))
    parts.append("</main></body></html>")
    return "".join(parts)


def _swimlanes(lir: Lir, fmt: FloatFormat) -> str:
    if not lir.steps:
        return ""
    active: dict[tuple[OpKind, int], dict[int, float]] = {}
    for step in lir.steps:
        for issue in step.issues:
            key = (issue.inst.kind, issue.inst.index)
            utilization = latency_of(issue.inst.kind, fmt) / max(step.latency, 1)
            active.setdefault(key, {})[step.index] = min(1.0, utilization)
    rows = ["<h2>Operator utilization</h2><table class='lanes'>"]
    for inst in lir.instances:
        key = (inst.kind, inst.index)
        color = _KIND_COLOR[inst.kind]
        cells = []
        for step in lir.steps:
            util = active.get(key, {}).get(step.index)
            style = f"background:{color};opacity:{util:.2f}" if util is not None else ""
            cells.append(f"<td><div class='cell' style='{style}'></div></td>")
        rows.append(f"<tr><td class='lane-name'>{inst.kind.value}_{inst.index}</td>{''.join(cells)}</tr>")
    rows.append("</table>")
    return "".join(rows)
