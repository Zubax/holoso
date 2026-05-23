"""Render a self-contained, light-themed single-page HTML report for a synthesized module."""

from __future__ import annotations

import html
from datetime import datetime

from ..format import FloatFormat
from ..lir import ConstRef, Issue, Lir, Operand, RegRef
from ..operators import OpKind, latency_of
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
/* Below the full-width Metrics + Utilization: schedule (left) beside interface + constants (right). */
.split { display: flex; gap: 28px; align-items: flex-start; }
.pane { flex: 1; min-width: 0; overflow-x: auto; }
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
/* Schedule: one tiny row per clock cycle; faint cycle lines, bold step separators; sizes to its content. */
.sched { border-collapse: collapse; }
.sched th { font-size: 10px; color: #4b5563; font-weight: 700; padding: 0 3px; border-bottom: 1px solid #000; vertical-align: bottom; text-align: left; }
.sched td { padding: 0 7px; height: 13px; font-size: 11px; line-height: 1; white-space: nowrap; border-top: 1px solid #c2c7cf; }
.sched tr.sstart td { border-top: 1px solid #000; }
.clk { color: #9aa1ab; text-align: right; font-size: 10px; border-right: 1px solid #c2c7cf !important; }
.stepn { font-weight: 700; color: #111827; }
.op { display: inline-block; padding: 0 5px; margin: 0 3px 0 0; border-radius: 3px; color: #fff; font-size: 10px; font-weight: 700; }
.lane { width: 14px; padding: 0 !important; border-right: 1px solid #c2c7cf; }
.vname { writing-mode: vertical-rl; font-size: 10px; }
.util { border-collapse: collapse; margin-top: 4px; }
.util td, .util th { border: 1px solid #c2c7cf; padding: 0; text-align: center; }
.util .name { padding: 1px 8px; text-align: left; color: #111827; border: none; font-size: 11px; }
.util .st { writing-mode: vertical-rl; font-size: 9px; color: #6b7280; font-weight: 400; padding: 3px 0; }
.cell { width: 11px; height: 15px; background: #fff; position: relative; }
.fill { position: absolute; bottom: 0; left: 0; right: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 22px; font-size: 12px; color: #374151; margin-top: 8px; align-items: center; }
.legend .box { display: inline-block; width: 12px; height: 15px; border: 1px solid #94a3b8; vertical-align: middle; margin-right: 5px; position: relative; }
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
    # Metrics and the wide utilization table span the full width; below them, the tall schedule sits beside the
    # interface and constants in a two-column split so the page width is not wasted.
    out.append(_metrics(interface, metrics, fmt))
    out.append(_utilization(lir, fmt))
    out.append("<div class='split'>")
    out.append(f"<div class='pane'>{_schedule(lir, fmt)}</div>")
    out.append(f"<div class='pane'>{_constants(lir)}{_interface(interface)}</div>")
    out.append("</div>")
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


def _schedule(lir: Lir, fmt: FloatFormat) -> str:
    if not lir.steps:
        return ""
    out = ["<h2>Schedule</h2><table class='sched'><tr><th class='clk'>clk</th><th>step</th><th>operations</th>"]
    for inst in lir.instances:
        out.append(
            f"<th class='lane'><span class='vname' style='color:{_KIND_COLOR[inst.kind]}'>"
            f"{inst.kind.value}_{inst.index}</span></th>"
        )
    out.append("</tr>")
    cycle = 0
    for step in lir.steps:
        latency = step.latency
        # how many cycles each issued operator is busy within this step (the rest of the step it idles at the barrier)
        busy = {
            (issue.inst.kind, issue.inst.index): min(latency_of(issue.inst.kind, fmt), latency) for issue in step.issues
        }
        ops = "".join(
            f"<span class='op' style='background:{_KIND_COLOR[issue.inst.kind]}'>{_esc(_issue_text(issue))}</span>"
            for issue in step.issues
        )
        for offset in range(latency):  # one row per clock cycle of the step
            out.append("<tr class='sstart'>" if offset == 0 else "<tr>")
            out.append(f"<td class='clk'>{cycle + offset}</td>")
            out.append(f"<td class='stepn'>S{step.index}</td><td>{ops}</td>" if offset == 0 else "<td></td><td></td>")
            for inst in lir.instances:
                cycles = busy.get((inst.kind, inst.index))
                shade = (
                    f" style='background:{_KIND_COLOR[inst.kind]}'" if cycles is not None and offset < cycles else ""
                )
                out.append(f"<td class='lane'{shade}></td>")
            out.append("</tr>")
        cycle += latency
    out.append("</table>")
    return "".join(out)


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
