"""
Render a self-contained, light-themed single-page HTML report for a synthesized module.

The stylesheet and the interactive layer live alongside this module as ``html.css`` and ``html.js`` (declared as
package data in ``pyproject.toml``); they are inlined into the self-contained report so it has no external dependency
beyond the web font.

Do not define any styles or colors here, do that in CSS.
"""

import html
import re
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

from ..._lir import Lir
from ..._operators import PooledHardwareOperator
from ..verilog import VerilogOutput
from ._schedule import render_schedule


@dataclass(frozen=True, slots=True)
class HtmlOutput:
    """One self-contained, single-page report document, with resources built-in."""

    html: str

    def __str__(self) -> str:
        return f"{type(self).__name__}(html_bytes={len(self.html.encode())})"


_CSS = resources.files(__package__).joinpath("html.css").read_text(encoding="utf-8")

_MODULE_HEADER_RE = re.compile(r"(?ms)^module\b.*?^\);")
_VERILOG_TOKEN_RE = re.compile(r"(?P<space>\s+)|(?P<ident>[A-Za-z_]\w*)|(?P<number>\d+)|(?P<other>.)")
_VERILOG_KEYWORDS = frozenset({"module", "parameter", "input", "output", "wire", "reg"})


def _esc(text: str) -> str:
    return html.escape(text)


def generate(lir: Lir, verilog_output: VerilogOutput) -> HtmlOutput:
    from holoso import __url__, __version__

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Module {_esc(lir.module_name)} - Holoso</title><style>{_CSS}</style></head><body>",
        f"<header><h1>Module {_esc(lir.module_name)}</h1>"
        f"<div class='sub'>Synthesized by <a href='{__url__}'>Holoso</a> v{__version__} at"
        f" {generated}</div></header><main>",
    ]
    # The compact summary sections share one wrapping row (metrics, then the narrow constants and interface) so they
    # do not waste page height; the wide register-grid schedule follows below.
    out.append("<div class='toprow'>")
    out.append(f"<div class='sec'>{_metrics(lir)}</div>")
    out.append(f"<div class='sec'>{_stage_config(lir)}</div>")
    constants = _constants(lir)
    if constants:
        out.append(f"<div class='sec'>{constants}</div>")
    out.append(f"<div class='sec'>{_interface(lir)}</div>")
    out.append(f"<div class='sec modhdrsec'>{_module_header(verilog_output.verilog)}</div>")
    out.append("</div>")
    out.append(render_schedule(lir))
    out.append("</main></body></html>")
    return HtmlOutput(html="".join(out))


def _metrics(lir: Lir) -> str:
    fmt = lir.float_format
    op_counts: dict[str, int] = {}
    for inst in lir.instances:
        op_counts[inst.operator.mnemonic] = op_counts.get(inst.operator.mnemonic, 0) + 1
    rows: list[tuple[str, object]] = [
        ("ZKF format", f"e{fmt.wexp}+m{fmt.wman} = {fmt.width}-bit"),
        ("operator instances", " ".join(f"{count}×{kind}" for kind, count in op_counts.items())),
        ("registers", lir.regfile.nreg),
        ("regfile R/W ports", f"{lir.regfile.nrd} / {lir.regfile.nwr}"),
        ("II min [cycles]", lir.initiation_interval),
    ]
    body = "".join(f"<tr><th>{_esc(label)}</th><td>{_esc(str(value))}</td></tr>" for label, value in rows)
    return f"<h2>Metrics</h2><table class='metrics'>{body}</table>"


def _stage_config(lir: Lir) -> str:
    out = [
        "<h2>Operator Params</h2><table class='metrics cfg'>",
        "<tr><th>operator</th><th>HDL param</th><th>value</th></tr>",
    ]
    seen: dict[PooledHardwareOperator, None] = {}  # distinct operators present, in instance order
    for inst in lir.instances:
        seen.setdefault(inst.operator, None)
    rows = 0
    for op in seen:
        for param, value in op.hdl_params().items():
            out.append(f"<tr><td>{_esc(op.instance_stem)}</td><td>{_esc(param)}</td><td>{value}</td></tr>")
            rows += 1
    if rows == 0:
        out.append("<tr><td colspan='3'>(defaults)</td></tr>")
    out.append("</table>")
    return "".join(out)


def _interface(lir: Lir) -> str:
    out = ["<h2>Interface</h2><div class='ifaces'>"]
    ctrl = lir.control_ports
    out.append(f"<div class='iface'><h3>ctrl ({len(ctrl)})</h3><table><tr><th>port</th><th>dir</th><th>bits</th></tr>")
    for control_port in ctrl:
        out.append(
            f"<tr><td>{_esc(control_port.name)}</td><td>{control_port.direction}</td>"
            f"<td>{control_port.width}</td></tr>"
        )
    out.append("</table></div>")
    for title, ports in (("in", lir.input_ports), ("out", lir.output_ports)):
        out.append(f"<div class='iface'><h3>{title} ({len(ports)})</h3><table><tr><th>port</th><th>bits</th></tr>")
        for data_port in ports:
            out.append(f"<tr><td>{_esc(data_port.name)}</td><td>{data_port.width}</td></tr>")
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
    if not lir.float_consts:
        return ""
    chips = "".join(
        f"<span class='const'>c{index} = {_esc(repr(value))}</span>" for index, value in enumerate(lir.float_consts)
    )
    return f"<h2>Constants</h2><div>{chips}</div>"
