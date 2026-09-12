"""
Render a self-contained, light-themed single-page HTML report for a synthesized module.

The stylesheet and the interactive layer live alongside this module as `html.css` and `html.js` (declared as
package data in `pyproject.toml`); they are inlined into the self-contained report so it has no external dependency
beyond the web font.

Do not define any styles or colors here, do that in CSS.
"""

import html
import re
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import assert_never

from ..._lir import (
    BoolBoundaryInstall,
    BoolOperand,
    BoolRegRef,
    Branch,
    InlineWriteSource,
    Lir,
    MoveWriteSource,
    OpWriteSource,
    RegRef,
    WideOperand,
    handshake_arms,
    read_sources_per_port,
    write_arms,
    write_events,
    write_sources_per_register,
)
from ..._operators import PooledHardwareOperator
from ..._legal import output_header
from ..verilog import VerilogOutput
from ._schedule import render_schedule


@dataclass(frozen=True, slots=True)
class HtmlOutput:
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
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Module {_esc(lir.module_name)} - Holoso</title><style>{_CSS}</style></head><body>",
        f"<header><h1>Module {_esc(lir.module_name)}</h1>"
        f"<div class='sub'>{generated} {_esc(output_header())}</div>"
        f"</header><main>",
    ]
    # The sections share one wrapping row; the constants come last because that block wants the full width, which
    # would push every narrower section onto a line of its own.
    out.append("<div class='toprow'>")
    out.append(f"<div class='sec'>{_metrics(lir)}</div>")
    out.append(f"<div class='sec'>{_interface(lir)}</div>")
    out.append(f"<div class='sec widesec'>{_module_header(verilog_output.verilog)}</div>")
    out.append(f"<div class='sec widesec'>{_operator_params(lir)}</div>")
    register_muxes = _register_muxes(lir)
    if register_muxes:
        out.append(f"<div class='sec widesec muxsec'>{register_muxes}</div>")
    constants = _constants(lir)
    if constants:
        out.append(f"<div class='sec'>{constants}</div>")
    out.append("</div>")
    out.append(render_schedule(lir))
    out.append("</main></body></html>")
    return HtmlOutput(html="".join(out))


def _metrics(lir: Lir) -> str:
    fmt = lir.float_format
    read_muxes = sum(_read_muxes_per_register(lir).values())
    write_muxes = sum(write_arms(lir).values())
    op_counts: dict[str, int] = {}
    for inst in lir.instances:
        op_counts[inst.operator.mnemonic] = op_counts.get(inst.operator.mnemonic, 0) + 1
    rows: list[tuple[str, object]] = [
        ("ZKF format", f"e{fmt.wexp}+m{fmt.wman} = {fmt.width}-bit"),
        ("integer format", str(lir.int_format)),
        ("operator instances", " ".join(f"{count}×{kind}" for kind, count in op_counts.items())),
        ("wide registers", f"{lir.regfile.nreg} × {lir.wide_register_width}-bit"),
        ("wide regfile R/W ports", f"{lir.regfile.nrd} / {lir.regfile.nwr}"),
        ("register muxes, both banks", f"{read_muxes} read + {write_muxes} write = {read_muxes + write_muxes}"),
        ("II min [cycles]", lir.initiation_interval),
    ]
    body = "".join(f"<tr><th>{_esc(label)}</th><td>{_esc(str(value))}</td></tr>" for label, value in rows)
    return f"<h2>Metrics</h2><table class='metrics'>{body}</table>"


def _operator_params(lir: Lir) -> str:
    """Most parameters (the float format above all) are shared, so the matrix is far shorter than one row per pair."""
    operators: dict[PooledHardwareOperator, None] = {}  # distinct operators present, in instance order
    for inst in lir.instances:
        operators.setdefault(inst.operator, None)
    names = sorted({name for op in operators for name in op.params})
    if not names:
        return "<h2>Operator Params</h2><table class='metrics cfg'><tr><td>(defaults)</td></tr></table>"
    head = "".join(f"<th class='v'>{_esc(op.mnemonic)}</th>" for op in operators)
    rows = "".join(
        f"<tr><th>{_esc(name)}</th>"
        + "".join(f"<td class='v'>{op.params.get(name, '')}</td>" for op in operators)
        + "</tr>"
        for name in names
    )
    return (
        "<h2>Operator Params</h2><div class='hscroll'><table class='metrics cfg'>"
        f"<tr><th>HDL param</th>{head}</tr>{rows}</table></div>"
    )


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
    if not lir.wide_consts:
        return ""
    chips = "".join(
        f"<span class='const'>c{index} = {_esc(repr(value))}</span>" for index, value in enumerate(lir.wide_consts)
    )
    return f"<h2>Constants</h2><div>{chips}</div>"


@dataclass(frozen=True, slots=True)
class _MuxTable:
    title: str
    read_label: str
    labels: list[str]
    reads: list[int]
    writes: list[int]

    def __post_init__(self) -> None:
        assert len(self.labels) == len(self.reads) == len(self.writes)

    @property
    def nreg(self) -> int:
        return len(self.labels)

    def render(self) -> str:
        assert self.nreg > 0
        head = "".join(
            f"<td class='rl{' bk' if index == self.nreg - 1 else ''}'>{_esc(label)}</td>"
            for index, label in enumerate(self.labels)
        )
        head += "<td class='rl'>total</td><td class='rl'>mean</td><td class='rl'>max</td>"
        rows = "".join(
            _mux_row(label, values, self.nreg)
            for label, values in ((self.read_label, self.reads), ("write mux", self.writes))
        )
        return (
            f"<h3>{_esc(self.title)}</h3><div class='hscroll'><table class='muxtab'>"
            f"<tr><th>register</th>{head}</tr>{rows}</table></div>"
        )


def _mux_row(label: str, values: list[int], nreg: int) -> str:
    cells = "".join(
        f"<td class='{'z' if value == 0 else 'n'}{' bk' if index == nreg - 1 else ''}'>{value}</td>"
        for index, value in enumerate(values)
    )
    total = sum(values)
    return (
        f"<tr><th>{_esc(label)}</th>{cells}"
        f"<td class='n'>{total}</td><td class='n'>{total / nreg:.2f}</td><td class='n'>{max(values)}</td></tr>"
    )


def _read_muxes_per_register(lir: Lir) -> dict[RegRef, int]:
    """Per wide register, how many operand read muxes it is an arm of; a constant arm belongs to no register."""
    muxes: dict[RegRef, int] = {}
    for sources in read_sources_per_port(lir).values():
        for source in sources:
            if isinstance(source, RegRef):
                muxes[source] = muxes.get(source, 0) + 1
    return muxes


def _bool_read_fanin(lir: Lir) -> dict[BoolRegRef, int]:
    """
    No read mux selects a boolean register -- the inline expressions, the moves, the branch conditions, the boundary
    state installs and the output taps all read the bank directly -- so what that bank costs is fan-in: the loads on
    each register's net, one per deduplicated write-select arm reading it plus one per direct reader.
    """
    fanin: dict[BoolRegRef, int] = {}

    def tally(operand: WideOperand | BoolOperand) -> None:
        if isinstance(operand.source, BoolRegRef):
            fanin[operand.source] = fanin.get(operand.source, 0) + 1

    for sources in write_sources_per_register(write_events(lir)).values():
        for source in sources:
            match source:
                case InlineWriteSource(operands=operands):
                    for operand in operands:
                        tally(operand)
                case MoveWriteSource(operand=operand):
                    tally(operand)
                case OpWriteSource():
                    pass
                case _:
                    assert_never(source)
    for arm in handshake_arms(lir).values():
        if isinstance(arm, BoolBoundaryInstall):
            tally(arm.source)
    for block in lir.blocks:
        if isinstance(block.terminator, Branch):
            fanin[block.terminator.cond] = fanin.get(block.terminator.cond, 0) + 1
    for wire in lir.bool_outputs:
        tally(wire.tap)
    return fanin


def _register_muxes(lir: Lir) -> str:
    read_muxes = _read_muxes_per_register(lir)
    fanin = _bool_read_fanin(lir)
    writes = write_arms(lir)
    wide = [RegRef(index) for index in range(lir.regfile.nreg)]
    boolean = [BoolRegRef(index) for index in range(lir.bool_regfile.nreg)]
    tables = [
        _MuxTable(
            title="Wide bank reach",
            read_label="read mux",
            labels=[reg.stable_label for reg in wide],
            reads=[read_muxes.get(reg, 0) for reg in wide],
            writes=[writes.get(reg, 0) for reg in wide],
        ),
        _MuxTable(
            title="Boolean bank reach",
            read_label="read fan-in",
            labels=[reg.stable_label for reg in boolean],
            reads=[fanin.get(reg, 0) for reg in boolean],
            writes=[writes.get(reg, 0) for reg in boolean],
        ),
    ]
    rendered = "".join(table.render() for table in tables if table.nreg)
    return f"<h2>Register Muxes</h2>{rendered}" if rendered else ""
