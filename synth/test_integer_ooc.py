"""
Out-of-context closure of the operators the example matrix cannot isolate: Holoso's integer cores, and the conversions
between the float and integer families. Each is built through Holoso's own operator construction from its options,
with the flow's synthesis attributes, rendered exactly as a kernel renders it, and wrapped in kept registers on every
port, so the paths measured are the operator's own. The integer rows state the port registers they were closed with.
"""

from dataclasses import dataclass, field
from pathlib import Path
import shutil

import pytest
import zkf
from amaranth.hdl import Shape
from amaranth.lib import wiring

from holoso import FFromIntOptions, FRintOptions, FloatFormat, FMulILog2Options, IntFormat, Options, OperatorOptions
from holoso import IAbssOptions, IAddsOptions, IDivsOptions, IMulsOptions, IPopcntOptions, IShftOptions
from holoso._operators import OpConfig
from synth import OocDesign, SourceFile

from synth._ooc import KEEP_ATTR
from synth._synth import BUILD_ROOT
from synth.flows import FlowId, make_flow

pytestmark = pytest.mark.synth

_WMULTIPLIER = 18
_FLOWS = ((FlowId.YOSYS_ECP5, 100.0), (FlowId.DIAMOND_ECP5, 100.0), (FlowId.VIVADO_ARTIX7, 150.0))
_E6M18, _E8M36 = FloatFormat(6, 18), FloatFormat(8, 36)
_FROM_INT = FFromIntOptions.CoreStages


@dataclass(frozen=True, slots=True)
class _Target:
    """One operator, the field of `OperatorOptions` naming it, configured by `operator`, on one flow."""

    kind: str
    label: str
    wint: int
    flow: FlowId
    target_frequency_MHz: float
    operator: OperatorOptions = field(default_factory=OperatorOptions)
    ffmt: FloatFormat = _E6M18
    expect_dsp: bool | None = None  # whether the operator must map to DSP tiles; None if it may either way

    @property
    def name(self) -> str:
        return f"{self.label}-{self.flow.value}-{self.target_frequency_MHz:g}MHz"

    def block(self) -> zkf.Block:
        """The block a kernel on this flow would render for the operator, built as a kernel builds it."""
        flow = make_flow(self.flow, self.target_frequency_MHz)
        options = flow.attributed(Options(operator=self.operator, ffmt=self.ffmt, wmultiplier=_WMULTIPLIER))
        ops = OpConfig(options.operator, options.ffmt, IntFormat(self.wint), options.wmultiplier)
        block: zkf.Block = getattr(ops, self.kind).block
        return block


_TARGETS = (
    *(
        _Target(kind, f"{kind}-w{width}", width, flow, frequency, operator)
        for kind, operator in (
            ("iabss", OperatorOptions(iabss=IAbssOptions(stage_input=1, stage_output=1))),
            ("ishft", OperatorOptions(ishft=IShftOptions(stage_input=1, stage_output=1))),
            ("ipopcnt", OperatorOptions(ipopcnt=IPopcntOptions(stage_input=1, stage_output=1))),
        )
        for width in (24, 44)
        for flow, frequency in _FLOWS
    ),
    *(
        _Target(
            "iadds",
            f"iadds-w{width}-f{int(fast)}",
            width,
            flow,
            frequency,
            OperatorOptions(iadds=IAddsOptions(stage_input=1, stage_output=1, fast=fast)),
        )
        for width in (24, 44)
        for fast in (False, True)
        for flow, frequency in _FLOWS
    ),
    *(
        _Target(
            "imuls",
            f"imuls-w{width}-m{stages}",
            width,
            flow,
            frequency,
            OperatorOptions(imuls=IMulsOptions(stage_input=1, stage_core_mul=stages, stage_output=1)),
            expect_dsp=True,
        )
        for width, flow, frequency, stages in (
            (24, FlowId.YOSYS_ECP5, 100.0, 3),
            # The grid's products registered: a multiplier feeding the sum of the grid directly missed 100 MHz.
            (24, FlowId.DIAMOND_ECP5, 100.0, 3),
            (24, FlowId.VIVADO_ARTIX7, 150.0, 1),
            (44, FlowId.YOSYS_ECP5, 100.0, 4),
            (44, FlowId.DIAMOND_ECP5, 100.0, 4),
            (44, FlowId.VIVADO_ARTIX7, 150.0, 4),
        )
    ),
    *(
        _Target(
            "idivs",
            f"idivs-w{width}",
            width,
            flow,
            frequency,
            OperatorOptions(idivs=IDivsOptions(stage_input=1, stage_output=1)),
            expect_dsp=False,
        )
        for width in (24, 44)
        for flow, frequency in _FLOWS
    ),
    *(
        _Target(
            "ffromint",
            f"ffromint-e{fmt.wexp}m{fmt.wman}-i{wint}",
            wint,
            flow,
            frequency,
            OperatorOptions(ffromint=FFromIntOptions(stage_input=0, stage_core=core, stage_output=output)),
            fmt,
        )
        for fmt, wint, flow, frequency, core, output in (
            (_E6M18, 24, FlowId.YOSYS_ECP5, 100.0, _FROM_INT.NORMALIZE, 0),
            (_E6M18, 24, FlowId.DIAMOND_ECP5, 100.0, _FROM_INT.NORMALIZE, 0),
            (_E6M18, 24, FlowId.VIVADO_ARTIX7, 150.0, _FROM_INT.NONE, 1),
            # The 44-bit magnitude's carry chain into the count of its leading zeros is split at the magnitude.
            (_E8M36, 44, FlowId.YOSYS_ECP5, 100.0, _FROM_INT.ABS | _FROM_INT.NORMALIZE, 0),
            (_E8M36, 44, FlowId.DIAMOND_ECP5, 100.0, _FROM_INT.ABS | _FROM_INT.NORMALIZE, 0),
            (_E8M36, 44, FlowId.VIVADO_ARTIX7, 150.0, _FROM_INT.ABS, 1),
        )
    ),
    *(
        _Target(
            "frint",
            f"frint-e{fmt.wexp}m{fmt.wman}-i{wint}",
            wint,
            flow,
            frequency,
            OperatorOptions(frint=FRintOptions(stage_input=0, stage_core=FRintOptions.CoreStages.MASK, stage_output=0)),
            fmt,
        )
        for fmt, wint in ((_E6M18, 24), (_E8M36, 44))
        for flow, frequency in _FLOWS
    ),
    *(
        _Target(
            "fmul_ilog2",
            f"fmul_ilog2-e{fmt.wexp}m{fmt.wman}-i{wint}",
            wint,
            flow,
            frequency,
            OperatorOptions(
                fmul_ilog2=FMulILog2Options(stage_input=0, stage_core=FMulILog2Options.CoreStages.NONE, stage_output=1)
            ),
            fmt,
            expect_dsp=False,
        )
        for fmt, wint in ((_E6M18, 24), (_E8M36, 44))
        for flow, frequency in _FLOWS
    ),
)


def _width(member: wiring.Member) -> int:
    return int(Shape.cast(member.shape).width)


def _wrapper(top: str, module: str, block: zkf.Block) -> str:
    """
    Kept registers on every port of the operator: each input loads from the one input bus by `in_sel`, and each output
    is captured and brought out through a registered selection, so nothing is pruned and every path is timed.
    """
    members = block.signature.members
    inputs = [(name, _width(member)) for name, member in members.items() if member.flow == wiring.In]
    outputs = [(name, _width(member)) for name, member in members.items() if member.flow == wiring.Out]
    wio = max(width for _, width in inputs + outputs)
    wsel_in = max(1, (len(inputs) - 1).bit_length())
    wsel_out = max(1, (len(outputs) - 1).bit_length())
    lines = [
        "`default_nettype none",
        "",
        f"module {top} (",
        "    input  wire clk,",
        "    input  wire rst,",
        f"    input  wire [{wsel_in - 1}:0] in_sel,",
        f"    input  wire [{wio - 1}:0] io_in,",
        f"    input  wire [{wsel_out - 1}:0] out_sel,",
        f"    output reg  [{wio - 1}:0] io_out",
        ");",
    ]
    lines += [f"    {KEEP_ATTR} reg [{width - 1}:0] r_{name};" for name, width in inputs]
    lines += [f"    wire [{width - 1}:0] y_{name};" for name, width in outputs]
    lines += [f"    {KEEP_ATTR} reg [{width - 1}:0] r_{name};" for name, width in outputs]
    ports = ([".clk(clk)", ".rst(rst)"] if block.clocked else []) + [f".{name}(r_{name})" for name, _ in inputs]
    ports += [f".{name}(y_{name})" for name, _ in outputs]
    lines += ["", f"    {module} dut (", *(f"        {port}," for port in ports[:-1]), f"        {ports[-1]}", "    );"]
    lines += ["", "    always @(posedge clk) begin", "        case (in_sel)"]
    lines += [f"            {index}: r_{name} <= io_in[{width - 1}:0];" for index, (name, width) in enumerate(inputs)]
    lines += ["            default: ;", "        endcase"]
    lines += [f"        r_{name} <= y_{name};" for name, _ in outputs]
    lines += ["        case (out_sel)"]
    lines += [f"            {index}: io_out <= r_{name};" for index, (name, _) in enumerate(outputs)]
    lines += [
        "            default: io_out <= 0;",
        "        endcase",
        "    end",
        "endmodule",
        "",
        "`default_nettype wire",
        "",
    ]
    return "\n".join(lines)


def _design(target: _Target) -> OocDesign:
    top = f"{target.label.replace('-', '_')}_ooc"
    prefix = f"_{top}_"
    block = target.block()
    files = [
        SourceFile(Path(f"{top}_dut.v"), block.compile(prefix=prefix)),
        SourceFile(Path(f"{top}.v"), _wrapper(top, prefix + block.name, block)),
    ]
    return OocDesign(top=top, files=files)


@pytest.mark.parametrize("target", _TARGETS, ids=lambda target: target.name)
def test_operator_closes_timing(target: _Target) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.name
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.name}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    if target.expect_dsp is not None:
        dsp_used = sum(
            use.used for name, use in report.resources.items() if "DSP" in name.upper() or "MULT" in name.upper()
        )
        assert (
            dsp_used > 0
        ) == target.expect_dsp, f"{target.name}: {dsp_used} DSP tiles; logs in {report.artifact_dir}"
