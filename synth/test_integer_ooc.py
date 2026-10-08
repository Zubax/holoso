from dataclasses import dataclass
from pathlib import Path
import shutil

import pytest
from holoso import FFromIntOptions, FRintOptions, FloatFormat, IPopcntOptions, IntFormat
from holoso._backend.verilog._support import support_files
from holoso._operators import FFromIntOperator, FRintOperator, HardwareOperator, IPopcntOperator
from synth import OocDesign, SourceFile

from synth._ooc import KEEP_ATTR
from synth._synth import BUILD_ROOT
from synth.flows import FlowId, make_flow

_SATURATING = ("holoso_iadds", "holoso_isubs", "holoso_iabss")


@dataclass(frozen=True, slots=True)
class _Target:
    operator: str
    width: int
    flow: FlowId
    target_frequency_MHz: float

    @property
    def label(self) -> str:
        return f"{self.operator}-w{self.width}-{self.flow.value}-{self.target_frequency_MHz:g}MHz"


@dataclass(frozen=True, slots=True)
class _MultiplierTarget:
    width: int
    flow: FlowId
    target_frequency_MHz: float
    stage_product: int
    latency: int

    def __post_init__(self) -> None:
        assert self.latency == 2 + self.stage_product

    @property
    def label(self) -> str:
        return f"holoso_imuls-w{self.width}-s{self.stage_product}-{self.flow.value}-{self.target_frequency_MHz:g}MHz"


@dataclass(frozen=True, slots=True)
class _DividerTarget:
    width: int
    quotient_floor: int
    flow: FlowId
    target_frequency_MHz: float
    latency: int

    def __post_init__(self) -> None:
        assert self.quotient_floor in (0, 1)
        assert self.latency == 3 + (self.width + 1) // 2

    @property
    def label(self) -> str:
        return (
            f"holoso_idivs-w{self.width}-f{self.quotient_floor}-" f"{self.flow.value}-{self.target_frequency_MHz:g}MHz"
        )


_TARGETS = tuple(
    _Target(operator, width, flow, frequency)
    for operator in (*_SATURATING, "holoso_icmp", "holoso_ishft", "holoso_ipopcnt")
    for width in (24, 44)
    for flow, frequency in (
        (FlowId.YOSYS_ECP5, 100.0),
        (FlowId.DIAMOND_ECP5, 100.0),
        (FlowId.VIVADO_ARTIX7, 150.0),
    )
)

_MULTIPLIER_TARGETS = (
    _MultiplierTarget(24, FlowId.YOSYS_ECP5, 100.0, stage_product=3, latency=5),
    _MultiplierTarget(24, FlowId.DIAMOND_ECP5, 100.0, stage_product=0, latency=2),
    _MultiplierTarget(24, FlowId.VIVADO_ARTIX7, 150.0, stage_product=1, latency=3),
    _MultiplierTarget(44, FlowId.YOSYS_ECP5, 100.0, stage_product=4, latency=6),
    _MultiplierTarget(44, FlowId.DIAMOND_ECP5, 100.0, stage_product=4, latency=6),
    _MultiplierTarget(44, FlowId.VIVADO_ARTIX7, 150.0, stage_product=4, latency=6),
)

_DIVIDER_TARGETS = (
    _DividerTarget(24, 0, FlowId.YOSYS_ECP5, 100.0, latency=15),
    _DividerTarget(24, 0, FlowId.DIAMOND_ECP5, 100.0, latency=15),
    _DividerTarget(24, 0, FlowId.VIVADO_ARTIX7, 150.0, latency=15),
    _DividerTarget(24, 1, FlowId.YOSYS_ECP5, 100.0, latency=15),
    _DividerTarget(24, 1, FlowId.DIAMOND_ECP5, 100.0, latency=15),
    _DividerTarget(24, 1, FlowId.VIVADO_ARTIX7, 150.0, latency=15),
    _DividerTarget(44, 0, FlowId.YOSYS_ECP5, 100.0, latency=25),
    _DividerTarget(44, 0, FlowId.DIAMOND_ECP5, 100.0, latency=25),
    _DividerTarget(44, 0, FlowId.VIVADO_ARTIX7, 150.0, latency=25),
    _DividerTarget(44, 1, FlowId.YOSYS_ECP5, 100.0, latency=25),
    _DividerTarget(44, 1, FlowId.DIAMOND_ECP5, 100.0, latency=25),
    _DividerTarget(44, 1, FlowId.VIVADO_ARTIX7, 150.0, latency=25),
)


@dataclass(frozen=True, slots=True)
class _FromIntTarget:
    wexp: int
    wman: int
    wint: int
    flow: FlowId
    target_frequency_MHz: float
    stage_input: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        assert self.wexp >= 2
        assert self.wman >= 4
        assert self.wint >= 2

    @property
    def _hardware_operator(self) -> HardwareOperator:
        return FFromIntOperator.build(
            FloatFormat(self.wexp, self.wman),
            IntFormat(self.wint),
            FFromIntOptions(
                stage_input=self.stage_input,
                stage_normalize=self.stage_normalize,
                stage_pack=self.stage_pack,
                stage_output=self.stage_output,
            ),
        )

    @property
    def operator(self) -> str:
        return self._hardware_operator.module_name

    @property
    def latency(self) -> int:
        return self._hardware_operator.latencies[0]

    @property
    def label(self) -> str:
        return (
            f"{self.operator}-e{self.wexp}m{self.wman}-i{self.wint}-"
            f"i{self.stage_input}n{self.stage_normalize}p{self.stage_pack}o{self.stage_output}-"
            f"{self.flow.value}-{self.target_frequency_MHz:g}MHz"
        )


@dataclass(frozen=True, slots=True)
class _RintTarget:
    wexp: int
    wman: int
    wint: int
    flow: FlowId
    target_frequency_MHz: float
    stage_input: int = 0
    stage_shift: int = 1
    stage_round: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        assert self.wexp >= 2
        assert self.wman >= 4
        assert self.wint >= 2

    @property
    def _hardware_operator(self) -> HardwareOperator:
        return FRintOperator.build(
            FloatFormat(self.wexp, self.wman),
            IntFormat(self.wint),
            FRintOptions(
                stage_input=self.stage_input,
                stage_shift=self.stage_shift,
                stage_round=self.stage_round,
                stage_output=self.stage_output,
            ),
        )

    @property
    def operator(self) -> str:
        return self._hardware_operator.module_name

    @property
    def latency(self) -> int:
        return self._hardware_operator.latencies[0]

    @property
    def label(self) -> str:
        return (
            f"{self.operator}-e{self.wexp}m{self.wman}-i{self.wint}-"
            f"i{self.stage_input}s{self.stage_shift}r{self.stage_round}o{self.stage_output}-"
            f"{self.flow.value}-{self.target_frequency_MHz:g}MHz"
        )


@dataclass(frozen=True, slots=True)
class _MulILog2Target:
    wexp: int
    wman: int
    wint: int
    flow: FlowId
    target_frequency_MHz: float
    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        assert self.wexp >= 2
        assert self.wman >= 4
        assert self.wint >= 2

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_decode

    @property
    def label(self) -> str:
        return (
            f"holoso_fmul_ilog2-e{self.wexp}m{self.wman}-i{self.wint}-i{self.stage_input}d{self.stage_decode}-"
            f"{self.flow.value}-{self.target_frequency_MHz:g}MHz"
        )


_FROM_INT_TARGETS = (
    *(
        _FromIntTarget(wexp, wman, wint, flow, frequency, stage_normalize=1)
        for wexp, wman, wint in ((6, 18, 24), (8, 36, 44))
        for flow, frequency in ((FlowId.YOSYS_ECP5, 100.0), (FlowId.DIAMOND_ECP5, 100.0))
    ),
    *(_FromIntTarget(wexp, wman, wint, FlowId.VIVADO_ARTIX7, 150.0) for wexp, wman, wint in ((6, 18, 24), (8, 36, 44))),
)

_RINT_TARGETS = (
    *(
        _RintTarget(6, 18, 24, flow, frequency)
        for flow, frequency in (
            (FlowId.YOSYS_ECP5, 100.0),
            (FlowId.DIAMOND_ECP5, 100.0),
            (FlowId.VIVADO_ARTIX7, 150.0),
        )
    ),
    _RintTarget(8, 36, 44, FlowId.YOSYS_ECP5, 100.0),
    _RintTarget(8, 36, 44, FlowId.DIAMOND_ECP5, 100.0),
    _RintTarget(8, 36, 44, FlowId.VIVADO_ARTIX7, 150.0),
)

_MUL_ILOG2_TARGETS = (
    *(
        _MulILog2Target(6, 18, 24, flow, frequency)
        for flow, frequency in (
            (FlowId.YOSYS_ECP5, 100.0),
            (FlowId.DIAMOND_ECP5, 100.0),
            (FlowId.VIVADO_ARTIX7, 150.0),
        )
    ),
    _MulILog2Target(8, 36, 44, FlowId.YOSYS_ECP5, 100.0),
    _MulILog2Target(8, 36, 44, FlowId.DIAMOND_ECP5, 100.0, stage_decode=1),
    _MulILog2Target(8, 36, 44, FlowId.VIVADO_ARTIX7, 150.0),
)

pytestmark = pytest.mark.synth


def _build_ooc_design(operator: str, width: int) -> OocDesign:
    top = f"{operator}_w{width}_ooc"
    if operator == "holoso_icmp":
        wrapper = _render_cmp_wrapper(top, width)
    elif operator == "holoso_ishft":
        wrapper = _render_shift_wrapper(top, width)
    elif operator == "holoso_ipopcnt":
        wrapper = _render_popcnt_wrapper(top, width)
    elif operator == "holoso_iabss":
        wrapper = _render_abs_wrapper(top, width)
    else:
        wrapper = _render_saturating_wrapper(top, operator, width)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _build_multiplier_ooc_design(target: _MultiplierTarget) -> OocDesign:
    top = f"holoso_imuls_w{target.width}_s{target.stage_product}_ooc"
    parameters = f".W({target.width}), .STAGE_PRODUCT({target.stage_product}), .LATENCY({target.latency})"
    wrapper = _render_saturating_wrapper(top, "holoso_imuls", target.width, parameters)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _build_divider_ooc_design(target: _DividerTarget) -> OocDesign:
    top = f"holoso_idivs_w{target.width}_f{target.quotient_floor}_ooc"
    wrapper = _render_divider_wrapper(top, target.width, target.latency, target.quotient_floor)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _build_ffromint_ooc_design(target: _FromIntTarget) -> OocDesign:
    top = f"{target.operator.removeprefix('holoso_')}_e{target.wexp}m{target.wman}_i{target.wint}_ooc"
    wrapper = _render_ffromint_wrapper(top, target)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _build_frint_ooc_design(target: _RintTarget) -> OocDesign:
    top = f"{target.operator.removeprefix('holoso_')}_e{target.wexp}m{target.wman}_i{target.wint}_ooc"
    wrapper = _render_frint_wrapper(top, target)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _build_fmul_ilog2_ooc_design(target: _MulILog2Target) -> OocDesign:
    top = f"fmul_ilog2_e{target.wexp}m{target.wman}_i{target.wint}_ooc"
    wrapper = _render_fmul_ilog2_wrapper(top, target)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _render_ffromint_wrapper(top: str, target: _FromIntTarget) -> str:
    wfull = target.wexp + target.wman
    wio = max(wfull, target.wint)
    zero_padding = f"{{{wio - wfull}{{1'b0}}}}"
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [{wio - 1}:0] io_in,
    output wire out_valid,
    output wire [{wio - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg signed [{target.wint - 1}:0] r_a;
    wire dut_out_valid;
    wire [{wfull - 1}:0] dut_y;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign io_out = {{{zero_padding}, r_y}};

    holoso_ffromint#(
        .WEXP({target.wexp}), .WMAN({target.wman}), .WINT({target.wint}),
        .STAGE_INPUT({target.stage_input}), .STAGE_NORMALIZE({target.stage_normalize}),
        .STAGE_PACK({target.stage_pack}), .STAGE_OUTPUT({target.stage_output}), .LATENCY({target.latency})
    ) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a(r_a),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        r_a <= io_in[{target.wint - 1}:0];
        r_y <= dut_y;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_frint_wrapper(top: str, target: _RintTarget) -> str:
    """Both results are registered and brought out, so neither tap's logic is pruned and both are timed."""
    wfull = target.wexp + target.wman
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [1:0] in_sel,
    input  wire [{wfull - 1}:0] io_in,
    output wire out_valid,
    output wire [{wfull + target.wint - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{wfull - 1}:0] r_a;
    {KEEP_ATTR} reg [1:0] r_a_sgnop;
    {KEEP_ATTR} reg [1:0] r_round_mode;
    wire dut_out_valid;
    wire [{wfull - 1}:0] dut_y_float;
    wire signed [{target.wint - 1}:0] dut_y_int;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg [{wfull - 1}:0] r_y_float;
    {KEEP_ATTR} reg signed [{target.wint - 1}:0] r_y_int;

    assign out_valid = r_out_valid;
    assign io_out = {{r_y_float, r_y_int}};

    holoso_frint#(
        .WEXP({target.wexp}), .WMAN({target.wman}), .WINT({target.wint}),
        .STAGE_INPUT({target.stage_input}), .STAGE_SHIFT({target.stage_shift}),
        .STAGE_ROUND({target.stage_round}), .STAGE_OUTPUT({target.stage_output}), .LATENCY({target.latency})
    ) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a_sgnop(r_a_sgnop), .round_mode(r_round_mode), .a(r_a),
        .out_valid(dut_out_valid), .y_float(dut_y_float), .y_int(dut_y_int)
    );

    always @(posedge clk) begin
        case (in_sel)
            2'd0: r_a <= io_in;
            2'd1: r_a_sgnop <= io_in[1:0];
            2'd2: r_round_mode <= io_in[1:0];
            default: ;
        endcase
        r_y_float <= dut_y_float;
        r_y_int <= dut_y_int;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_fmul_ilog2_wrapper(top: str, target: _MulILog2Target) -> str:
    wfull = target.wexp + target.wman
    wio = max(wfull, target.wint)
    zero_padding = f"{{{wio - wfull}{{1'b0}}}}"
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [1:0] in_sel,
    input  wire [{wio - 1}:0] io_in,
    output wire out_valid,
    output wire [{wio - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{wfull - 1}:0] r_a;
    {KEEP_ATTR} reg signed [{target.wint - 1}:0] r_k;
    {KEEP_ATTR} reg [1:0] r_a_sgnop;
    wire dut_out_valid;
    wire [{wfull - 1}:0] dut_y;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign io_out = {{{zero_padding}, r_y}};

    holoso_fmul_ilog2#(
        .WEXP({target.wexp}), .WMAN({target.wman}), .WINT({target.wint}),
        .STAGE_INPUT({target.stage_input}), .STAGE_DECODE({target.stage_decode}), .LATENCY({target.latency})
    ) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a_sgnop(r_a_sgnop), .a(r_a), .k(r_k),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        case (in_sel)
            2'd0: r_a <= io_in[{wfull - 1}:0];
            2'd1: r_k <= io_in[{target.wint - 1}:0];
            2'd2: r_a_sgnop <= io_in[1:0];
            default: ;
        endcase
        r_y <= dut_y;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_divider_wrapper(top: str, width: int, latency: int, quotient_floor: int) -> str:
    assert quotient_floor in (0, 1)
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire in_sel,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    input  wire [1:0] out_sel,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_num;
    {KEEP_ATTR} reg [{width - 1}:0] r_den;
    wire dut_out_valid;
    wire [{width - 1}:0] dut_quo;
    wire [{width - 1}:0] dut_rem;
    wire dut_saturated;
    wire dut_div0;
    {KEEP_ATTR} reg [{width - 1}:0] r_quo;
    {KEEP_ATTR} reg [{width - 1}:0] r_rem;
    {KEEP_ATTR} reg r_saturated;
    {KEEP_ATTR} reg r_div0;
    {KEEP_ATTR} reg r_dut_valid;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_io_out;

    assign out_valid = r_out_valid;
    assign io_out = r_io_out;

    holoso_idivs#(.W({width}), .QUOTIENT_FLOOR({quotient_floor}), .LATENCY({latency})) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .num(r_num), .den(r_den),
        .out_valid(dut_out_valid), .quo(dut_quo), .rem(dut_rem), .saturated(dut_saturated), .div0(dut_div0)
    );

    always @(posedge clk) begin
        if (in_sel) r_den <= io_in;
        else        r_num <= io_in;
        r_quo <= dut_quo;
        r_rem <= dut_rem;
        r_saturated <= dut_saturated;
        r_div0 <= dut_div0;
        case (out_sel)
            2'd0: r_io_out <= r_quo;
            2'd1: r_io_out <= r_rem;
            2'd2: r_io_out <= {{{width}{{r_saturated}}}};
            default: r_io_out <= {{{width}{{r_div0}}}};
        endcase
        if (rst) begin
            r_in_valid <= 1'b0;
            r_dut_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_dut_valid <= dut_out_valid;
            r_out_valid <= r_dut_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_saturating_wrapper(top: str, operator: str, width: int, parameters: str | None = None) -> str:
    parameters = parameters or f".W({width}), .LATENCY(2)"
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire in_sel,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    input  wire out_sel,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_a;
    {KEEP_ATTR} reg [{width - 1}:0] r_b;
    wire dut_out_valid;
    wire [{width - 1}:0] dut_y;
    wire dut_saturated;
    {KEEP_ATTR} reg [{width - 1}:0] r_y;
    {KEEP_ATTR} reg r_saturated;
    {KEEP_ATTR} reg r_out_valid;

    assign out_valid = r_out_valid;
    assign io_out = out_sel ? {{{width}{{r_saturated}}}} : r_y;

    {operator}#({parameters}) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a(r_a), .b(r_b),
        .out_valid(dut_out_valid), .y(dut_y), .saturated(dut_saturated)
    );

    always @(posedge clk) begin
        if (in_sel) r_b <= io_in;
        else        r_a <= io_in;
        r_y <= dut_y;
        r_saturated <= dut_saturated;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_abs_wrapper(top: str, width: int) -> str:
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    input  wire out_sel,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_a;
    wire dut_out_valid;
    wire [{width - 1}:0] dut_y;
    wire dut_saturated;
    {KEEP_ATTR} reg [{width - 1}:0] r_y;
    {KEEP_ATTR} reg r_saturated;
    {KEEP_ATTR} reg r_out_valid;

    assign out_valid = r_out_valid;
    assign io_out = out_sel ? {{{width}{{r_saturated}}}} : r_y;

    holoso_iabss#(.W({width}), .LATENCY(2)) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .x(r_a),
        .out_valid(dut_out_valid), .y(dut_y), .saturated(dut_saturated)
    );

    always @(posedge clk) begin
        r_a <= io_in;
        r_y <= dut_y;
        r_saturated <= dut_saturated;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_cmp_wrapper(top: str, width: int) -> str:
    zero_padding = f"{{{width - 1}{{1'b0}}}}"
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire in_sel,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    input  wire [1:0] out_sel,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_a;
    {KEEP_ATTR} reg [{width - 1}:0] r_b;
    wire dut_out_valid;
    wire dut_a_gt_b;
    wire dut_a_eq_b;
    wire dut_a_lt_b;
    {KEEP_ATTR} reg r_a_gt_b;
    {KEEP_ATTR} reg r_a_eq_b;
    {KEEP_ATTR} reg r_a_lt_b;
    {KEEP_ATTR} reg r_out_valid;
    reg [{width - 1}:0] io_out_mux;

    assign out_valid = r_out_valid;
    assign io_out = io_out_mux;

    always @* begin
        case (out_sel)
            2'd0: io_out_mux = {{{zero_padding}, r_a_gt_b}};
            2'd1: io_out_mux = {{{zero_padding}, r_a_eq_b}};
            default: io_out_mux = {{{zero_padding}, r_a_lt_b}};
        endcase
    end

    holoso_icmp#(.W({width}), .LATENCY(2)) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a(r_a), .b(r_b), .out_valid(dut_out_valid),
        .a_gt_b(dut_a_gt_b), .a_eq_b(dut_a_eq_b), .a_lt_b(dut_a_lt_b)
    );

    always @(posedge clk) begin
        if (in_sel) r_b <= io_in;
        else        r_a <= io_in;
        r_a_gt_b <= dut_a_gt_b;
        r_a_eq_b <= dut_a_eq_b;
        r_a_lt_b <= dut_a_lt_b;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_shift_wrapper(top: str, width: int) -> str:
    """One output and no sideband, so the result needs no selector; the direction bit is registered as an operand is."""
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [1:0] in_sel,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg r_right;
    {KEEP_ATTR} reg [{width - 1}:0] r_x;
    {KEEP_ATTR} reg [{width - 1}:0] r_shamt;
    wire dut_out_valid;
    wire [{width - 1}:0] dut_shft;
    {KEEP_ATTR} reg [{width - 1}:0] r_shft;
    {KEEP_ATTR} reg r_out_valid;

    assign out_valid = r_out_valid;
    assign io_out = r_shft;

    holoso_ishft#(.W({width}), .LATENCY(2)) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .right(r_right), .x(r_x), .shamt(r_shamt),
        .out_valid(dut_out_valid), .shft(dut_shft)
    );

    always @(posedge clk) begin
        case (in_sel)
            2'd0: r_x <= io_in;
            2'd1: r_shamt <= io_in;
            2'd2: r_right <= io_in[0];
            default: ;
        endcase
        r_shft <= dut_shft;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_popcnt_wrapper(top: str, width: int) -> str:
    """
    One operand, one output and no sideband, so neither selector is needed. The result register stays as narrow as
    the count port and the zero fill happens on the way out: a full-width boundary register would hold the constant
    high bits under the keep attribute and charge them to the measurement.
    """
    operator = IPopcntOperator.build(IntFormat(width), IPopcntOptions())
    count_width = operator.params["WY"]
    parameters = ", ".join(f".{name}({value})" for name, value in operator.params.items())
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_x;
    wire dut_out_valid;
    wire [{count_width - 1}:0] dut_y;
    {KEEP_ATTR} reg [{count_width - 1}:0] r_y;
    {KEEP_ATTR} reg r_out_valid;

    assign out_valid = r_out_valid;
    assign io_out = {{{{{width - count_width}{{1'b0}}}}, r_y}};

    holoso_ipopcnt#({parameters}) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .x(r_x),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        r_x <= io_in;
        r_y <= dut_y;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


@pytest.mark.parametrize("target", _TARGETS, ids=lambda target: target.label)
def test_integer_operator_closes_timing(target: _Target) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_ooc_design(target.operator, target.width)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )


@pytest.mark.parametrize("target", _MULTIPLIER_TARGETS, ids=lambda target: target.label)
def test_integer_multiplier_closes_timing(target: _MultiplierTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_multiplier_ooc_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    dsp_used = sum(
        resource.used for name, resource in report.resources.items() if "DSP" in name.upper() or "MULT" in name.upper()
    )
    assert dsp_used > 0, f"{target.label}: no DSP resources reported; logs in {report.artifact_dir}"


@pytest.mark.parametrize("target", _DIVIDER_TARGETS, ids=lambda target: target.label)
def test_integer_divider_closes_timing(target: _DividerTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_divider_ooc_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    dsp_used = sum(
        resource.used for name, resource in report.resources.items() if "DSP" in name.upper() or "MULT" in name.upper()
    )
    assert dsp_used == 0, f"{target.label}: unexpected DSP resources reported; logs in {report.artifact_dir}"


@pytest.mark.parametrize("target", _FROM_INT_TARGETS, ids=lambda target: target.label)
def test_ffromint_closes_timing(target: _FromIntTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_ffromint_ooc_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )


@pytest.mark.parametrize("target", _RINT_TARGETS, ids=lambda target: target.label)
def test_frint_closes_timing(target: _RintTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_frint_ooc_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )


@pytest.mark.parametrize("target", _MUL_ILOG2_TARGETS, ids=lambda target: target.label)
def test_fmul_ilog2_closes_timing(target: _MulILog2Target) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_fmul_ilog2_ooc_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    dsp_used = sum(
        resource.used for name, resource in report.resources.items() if "DSP" in name.upper() or "MULT" in name.upper()
    )
    assert dsp_used == 0, f"{target.label}: unexpected DSP resources reported; logs in {report.artifact_dir}"
