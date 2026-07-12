from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shutil

import pytest
from holoso._backend.verilog._support import support_files
from synth import OocDesign, SourceFile

from synth._ooc import KEEP_ATTR
from synth._synth import BUILD_ROOT
from synth.flows import FlowId, make_flow

_SATURATING = ("holoso_iadds", "holoso_isubs", "holoso_iabss")
_SINGLE_OUTPUT = ("holoso_ashift",)
_UNARY = frozenset(("holoso_iabss",))


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
        assert self.latency == 2 + (self.width + 1) // 2

    @property
    def label(self) -> str:
        return (
            f"holoso_idivs-w{self.width}-f{self.quotient_floor}-" f"{self.flow.value}-{self.target_frequency_MHz:g}MHz"
        )


_TARGETS = tuple(
    _Target(operator, width, flow, frequency)
    for operator in (*_SATURATING, "holoso_icmp", *_SINGLE_OUTPUT)
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
    _DividerTarget(24, 0, FlowId.YOSYS_ECP5, 100.0, latency=14),
    _DividerTarget(24, 0, FlowId.DIAMOND_ECP5, 100.0, latency=14),
    _DividerTarget(24, 0, FlowId.VIVADO_ARTIX7, 150.0, latency=14),
    _DividerTarget(24, 1, FlowId.YOSYS_ECP5, 100.0, latency=14),
    _DividerTarget(24, 1, FlowId.DIAMOND_ECP5, 100.0, latency=14),
    _DividerTarget(24, 1, FlowId.VIVADO_ARTIX7, 150.0, latency=14),
    _DividerTarget(44, 0, FlowId.YOSYS_ECP5, 100.0, latency=24),
    _DividerTarget(44, 0, FlowId.DIAMOND_ECP5, 100.0, latency=24),
    _DividerTarget(44, 0, FlowId.VIVADO_ARTIX7, 150.0, latency=24),
    _DividerTarget(44, 1, FlowId.YOSYS_ECP5, 100.0, latency=24),
    _DividerTarget(44, 1, FlowId.DIAMOND_ECP5, 100.0, latency=24),
    _DividerTarget(44, 1, FlowId.VIVADO_ARTIX7, 150.0, latency=24),
)


@dataclass(frozen=True, slots=True)
class _MixedTarget(ABC):
    wexp: int
    wman: int
    wint: int
    flow: FlowId
    target_frequency_MHz: float

    def _validate(self) -> None:
        assert self.wexp >= 2
        assert self.wman >= 4
        assert self.wint >= 2

    @property
    @abstractmethod
    def operator(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def latency(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def stage_label(self) -> str:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return (
            f"{self.operator}-e{self.wexp}m{self.wman}-i{self.wint}-{self.stage_label}-"
            f"{self.flow.value}-{self.target_frequency_MHz:g}MHz"
        )


@dataclass(frozen=True, slots=True)
class _FromIntTarget(_MixedTarget):
    stage_input: int = 0
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        self._validate()

    @property
    def operator(self) -> str:
        return "holoso_ffromint"

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_normalize + self.stage_pack + self.stage_output

    @property
    def stage_label(self) -> str:
        return f"i{self.stage_input}n{self.stage_normalize}p{self.stage_pack}o{self.stage_output}"


@dataclass(frozen=True, slots=True)
class _ToIntTarget(_MixedTarget):
    stage_input: int = 0

    def __post_init__(self) -> None:
        self._validate()

    @property
    def operator(self) -> str:
        return "holoso_ftoint"

    @property
    def latency(self) -> int:
        return 4 + self.stage_input

    @property
    def stage_label(self) -> str:
        return f"i{self.stage_input}"


@dataclass(frozen=True, slots=True)
class _MulILog2Target(_MixedTarget):
    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        self._validate()

    @property
    def operator(self) -> str:
        return "holoso_fmul_ilog2"

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_decode

    @property
    def stage_label(self) -> str:
        return f"i{self.stage_input}d{self.stage_decode}"


_MIXED_TARGETS: tuple[_MixedTarget, ...] = (
    *(
        _FromIntTarget(wexp, wman, wint, flow, frequency, stage_normalize=1)
        for wexp, wman, wint in ((6, 18, 24), (8, 36, 44))
        for flow, frequency in ((FlowId.YOSYS_ECP5, 100.0), (FlowId.DIAMOND_ECP5, 100.0))
    ),
    *(_FromIntTarget(wexp, wman, wint, FlowId.VIVADO_ARTIX7, 150.0) for wexp, wman, wint in ((6, 18, 24), (8, 36, 44))),
    *(
        _ToIntTarget(6, 18, 24, flow, frequency)
        for flow, frequency in (
            (FlowId.YOSYS_ECP5, 100.0),
            (FlowId.DIAMOND_ECP5, 100.0),
            (FlowId.VIVADO_ARTIX7, 150.0),
        )
    ),
    _ToIntTarget(8, 36, 44, FlowId.YOSYS_ECP5, 100.0, stage_input=1),
    _ToIntTarget(8, 36, 44, FlowId.DIAMOND_ECP5, 100.0),
    _ToIntTarget(8, 36, 44, FlowId.VIVADO_ARTIX7, 150.0),
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
    elif operator in _SINGLE_OUTPUT:
        wrapper = _render_shift_wrapper(top, width)
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


def _build_mixed_ooc_design(target: _MixedTarget) -> OocDesign:
    top = f"{target.operator.removeprefix('holoso_')}_e{target.wexp}m{target.wman}_i{target.wint}_ooc"
    wrapper = _render_mixed_wrapper(top, target)
    files = [SourceFile(Path(name), content) for name, content in support_files().items()]
    files.append(SourceFile(Path(f"{top}.v"), wrapper))
    return OocDesign(top=top, files=files)


def _render_mixed_wrapper(top: str, target: _MixedTarget) -> str:
    if isinstance(target, _FromIntTarget):
        return _render_ffromint_wrapper(top, target)
    if isinstance(target, _ToIntTarget):
        return _render_ftoint_wrapper(top, target)
    assert isinstance(target, _MulILog2Target)
    return _render_fmul_ilog2_wrapper(top, target)


def _render_ffromint_wrapper(top: str, target: _FromIntTarget) -> str:
    wfull = target.wexp + target.wman
    wio = max(wfull, target.wint)
    zero_padding = f"{{{wio - wfull}{{1'b0}}}}"
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire in_sel,
    input  wire [{wio - 1}:0] io_in,
    output wire out_valid,
    output wire [{wio - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg signed [{target.wint - 1}:0] r_a;
    {KEEP_ATTR} reg [1:0] r_y_sgnop;
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
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a(r_a), .y_sgnop(r_y_sgnop),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        if (in_sel) r_y_sgnop <= io_in[1:0];
        else        r_a <= io_in[{target.wint - 1}:0];
        r_y <= dut_y;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
            r_y_sgnop <= 2'b00;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
    end
endmodule

`default_nettype wire
"""


def _render_ftoint_wrapper(top: str, target: _ToIntTarget) -> str:
    wfull = target.wexp + target.wman
    wio = max(wfull, target.wint)
    zero_padding = f"{{{wio - target.wint}{{1'b0}}}}"
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
    {KEEP_ATTR} reg [1:0] r_a_sgnop;
    {KEEP_ATTR} reg [1:0] r_round_mode;
    wire dut_out_valid;
    wire signed [{target.wint - 1}:0] dut_y;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg signed [{target.wint - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign io_out = {{{zero_padding}, r_y}};

    holoso_ftoint#(
        .WEXP({target.wexp}), .WMAN({target.wman}), .WINT({target.wint}),
        .STAGE_INPUT({target.stage_input}), .LATENCY({target.latency})
    ) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a_sgnop(r_a_sgnop), .round_mode(r_round_mode), .a(r_a),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        case (in_sel)
            2'd0: r_a <= io_in[{wfull - 1}:0];
            2'd1: r_a_sgnop <= io_in[1:0];
            2'd2: r_round_mode <= io_in[1:0];
            default: ;
        endcase
        r_y <= dut_y;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
            r_a_sgnop <= 2'b00;
            r_round_mode <= 2'b00;
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
    {KEEP_ATTR} reg [1:0] r_y_sgnop;
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
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a_sgnop(r_a_sgnop), .y_sgnop(r_y_sgnop), .a(r_a), .k(r_k),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        case (in_sel)
            2'd0: r_a <= io_in[{wfull - 1}:0];
            2'd1: r_k <= io_in[{target.wint - 1}:0];
            2'd2: r_a_sgnop <= io_in[1:0];
            default: r_y_sgnop <= io_in[1:0];
        endcase
        r_y <= dut_y;
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
            r_a_sgnop <= 2'b00;
            r_y_sgnop <= 2'b00;
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
    {KEEP_ATTR} reg [{width - 1}:0] r_num;
    {KEEP_ATTR} reg [{width - 1}:0] r_den;
    wire dut_out_valid;
    wire [{width - 1}:0] dut_quo;
    wire [{width - 1}:0] dut_rem;
    wire dut_saturated;
    wire dut_div0;
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
        case (out_sel)
            2'd0: r_io_out <= dut_quo;
            2'd1: r_io_out <= dut_rem;
            2'd2: r_io_out <= {{{zero_padding}, dut_saturated}};
            default: r_io_out <= {{{zero_padding}, dut_div0}};
        endcase
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


def _render_saturating_wrapper(top: str, operator: str, width: int, parameters: str | None = None) -> str:
    binary_port = f"    input  wire in_sel,\n" if operator not in _UNARY else ""
    operand_reg = f"    {KEEP_ATTR} reg [{width - 1}:0] r_b;\n" if operator not in _UNARY else ""
    operand_port = ", .b(r_b)" if operator not in _UNARY else ""
    first_operand_port = "x" if operator in _UNARY else "a"
    operand_load = (
        "        if (in_sel) r_b <= io_in;\n        else        r_a <= io_in;"
        if operator not in _UNARY
        else ("        r_a <= io_in;")
    )
    parameters = parameters or f".W({width}), .LATENCY(2)"
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
{binary_port}    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    input  wire out_sel,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_a;
{operand_reg}    wire dut_out_valid;
    wire [{width - 1}:0] dut_y;
    wire dut_saturated;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_y;
    {KEEP_ATTR} reg r_saturated;
    {KEEP_ATTR} reg [{width - 1}:0] r_io_out;

    assign out_valid = r_out_valid;
    assign io_out = r_io_out;

    {operator}#({parameters}) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .{first_operand_port}(r_a){operand_port},
        .out_valid(dut_out_valid), .y(dut_y), .saturated(dut_saturated)
    );

    always @(posedge clk) begin
{operand_load}
        r_y <= dut_y;
        r_saturated <= dut_saturated;
        r_io_out <= out_sel ? r_saturated : r_y;
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
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg r_a_gt_b;
    {KEEP_ATTR} reg r_a_eq_b;
    {KEEP_ATTR} reg r_a_lt_b;
    {KEEP_ATTR} reg [{width - 1}:0] r_io_out;

    assign out_valid = r_out_valid;
    assign io_out = r_io_out;

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
        case (out_sel)
            2'd0: r_io_out <= {{{zero_padding}, r_a_gt_b}};
            2'd1: r_io_out <= {{{zero_padding}, r_a_eq_b}};
            default: r_io_out <= {{{zero_padding}, r_a_lt_b}};
        endcase
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
    return f"""`default_nettype none

module {top} (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire in_sel,
    input  wire [{width - 1}:0] io_in,
    output wire out_valid,
    output wire [{width - 1}:0] io_out
);
    {KEEP_ATTR} reg r_in_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_x;
    {KEEP_ATTR} reg [{width - 1}:0] r_shamt;
    wire dut_out_valid;
    wire [{width - 1}:0] dut_y;
    {KEEP_ATTR} reg r_out_valid;
    {KEEP_ATTR} reg [{width - 1}:0] r_y;
    {KEEP_ATTR} reg [{width - 1}:0] r_io_out;

    assign out_valid = r_out_valid;
    assign io_out = r_io_out;

    holoso_ashift#(.W({width}), .LATENCY(2)) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .x(r_x), .shamt(r_shamt),
        .out_valid(dut_out_valid), .y(dut_y)
    );

    always @(posedge clk) begin
        if (in_sel) r_shamt <= io_in;
        else        r_x <= io_in;
        r_y <= dut_y;
        r_io_out <= r_y;
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


@pytest.mark.parametrize("target", _MIXED_TARGETS, ids=lambda target: target.label)
def test_mixed_operator_closes_timing(target: _MixedTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    directory = BUILD_ROOT / "integer" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(_build_mixed_ooc_design(target)).synthesize(directory)
    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    if target.operator == "holoso_fmul_ilog2":
        dsp_used = sum(
            resource.used
            for name, resource in report.resources.items()
            if "DSP" in name.upper() or "MULT" in name.upper()
        )
        assert dsp_used == 0, f"{target.label}: unexpected DSP resources reported; logs in {report.artifact_dir}"


def test_mixed_wrapper_registers_native_result_width() -> None:
    from_int = _render_ffromint_wrapper("top", _FromIntTarget(6, 18, 44, FlowId.YOSYS_ECP5, 100.0))
    to_int = _render_ftoint_wrapper("top", _ToIntTarget(8, 36, 24, FlowId.YOSYS_ECP5, 100.0))
    mul_ilog2 = _render_fmul_ilog2_wrapper("top", _MulILog2Target(6, 18, 44, FlowId.YOSYS_ECP5, 100.0))
    assert "reg [23:0] r_y;" in from_int
    assert "reg signed [23:0] r_y;" in to_int
    assert "input  wire [1:0] in_sel," in to_int
    assert "reg [1:0] r_round_mode;" in to_int
    assert ".round_mode(r_round_mode)" in to_int
    assert "reg [23:0] r_y;" in mul_ilog2
    assert "r_io_out" not in from_int
    assert "r_io_out" not in to_int
    assert "r_io_out" not in mul_ilog2
