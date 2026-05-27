"""
End-to-end checks for the OOC synthesis-evaluation harness.
"""

from pathlib import Path

import pytest
import synth.__main__ as synth_cli

from holoso import synthesize, SynthesisResult, FloatFormat
from holoso import FAddOperator, FDivOperator, FMulILog2OperatorFamily, FMulOperator, OpConfig

from synth import SynthReport, build_ooc_wrapper
from synth._synth import SynthArtifact
from synth.flows.diamond import DiamondEcp5Flow
from synth.flows.vivado import VivadoFlow
from synth.flows.yosys import YosysEcp5Flow

FMT = FloatFormat(wexp=8, wman=24)  # 32-bit ports
REPO_ROOT = Path(__file__).resolve().parents[1]
KULIBIN_HDL = sorted((REPO_ROOT / "lib" / "kulibin" / "float" / "hdl").glob("*.v"))


def kern(a: float, b: float) -> float:
    # fadd + fmul + multiply-by-2^-2.
    return (a - b) * 0.25 + a * b


def wide(a: float, b: float, c: float, d: float, e: float, f: float) -> list[float]:
    # Six inputs and three outputs, so both selectors are multi-bit (exercises the mux decode).
    return [a * b + c, d - e * f, a + d]


OPS = OpConfig(
    fadd=FAddOperator(FMT), fmul=FMulOperator(FMT), fdiv=FDivOperator(FMT), fmul_ilog2=FMulILog2OperatorFamily(FMT)
)
KERN: SynthesisResult = synthesize(kern, ops=OPS, name="kern")
WIDE: SynthesisResult = synthesize(wide, ops=OPS, name="wide")

requires_diamond = pytest.mark.skipif(not DiamondEcp5Flow().available(), reason="Lattice Diamond not found")
requires_vivado = pytest.mark.skipif(not VivadoFlow().available(), reason="Vivado not found")


def _parse_cli_flow_requests(*flow_specs: str) -> list[synth_cli._FlowRequest]:
    args = ["kernel.py", "entry", "--rtl", "lib/kulibin/float/hdl"]
    for flow_spec in flow_specs:
        args += ["--flow", flow_spec]
    return synth_cli._parse_args(args).flow_requests


def _stage_override_map(request: synth_cli._FlowRequest) -> dict[tuple[str, str], object]:
    return {(override.operator_name, override.field_name): override.value for override in request.stage_overrides}


def test_cli_flow_stage_overrides_are_sparse_per_flow() -> None:
    yosys, vivado = _parse_cli_flow_requests(
        "yosys-ecp5:freq=42,fadd.decode=1,fmul.input=1",
        "vivado:freq=60,fmul.stage_output=0,fdiv.input=1,fmul_ilog2.decode=1",
    )
    yosys_stages = _stage_override_map(yosys)
    vivado_stages = _stage_override_map(vivado)

    assert yosys.flow_id == "yosys-ecp5"
    assert yosys_stages[("fadd", "stage_decode")] == 1
    assert ("fadd", "stage_align") not in yosys_stages
    assert yosys_stages[("fmul", "stage_input")] == 1
    assert ("fmul", "stage_product") not in yosys_stages

    assert vivado.flow_id == "vivado"
    assert vivado_stages[("fmul", "stage_output")] == 0
    assert vivado_stages[("fdiv", "stage_input")] == 1
    assert vivado_stages[("fmul_ilog2", "stage_decode")] == 1


def test_cli_operator_config_preserves_defaults_for_unset_stages() -> None:
    (request,) = _parse_cli_flow_requests("yosys-ecp5:freq=42,fadd.decode=1,fmul.output=0")
    ops = synth_cli._op_config(FMT, request.stage_overrides)

    assert ops.fadd.stage_decode == 1
    assert ops.fadd.stage_align == FAddOperator(FMT).stage_align
    assert ops.fadd.stage_output == FAddOperator(FMT).stage_output
    assert ops.fmul.stage_input == FMulOperator(FMT).stage_input
    assert ops.fmul.stage_product == FMulOperator(FMT).stage_product
    assert ops.fmul.stage_output == 0
    assert ops.fdiv == FDivOperator(FMT)
    assert ops.fmul_ilog2 == FMulILog2OperatorFamily(FMT)


@pytest.mark.parametrize(
    "flow_spec",
    [
        "yosys-ecp5:freq=42,fadd.decode=no",
        "yosys-ecp5:freq=42,fadd.foo=1",
        "yosys-ecp5:freq=42,unknown.decode=1",
    ],
)
def test_cli_rejects_invalid_stage_fields(flow_spec: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_cli_flow_requests(flow_spec)
    assert exc_info.value.code == 2


def test_cli_parser_does_not_validate_stage_value_range() -> None:
    (request,) = _parse_cli_flow_requests("yosys-ecp5:freq=42,fadd.decode=2")
    assert _stage_override_map(request)[("fadd", "stage_decode")] == 2


def _native_data_bits(result: SynthesisResult) -> int:
    return sum(p.width for p in result.input_ports) + sum(p.width for p in result.output_ports)


def test_wrapper_reduces_io_to_bounded_words() -> None:
    # The headline property: the wrapper exposes ~two data words + selectors + control, independent of how many
    # data bits the DUT actually has, so even wide kernels map to real device pins.
    kern_w = build_ooc_wrapper(KERN)
    assert kern_w.top == "kern_ooc"
    assert kern_w.io_in_width == 32 and kern_w.io_out_width == 32
    assert kern_w.in_sel_width == 1  # two inputs
    assert kern_w.out_sel_width == 1  # one output + err_pc => two slots
    assert kern_w.primary_io_bits == 32 + 32 + 1 + 1 + 6

    wide_w = build_ooc_wrapper(WIDE)
    assert wide_w.in_sel_width == 3  # six inputs
    assert wide_w.out_sel_width == 2  # three outputs + err_pc => four slots
    # The reduction is real: far fewer primary bits than the DUT's native data IO.
    assert wide_w.primary_io_bits < _native_data_bits(WIDE)
    assert wide_w.primary_io_bits <= 2 * wide_w.io_in_width + 16


def _assert_sane(report: SynthReport, flow: str) -> None:
    assert report.flow == flow
    assert report.fmax_MHz > 0.0
    assert isinstance(report.slack_ns, float)
    assert report.resources, "no fabric resources were parsed"
    assert report.artifact_dir.is_dir()


def test_yosys_ecp5_end_to_end() -> None:
    wrapper = build_ooc_wrapper(KERN)
    report = YosysEcp5Flow(target_frequency_MHz=100.0).prepare(KERN, KULIBIN_HDL).synthesize()
    _assert_sane(report, "yosys-ecp5")
    # Out of context: no IO pads, so the bounded primary IO is the only boundary.
    assert report.resources["TRELLIS_IO"].used == 0
    # The DUT survived optimization -- a collapsed datapath would be a few boundary flops, not ~hundreds of LUTs.
    assert report.resources["TRELLIS_COMB"].used > 100
    assert wrapper.primary_io_bits < _native_data_bits(KERN) + 64


def test_yosys_ecp5_wide_selectors_synthesize() -> None:
    # A kernel whose selectors are multi-bit must still elaborate and route.
    report = YosysEcp5Flow(target_frequency_MHz=100.0).prepare(WIDE, KULIBIN_HDL).synthesize()
    _assert_sane(report, "yosys-ecp5")
    assert report.resources["TRELLIS_IO"].used == 0


@requires_diamond
def test_diamond_ecp5_end_to_end() -> None:
    report = DiamondEcp5Flow(target_frequency_MHz=100.0).prepare(KERN, KULIBIN_HDL).synthesize()
    _assert_sane(report, "diamond-ecp5")
    # The muxed wrapper bounds the pin count, so Diamond's PIO usage stays small and the design fits real pins.
    pio = report.resources.get("PIO")
    assert pio is not None and pio.used == build_ooc_wrapper(KERN).primary_io_bits


@requires_vivado
def test_vivado_end_to_end() -> None:
    report = VivadoFlow(target_frequency_MHz=100.0).prepare(KERN, KULIBIN_HDL).synthesize()
    _assert_sane(report, "vivado")
    assert report.resources["Slice LUTs"].used > 0
    assert report.resources["Slice Registers"].used > 0
