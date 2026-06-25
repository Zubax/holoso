import re

import pytest

from holoso import synthesize, SynthesisResult, FloatFormat
from holoso import FAddOperator, FCmpOperator, FDivOperator, FMulILog2OperatorFamily, FMulOperator, OpConfig

from synth import SynthReport, build_ooc_wrapper
from synth.flows import FlowId
from synth.flows.diamond import DiamondEcp5Flow
from synth.flows.vivado import VivadoArtix7Flow
from synth.flows.yosys import YosysEcp5Flow

FMT = FloatFormat(wexp=8, wman=24)  # 32-bit ports


def kern(a: float, b: float) -> float:
    # fadd + fmul + multiply-by-2^-2.
    return (a - b) * 0.25 + a * b


def wide(a: float, b: float, c: float, d: float, e: float, f: float) -> list[float]:
    # Six inputs and three outputs, so both selectors are multi-bit (exercises the mux decode).
    return [a * b + c, d - e * f, a + d]


def bool_gate(a: bool, b: bool) -> bool:
    # All data inputs are 1-bit, so io_in_width == 1 -- the case where io_in must be a vector to stay bit-sliceable.
    return a and b


OPS = OpConfig(
    fadd=FAddOperator(FMT),
    fmul=FMulOperator(FMT),
    fdiv=FDivOperator(FMT),
    fmul_ilog2=FMulILog2OperatorFamily(FMT),
    fcmp=FCmpOperator(FMT),
)
KERN: SynthesisResult = synthesize(kern, ops=OPS, name="kern")
WIDE: SynthesisResult = synthesize(wide, ops=OPS, name="wide")

requires_diamond = pytest.mark.skipif(not DiamondEcp5Flow().available(), reason="Lattice Diamond not found")
requires_vivado = pytest.mark.skipif(not VivadoArtix7Flow().available(), reason="Vivado not found")


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


def test_wrapper_buses_single_bit_input_word() -> None:
    # Regression: an all-boolean-input kernel makes io_in 1 bit wide. The wrapper still bit-slices io_in in the input
    # load, so io_in must be declared as a vector -- a collapsed scalar wire is rejected by strict elaborators
    # (Diamond/Vivado: "cannot index into non-array io_in"), though lenient Yosys accepts it.
    wrapper = build_ooc_wrapper(synthesize(bool_gate, ops=OPS, name="bool_gate"))
    assert wrapper.io_in_width == 1
    assert re.search(r"input\s+wire\s+\[\d+:0\]\s+io_in", wrapper.verilog), "io_in must be declared as a vector"
    assert "io_in[0:0]" in wrapper.verilog  # the load slice that requires io_in to be indexable


def _assert_sane(report: SynthReport, flow: FlowId) -> None:
    assert report.flow == flow
    assert report.fmax_MHz > 0.0
    assert isinstance(report.slack_ns, float)
    assert report.resources, "no fabric resources were parsed"
    assert report.artifact_dir.is_dir()


def test_yosys_ecp5_end_to_end() -> None:
    wrapper = build_ooc_wrapper(KERN)
    report = YosysEcp5Flow(target_frequency_MHz=100.0).prepare(KERN).synthesize()
    _assert_sane(report, FlowId.YOSYS_ECP5)
    # Out of context: no IO pads, so the bounded primary IO is the only boundary.
    assert report.resources["TRELLIS_IO"].used == 0
    # The DUT survived optimization -- a collapsed datapath would be a few boundary flops, not ~hundreds of LUTs.
    assert report.resources["TRELLIS_COMB"].used > 100
    assert wrapper.primary_io_bits < _native_data_bits(KERN) + 64


def test_yosys_ecp5_wide_selectors_synthesize() -> None:
    # A kernel whose selectors are multi-bit must still elaborate and route.
    report = YosysEcp5Flow(target_frequency_MHz=100.0).prepare(WIDE).synthesize()
    _assert_sane(report, FlowId.YOSYS_ECP5)
    assert report.resources["TRELLIS_IO"].used == 0


@requires_diamond
def test_diamond_ecp5_end_to_end() -> None:
    report = DiamondEcp5Flow(target_frequency_MHz=100.0).prepare(KERN).synthesize()
    _assert_sane(report, FlowId.DIAMOND_ECP5)
    # The muxed wrapper bounds the pin count, so Diamond's PIO usage stays small and the design fits real pins.
    pio = report.resources.get("PIO")
    assert pio is not None and pio.used == build_ooc_wrapper(KERN).primary_io_bits


@requires_vivado
def test_vivado_end_to_end() -> None:
    report = VivadoArtix7Flow(target_frequency_MHz=100.0).prepare(KERN).synthesize()
    _assert_sane(report, FlowId.VIVADO_ARTIX7)
    assert report.resources["Slice LUTs"].used > 0
    assert report.resources["Slice Registers"].used > 0
