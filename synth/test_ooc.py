"""
End-to-end checks for the OOC synthesis-evaluation harness.
"""

from pathlib import Path

import pytest

from holoso import synthesize, SynthesisResult, FloatFormat
from holoso import FAddOp, FDivOp, FMulILog2GenericOp, FMulOp, OpConfig

from synth import SynthReport, build_ooc_wrapper
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


OPS = OpConfig(fadd=FAddOp(), fmul=FMulOp(), fdiv=FDivOp(), fmul_ilog2=FMulILog2GenericOp())
KERN: SynthesisResult = synthesize(kern, float_format=FMT, ops=OPS, name="kern")
WIDE: SynthesisResult = synthesize(wide, float_format=FMT, ops=OPS, name="wide")

requires_diamond = pytest.mark.skipif(not DiamondEcp5Flow().available(), reason="Lattice Diamond not found")
requires_vivado = pytest.mark.skipif(not VivadoFlow().available(), reason="Vivado not found")


def _native_data_bits(result: SynthesisResult) -> int:
    iface = result.interface
    return sum(p.width for p in iface.input_ports) + sum(p.width for p in iface.output_ports)


def test_wrapper_reduces_io_to_bounded_words() -> None:
    # The headline property: the wrapper exposes ~two data words + selectors + control, independent of how many
    # data bits the DUT actually has, so even wide kernels map to real device pins.
    kern_w = build_ooc_wrapper(KERN)
    assert kern_w.top == "kern_ooc"
    assert kern_w.io_in_width == 32 and kern_w.io_out_width == 32
    assert kern_w.in_sel_width == 1  # two inputs
    assert kern_w.out_sel_width == 1  # one output + err_cyc => two slots
    assert kern_w.primary_io_bits == 32 + 32 + 1 + 1 + 6

    wide_w = build_ooc_wrapper(WIDE)
    assert wide_w.in_sel_width == 3  # six inputs
    assert wide_w.out_sel_width == 2  # three outputs + err_cyc => four slots
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
