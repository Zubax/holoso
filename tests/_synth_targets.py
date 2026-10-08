"""
The out-of-context synthesis matrix: a flat list of SynthTarget records, one per (kernel, flow, target f_max,
operator configuration). This is the single source of truth for what the example synthesis suite
(test_synth_examples) asserts can close timing.

Deliberately a dumb-simple data table -- literal frequencies, one explicit row per (example, flow), and a full typed
Options per row; duplication is preferred over indirection here. Catalogued example targets reuse the shared example
registry (_examples.SPECS) so each kernel is constructed once. Off-catalogue regression cores may be added as
SynthTarget records with example=None.

The bar (all at minimum speed grade): yosys-ecp5 and diamond-ecp5 at 100 MHz, vivado-artix7 at 150 MHz. Almost every
kernel closes lean; stage knobs are row-local and appear only where measured closure or its margin needs them. Lean
keeps the product stage that brackets a DSP tile or splits a multiplicand wider than one (`stage_product`); a row's
history names that knob only where it goes beyond lean. Vivado's result does not reproduce from one machine to the
next, so a Vivado row closing with only hundredths of a nanosecond to spare takes one more stage where one helps.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from holoso import (
    FAddOptions,
    FCmpOptions,
    FCordicOptions,
    FDivsqrtOptions,
    FExp2Options,
    FFmaOptions,
    FFromIntOptions,
    FILog2Options,
    FLog2Options,
    FloatFormat,
    FMulILog2Options,
    FMulOptions,
    FRintOptions,
    FSortOptions,
    OperatorOptions,
    Options,
)
from synth.flows import DeviceClass, FlowId

from ._examples import SPECS, ekf1_stateful, imu_fusion, polar, rigid_body_rates

_F_e6m18 = FloatFormat(6, 18)
_F_e8m36 = FloatFormat(8, 36)
_F_e4m8 = FloatFormat(4, 8)  # the narrowest datapath; the float-free UART ignores it and takes its 16-bit floor


def _op_config(
    fmt: FloatFormat,
    *,
    fadd: FAddOptions | None = None,
    fmul: FMulOptions | None = None,
    fdivsqrt: FDivsqrtOptions | None = None,
    fmul_ilog2: FMulILog2Options | None = None,
    filog2: FILog2Options | None = None,
    fcmp: FCmpOptions | None = None,
    ffma: FFmaOptions | None = None,
    fexp2: FExp2Options | None = None,
    flog2: FLog2Options | None = None,
    fcordic: FCordicOptions | None = None,
    fsort: FSortOptions | None = None,
    frint: FRintOptions | None = None,
    ffromint: FFromIntOptions | None = None,
    wmultiplier: int | None = None,
    wint_min: int = Options(OperatorOptions()).wint_min,
) -> Options:
    """
    The Options for fmt; pass an options object to give an operator stage knobs, else that operator is lean.
    ffma/fexp2/flog2/fcordic/fsort/frint/ffromint are absent unless supplied, so MAC chains stay expanded (fmul + fadd)
    and a kernel that uses no transcendental, min/max, rounding or integer conversion needs no such module.
    """
    return Options(
        OperatorOptions(
            fadd=fadd or FAddOptions(),
            fmul=fmul or FMulOptions(),
            fdivsqrt=fdivsqrt or FDivsqrtOptions(),
            fmul_ilog2=fmul_ilog2 or FMulILog2Options(),
            filog2=filog2 or FILog2Options(),
            fcmp=fcmp or FCmpOptions(),
            ffma=ffma,
            fexp2=fexp2,
            flog2=flog2,
            fcordic=fcordic,
            fsort=fsort,
            frint=frint,
            ffromint=ffromint,
        ),
        ffmt=fmt,
        wmultiplier=wmultiplier,
        wint_min=wint_min,
    )


@dataclass(frozen=True, slots=True)
class SynthTarget:
    """One synthesis closure target: a kernel synthesized for one flow under one Options, asserted to meet f_max."""

    kernel: Callable[[], Callable[..., object]]
    flow: FlowId
    target_frequency_MHz: float
    ops: Options
    name: str  # descriptive module/report label; unique per flow
    example: str | None = None  # the SPECS name this exercises; None for an off-catalogue regression core
    device_class: DeviceClass = DeviceClass.DEFAULT  # LARGE for a kernel the flow's default device cannot hold

    @property
    def label(self) -> str:
        return f"{self.name}-{self.flow.value}"


_SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def _for_example(
    example: str,
    flow: FlowId,
    target_frequency_MHz: float,
    ops: Options,
    *,
    kernel: Callable[[], Callable[..., object]] | None = None,
    name: str | None = None,
    device_class: DeviceClass = DeviceClass.DEFAULT,
) -> SynthTarget:
    """
    A target whose kernel is the catalogued example `example`. The kernel defaults to the SPEC factory, but a kernel
    the SPEC constructs differently for cosim than the bundled example ships (e.g. ekf1_stateful, whose SPEC reset is
    divisor-safe for the test vectors) must pass `kernel` explicitly, so the matrix synthesizes the shipped circuit
    rather than the cosim variant.
    """
    spec = _SPEC_BY_NAME[example]  # KeyError guards a typo'd example name
    fmt = ops.ffmt
    return SynthTarget(
        kernel=kernel or spec.make_kernel,
        flow=flow,
        target_frequency_MHz=target_frequency_MHz,
        ops=ops,
        name=name or f"{example}_e{fmt.wexp}m{fmt.wman}",
        example=example,
        device_class=device_class,
    )


def _ekf1_stateful_kernel() -> Callable[..., object]:
    # The bundled example's default-constructed kernel -- what examples/ekf1_stateful.py ships and the matrix must
    # synthesize. SPEC.make_kernel instead uses _fresh_stateful_ekf, a cosim-only divisor-safe reset that folds
    # different constants into different RTL, so the synth rows pass this explicitly.
    return ekf1_stateful.Ekf1().update


def _rigid_body_rates_kernel() -> Callable[..., object]:
    # Off-catalogue: the shaped matrix/vector ports have no scalar-lane SPEC (the cosim registry drives the
    # rigid_body_scalar wrapper instead); the matrix synthesizes the shipped example circuit directly.
    return rigid_body_rates.update


def _imu_fusion_kernel() -> Callable[..., object]:
    # The bundled example's realistic-config kernel -- what examples/imu_fusion.py ships. SPEC.make_kernel instead
    # uses an oracle-safe reset and clamp/clip overrides that fold different constants into different RTL, so the
    # rows pass this explicitly.
    return imu_fusion.ImuFusion().update


def _to_polar_kernel() -> Callable[..., object]:
    # Off-catalogue (2-vector I/O); exercises the fused atan2+hypot CORDIC.
    return polar.to_polar


def _from_polar_kernel() -> Callable[..., object]:
    # Off-catalogue (2-vector I/O); exercises the coalesced sin+cos CORDIC.
    return polar.from_polar


def _root_only(x: float) -> float:
    return math.sqrt(x)


def _root_only_kernel() -> Callable[..., object]:
    # Off-catalogue: every bundled kernel that takes a root also divides, so nothing else elaborates the divider for
    # the root alone -- the build whose digit selection a resource-sharing synthesizer folds unless told to keep it.
    return _root_only


TARGETS: list[SynthTarget] = [
    # Most of the catalogue closes the bar at lean (no optional stages) on all three tools -- verified by a full lean
    # baseline. One explicit row per (example, flow); duplication is intentional.
    _for_example("madd", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    # 99.4 MHz lean: the multiplier's DSP product register through its pack rounding into the register file (then
    # 103.8).
    _for_example("madd", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, fmul=FMulOptions(stage_pack=1))),
    _for_example("madd", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("poly3", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fmul=FMulOptions(stage_pack=1))),
    _for_example("poly3", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("poly3", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("signal_window", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("signal_window", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("signal_window", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("iir1_hpf", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fadd=FAddOptions(stage_output=1))),
    _for_example("iir1_hpf", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    # 153.0 MHz lean. For margin, the adder's close-cancellation normalizer (then 156.0).
    _for_example("iir1_hpf", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("iir1_lpf", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fadd=FAddOptions(stage_output=1))),
    _for_example("iir1_lpf", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("iir1_lpf", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("pid", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("pid", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("pid", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("schmitt_trigger", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("schmitt_trigger", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("schmitt_trigger", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("quadrature_encoder", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("quadrature_encoder", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("quadrature_encoder", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("phase_frequency_detector", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("phase_frequency_detector", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("phase_frequency_detector", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("latching_fault_register", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("latching_fault_register", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("latching_fault_register", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("majority_voter", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("majority_voter", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("majority_voter", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    # nextpnr's critical path runs from the microcode word through the adder's operand read mux into its magnitude
    # compare, route-dominated; the input stage splits it at the read mux.
    _for_example("recip_newton", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1))),
    # 93.0 MHz lean: the multiplier's DSP product register through its pack rounding into the register file (then
    # 112.1).
    _for_example("recip_newton", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, fmul=FMulOptions(stage_pack=1))),
    # Vivado closes this lean only on some releases: 2025.2 clears 150 MHz, 2026.1 misses by 0.078 ns on the adder's
    # normalize cascade. One barrier holds it on both.
    _for_example("recip_newton", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("integrator", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("integrator", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "integrator",
        FlowId.VIVADO_ARTIX7,
        150,
        # 152.9 MHz with the normalizer stage alone. For margin, the register file through the adder's magnitude compare
        # and operand select into its exponent difference (then 165.9).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_decode=1, stage_normalize=1)),
    ),
    # Straight-line multiply-accumulate chains; both close lean on all three flows.
    _for_example("fir", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("fir", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("fir", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("biquad", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("biquad", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("biquad", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("uart_tx", FlowId.YOSYS_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_tx", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_tx", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e4m8)),
    _for_example("uart_rx", FlowId.YOSYS_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_rx", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_rx", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e4m8)),
    # tunable_lowpass: the one bundled kernel whose shift counts are run-time operands, hence the one that builds the
    # barrel shifter; at the 32-bit word its script ships.
    _for_example("tunable_lowpass", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, wint_min=32)),
    _for_example("tunable_lowpass", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wint_min=32)),
    _for_example("tunable_lowpass", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wint_min=32)),
    _for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        # 67.2 MHz lean: the microcode operand read mux into the multiplier's DSP product, the same read mux into the
        # adder's magnitude compare and exponent select (76.6 MHz), the multiplier's product register through its
        # rounder and pack into the register-file write select (96.3 MHz), the adder's pack-input select through its
        # rounder and result select into the register file (94.7 MHz), the microcode operand read mux into the scaler's
        # exponent adder and overflow detect (97.1 MHz, then 100.5). For margin, the scaler's latched exponent through
        # that adder and overflow detect into its output register (then 111.1).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        # 87.9 MHz lean: the microcode through the adder's operand read mux into its exponent difference, the
        # multiplier's DSP product through its pack rounding into the register file (90.9 MHz), the microcode through
        # the multiplier's operand read mux into its unregistered DSP operand (98.2 MHz, then 105.8).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1), fmul=FMulOptions(stage_input=1, stage_pack=1)),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        # 131.0 MHz lean: the register file through the adder's operand read mux into its magnitude compare and exponent
        # difference, the microcode word through the multiplier's operand read mux into the unregistered DSP input
        # (131.5 MHz), the adder's pack logic into the register file (151.6 MHz with 0.07 ns to spare; then 159.8).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1, stage_pack=1), fmul=FMulOptions(stage_input=1)),
    ),
    # The same EKF with a second multiplier: the allocator binds the co-issued products across the two instances.
    # Against the one-multiplier row at the same stages: 14 percent more LUTs for a 22 percent shorter transaction
    # (DESIGN.md).
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        # 136.2 MHz lean; held to the one-multiplier row's stages so that the pair compares like for like (then 154.8).
        _op_config(
            _F_e6m18, fadd=FAddOptions(stage_input=1, stage_pack=1), fmul=FMulOptions(instances=2, stage_input=1)
        ),
        name="ekf1_stateless_e6m18_fmul2",
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.YOSYS_ECP5,
        100,
        # 73.8 MHz lean: the microcode operand read mux into the multiplier's DSP product, the same read mux into the
        # adder's magnitude compare and exponent select (82.5 MHz), the adder's rounder and pack across a long route
        # into the register-file write select (98.9 MHz, then 102.4).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1, stage_output=1), fmul=FMulOptions(stage_input=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # 92.2 MHz lean: the microcode through the adder's operand read mux into its exponent difference, the
        # multiplier's DSP product through its pack rounding into the register file (99.8 MHz, then 104.6).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1), fmul=FMulOptions(stage_pack=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        # 131.3 MHz lean: the microcode word through the multiplier's operand read mux into the unregistered DSP input
        # (then 155.3).
        _op_config(_F_e6m18, fmul=FMulOptions(stage_input=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "cordic_sincos",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_decode=1),
            # The microcode immediate out of the EBR (5.6 ns clock-to-out) into the scaler's exponent adder (83.6
            # MHz): the scaler's input stage.
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
        ),
    ),
    _for_example("cordic_sincos", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("cordic_sincos", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("octave_index", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("octave_index", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("octave_index", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    # octave_index's transcendental sibling: the exp2 and log2 operators, with their lookup tables and Horner products.
    _for_example(
        "equal_temperament",
        FlowId.YOSYS_ECP5,
        100,
        # 52.9 MHz lean: log2's result normalizer through its rounder into the register file, exp2's Horner multiply
        # from its operand capture through the DSP tiles and the two partial-product adders (83.8 MHz), the second half
        # of log2's normalizer through the rounder into the register file (84.2 MHz), exp2's Horner accumulator through
        # the sticky reduction and the rounder into the register file (99.2 MHz, then 103.9).
        _op_config(
            _F_e6m18,
            fexp2=FExp2Options(stage_product=2, stage_pack=1),
            flog2=FLog2Options(stage_product=1, stage_product_final=1, stage_normalize=1, stage_pack=1),
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        # 45.8 MHz lean: the log2 normalizer through its rounder into the register file, the normalizer's second half
        # through the rounder into the register file (77.6 MHz), the exp2 Horner multiply built as one multiplier tile
        # cascaded through two adder tiles in a single cycle (89.7 MHz), the multiplier's DSP product register through
        # its pack rounding into the register file (99.1 MHz, then 105.4).
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_pack=1),
            fexp2=FExp2Options(stage_product=2),
            flog2=FLog2Options(stage_product=1, stage_product_final=1, stage_normalize=1, stage_pack=1),
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.VIVADO_ARTIX7,
        150,
        # 98.0 MHz lean: log2's result normalizer, whose whole leading-zero-count-and-shift cascade ran on through the
        # rounder into the register file in one cycle, then the first half of that normalizer (152.0 MHz with 0.09 ns to
        # spare; then 154.9).
        _op_config(
            _F_e6m18,
            fexp2=FExp2Options(stage_product=1),
            flog2=FLog2Options(stage_product=1, stage_product_final=1, stage_normalize=2),
        ),
    ),
    _for_example("remainder", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("remainder", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("remainder", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        # 58.7 MHz lean: the microcode operand select through the adder's read mux into its magnitude compare and
        # exponent difference, the multiplier's 36-bit product from its operand capture through the DSP tiles and the
        # two partial-product adders (74.8 MHz), the adder's exponent pack through its result into the register file
        # (82.5 MHz), the microcode operand select through the scaler's exponent adder and overflow classification into
        # the clear of its output register (79.0 MHz), the multiplier's product register through the sticky reduction
        # and the rounder into the register file (83.1 MHz), the adder's close-cancellation normalizer (91.4 MHz), the
        # microcode operand select through the multiplier's read mux into its exponent adder (90.6 MHz, then 105.8).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_product=2, stage_input=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_decode=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        # 76.2 MHz lean: the microcode ROM through the adder's operand read mux into its magnitude compare and first-
        # stage exponent difference, the multiplier's post-product cone into the register file (92.9 MHz), the adder's
        # last stage through the packer's rounding increment and the register file's write mux (88.3 MHz), the adder's
        # close-cancellation normalizer (93.1 MHz, then 100.1). For margin, the microcode ROM through the scaler's read
        # mux and exponent adder into its output register (then 101.6), the adder's input register through its
        # magnitude compare into the exponent difference (then 101.8), the microcode ROM through the multiplier's read
        # mux into its operand capture (then 104.1); that last stage without the adder's decode stage measured 94.0 MHz.
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_decode=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        # 106.9 MHz lean: the multiplier's product-completion carry chain through its rounder into the register file,
        # the microcode word through the adder's operand read mux and finiteness decode into the exponent-difference
        # subtractor (138.6 MHz), the adder's close-cancellation normalizer (140.9 MHz), the adder's rounder and
        # special-case select into the register file (141.5 MHz, then 159.8).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_product=1, stage_pack=1),
        ),
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.YOSYS_ECP5,
        100,
        # 70.3 MHz lean: the microcode operand select through the adder's read mux into its magnitude compare and
        # exponent difference, the multiplier's 36-bit product from its operand capture through the DSP tiles and the
        # two partial-product adders (72.3 MHz), a state register through the scaler's read mux, exponent adder and
        # overflow classification into the clear of its output register (76.6 MHz), the multiplier's product register
        # through the sticky reduction and the rounder into the register file (80.4 MHz), the adder's exponent pack
        # through its result into the register file (82.0 MHz), the adder's close-cancellation normalizer (89.5 MHz),
        # the microcode operand select through the multiplier's read mux into its exponent adder (95.6 MHz, then 104.4).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_product=2, stage_input=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_decode=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # 80.8 MHz lean: the microcode ROM through the scaler's read mux, exponent add and overflow compare into its
        # output register, the register file through the adder's operand read mux into its magnitude compare and first-
        # stage exponent difference (78.7 MHz), the adder's last stage through the pack range classification and the
        # register file's write mux (86.5 MHz), the multiplier's post-product cone through the packer's rounding
        # increment into the register file (88.2 MHz), the adder's close-cancellation normalizer (86.8 MHz, then 104.2).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_product=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        # 102.3 MHz lean: the multiplier's product-completion carry chain through its rounder into the register file,
        # the microcode word through the adder's operand read mux and magnitude comparison into the exponent-difference
        # subtractor (139.3 MHz), the adder's close-cancellation normalizer (144.8 MHz), the adder's normalize shift
        # count through the exponent adjustment and the packer's special-case select into the register file (145.2 MHz,
        # then 157.1).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_product=1, stage_pack=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    # polar: two off-catalogue 2-vector CORDIC kernels (no scalar-lane SPEC). to_polar fuses atan2+hypot into one
    # vectoring; from_polar coalesces sin+cos into one rotation. Every CORDIC row in the matrix is closed from lean, one
    # stage per critical path in the order its comment lists, each figure the f_max at which that path was critical;
    # the normalizer, the rounder and the multiplier named there are the CORDIC's own. The unrolling stays at its
    # default except where a comment says the iteration itself was the path. An instance running one mode is
    # elaborated for it alone, so only foc's CORDIC carries both modes' datapaths.
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        # 40.3 MHz lean: the normalizer (the post-product angle register through the whole normalizer, exponent subtract
        # and rounder into the register file), the normalizer's second half through the rounder (74.3 MHz), the multiply
        # -- operand register through the DSP tiles and their fabric sum (78.5 MHz), the multiplier's output through the
        # normalizer's first half (88.9 MHz, then 106.9).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=2, stage_normalize=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 38.2 MHz lean: the normalizer, its second half through the rounder into the register file (62.6 MHz), the
        # multiplier's output into the normalizer's first half (79.4 MHz, then 103.5).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 71.9 MHz lean: the CORDIC multiplier's fabric sum tail through the normalizer and the rounder into the
        # register file, the same tail through the normalizer's first half into its barrier (117.4 MHz), the
        # normalizer's second half through the rounder into the register file (151.9 MHz with 0.08 ns to spare; then
        # 156.6). Splitting the product instead of the second barrier made that cone worse (104.3 MHz), retiming pulling
        # the reduction's sum register back into the adder.
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        # 50.8 MHz lean: the normalizer (through the rounder into the register file), the multiply (84.5 MHz), the
        # normalizer's second half through the rounder (88.9 MHz, then 105.4).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=2, stage_normalize=1, stage_pack=1)),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 47.4 MHz lean: the normalizer, fmul's pack rounding into the register file (99.0 MHz, then 101.2). No stage
        # buys margin: the normalizer's second barrier, the CORDIC's pack, and both measured 100.1, 99.9 and 98.9 MHz,
        # the CORDIC iteration being the worst path by then.
        ops=_op_config(
            _F_e6m18, fmul=FMulOptions(stage_pack=1), fcordic=FCordicOptions(stage_product=1, stage_normalize=1)
        ),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 96.8 MHz lean: the CORDIC's normalizer through the exponent and the rounder into the held sine (then 152.5).
        # For margin, the microcode through fmul's read mux into its unregistered DSP operand (then 156.7).
        ops=_op_config(
            _F_e6m18, fmul=FMulOptions(stage_input=1), fcordic=FCordicOptions(stage_product=1, stage_normalize=1)
        ),
        name="from_polar_e6m18",
    ),
    # rigid_body_rates: the pivoted 3x3 Gauss-Jordan inversion -- conditional-swap select networks feeding one pooled
    # divider. Lean start per the closure procedure.
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        # 75.9 MHz lean: the microcode operand read mux into the divider's folded first-digit subtract and digit select,
        # the read mux into the multiplier's DSP product (76.6 MHz), the read mux into the adder's magnitude compare and
        # exponent select (80.8 MHz), the adder's normalize-shift exponent subtract through the pack's result select and
        # a long route into the register-file write select (97.8 MHz, then 101.8). For margin, the divider's folded
        # first digit from its latched divisor (then 103.5).
        ops=_op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1, stage_decode=1),
        ),
        name="rigid_body_rates_e6m18",
    ),
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 79.7 MHz lean: the microcode through the adder's operand read mux into its exponent difference, the
        # multiplier's DSP product through its pack rounding into the register file (89.9 MHz), the adder's close-
        # cancellation normalize cascade (90.6 MHz, then 105.9).
        ops=_op_config(_F_e6m18, fadd=FAddOptions(stage_input=1, stage_normalize=1), fmul=FMulOptions(stage_pack=1)),
        name="rigid_body_rates_e6m18",
    ),
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 139.8 MHz lean: the microcode word through the adder's operand read mux into its magnitude compare and
        # exponent difference, the register file through the multiplier's operand read mux into the unregistered DSP
        # input (143.2 MHz), the register file through the divider's operand read mux into its folded first-digit
        # subtracts (146.2 MHz, then 157.9).
        ops=_op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
        ),
        name="rigid_body_rates_e6m18",
    ),
    # flux_observer: a short stateful clamped fadd/fmul/fsort integrator feeding the same CORDIC vectoring as to_polar.
    _for_example(
        "flux_observer",
        FlowId.YOSYS_ECP5,
        100,
        # 42.3 MHz lean: the normalizer, its second half through the rounder (79.4 MHz), the multiply (81.5 MHz), the
        # normalizer's first half off the post-product angle register (89.3 MHz), the microcode through the adder's read
        # mux into its magnitude compare and exponent difference (92.1 MHz, then 101.7). For margin, the adder's pack
        # through its rounder into the register-file write select (then 105.6).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fcordic=FCordicOptions(stage_product=2, stage_normalize=2, stage_pack=1),
            fsort=FSortOptions(),
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.DIAMOND_ECP5,
        100,
        # 39.7 MHz lean: the normalizer, its second half through the rounder into the register file (73.8 MHz), the
        # multiplier's output into the normalizer's first half (78.6 MHz), fmul's pack rounding into the register file
        # (97.8 MHz, then 104.8).
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_pack=1),
            fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1),
            fsort=FSortOptions(),
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.VIVADO_ARTIX7,
        150,
        # 80.2 MHz lean: the CORDIC multiplier's fabric sum tail through the normalizer and the rounder into the
        # register file, the same tail through the normalizer's first half into its barrier (119.4 MHz), the
        # normalizer's second half through the rounder into the register file (144.5 MHz), the microcode fetch through
        # the adder's read mux and decode into its exponent difference (147.3 MHz), the multiplier's fabric sum tail
        # through the normalizer's top level into its first barrier, which with both barriers taken only the product
        # split reaches (149.7 MHz, then 156.8); the two-by-two split made it worse (141.6 MHz), the three-by-three
        # closes it.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fcordic=FCordicOptions(stage_product=3, stage_normalize=2, stage_pack=1),
            fsort=FSortOptions(),
        ),
    ),
    # foc: that observer embedded in a full current controller -- one CORDIC serving the observer's atan2 and the Park
    # rotation, the sorter, and the divides of the limiter and the modulator, over the widest microcode word in the
    # matrix. Only the Vivado flow is measured here; the ECP5 rows are absent rather than guessed, since closure is
    # only ever established by running the flow.
    _for_example(
        "foc",
        FlowId.VIVADO_ARTIX7,
        150,
        # 66.6 MHz lean: the CORDIC's normalizer, entered from the product's fabric sum and run through the rounder into
        # the held magnitude, the product's fabric sum into the normalizer's first half (111.4 MHz), and again with the
        # split product's final adder retimed into that same half (113.0 MHz), the microcode through the adder's read
        # mux into its magnitude compare and exponent difference (134.4 MHz), the microcode through fmul's read mux
        # into the DSP operand (134.9 MHz), the register file through the divider's read mux into its folded first
        # digit (139.4 MHz), the CORDIC's back-end select through the normalizer's top two levels into its barrier
        # (136.7 MHz), the sorter's compare and select into the register file (145.0 MHz), the register file through the
        # scaler's read mux into its exponent adder and overflow check (142.8 MHz, then 152.2 with 0.10 ns to spare). It
        # stops there: the next three paths' stages -- the divider's first digit, the CORDIC's pack, the adder's
        # normalizer -- each left it at or under that (152.0, 152.0, 150.4 MHz), as did the first two together and all
        # three (152.0, 150.4 MHz), the design being congestion-bound by then.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fcordic=FCordicOptions(stage_product=3, stage_normalize=2),
            fsort=FSortOptions(stage_input=1),
        ),
    ),
    # imu_fusion: the fusion capstone -- three norm/rsqrt chains (a root and a division each on the one divider, one
    # feeding the coarse alignment), the sorter-backed clamp, and real gate branches over the heaviest register pressure
    # in the matrix, in the plain and the ffma-contracted datapaths. The Euclidean norms expand by exact exponent
    # scaling, a `filog2` per leg and its scalings.
    _for_example(
        "imu_fusion",
        FlowId.YOSYS_ECP5,
        100,
        # 68.4 MHz lean: the microcode's operand select through the multiplier's read mux into the DSP product, the
        # scaler's operand select off the microcode into its exponent adder and overflow check (75.4 MHz), the adder's
        # operand select off the microcode into its magnitude compare (75.7 MHz), the divisor straight off the register
        # file into the divider's folded first digit (88.2 MHz), the adder's pack rounding into the register file (89.2
        # MHz), the multiplier's pack rounding into the register file (97.7 MHz), the divider's folded first digit from
        # its latched divisor (99.2 MHz, then 102.9).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1, stage_decode=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsort=FSortOptions(),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        # 82.1 MHz lean: the microcode through the adder's operand read mux into its exponent difference, the microcode
        # through the scaler's read mux into its zero/infinity classification (93.0 MHz), the multiplier's DSP product
        # through its pack rounding into the register file (88.9 MHz), the microcode through the multiplier's operand
        # read mux into its DSP operand (98.6 MHz), the adder's pack tail into the register file (91.2 MHz, then 104.1).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsort=FSortOptions(),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        # 132.0 MHz lean: the adder's exponent correction through its pack into the register file, the microcode through
        # the multiplier's read mux into the DSP operand (135.5 MHz), the register file through the divider's read mux
        # into its folded first digit (132.4 MHz), the register file through the scaler's read mux into its exponent
        # adder and overflow check (140.8 MHz), the register file through the adder's read mux into its magnitude
        # compare and exponent difference (148.4 MHz), the microcode through the sorter's read mux and compare into its
        # max register (151.1 MHz with 0.05 ns to spare; then 157.3).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_output=1, stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsort=FSortOptions(stage_input=1),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.YOSYS_ECP5,
        100,
        # 67.3 MHz lean: the microcode's operand select through the multiplier's read mux into the DSP product, the same
        # into the fma's product (67.5 MHz), the scaler's operand select off the microcode into its exponent adder (71.4
        # MHz), the adder's operand select off the microcode into its magnitude compare (79.4 MHz), the fma's sticky
        # through its pack rounding into the register file (74.4 MHz), the dividend select off the microcode into the
        # divider's folded first digit (83.8 MHz), the fma's normalize shift (86.8 MHz), the adder's exponent through
        # its pack rounding into the register file (81.1 MHz), the divider's folded first digit from its latched divisor
        # (95.2 MHz), the multiplier's exponent adjust through its pack into the register file (95.5 MHz, then 100.1).
        # For margin, the fma's product exponent adjust through its alignment-shift count into the alignment register
        # (then 105.1).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1, stage_decode=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_output=1, stage_normalize=1),
            fsort=FSortOptions(),
            wmultiplier=18,
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        # 82.9 MHz lean: the multiplier's DSP product through its pack rounding into the register file, the microcode
        # through the adder's operand read mux into its exponent difference (84.2 MHz), the microcode through the
        # scaler's read mux into its exponent adder and overflow detect (85.4 MHz), the fma's close-cancellation
        # normalize cascade (84.0 MHz), the fma's sticky through its pack rounding into the register file (90.2 MHz),
        # the microcode through the fma's operand read mux into its DSP operand (97.9 MHz, then 102.5).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fsort=FSortOptions(),
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        # 133.8 MHz lean: the microcode through the fma's read mux into its DSP operand, the divisor off the register
        # file into the divider's folded first digit (134.7 MHz), the microcode through the multiplier's read mux into
        # its DSP operand (137.1 MHz), the register file through the scaler's read mux into its exponent adder and
        # overflow check (135.2 MHz), the fma's close-cancellation normalize shift (149.4 MHz), the adder's exponent
        # difference, retimed back toward its inputs, through its alignment shift (150.5 MHz with 0.02 ns to spare), the
        # microcode through the sorter's read mux into its max register (150.4 MHz, then 154.5).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_normalize=1),
            fsort=FSortOptions(stage_input=1),
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    # kepler: a CORDIC rotation inside a data-dependent Newton back-edge loop -- the only II>1 operator in a loop in the
    # matrix.
    _for_example(
        "kepler",
        FlowId.YOSYS_ECP5,
        100,
        # 53.5 MHz lean: the normalizer (through the rounder into the register file), the multiply (87.5 MHz), the
        # normalizer's second half through the rounder (95.0 MHz, then 103.6).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=2, stage_normalize=1, stage_pack=1)),
    ),
    _for_example(
        "kepler",
        FlowId.DIAMOND_ECP5,
        100,
        # 49.8 MHz lean: the normalizer, its first half (94.0 MHz, then 101.0). For margin, the normalizer's second half
        # through the rounder into the register file (then 104.9).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1)),
    ),
    _for_example(
        "kepler",
        FlowId.VIVADO_ARTIX7,
        150,
        # 96.7 MHz lean: the CORDIC's normalizer through the exponent and the rounder into the held sine (then 153.4).
        # For margin, the CORDIC's back-end magnitude through the normalizer's top two levels into its barrier (then
        # 154.7); fmul's input stage on top of it, or instead of it, measured lower (153.9, 150.9 MHz).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=2)),
    ),
    # iq_oscillator: the phase accumulator's float-to-integer conversion on the rounder, beside a CORDIC rotation.
    _for_example(
        "iq_oscillator",
        FlowId.YOSYS_ECP5,
        100,
        # 53.2 MHz lean: the CORDIC's normalizer, the integer-to-float converter's normalizer through its rounder into
        # the register file (56.4 MHz), the CORDIC's multiply (86.0 MHz), the CORDIC normalizer's second half through
        # the rounder (82.7 MHz), the converter's normalizer's second half through its rounder (98.5 MHz, then 101.3).
        # For margin, the CORDIC normalizer's second barrier (then 104.9): the worst path is the CORDIC iteration, which
        # no stage splits and whose halved unrolling made worse (97.3 MHz), so the gain is one of placement.
        _op_config(
            _F_e6m18,
            fcordic=FCordicOptions(stage_product=2, stage_normalize=2, stage_pack=1),
            frint=FRintOptions(),
            ffromint=FFromIntOptions(stage_normalize=1, stage_pack=1),
            wint_min=34,
        ),
    ),
    _for_example(
        "iq_oscillator",
        FlowId.DIAMOND_ECP5,
        100,
        # 40.4 MHz lean: the CORDIC's normalizer, the integer-to-float converter's normalizer through its rounder into
        # the register file (61.6 MHz), that converter's normalizer into its rounder's input register (75.0 MHz), the
        # CORDIC normalizer's second half through the rounder and the register file's write multiplexer (75.4 MHz),
        # fmul's pack rounding into the register file (92.8 MHz), the CORDIC iteration, which no stage splits and takes
        # the halved unrolling (98.5 MHz, then 103.4).
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_pack=1),
            fcordic=FCordicOptions(stage_product=1, stage_normalize=1, stage_pack=1, unroll100=50),
            frint=FRintOptions(),
            ffromint=FFromIntOptions(stage_normalize=1, stage_pack=1),
            wint_min=34,
        ),
    ),
    _for_example(
        "iq_oscillator",
        FlowId.VIVADO_ARTIX7,
        150,
        # 91.7 MHz lean: the CORDIC's normalizer through the rounder into fmul's DSP operand, the integer-to-float
        # converter's normalizer through its rounder into the register file (116.2 MHz, then 153.9). For margin, the
        # CORDIC normalizer's second half through the rounder into the register file (then 156.2).
        _op_config(
            _F_e6m18,
            fcordic=FCordicOptions(stage_product=1, stage_normalize=1, stage_pack=1),
            frint=FRintOptions(),
            ffromint=FFromIntOptions(stage_normalize=1),
            wint_min=34,
        ),
    ),
    # The divider elaborated for the root alone. Diamond takes it at the wide format, where the digit selection LSE
    # folds under resource sharing is the difference between 81 and 135 MHz.
    SynthTarget(
        kernel=_root_only_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e6m18),
        name="root_only_e6m18",
    ),
    SynthTarget(
        kernel=_root_only_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e8m36),
        name="root_only_e8m36",
    ),
    SynthTarget(
        kernel=_root_only_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(_F_e6m18),
        name="root_only_e6m18",
    ),
    # equal_temperament at the wide format, where the exp2 and log2 tables are 256 and 725 entries of 204 bits: twelve
    # block RAMs, or in LUT logic three times the whole machine, which is what this row guards against. Its 55
    # multipliers outgrow the default device.
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        # 30.9 MHz lean, which here has every product split in two for the 18-bit multipliers, a partial-product sum
        # taking it to three: the log2 normalizer through its rounder into the register file, the exp2 Horner product's
        # nine partial products summed in one cycle (47.9 MHz), the log2 normalizer's second half through the rounder
        # into the register file (56.5 MHz), the log2 Horner product's partial-product sum (80.3 MHz), the log2
        # normalizer's leading-zero detect into its mid-cascade register (75.5 MHz), the log2 final product's partial-
        # product sum (81.6 MHz), the multiplier's product register through its pack rounding into the register file
        # (89.7 MHz), the adder's alignment shifter (91.5 MHz, then 94.5). There the worst path is one net between the
        # log2 normalizer's two barriers that no stage shortens: a zero-detect that LSE makes the synchronous reset of
        # sixteen registers and place-and-route then carries on primary clock routing. The paths behind it are what
        # yield: the log2 normalizer's output through its rounder into the register file (92.2 MHz with that split), the
        # adder's cancellation normalizer (then 105.9).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_align=1, stage_normalize=1),
            fmul=FMulOptions(stage_product=2, stage_pack=1),
            fexp2=FExp2Options(stage_product=3),
            flog2=FLog2Options(
                stage_product=3, stage_product_final=3, stage_normalize=2, stage_normalize_output=1, stage_pack=1
            ),
            wmultiplier=18,
        ),
        device_class=DeviceClass.LARGE,
    ),
]

assert len({t.label for t in TARGETS}) == len(TARGETS)  # labels key build dirs and pytest ids
