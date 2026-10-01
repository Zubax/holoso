"""
The out-of-context synthesis matrix: a flat list of SynthTarget records, one per (kernel, flow, target f_max,
operator configuration). This is the single source of truth for what the example synthesis suite
(test_synth_examples) asserts can close timing.

Deliberately a dumb-simple data table -- literal frequencies, one explicit row per (example, flow), and a full typed
Options per row; duplication is preferred over indirection here. Catalogued example targets reuse the shared example
registry (_examples.SPECS) so each kernel is constructed once. Off-catalogue regression cores may be added as
SynthTarget records with example=None.

The bar (all at minimum speed grade): yosys-ecp5 and diamond-ecp5 at 100 MHz, vivado-artix7 at 150 MHz. Almost every
kernel closes lean; stage knobs are row-local and appear only where measured closure needs them.
"""

from collections.abc import Callable
from dataclasses import dataclass

from holoso import (
    FAddOptions,
    FCmpOptions,
    FCordicOptions,
    FDivOptions,
    FExp2Options,
    FFmaOptions,
    FILog2Options,
    FLog2Options,
    FloatFormat,
    FMulILog2Options,
    FMulOptions,
    FSortOptions,
    FSqrtOptions,
    OperatorOptions,
    Options,
)
from synth.flows import FlowId

from ._examples import SPECS, ekf1_stateful, imu_fusion, polar, rigid_body_rates

_F_e6m18 = FloatFormat(6, 18)
_F_e8m36 = FloatFormat(8, 36)
_F_e4m8 = FloatFormat(4, 8)  # the narrowest datapath; the float-free UART ignores it and takes its 16-bit floor


def _op_config(
    fmt: FloatFormat,
    *,
    fadd: FAddOptions | None = None,
    fmul: FMulOptions | None = None,
    fdiv: FDivOptions | None = None,
    fmul_ilog2: FMulILog2Options | None = None,
    filog2: FILog2Options | None = None,
    fcmp: FCmpOptions | None = None,
    ffma: FFmaOptions | None = None,
    fexp2: FExp2Options | None = None,
    flog2: FLog2Options | None = None,
    fsqrt: FSqrtOptions | None = None,
    fcordic: FCordicOptions | None = None,
    fsort: FSortOptions | None = None,
    wmultiplier: int | None = None,
) -> Options:
    """
    The Options for fmt; pass an options object to give an operator stage knobs, else that operator is lean.
    ffma/fexp2/flog2/fsqrt/fcordic/fsort are absent unless supplied, so MAC chains stay expanded (fmul + fadd)
    and a kernel that uses no transcendental, root, or min/max needs no such module.
    """
    return Options(
        OperatorOptions(
            fadd=fadd or FAddOptions(),
            fmul=fmul or FMulOptions(),
            fdiv=fdiv or FDivOptions(),
            fmul_ilog2=fmul_ilog2 or FMulILog2Options(),
            filog2=filog2 or FILog2Options(),
            fcmp=fcmp or FCmpOptions(),
            ffma=ffma,
            fexp2=fexp2,
            flog2=flog2,
            fsqrt=fsqrt,
            fcordic=fcordic,
            fsort=fsort,
        ),
        ffmt=fmt,
        wmultiplier=wmultiplier,
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


TARGETS: list[SynthTarget] = [
    # Most of the catalogue closes the bar at lean (no optional stages) on all three tools -- verified by a full lean
    # baseline. One explicit row per (example, flow); duplication is intentional.
    _for_example("madd", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("madd", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("madd", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("poly3", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fmul=FMulOptions(stage_pack=1))),
    _for_example("poly3", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("poly3", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("signal_window", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("signal_window", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("signal_window", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("iir1_hpf", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fadd=FAddOptions(stage_output=1))),
    _for_example("iir1_hpf", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("iir1_hpf", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("iir1_lpf", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, fadd=FAddOptions(stage_output=1))),
    _for_example("iir1_lpf", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("iir1_lpf", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example(
        "pid",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_output=1),
            fdiv=FDivOptions(stage_output=1),
        ),
    ),
    # Diamond's critical path is the fmul post-product cone (DSP product register through pack into a seven-input
    # register write select), 21 logic levels; one pack stage splits it, as on recip_newton.
    _for_example(
        "pid",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1, stage_output=1), fmul=FMulOptions(stage_pack=1)),
    ),
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
    # Diamond's critical path here is the fmul post-product cone (DSP product register through pack/normalize into
    # the register file), 18 logic levels at 58% route. One pack stage splits it; the other two flows close lean.
    _for_example("recip_newton", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, fmul=FMulOptions(stage_pack=1))),
    # Vivado closes this lean only on some releases: 2025.2 clears 150 MHz, 2026.1 misses by 0.078 ns on the same
    # adder normalize cascade the Diamond rows above split. One barrier holds it on both.
    _for_example("recip_newton", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("integrator", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "integrator",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_output=1),
            fmul=FMulOptions(stage_output=1),
            fdiv=FDivOptions(stage_output=1),
        ),
    ),
    _for_example("integrator", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    # Straight-line multiply-accumulate chains; both close lean on all three flows.
    _for_example("fir", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("fir", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("fir", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("biquad", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    # Diamond routes the adder's close-cancellation normalize cascade long enough to miss 100 MHz once the
    # composed scaling shortens the schedule around it; one barrier splits it.
    _for_example("biquad", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("biquad", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("uart_tx", FlowId.YOSYS_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_tx", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_tx", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e4m8)),
    _for_example("uart_rx", FlowId.YOSYS_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_rx", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e4m8)),
    _for_example("uart_rx", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e4m8)),
    # nextpnr's critical path runs from the microcode word through the ilog2 multiplier's operand read mux into its
    # exponent adders, route-dominated; the input stage splits it at the read mux.
    _for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_decode=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            fdiv=FDivOptions(stage_input=1, stage_output=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_product=1),
        ),
    ),
    # The same EKF with a second multiplier: the allocator binds the co-issued products across the two instances.
    # Against the one-multiplier row: 18 percent more LUTs for a 15 percent shorter transaction (DESIGN.md).
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_product=1, instances=2),
        ),
        name="ekf1_stateless_e6m18_fmul2",
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_decode=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            # The scaler's exponent add and overflow compare into its output register's reset (96.7 MHz).
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # Lean, the microcode word through the adder's b-port read mux into its exponent difference is critical
        # (92.1 MHz) and takes the adder's input stage; the fmul post-product cone (DSP product register through
        # pack rounding into the register file, 19 logic levels) then limits it to 93.9 MHz and takes the pack stage;
        # the adder's close-cancellation normalize shift (s2 to s3, 92.4 MHz) then takes the normalize stage.
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1, stage_normalize=1), fmul=FMulOptions(stage_pack=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1),
            fmul=FMulOptions(stage_product=1),
        ),
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
    _for_example(
        "cordic_sincos",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_decode=1, stage_normalize=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_decode=1),
        ),
    ),
    _for_example("cordic_sincos", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example("octave_index", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "octave_index",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_output=1),
            fdiv=FDivOptions(stage_output=1),
        ),
    ),
    _for_example("octave_index", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    # octave_index's transcendental sibling. The exp2/log2 Horner products and log2's final f*C(f) product need
    # registered partial-product reduction; this one config closes all three flows.
    _for_example(
        "equal_temperament",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_pack=1),
            # The exp2 pack rounding carry (93.3 MHz): its pack stage.
            fexp2=FExp2Options(stage_reduce=1, stage_product=2, stage_pack=1),
            flog2=FLog2Options(stage_product=2, stage_product_final=2, stage_normalize=2, stage_pack=1),
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_normalize=1),
            # The DSP product through the multiplier's pack rounding into the register file (87.9 MHz): the pack
            # stage.
            fmul=FMulOptions(stage_pack=1),
            fexp2=FExp2Options(stage_product=2),
            flog2=FLog2Options(stage_product=2, stage_product_final=2, stage_normalize=1, stage_pack=1),
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_normalize=1),
            fexp2=FExp2Options(stage_product=2),
            flog2=FLog2Options(stage_product=2, stage_product_final=2, stage_normalize=1, stage_pack=1),
        ),
    ),
    _for_example("remainder", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "remainder",
        FlowId.DIAMOND_ECP5,
        100,
        # The adder's subtraction result through its normalization shift (99.3 MHz): the normalize stage.
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1), fdiv=FDivOptions(stage_input=1, stage_output=1)),
    ),
    _for_example("remainder", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1))),
    _for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        # Closed lean from the product stage alone (60.2 MHz), one stage per critical path in this order: the
        # register file into the adder's exponent difference, the partial-product sum, the multiplier's pack rounding,
        # the register file into the scaler's exponent adder, the adder's pack into the register file, the adder's
        # normalization, the register file into the multiplier's exponent adder, the scaler's decode, the
        # multiplier's pack rounding into the register file.
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_product=2, stage_pack=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(
                stage_input=1, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1, stage_output=1
            ),
            fmul=FMulOptions(stage_input=1, stage_product=1, stage_pack=1, stage_output=1),
            fdiv=FDivOptions(stage_input=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
            fdiv=FDivOptions(stage_input=1, stage_pack=1, stage_output=1),
        ),
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_product=2, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_decode=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # Closed lean from the product stage alone (81.0 MHz), one stage per critical path in this order: the register
        # file into the adder's exponent difference, the multiplier's pack rounding into the register file (84.9 MHz),
        # the adder's normalize shift, the adder's pack tail into the register file (91.4 MHz), the multiplier's DSP
        # cascade, which takes the split product (96.5 MHz, then 106.5).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_product=2, stage_pack=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_decode=1, stage_align=1, stage_normalize=2, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
            fdiv=FDivOptions(stage_input=1, stage_pack=1, stage_output=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    # polar: two off-catalogue 2-vector CORDIC kernels (no scalar-lane SPEC). to_polar fuses atan2+hypot into one
    # vectoring; from_polar coalesces sin+cos into one rotation. Every CORDIC row in the matrix is closed lean (the
    # unrolling at its default), one stage per critical path in the order its comment lists, each figure the f_max at
    # which that path was critical; the normalizer, the rounder and the multiplier named there are the CORDIC's own.
    # An instance running one mode is elaborated for it alone, so only foc's CORDIC carries both modes' datapaths.
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        # 41.7 MHz lean: the normalizer, the multiplier's operand (75.9 MHz), the multiply (79.5 MHz), the normalizer's
        # second half through the rounder (84.4 MHz), its first half (88.5 MHz, then 112.9).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=2, stage_product=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 40.8 MHz lean: the normalizer, its first half (69.9 MHz), its second half through the rounder (76.1 MHz),
        # the multiplier's operand into its DSP (93.8 MHz), the multiplier's output into the normalizer's first barrier
        # (89.7 MHz, then 111.1).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=2, stage_product=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 70.1 MHz lean: the vectoring's post-product logic through the normalizer, and into its barrier (116.5 MHz),
        # the multiplier's operand into its DSP cascade (141.6 MHz), the normalizer's second half through the rounder
        # (149.6 MHz, then 159.4).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=2, stage_product=1, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        # 55.4 MHz lean: the normalizer, the multiplier's operand (70.6 MHz), the multiply (86.7 MHz), the normalizer's
        # second half through the rounder (92.3 MHz, then 109.6).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=1, stage_product=2, stage_pack=1)),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 49.0 MHz lean: the normalizer, the CORDIC's done flag into the multiplier's operand (88.6 MHz), fmul's pack
        # rounding into the register file (96.6 MHz), the normalizer's second half through the rounder (94.6 MHz), the
        # CORDIC iteration, which takes the halved unrolling (99.98 MHz, then 106.6).
        ops=_op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_pack=1),
            fcordic=FCordicOptions(unroll100=50, stage_normalize=1, stage_product=1, stage_pack=1),
        ),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 102.9 MHz lean: the normalizer, the DSP cascade (129.3 MHz, then 152.5).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=1, stage_product=1)),
        name="from_polar_e6m18",
    ),
    # rigid_body_rates: the pivoted 3x3 Gauss-Jordan inversion -- conditional-swap select networks feeding one pooled
    # divider. Lean start per the closure procedure; stage knobs appear only where measured closure needs them.
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
        ),
        name="rigid_body_rates_e6m18",
    ),
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_output=1),
            fdiv=FDivOptions(stage_input=1),
        ),
        name="rigid_body_rates_e6m18",
    ),
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(_F_e6m18, fadd=FAddOptions(stage_input=1), fmul=FMulOptions(stage_input=1)),
        name="rigid_body_rates_e6m18",
    ),
    # flux_observer: a short stateful clamped fadd/fmul/fsort integrator feeding the same CORDIC vectoring as to_polar.
    _for_example(
        "flux_observer",
        FlowId.YOSYS_ECP5,
        100,
        # 45.0 MHz lean: the normalizer, the multiplier's operand (72.8 MHz), the multiply (78.5 MHz), the product into
        # the normalizer's first half (81.0 MHz), the microcode into the adder's exponent difference (93.8 MHz), the
        # normalizer's second half through the rounder (82.5 MHz), fmul's pack rounding (99.6 MHz, then 105.4).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_pack=1),
            fcordic=FCordicOptions(stage_normalize=2, stage_product=2, stage_pack=1),
            fsort=FSortOptions(),
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.DIAMOND_ECP5,
        100,
        # 38.8 MHz lean: the multiplier's output through the normalizer, its second half through the rounder (72.7 MHz),
        # its first half (79.6 MHz), the multiplier's output into its first barrier (82.6 and 88.4 MHz, the product
        # stage twice), the rounder's own rounding (99.0 MHz), fmul's pack rounding (99.5 MHz), then the adder's
        # normalize shift (97.0 MHz, then 101.9): the vectoring magnitude into the normalizer's first barrier was
        # critical there, which no knob splits, and the shift was the next path in that report.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_normalize=1),
            fmul=FMulOptions(stage_pack=1),
            fcordic=FCordicOptions(stage_normalize=2, stage_product=2, stage_pack=2),
            fsort=FSortOptions(),
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.VIVADO_ARTIX7,
        150,
        # 78.9 MHz lean: the vectoring's post-product logic through the normalizer, and into its barrier (110.9 MHz),
        # the multiplier's operand into its DSP cascade (134.9 MHz), the normalizer's second half through the rounder
        # (138.2 MHz), the DSP's output (149.1 MHz), the post-product logic into the normalizer's first barrier
        # (144.8 MHz, then 150.9).
        _op_config(
            _F_e6m18,
            fcordic=FCordicOptions(stage_normalize=2, stage_product=3, stage_pack=1),
            fsort=FSortOptions(),
        ),
    ),
    # foc: that observer embedded in a full current controller -- one CORDIC serving the observer's atan2 and the Park
    # rotation, the sorter, and the divides of the limiter and the modulator, over the widest microcode word in the
    # matrix. 65.9 MHz lean: the product's fabric sum, and again (66.7 MHz), the normalizer (77.1 MHz), the
    # post-product logic into it (102.7 MHz) and into its first barrier (127.7 MHz), the microcode into the adder's
    # exponent difference (136.2 MHz) and into fmul's DSP operand (140.8 MHz), the adder's pack tail into the register
    # file (142.9 MHz), the register file into the scaler (143.5 MHz, then 151.6). Only the Vivado flow is measured
    # here; the ECP5 rows are absent rather than guessed, since closure is only ever established by running the flow.
    _for_example(
        "foc",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsort=FSortOptions(),
            fsqrt=FSqrtOptions(),
            fcordic=FCordicOptions(stage_normalize=2, stage_product=3),
        ),
    ),
    # The Euclidean norms expand by exact exponent scaling, a `filog2` per leg and its scalings. The ffma Vivado row
    # fell just under its fence on that (148.22) and took one stage on the hop its critical path named: the scaler's
    # input.
    # imu_fusion: the fusion capstone -- three norm/rsqrt chains (fsqrt/fdiv, one feeding the coarse alignment), the
    # sorter-backed clamp, and real gate branches over the heaviest register pressure in the matrix, in the plain and
    # the ffma-contracted datapaths. The native root retired the wall these rows used to close against (the flog2
    # Horner pmul). On the plain yosys row the deepest path is the sorter's compare cone entered straight off the
    # register file, which fsort stage_input splits. The exact install scheduling shifted the ffma Vivado row a hair
    # under its fence (19 ps) on the fadd subtract-normalize cone, which fadd stage_normalize splits.
    _for_example(
        "imu_fusion",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            # The adder's pack rounding through the register file's write select (98.5 MHz): the output stage.
            fadd=FAddOptions(stage_input=1, stage_pack=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(stage_input=1),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        # Closed lean (88.9 MHz), one stage per critical path in this order: the microcode word through the adder's
        # read mux into its exponent difference, the multiplier's DSP product through pack rounding into the register
        # file (78.5 MHz), the register file through the scaler's read mux into its exponent adder (97.4 MHz), the
        # adder's pack tail into the register file (99.7 MHz, then 101.3).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_pack=1),
            fmul=FMulOptions(stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            # The register file into the scaler's decode (-0.07 ns): the scaler's input stage.
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.YOSYS_ECP5,
        100,
        # Closed lean (66.1 MHz), one stage per critical path in this order: the microcode into the fma's DSP operand,
        # the microcode through the scaler's read mux into its overflow compare, the microcode into the multiplier's
        # DSP operand, the fma's normalize shift, the fma's pack rounding into the register file, the microcode into
        # the adder's exponent difference, the fma's product into its exponent decode, the adder's pack rounding into
        # the register file (at its output), the multiplier's pack rounding (101.7 MHz).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_normalize=1, stage_pack=1),
            wmultiplier=18,
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    # Closed lean (76.3 MHz), one stage per critical path in this order: the fma's sticky through its pack rounding
    # into the register file, the multiplier's pack rounding (83.3 MHz), the microcode through the scaler's read mux
    # into its exponent adder (89.0 MHz), the fma's normalize shift (90.7 MHz, then 101.0).
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(),
            ffma=FFmaOptions(stage_normalize=1, stage_pack=1),
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            # Two hops here are long and mostly route, so this row closes on one machine and not on another: the
            # microcode word into the adder's decode, and the register file into the sorter. A stage on each buys
            # 0.15 ns of slack to 0.44. The scaler's own decode is the next hop and must NOT be staged -- measured,
            # that costs 0.28 ns, the design being congestion-bound rather than deep by then.
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
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
        # 51.3 MHz lean: the normalizer, the multiplier's operand (72.0 MHz), the multiply (86.0 MHz), the normalizer's
        # second half through the rounder (88.7 MHz, then 103.0).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=1, stage_product=2, stage_pack=1)),
    ),
    _for_example(
        "kepler",
        FlowId.DIAMOND_ECP5,
        100,
        # 50.9 MHz lean: the normalizer, the CORDIC's done flag into the multiplier's operand (88.8 MHz), the
        # normalizer's second half through the rounder (96.4 MHz, then 106.3).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=1, stage_product=1, stage_pack=1)),
    ),
    _for_example(
        "kepler",
        FlowId.VIVADO_ARTIX7,
        150,
        # 100.5 MHz lean: the normalizer, the DSP cascade (130.5 MHz), the normalizer's first half (149.3 MHz, then
        # 152.8).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_normalize=2, stage_product=1)),
    ),
]

assert len({t.label for t in TARGETS}) == len(TARGETS)  # labels key build dirs and pytest ids
