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
    FAtan2Options,
    FCmpOptions,
    FDivOptions,
    FExp2Options,
    FFmaOptions,
    FILog2Options,
    FLog2Options,
    FloatFormat,
    FMulILog2Options,
    FMulOptions,
    FSincosOptions,
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
    fsincos: FSincosOptions | None = None,
    fatan2: FAtan2Options | None = None,
    fsort: FSortOptions | None = None,
    wmultiplier: int | None = None,
) -> Options:
    """
    The Options for fmt; pass an options object to give an operator stage knobs, else that operator is lean.
    ffma/fexp2/flog2/fsqrt/fsincos/fatan2/fsort are absent unless supplied, so MAC chains stay expanded (fmul + fadd)
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
            fsincos=fsincos,
            fatan2=fatan2,
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


# One measured CORDIC config per polar kernel closes all three flows, so the three per-flow rows share it (unlike the
# per-flow stage knobs elsewhere in the matrix).
_TO_POLAR_FATAN2 = FAtan2Options(unroll100=50, stage_pack=1, stage_normalize=2, stage_product=3)
_FROM_POLAR_FSINCOS = FSincosOptions(stage_pack=1, stage_product=2, stage_normalize=2)
_FLUX_OBSERVER_DIAMOND_FATAN2 = FAtan2Options(unroll100=50, stage_pack=1, stage_normalize=2, stage_product=2)
# kepler's fsincos (coalesced sin+cos per Newton iteration) dominates timing, so its measured closure coincides with
# from_polar's -- the same operator.
_KEPLER_FSINCOS = FSincosOptions(stage_pack=1, stage_product=2, stage_normalize=2)


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
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # The register file through the adder's b-port read mux into its exponent difference (81.5 MHz with no stage)
        # takes the adder's input stage, and its normalizing shift the normalize stage.
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1, stage_normalize=1)),
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
            fadd=FAddOptions(stage_input=1, stage_decode=1, stage_normalize=2, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_product=2, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_decode=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=2, stage_decode=1, stage_align=1, stage_normalize=2, stage_pack=1),
            fmul=FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
            fdiv=FDivOptions(stage_input=3, stage_pack=1, stage_output=1),
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
    # CORDIC; from_polar coalesces sin+cos.
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e6m18, fatan2=_TO_POLAR_FATAN2),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e6m18, fatan2=_TO_POLAR_FATAN2, fmul=FMulOptions(stage_output=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(_F_e6m18, fatan2=_TO_POLAR_FATAN2),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e6m18, fsincos=_FROM_POLAR_FSINCOS),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e6m18, fmul=FMulOptions(stage_pack=1), fsincos=_FROM_POLAR_FSINCOS),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(_F_e6m18, fmul=FMulOptions(stage_input=1), fsincos=_FROM_POLAR_FSINCOS),
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
    # flux_observer: a short stateful clamped fadd/fmul/fsort integrator feeding the same CORDIC atan2 as to_polar,
    # so the rows start from that operator's measured configuration. Diamond uses the smaller 2×2 product grid because
    # the 3×3 topology's extra registers congest the shared normalizer. On Vivado the register file reaches both the
    # sorter's compare cone and the DSP operand mux inside one period, so those two entries take an input stage each.
    _for_example(
        "flux_observer",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            # The adder's normalize shift through its exponent bias into the pack rounding (97.9 MHz): the pack stage.
            fadd=FAddOptions(stage_input=1, stage_pack=1),
            # The multiplier's pack rounding carry (98.1 MHz): its pack stage.
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fatan2=_TO_POLAR_FATAN2,
            fsort=FSortOptions(),
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            # The DSP product through the multiplier's pack rounding into the register file (98.1 MHz): the pack
            # stage.
            fmul=FMulOptions(stage_pack=1),
            fatan2=_FLUX_OBSERVER_DIAMOND_FATAN2,
            fsort=FSortOptions(),
        ),
    ),
    # The register file into the adder's exponent difference (-0.03 ns): the adder's input stage.
    _for_example(
        "flux_observer",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fatan2=_TO_POLAR_FATAN2,
            fsort=FSortOptions(stage_input=1),
        ),
    ),
    # foc: that observer embedded in a full current controller -- the same CORDIC atan2 plus from_polar's fsincos,
    # the sorter, and the divides of the limiter and the modulator, over the widest microcode word in the matrix.
    # Input stages split the DSP operand mux, the adder's exponent-difference carry chain, and the sorter's compare
    # cone; the adder's pack stage splits exponent correction from packing and register-file writeback.
    # Only the Vivado flow is measured here; the ECP5 rows are absent rather than guessed, since closure is only
    # ever established by running the flow.
    _for_example(
        "foc",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1),
            fsort=FSortOptions(stage_input=1),
            fsqrt=FSqrtOptions(),
            fatan2=_TO_POLAR_FATAN2,
            fsincos=_FROM_POLAR_FSINCOS,
        ),
    ),
    # The Euclidean norms expand by exact exponent scaling, a `filog2` per leg and its scalings. Two rows fell just
    # under their fence on that (99.19 and 148.22) and each took one stage on the hop its critical path named: the
    # extractor's input on diamond, the scaler's input on the ffma Vivado row.
    # imu_fusion: the fusion capstone -- three norm/rsqrt chains (fsqrt/fdiv, one feeding the coarse alignment), the
    # sorter-backed clamp, and real gate branches over the heaviest register pressure in the matrix, in the plain and
    # the ffma-contracted datapaths. The native root retired the wall these rows used to close against (the flog2
    # Horner pmul); the two ffma ECP5 rows each needed one more stage once that wall left. On yosys the deepest path
    # is the sorter's compare cone entered straight off the register file, which fsort stage_input splits; on diamond
    # it is the fadd normalize/pack tail into the register file, which fadd stage_pack splits -- and the plain
    # diamond row's fadd stage_output stays, since removing it only exposes the same tail one stage earlier. The
    # exact install scheduling shifted the plain diamond and ffma Vivado rows a hair under their fences (12 and
    # 19 ps): the diamond miss was a lone controller clock-enable route, re-closed by splitting the fadd pack tail;
    # the Vivado miss was the fadd subtract-normalize cone, which fadd stage_normalize splits.
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
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            filog2=FILog2Options(stage_input=1),
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
        _op_config(
            _F_e6m18,
            # The rounding carry chain plus register-file writeback exceeds the period; separate them at the output.
            fadd=FAddOptions(stage_input=1, stage_pack=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
            wmultiplier=18,
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_pack=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fsqrt=FSqrtOptions(),
            fsort=FSortOptions(),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
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
    # kepler: fsincos inside a data-dependent Newton back-edge loop -- the only II>1 operator in a loop in the matrix.
    # The adder's aligned subtraction through its pack rounding carry (99.3 MHz): the pack stage.
    _for_example(
        "kepler",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(_F_e6m18, fadd=FAddOptions(stage_pack=1), fsincos=_KEPLER_FSINCOS),
    ),
    _for_example(
        "kepler",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(_F_e6m18, fsincos=FSincosOptions(stage_pack=1, stage_product=2, stage_normalize=1)),
    ),
    # At lean the adder's normalize shifter is one combinational barrel shift between the s2 and s3 boundaries,
    # and on artix7 it is what this row's critical path runs through (s2_raw_result -> s3_sub_aligned, ~77% route)
    # once the sincos is staged off it. One barrier splits the shift and puts the path back on the multiplier.
    _for_example(
        "kepler",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1), fsincos=_KEPLER_FSINCOS),
    ),
]

assert len({t.label for t in TARGETS}) == len(TARGETS)  # labels key build dirs and pytest ids
