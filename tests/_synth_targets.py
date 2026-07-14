"""
The out-of-context synthesis matrix: a flat list of SynthTarget records, one per (kernel, flow, target f_max,
operator configuration). This is the single source of truth for what the example synthesis suite
(test_synth_examples) asserts can close timing.

Deliberately a dumb-simple data table -- literal frequencies, one explicit row per (example, flow), and a full typed
OpConfig per row; duplication is preferred over indirection here. Catalogued example targets reuse the shared example
registry (_examples.SPECS) so each kernel is constructed once. Off-catalogue regression cores may be added as
SynthTarget records with example=None.

The bar (all at minimum speed grade): yosys-ecp5 and diamond-ecp5 at 100 MHz, vivado-artix7 at 150 MHz. Almost every
kernel closes lean; stage knobs are row-local and appear only where measured closure needs them.
"""

from collections.abc import Callable
from dataclasses import dataclass

from holoso import (
    FAddOperator,
    FAtan2Operator,
    FCmpOperator,
    FDivOperator,
    FExp2Operator,
    FFmaOperator,
    FloatFormat,
    FLog2Operator,
    FMulILog2OperatorFamily,
    FMulOperator,
    FSincosOperator,
    OpConfig,
)
from synth.flows import FlowId

from ._examples import SPECS, ekf1_stateful, imu_frame_transform, polar

F_e6m18 = FloatFormat(6, 18)
F_e8m36 = FloatFormat(8, 36)
F_e4m8 = FloatFormat(4, 8)  # the narrow uart byte format (per examples/uart.py)


def op_config(
    fmt: FloatFormat,
    *,
    fadd: FAddOperator | None = None,
    fmul: FMulOperator | None = None,
    fdiv: FDivOperator | None = None,
    fmul_ilog2: FMulILog2OperatorFamily | None = None,
    fcmp: FCmpOperator | None = None,
    ffma: FFmaOperator | None = None,
    fexp2: FExp2Operator | None = None,
    flog2: FLog2Operator | None = None,
    fsincos: FSincosOperator | None = None,
    fatan2: FAtan2Operator | None = None,
) -> OpConfig:
    """
    The OpConfig for fmt; pass a fully-constructed operator to give it stage knobs, else that operator is lean.
    ffma/fexp2/flog2/fsincos/fatan2 are absent unless supplied, so MAC chains stay expanded (fmul + fadd) and a
    kernel that uses no transcendental needs no such module.
    """
    return OpConfig(
        fadd=fadd or FAddOperator(fmt),
        fmul=fmul or FMulOperator(fmt),
        fdiv=fdiv or FDivOperator(fmt),
        fmul_ilog2=fmul_ilog2 or FMulILog2OperatorFamily(fmt),
        fcmp=fcmp or FCmpOperator(fmt),
        ffma=ffma,
        fexp2=fexp2,
        flog2=flog2,
        fsincos=fsincos,
        fatan2=fatan2,
    )


def op_config_staged_output(fmt: FloatFormat) -> OpConfig:
    """
    op_config(fmt) with an output register stage on the wide arithmetic operators (fadd/fmul/fdiv), enabled locally on
    just the rows whose timing closure needs it. fmul_ilog2 and fcmp have no output stage and stay lean.
    """
    return op_config(
        fmt,
        fadd=FAddOperator(fmt, stage_output=1),
        fmul=FMulOperator(fmt, stage_output=1),
        fdiv=FDivOperator(fmt, stage_output=1),
    )


@dataclass(frozen=True, slots=True)
class SynthTarget:
    """One synthesis closure target: a kernel synthesized for one flow under one OpConfig, asserted to meet f_max."""

    kernel: Callable[[], Callable[..., object]]
    flow: FlowId
    target_frequency_MHz: float
    ops: OpConfig
    name: str  # descriptive module/report label; unique per flow
    # The example name this exercises -- a SPECS name where one exists, or the bundled example's own name for an
    # off-catalogue kernel (imu, polar). Keys the FIR_PARITY_PENDING registry; None for a pure regression core.
    example: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name}-{self.flow.value}"


_SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def for_example(
    example: str,
    flow: FlowId,
    target_frequency_MHz: float,
    ops: OpConfig,
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
    fmt = ops.float_format
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


def _imu_frame_transform_kernel() -> Callable[..., object]:
    # Off-catalogue: the shaped matrix/vector ports have no scalar-lane SPEC, so this stateless kernel is referenced
    # directly rather than through the cosim registry.
    return imu_frame_transform.transform


def _to_polar_kernel() -> Callable[..., object]:
    # Off-catalogue (2-vector I/O); exercises the fused atan2+hypot CORDIC.
    return polar.to_polar


def _from_polar_kernel() -> Callable[..., object]:
    # Off-catalogue (2-vector I/O); exercises the coalesced sin+cos CORDIC.
    return polar.from_polar


# One measured CORDIC config per polar kernel closes all three flows, so the three per-flow rows share it (unlike the
# per-flow stage knobs elsewhere in the matrix).
_TO_POLAR_FATAN2 = FAtan2Operator(F_e6m18, unroll100=50, stage_pack=1, stage_normalize=2, stage_product=3)
_FROM_POLAR_FSINCOS = FSincosOperator(F_e6m18, stage_pack=1, stage_product=2, stage_normalize=2)
# kepler's fsincos (coalesced sin+cos per Newton iteration) dominates timing, so its measured closure coincides with
# from_polar's -- the same operator -- and one config closes all three flows.
_KEPLER_FSINCOS = FSincosOperator(F_e6m18, stage_pack=1, stage_product=2, stage_normalize=2)


TARGETS: list[SynthTarget] = [
    # Most of the catalogue closes the bar at lean (no optional stages) on all three tools -- verified by a full lean
    # baseline. One explicit row per (example, flow); duplication is intentional.
    for_example("madd", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("madd", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("madd", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("poly3", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("poly3", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("poly3", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_normalize=1))),
    for_example("signal_window", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("signal_window", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("signal_window", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("iir1_lpf", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_output=1))),
    for_example("iir1_lpf", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("iir1_lpf", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example(
        "pid",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_output=1),
            fdiv=FDivOperator(F_e6m18, stage_output=1),
        ),
    ),
    for_example(
        "pid", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_input=1, stage_output=1))
    ),
    for_example("pid", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("schmitt_trigger", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("schmitt_trigger", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("schmitt_trigger", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("quadrature_encoder", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("quadrature_encoder", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("quadrature_encoder", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("phase_frequency_detector", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("phase_frequency_detector", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("phase_frequency_detector", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("latching_fault_register", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("latching_fault_register", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("latching_fault_register", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("majority_voter", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("majority_voter", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("majority_voter", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("recip_newton", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("recip_newton", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("recip_newton", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("integrator", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("integrator", FlowId.DIAMOND_ECP5, 100, op_config_staged_output(F_e6m18)),
    for_example(
        "integrator", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_normalize=1))
    ),
    for_example("uart_tx", FlowId.YOSYS_ECP5, 100, op_config(F_e4m8)),
    for_example("uart_tx", FlowId.DIAMOND_ECP5, 100, op_config(F_e4m8)),
    for_example("uart_tx", FlowId.VIVADO_ARTIX7, 150, op_config(F_e4m8)),
    for_example("uart_rx", FlowId.YOSYS_ECP5, 100, op_config(F_e4m8)),
    for_example("uart_rx", FlowId.DIAMOND_ECP5, 100, op_config(F_e4m8)),
    for_example("uart_rx", FlowId.VIVADO_ARTIX7, 150, op_config(F_e4m8)),
    for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_decode=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_input=1, stage_output=1),
        ),
    ),
    for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_input=1, stage_output=1),
            fdiv=FDivOperator(F_e6m18, stage_input=1, stage_output=1),
        ),
    ),
    for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_product=1),
        ),
    ),
    for_example(
        "ekf1_stateful",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_decode=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_input=1, stage_output=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_decode=1, stage_normalize=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_pack=1),
            fdiv=FDivOperator(F_e6m18, stage_input=1, stage_output=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_normalize=1),
            fmul=FMulOperator(F_e6m18, stage_product=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "cordic_sincos",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_decode=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e6m18, stage_decode=1),
        ),
    ),
    for_example(
        "cordic_sincos",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_decode=1, stage_normalize=1, stage_output=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e6m18, stage_decode=1),
        ),
    ),
    for_example(
        "cordic_sincos", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_normalize=1))
    ),
    for_example("octave_index", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example(
        "octave_index",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_output=1),
            fdiv=FDivOperator(F_e6m18, stage_output=1),
        ),
    ),
    for_example("octave_index", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    # octave_index's transcendental sibling. The exp2/log2 Horner products and log2's final f*C(f) product need
    # registered partial-product reduction; this one config closes all three flows.
    for_example(
        "equal_temperament",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e6m18,
            fmul=FMulOperator(F_e6m18, stage_pack=1),
            fexp2=FExp2Operator(F_e6m18, stage_reduce=1, stage_product=2),
            flog2=FLog2Operator(F_e6m18, stage_product=2, stage_product_final=2, stage_normalize=2, stage_pack=1),
        ),
    ),
    for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e6m18,
            fexp2=FExp2Operator(F_e6m18, stage_product=2),
            flog2=FLog2Operator(F_e6m18, stage_product=2, stage_product_final=2, stage_normalize=1, stage_pack=1),
        ),
    ),
    for_example(
        "equal_temperament",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_normalize=1),
            fexp2=FExp2Operator(F_e6m18, stage_product=2),
            flog2=FLog2Operator(F_e6m18, stage_product=2, stage_product_final=2, stage_normalize=1, stage_pack=1),
        ),
    ),
    for_example("remainder", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example(
        "remainder",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(F_e6m18, fdiv=FDivOperator(F_e6m18, stage_input=1, stage_output=1)),
    ),
    for_example(
        "remainder", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_normalize=1))
    ),
    for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e8m36,
            fadd=FAddOperator(F_e8m36, stage_input=1, stage_decode=1, stage_normalize=2, stage_pack=1, stage_output=1),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=2, stage_pack=1, stage_output=1),
            fdiv=FDivOperator(F_e8m36, stage_input=1, stage_output=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e8m36, stage_input=1, stage_decode=1),
        ),
    ),
    for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e8m36,
            fadd=FAddOperator(
                F_e8m36, stage_input=2, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1, stage_output=1
            ),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=1, stage_pack=1, stage_output=1),
            fdiv=FDivOperator(F_e8m36, stage_input=1, stage_pack=1, stage_output=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e8m36, stage_decode=1),
        ),
    ),
    for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(
            F_e8m36,
            fadd=FAddOperator(F_e8m36, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=1, stage_pack=1),
            fdiv=FDivOperator(F_e8m36, stage_input=1, stage_pack=1, stage_output=1),
        ),
    ),
    for_example(
        "ekf1_stateful",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e8m36,
            fadd=FAddOperator(F_e8m36, stage_input=1, stage_decode=1, stage_normalize=2, stage_pack=1),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=2, stage_output=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e8m36, stage_input=1, stage_decode=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e8m36,
            fadd=FAddOperator(F_e8m36, stage_input=2, stage_decode=1, stage_align=1, stage_normalize=2, stage_pack=1),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=1, stage_pack=1),
            fdiv=FDivOperator(F_e8m36, stage_input=3, stage_pack=1, stage_output=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(
            F_e8m36,
            fadd=FAddOperator(F_e8m36, stage_decode=1, stage_align=1, stage_normalize=2, stage_pack=1),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=1, stage_pack=1),
            fdiv=FDivOperator(F_e8m36, stage_input=1, stage_pack=1, stage_output=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    # imu_frame_transform: a stack of static 3x3 matrix products -- the widest, most multiply-heavy datapath in the
    # suite -- off-catalogue (matrix/vector ports have no scalar-lane SPEC). Two datapaths per flow: the plain
    # fmul+fadd expansion, and the ffma-contracted form where each dot-product multiply-accumulate fuses into a single
    # rounding. Stage knobs are the measured lean-start closure per (flow, datapath).
    SynthTarget(
        kernel=_imu_frame_transform_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_decode=1),
            fmul=FMulOperator(F_e6m18, stage_product=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e6m18, stage_decode=1),
        ),
        name="imu_frame_transform_e6m18",
        example="imu_frame_transform",
    ),
    SynthTarget(
        kernel=_imu_frame_transform_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_input=1)),
        name="imu_frame_transform_e6m18",
        example="imu_frame_transform",
    ),
    SynthTarget(
        kernel=_imu_frame_transform_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_input=1),
        ),
        name="imu_frame_transform_e6m18",
        example="imu_frame_transform",
    ),
    SynthTarget(
        kernel=_imu_frame_transform_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_decode=1, stage_pack=1),
            fmul=FMulOperator(F_e6m18, stage_product=1),
            ffma=FFmaOperator(F_e6m18, stage_product=1, stage_decode=1, stage_normalize=1, stage_pack=1),
        ),
        name="imu_frame_transform_e6m18_fma",
        example="imu_frame_transform",
    ),
    SynthTarget(
        kernel=_imu_frame_transform_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_normalize=1),
            fmul=FMulOperator(F_e6m18, stage_output=1),
            ffma=FFmaOperator(F_e6m18, stage_normalize=1, stage_pack=1),
        ),
        name="imu_frame_transform_e6m18_fma",
        example="imu_frame_transform",
    ),
    SynthTarget(
        kernel=_imu_frame_transform_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=op_config(
            F_e6m18,
            fmul=FMulOperator(F_e6m18, stage_product=1),
            ffma=FFmaOperator(F_e6m18, stage_product=1, stage_normalize=1),
        ),
        name="imu_frame_transform_e6m18_fma",
        example="imu_frame_transform",
    ),
    # polar: two off-catalogue 2-vector CORDIC kernels (no scalar-lane SPEC). to_polar fuses atan2+hypot into one
    # CORDIC; from_polar coalesces sin+cos.
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=op_config(F_e6m18, fatan2=_TO_POLAR_FATAN2),
        name="to_polar_e6m18",
        example="polar_to",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=op_config(F_e6m18, fatan2=_TO_POLAR_FATAN2),
        name="to_polar_e6m18",
        example="polar_to",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=op_config(F_e6m18, fatan2=_TO_POLAR_FATAN2),
        name="to_polar_e6m18",
        example="polar_to",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=op_config(F_e6m18, fsincos=_FROM_POLAR_FSINCOS),
        name="from_polar_e6m18",
        example="polar_from",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=op_config(F_e6m18, fsincos=_FROM_POLAR_FSINCOS),
        name="from_polar_e6m18",
        example="polar_from",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=op_config(F_e6m18, fmul=FMulOperator(F_e6m18, stage_input=1), fsincos=_FROM_POLAR_FSINCOS),
        name="from_polar_e6m18",
        example="polar_from",
    ),
    # kepler: fsincos inside a data-dependent Newton back-edge loop -- the only II>1 operator in a loop in the matrix.
    for_example("kepler", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18, fsincos=_KEPLER_FSINCOS)),
    for_example("kepler", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18, fsincos=_KEPLER_FSINCOS)),
    for_example("kepler", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18, fsincos=_KEPLER_FSINCOS)),
]

assert len({t.label for t in TARGETS}) == len(TARGETS)  # labels key build dirs and pytest ids
