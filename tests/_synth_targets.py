"""
The out-of-context synthesis matrix: a flat list of SynthTarget records, one per (kernel, flow, target f_max,
operator configuration). This is the single source of truth for what the example synthesis suite
(test_synth_examples) asserts can close timing.

Deliberately a dumb-simple data table -- literal frequencies, one explicit row per (example, flow), and a full typed
OpConfig per row; duplication is preferred over indirection here. Catalogued example targets reuse the shared example
registry (_examples.SPECS) so each kernel is constructed once; test_synth_targets checks that every catalogued example
appears. Off-catalogue regression cores may be added as SynthTarget records with example=None.

The bar (all at minimum speed grade): yosys-ecp5 and diamond-ecp5 at 100 MHz, vivado-artix7 at 150 MHz. Almost every
kernel closes lean; stage knobs are row-local and appear only where measured closure needs them.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)
from synth.flows import FlowId

from ._examples import SPECS, ekf1_stateful

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
) -> OpConfig:
    """The OpConfig for fmt; pass a fully-constructed operator to give it stage knobs, else that operator is lean."""
    return OpConfig(
        fadd=fadd or FAddOperator(fmt),
        fmul=fmul or FMulOperator(fmt),
        fdiv=fdiv or FDivOperator(fmt),
        fmul_ilog2=fmul_ilog2 or FMulILog2OperatorFamily(fmt),
        fcmp=fcmp or FCmpOperator(fmt),
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
    name: str  # descriptive module/report label; unique per flow (guarded by test_synth_targets)
    env: Mapping[str, str] = field(default_factory=dict)
    example: str | None = None  # the SPECS name this exercises; None for an off-catalogue regression core

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
    env: Mapping[str, str] | None = None,
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
        env=env or {},
        example=example,
    )


def _ekf1_stateful_kernel() -> Callable[..., object]:
    # The bundled example's default-constructed kernel -- what examples/ekf1_stateful.py ships and the matrix must
    # synthesize. SPEC.make_kernel instead uses _fresh_stateful_ekf, a cosim-only divisor-safe reset that folds
    # different constants into different RTL, so the synth rows pass this explicitly.
    return ekf1_stateful.Ekf1().update


TARGETS: list[SynthTarget] = [
    # Most of the catalogue closes the bar at lean (no optional stages) on all three tools -- verified by a full lean
    # baseline. One explicit row per (example, flow); duplication is intentional.
    for_example("madd", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("madd", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("madd", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("poly3", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example("poly3", FlowId.DIAMOND_ECP5, 100, op_config(F_e6m18)),
    for_example("poly3", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
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
    for_example("integrator", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
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
            fadd=FAddOperator(F_e6m18, stage_input=1),
            fmul=FMulOperator(F_e6m18, stage_output=1),
            fdiv=FDivOperator(F_e6m18, stage_input=1, stage_output=1),
        ),
        env={"HOLOSO_DIAMOND_HARD": "1"},
    ),
    for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_input=1), fmul=FMulOperator(F_e6m18, stage_product=1)),
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
            fdiv=FDivOperator(F_e6m18, stage_input=1, stage_output=1),
        ),
        env={"HOLOSO_DIAMOND_HARD": "1"},
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        op_config(F_e6m18, fmul=FMulOperator(F_e6m18, stage_product=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    for_example(
        "cordic_sincos",
        FlowId.YOSYS_ECP5,
        100,
        op_config(F_e6m18, fadd=FAddOperator(F_e6m18, stage_decode=1)),
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
    for_example("cordic_sincos", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("octave_index", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example(
        "octave_index",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(
            F_e6m18,
            fadd=FAddOperator(F_e6m18, stage_input=1, stage_output=1),
            fmul=FMulOperator(F_e6m18, stage_output=1),
            fdiv=FDivOperator(F_e6m18, stage_output=1),
        ),
    ),
    for_example("octave_index", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example("remainder", FlowId.YOSYS_ECP5, 100, op_config(F_e6m18)),
    for_example(
        "remainder",
        FlowId.DIAMOND_ECP5,
        100,
        op_config(F_e6m18, fdiv=FDivOperator(F_e6m18, stage_input=1, stage_output=1)),
    ),
    for_example("remainder", FlowId.VIVADO_ARTIX7, 150, op_config(F_e6m18)),
    for_example(
        "ekf1_stateless",
        FlowId.YOSYS_ECP5,
        100,
        op_config(
            F_e8m36,
            fadd=FAddOperator(F_e8m36, stage_input=1, stage_decode=1, stage_normalize=1, stage_pack=1, stage_output=1),
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
        env={"HOLOSO_DIAMOND_HARD": "1"},
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
            fadd=FAddOperator(F_e8m36, stage_input=1, stage_decode=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOperator(F_e8m36, stage_input=1, stage_product=2, stage_output=1),
            fmul_ilog2=FMulILog2OperatorFamily(F_e8m36, stage_input=1, stage_decode=1),
        ),
        env={"HOLOSO_YOSYS_HARD": "1"},
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
        env={"HOLOSO_DIAMOND_HARD": "1"},
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
]


# Every environment-variable key any target sets. The harness clears these before applying a target's own env, so an
# ambient value (e.g. a shell HOLOSO_DIAMOND_HARD=1) cannot leak into a lean row and mask a closure regression by
# silently running the hard strategy.
TARGET_ENV_KEYS: frozenset[str] = frozenset(key for target in TARGETS for key in target.env)
