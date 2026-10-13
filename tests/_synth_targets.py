"""
The out-of-context synthesis matrix: a flat list of SynthTarget records, one per (kernel, flow, target f_max,
operator configuration). Together with the example scripts (below) this is what the example synthesis suite
(test_synth_examples) asserts can close timing.

Deliberately a dumb-simple data table -- literal frequencies, one explicit row per (example, flow), and a full typed
Options per row; duplication is preferred over indirection here. Catalogued example targets reuse the shared example
registry (_examples.SPECS) so each kernel is constructed once. Off-catalogue regression cores may be added as
SynthTarget records with example=None.

The bundled example scripts are themselves synthesized on yosys-ecp5 (test_synth_examples runs each one and synthesizes
what it produces), so a kernel has no yosys-ecp5 row here at the format and word width its script ships.

The bar (all at minimum speed grade): yosys-ecp5 and diamond-ecp5 at 100 MHz, vivado-artix7 at 150 MHz. More than half
of the rows close lean; stage knobs are row-local and appear only where measured closure or its margin needs them. Lean
keeps the product stages without which one multiplier would feed another within a cycle, and those that split a product
wider than a multiplier where the row states that width (`wmultiplier`): `stage_product=1` on the 36-bit multiplier (2
where split), `stage_product_chain=1` in exp2 and log2 (2 where split, as `stage_product` there) and
`stage_product_final=1` in log2 (2 where split). A row's history names those knobs only where they go beyond lean. A
closed row is then trimmed: each stage is taken out in turn, and one the row closes without stays out, which its history
says. Vivado's result does not reproduce from one machine to the next, so a Vivado row closing with less than about 0.15
ns to spare takes one more stage where one helps.
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
    frint: FRintOptions | None = None,
    ffromint: FFromIntOptions | None = None,
    wmultiplier: int | None = None,
    wint_min: int = Options(OperatorOptions()).wint_min,
) -> Options:
    """
    The Options for fmt; pass an options object to give an operator stage knobs, else that operator is lean.
    ffma/fexp2/flog2/fcordic/frint/ffromint are absent unless supplied, so a kernel that uses no fused multiply-add,
    transcendental, rounding or integer conversion needs no such module.
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
    # baseline. One explicit row per (example, flow), less the yosys-ecp5 rows the example scripts stand in for;
    # duplication is intentional.
    _for_example(
        "madd",
        FlowId.DIAMOND_ECP5,
        100,
        # 96.1 MHz lean: the multiplier's DSP product through its pack rounding into the register file (then 108.5).
        _op_config(_F_e6m18, fmul=FMulOptions(stage_output=1)),
    ),
    _for_example(
        "madd",
        FlowId.VIVADO_ARTIX7,
        150,
        # 152.4 MHz lean with 0.11 ns to spare. For margin, the adder's close-cancellation normalizer (then 154.9).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1)),
    ),
    _for_example("poly3", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "poly3",
        FlowId.VIVADO_ARTIX7,
        150,
        # 150.9 MHz lean with 0.04 ns to spare. For margin, the adder's close-cancellation normalizer (then 157.1).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1)),
    ),
    _for_example("signal_window", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("signal_window", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("signal_window", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("iir1_hpf", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("iir1_hpf", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("iir1_hpf", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("iir1_lpf", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("iir1_lpf", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "iir1_lpf",
        FlowId.VIVADO_ARTIX7,
        150,
        # 153.1 MHz lean with 0.14 ns to spare. For margin, the adder's close-cancellation normalizer (then 169.4).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1)),
    ),
    _for_example(
        "pid",
        FlowId.YOSYS_ECP5,
        100,
        # 98.5 MHz lean, the worst path an output register through the adder's operand read mux into its magnitude
        # compare and exponent difference. Staging that read only moved the worst path on, around the adder, the
        # multiplier and the scaler, for four stages and eleven cycles (then 109.4); of those and their neighbours the
        # multiplier's output register alone closes the row (101.0 MHz), place-and-route being that sensitive this
        # close to the bar.
        _op_config(_F_e6m18, fmul=FMulOptions(stage_output=1)),
    ),
    _for_example(
        "pid",
        FlowId.DIAMOND_ECP5,
        100,
        # 95.2 MHz lean: the multiplier's DSP product through its pack rounding into the register file (then 102.7).
        _op_config(_F_e6m18, fmul=FMulOptions(stage_output=1)),
    ),
    _for_example("pid", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("schmitt_trigger", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("schmitt_trigger", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("schmitt_trigger", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("quadrature_encoder", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("quadrature_encoder", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("phase_frequency_detector", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("phase_frequency_detector", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("latching_fault_register", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("latching_fault_register", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("majority_voter", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("majority_voter", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("majority_voter", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("recip_newton", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("recip_newton", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "recip_newton",
        FlowId.VIVADO_ARTIX7,
        150,
        # 152.7 MHz lean with 0.12 ns to spare. For margin, the adder's close-cancellation normalizer (then 155.5).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1)),
    ),
    _for_example("integrator", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("integrator", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "integrator",
        FlowId.VIVADO_ARTIX7,
        150,
        # 152.9 MHz lean with 0.13 ns to spare. For margin, the adder's close-cancellation normalizer (then 163.4).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1)),
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
    _for_example("tunable_lowpass", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wint_min=32)),
    _for_example("tunable_lowpass", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wint_min=32)),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        # 89.3 MHz lean: the register file through the adder's operand read mux into its exponent difference, the
        # multiplier's DSP product through its pack rounding into the register file (94.8 MHz, then 100.4).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1), fmul=FMulOptions(stage_output=1)),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        # 134.0 MHz lean: the microcode word through the multiplier's operand read mux into the unregistered DSP
        # input, the microcode word through the scaler's read mux into its exponent adder and overflow check (143.8
        # MHz), the multiplier's DSP product register through its sticky and pack rounding into the register file
        # (145.9 MHz), the register file through the adder's operand read mux into its magnitude compare and exponent
        # difference (135.7 MHz), the adder's close-cancellation normalizer (146.6 MHz, then 150.3 with 0.01 ns to
        # spare). For margin, the adder's normalize-count exponent subtract through its pack classification into the
        # register file (then 165.0). The multiplier's pack stage is redundant beside the adder's stages; without it
        # 158.4 MHz with 0.35 ns to spare. The row closes without the adder's input stage and the scaler's too (153.4
        # MHz), but keeps the stages its two-multiplier sibling below needs.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
    ),
    # The same EKF with a second multiplier: the allocator binds the co-issued products across the two instances.
    # Against the one-multiplier row at the same stages: 15 percent more LUTs for a 15 percent shorter transaction
    # (DESIGN.md).
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        # 130.8 MHz lean; held to the one-multiplier row's stages so that the pair compares like for like (then 159.2
        # with 0.39 ns to spare). Without the adder's input stage it has 0.11 ns left, and without the scaler's as
        # well it fails, which is what keeps them on both rows.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(instances=2, stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
        name="ekf1_stateless_e6m18_fmul2",
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # 97.5 MHz lean: the multiplier's DSP product through its pack rounding into the register file, the microcode
        # through the adder's operand read mux into its magnitude compare and exponent difference (90.6 MHz, then
        # 102.3).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1), fmul=FMulOptions(stage_output=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        # 130.5 MHz lean: the register file through the adder's operand read mux into its magnitude compare and
        # exponent difference, the register file through the multiplier's operand read mux into the unregistered DSP
        # input (130.8 MHz), the adder's close-cancellation normalizer (144.4 MHz), the adder's normalize-count
        # exponent subtract through its pack classification into the register file (148.9 MHz, then 157.4). With the
        # adder's output stage in, its input and normalizer stages are redundant; without them 155.7 MHz with 0.24 ns
        # to spare.
        _op_config(_F_e6m18, fadd=FAddOptions(stage_output=1), fmul=FMulOptions(stage_input=1)),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "cordic_sincos",
        FlowId.YOSYS_ECP5,
        100,
        # 100.00 MHz lean, short of the bar by a fraction of a picosecond: the microcode operand select through the
        # adder's read mux and magnitude compare into its alignment shift amount (then 109.1).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_decode=1)),
    ),
    _for_example("cordic_sincos", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("cordic_sincos", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    _for_example("octave_index", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("octave_index", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("octave_index", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18)),
    # octave_index's transcendental sibling: the exp2 and log2 operators, with their lookup tables and polynomial
    # products.
    _for_example(
        "equal_temperament",
        FlowId.YOSYS_ECP5,
        100,
        # 68.5 MHz lean: exp2's coefficient table, read straight out of block RAM, through the DSP tiles of its widest
        # product and their partial-product sum into the product register, the same out of log2's table into its
        # widest product (69.2 MHz, then 100.1). Neither operand register costs a cycle, the kernel's latency being
        # its chain's.
        _op_config(
            _F_e6m18,
            fexp2=FExp2Options(stage_product=1, stage_product_chain=1),
            flog2=FLog2Options(stage_product=1, stage_product_chain=1, stage_product_final=1),
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        # 69.9 MHz lean: the exp2 table's block-RAM read straight into its widest product, one multiplier tile
        # cascaded into an adder tile in a single cycle, the same path in log2 (69.7 MHz), the multiplier's DSP
        # product through its pack rounding into the register file (91.9 MHz, then 106.4).
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_output=1),
            fexp2=FExp2Options(stage_product=1, stage_product_chain=1),
            flog2=FLog2Options(stage_product=1, stage_product_chain=1, stage_product_final=1),
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.VIVADO_ARTIX7,
        150,
        # 123.1 MHz lean: log2's argument square, built in fabric, feeding the chain multiply's DSP tile unregistered,
        # exp2's coefficient table read straight out of block RAM into the unregistered DSP operand of its widest
        # product (137.8 MHz, then 150.9 with 0.04 ns to spare). For margin, in turn: log2's final multiply as two
        # cascaded DSP tiles in one cycle, split into tile-sized slices (151.3 MHz), the adder's close-cancellation
        # normalizer (151.8 MHz), exp2's last three-operand sum through its result assembly into the register file
        # (152.1 MHz), and exp2's pack register through its result assembly and the multiplier's operand read mux into
        # the multiplier's operand register (154.6 MHz). The adder's normalizer stage is redundant beside exp2's two;
        # without it 156.0 MHz with 0.26 ns to spare.
        _op_config(
            _F_e6m18,
            fexp2=FExp2Options(stage_product=1, stage_product_chain=1, stage_pack=1, stage_output=1),
            flog2=FLog2Options(stage_product_square=1, stage_product_chain=1, stage_product_final=2),
            wmultiplier=18,
        ),
    ),
    _for_example("remainder", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18)),
    _for_example("remainder", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18)),
    _for_example(
        "remainder",
        FlowId.VIVADO_ARTIX7,
        150,
        # 153.4 MHz lean with 0.15 ns to spare. For margin, the adder's close-cancellation normalizer (then 165.7).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_normalize=1)),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        # 72.7 MHz lean: the microcode through the adder's operand read mux into its exponent difference, the
        # multiplier's product through the sticky reduction, the rounder and the result into the register file's write
        # select (88.6 MHz), the adder's close-cancellation normalizer (84.1 MHz), the adder's last stage through the
        # exponent adjustment, the pack overflow classification and the result into the register file's write select
        # (98.4 MHz, then 101.0).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_output=1),
            fmul=FMulOptions(stage_product=1, stage_pack=1),
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        # 103.8 MHz lean: the multiplier's product-completion carry chain through its rounder into the register file,
        # the adder's normalize shift count through the exponent correction and the pack's range classification into
        # the register file's clear (131.2 MHz), the adder's raw sum through its close-cancellation normalizer (127.8
        # MHz), the microcode word through the adder's operand read mux and magnitude compare into its operand select
        # (145.6 MHz, then 155.6 with 0.24 ns to spare).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_normalize=1, stage_pack=1),
            fmul=FMulOptions(stage_product=1, stage_pack=1),
        ),
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        # 82.4 MHz lean: the register file through the adder's operand read mux into its exponent difference, the
        # multiplier's product through the sticky reduction, the rounder and the result into the register file's write
        # select (80.5 MHz), the adder's close-cancellation normalizer (91.8 MHz), the adder's alignment shifter
        # through its sticky reduction (99.4 MHz, then 101.1).
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_input=1, stage_align=1, stage_normalize=1),
            fmul=FMulOptions(stage_product=1, stage_pack=1),
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        # 104.5 MHz lean: the multiplier's product-completion carry chain through its rounder into the register file,
        # the microcode word through the adder's operand read mux and magnitude compare into its exponent difference
        # (137.9 MHz), the adder's raw sum through its close-cancellation normalizer (132.2 MHz), the adder's pack-
        # input select through its rounder and special-case select into the register file (146.3 MHz, then 158.9 with
        # 0.37 ns to spare).
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
    # the normalizer, the rounder and the multiplier named there are the CORDIC's own unless the comment says fmul's.
    # The unrolling stays at its default throughout. An instance running one mode is elaborated for it alone, so only
    # foc's CORDIC carries both modes' datapaths.
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 41.2 MHz lean: the CORDIC's normalizer (the vectoring magnitude select through the whole normalizer and the
        # rounder into the register file), the normalizer's second half through the sticky reduction, the rounder and
        # the result select into the register file (69.3 MHz), the CORDIC multiplier's DSP output through the
        # magnitude select and the normalizer's first two levels into its mid-cascade register (78.3 MHz), the
        # divider's result through the operand select into the CORDIC multiplier's unregistered DSP input (90.3 MHz,
        # then 102.5).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 69.4 MHz lean: the CORDIC multiplier's fabric sum tail through the whole normalizer and the rounder into the
        # register file, the same tail through the normalizer's first half into its barrier (122.8 MHz), the CORDIC
        # multiplier's operand select off the divider's valid flag into the unregistered DSP operand and on through
        # the cascaded tile (137.8 MHz, then 150.7 with 0.03 ns to spare). For margin, the normalizer's second half
        # through the exponent subtract and the rounder into the held magnitude (then 159.3).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1)),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 47.8 MHz lean: the CORDIC's normalizer through the rounder into the register file, the engine's residual-
        # angle register through the operand select into the CORDIC multiplier's unregistered DSP input (88.0 MHz),
        # fmul's DSP product through its pack rounding into the register file's write select (97.1 MHz, then 103.1).
        ops=_op_config(
            _F_e6m18, fmul=FMulOptions(stage_output=1), fcordic=FCordicOptions(stage_product=1, stage_normalize=1)
        ),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 104.4 MHz lean: the CORDIC's normalizer, entered from its back-end magnitude register and run through the
        # exponent subtract and the rounder into the held sine, the CORDIC multiplier's operand register through the
        # operand select into the unregistered DSP operand and on through the cascaded tile (128.2 MHz, then 154.3
        # with 0.19 ns to spare).
        ops=_op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=1)),
        name="from_polar_e6m18",
    ),
    # rigid_body_rates: the pivoted 3x3 Gauss-Jordan inversion -- conditional-swap select networks feeding one pooled
    # divider. Lean start per the closure procedure.
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        # 93.2 MHz lean: the microcode through the adder's operand read mux into its magnitude compare and exponent
        # difference (then 101.2).
        ops=_op_config(_F_e6m18, fadd=FAddOptions(stage_input=1)),
        name="rigid_body_rates_e6m18",
    ),
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        # 136.1 MHz lean: the register file through the adder's operand read mux into its magnitude compare and
        # exponent difference, the register file through the divider's operand read mux into its folded first-digit
        # subtracts (142.5 MHz), the adder's sticky through its pack rounding and result select across a long route
        # into the register file (143.5 MHz), the adder's close-cancellation normalizer (144.8 MHz), the microcode
        # word through the multiplier's operand read mux into the unregistered DSP input (143.1 MHz, then 155.9). With
        # the multiplier's and the divider's input stages in, the three adder stages are redundant, each paying its
        # cycles for nothing: without them 153.7 MHz with 0.16 ns to spare, eighteen cycles shorter.
        ops=_op_config(_F_e6m18, fmul=FMulOptions(stage_input=1), fdivsqrt=FDivsqrtOptions(stage_input=1)),
        name="rigid_body_rates_e6m18",
    ),
    # flux_observer: a short stateful clamped fadd/fmul/fcmp integrator feeding the same CORDIC vectoring as to_polar.
    _for_example(
        "flux_observer",
        FlowId.YOSYS_ECP5,
        100,
        # 41.8 MHz lean: the CORDIC's back-end result select through the whole normalizer, the exponent subtract and
        # the rounder into the register file, the CORDIC's multiply -- the operand select through the DSP tiles and
        # their fabric sum (75.5 MHz), the microcode operand select through the adder's read mux into its magnitude
        # compare and exponent difference (79.7 MHz), the CORDIC normalizer's second half through the exponent
        # subtract and the rounder into the register file (75.6 MHz), the CORDIC's multiply again, from its operand
        # register through the DSP tiles and the two partial-product adders, which takes the split into registered
        # tile-sized partial products (82.7 MHz), fmul's product register through its rounder and a long route into
        # the register-file write select (84.1 MHz), the CORDIC's back-end result select through the normalizer's
        # first half into its barrier (86.7 MHz), the adder's pack-input select through its rounder and result select
        # into the register file (98.6 MHz, then 107.1). With the normalizer's two barriers in, the output stages of
        # the multiplier and the adder are redundant; without them 102.0 MHz.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fcordic=FCordicOptions(stage_product=2, stage_normalize=2, stage_pack=1),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.DIAMOND_ECP5,
        100,
        # 40.0 MHz lean: the CORDIC multiplier's DSP output through the whole normalizer and the rounder into the
        # register file, the normalizer's second half through the exponent subtract, the rounder and the write select
        # into the register file (77.7 MHz), the CORDIC multiplier's DSP output through the magnitude select and the
        # normalizer's first two levels into its mid-cascade register (84.3 MHz), the same DSP output through the
        # magnitude select and the normalizer's top level into its first barrier, a route-dominated path sitting
        # behind the CORDIC iteration (89.9 MHz), the divider's result through the operand select into the CORDIC
        # multiplier's unregistered DSP input (92.7 MHz), the CORDIC pack register through the rounder's result select
        # and the write select into the register file (96.0 MHz), fmul's DSP product through its pack rounding into
        # the register file (95.5 MHz, then 101.5). With the later stages in, the normalizer's input register and the
        # CORDIC's output stage are redundant; without them 101.6 MHz.
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_output=1),
            fcordic=FCordicOptions(stage_product=1, stage_normalize=2, stage_pack=1),
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.VIVADO_ARTIX7,
        150,
        # 79.5 MHz lean: the CORDIC multiplier's fabric sum tail through the whole normalizer and the rounder into the
        # register file, the same tail through the normalizer's first half into its barrier (117.0 MHz), the CORDIC
        # multiplier's operand select off the divider's quotient into the unregistered DSP operand and on through the
        # cascaded tile (139.7 MHz), the register file through the adder's read mux and magnitude compare into its
        # exponent difference (141.8 MHz), the CORDIC normalizer's second half through the rounder into the register
        # file (143.7 MHz), the CORDIC multiply itself, from its operand capture through both cascaded tiles into a
        # product register that the placer had pulled out of the tile because the hop behind it (the sum tail through
        # the normalizer's top level into its first barrier) was starving, which the normalizer's input register
        # relieves (149.3 MHz, then 150.2 with 0.01 ns to spare). For margin, the adder's close-cancellation
        # normalizer (then 156.3).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_normalize=1),
            fcordic=FCordicOptions(stage_product=1, stage_normalize=3, stage_pack=1),
        ),
    ),
    # foc: that observer embedded in a full current controller -- one CORDIC serving the observer's atan2 and the Park
    # rotation, the clamp on the comparator, and the divides of the limiter and the modulator, over the widest
    # microcode word in the matrix. Only the Vivado flow is measured here; a Diamond row is absent rather than
    # guessed, since closure is only ever established by running the flow.
    _for_example(
        "foc",
        FlowId.VIVADO_ARTIX7,
        150,
        # 66.7 MHz lean: the CORDIC's normalizer, entered from the product's fabric sum and run through the rounder
        # into the held result, the product's fabric sum through the normalizer's first half into its barrier (114.7
        # MHz), the CORDIC multiplier's two-level operand select off its phase register into the unregistered DSP
        # operand and on through the cascaded tile (115.0 MHz), the register file through the adder's read mux and
        # magnitude compare into its exponent difference (137.6 MHz), the product's fabric sum, off a product register
        # pulled out of its tile, through the normalizer's top level into its first barrier, which the normalizer's
        # input register splits (139.1 MHz), the microcode through fmul's read mux into its unregistered DSP operand
        # (139.4 MHz), the CORDIC normalizer's second half through the rounder into the register file (138.2 MHz), the
        # register file through the divider's read mux into its folded first-digit subtracts (144.4 MHz), the adder's
        # last-stage exponent through the exponent correction and the pack's range classification into the register
        # file's clear (149.3 MHz, then 152.7 with 0.12 ns to spare). For margin, the register file through the
        # comparator's read mux and compare into the select of its minimum, which alone only moves the worst path on
        # (151.6 MHz), and the register file through the scaler's read mux, exponent adder and overflow check into the
        # clear of its output register (156.8 MHz). The comparator's stage is redundant beside the scaler's; without
        # it 153.6 MHz with 0.16 ns to spare.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_pack=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            fcordic=FCordicOptions(stage_product=1, stage_normalize=3, stage_pack=1),
        ),
    ),
    # imu_fusion: the fusion capstone -- three norm/rsqrt chains (a root and a division each on the one divider, one
    # feeding the coarse alignment), the comparator-backed clamp, and real gate branches over the heaviest register
    # pressure in the matrix, in the plain and the ffma datapaths. The Euclidean norms expand by exact exponent
    # scaling, a `filog2` per leg and its scalings.
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        # 91.2 MHz lean: the microcode through the adder's operand read mux into its exponent select, the multiplier's
        # DSP product through its pack rounding into the register file (90.8 MHz, then 103.1).
        _op_config(_F_e6m18, fadd=FAddOptions(stage_input=1), fmul=FMulOptions(stage_output=1)),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        # 128.3 MHz lean: the register file through the multiplier's operand read mux into the unregistered DSP input,
        # the register file through the scaler's read mux into its exponent adder and overflow check (125.4 MHz), the
        # register file through the divider's operand read mux into its folded first-digit subtracts (140.6 MHz), the
        # register file through the adder's operand read mux into its magnitude compare and exponent select (144.8
        # MHz, then 152.9 with 0.13 ns to spare). The divider's decode stage, which that run's worst path asks for,
        # does not help (152.4 MHz): behind it sits one control net, the transaction-state flag through one LUT into
        # the register file, which no stage shortens.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.YOSYS_ECP5,
        100,
        # 70.8 MHz lean: the microcode operand select through the fma's read mux into its DSP product, the same into
        # the multiplier's DSP product (73.5 MHz), the microcode operand select through the adder's read mux into its
        # magnitude compare and exponent difference (77.2 MHz), the microcode operand select through the scaler's read
        # mux into its exponent adder and overflow check and the clear of its output register (75.6 MHz), the fma's
        # normalized difference through the sticky reduction and its pack rounding into the register file (83.5 MHz),
        # the adder's normalize-count exponent subtract through its pack and a long route into the register-file write
        # select (85.0 MHz), the microcode divisor select into the divider's folded first-digit subtracts and digit
        # select (83.0 MHz), the fma's close-cancellation normalizer (88.0 MHz), the multiplier's product register
        # through its rounder and a long route into the register-file write select (88.7 MHz), the fma's product
        # register through its exponent adjust and exponent difference into the magnitude ordering of the product and
        # the addend (92.9 MHz, then 106.2).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_input=1, stage_output=1),
            fdivsqrt=FDivsqrtOptions(stage_decode=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_normalize=1, stage_pack=1),
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    _for_example(
        "imu_fusion",
        FlowId.DIAMOND_ECP5,
        100,
        # 88.6 MHz lean: the fma's close-cancellation normalizer, the microcode through the adder's operand read mux
        # into its magnitude compare and exponent difference (93.4 MHz), the multiplier's DSP product through its pack
        # rounding into the register file (88.4 MHz), the fma's normalized sum through its pack rounding into the
        # register file (90.4 MHz), the adder's sticky through its pack into the register file (99.5 MHz, then 101.3).
        # The fma's output stage is redundant beside the adder's; without it 103.2 MHz.
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1, stage_output=1),
            fmul=FMulOptions(stage_output=1),
            ffma=FFmaOptions(stage_normalize=1),
        ),
        kernel=_imu_fusion_kernel,
        name="imu_fusion_e6m18_fma",
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        # 134.5 MHz lean: the register file through the fma's operand read mux into its unregistered DSP input, the
        # microcode word through the adder's operand read mux into its magnitude compare and exponent difference
        # (136.8 MHz), the register file through the multiplier's operand read mux into the unregistered DSP input
        # (139.3 MHz), the register file through the scaler's read mux into its exponent adder and overflow check
        # (136.3 MHz), the fma's close-cancellation normalizer (143.1 MHz), the register file through the divider's
        # operand read mux into its folded first-digit subtracts (137.7 MHz), the fma's product and addend through
        # their ordering by magnitude into the alignment operand select (144.7 MHz, then 153.5 with 0.15 ns to spare).
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(stage_input=1),
            fmul=FMulOptions(stage_input=1),
            fdivsqrt=FDivsqrtOptions(stage_input=1),
            fmul_ilog2=FMulILog2Options(stage_input=1),
            ffma=FFmaOptions(stage_input=1, stage_decode=1, stage_normalize=1),
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
        # 51.4 MHz lean: the CORDIC's back-end magnitude through the whole normalizer, the exponent subtract and the
        # rounder into the register file, the multiply -- the phase register through the operand select, the DSP tiles
        # and their fabric sum (72.0 MHz), the normalizer's second half through the exponent subtract and the rounder
        # into the register file (85.2 MHz), the multiply again, from its operand register through the DSP tiles and
        # the two partial-product adders, which takes the split into registered tile-sized partial products (82.0 MHz,
        # then 100.9).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=2, stage_normalize=1, stage_pack=1), wmultiplier=18),
    ),
    _for_example(
        "kepler",
        FlowId.DIAMOND_ECP5,
        100,
        # 46.8 MHz lean: the CORDIC's normalizer (the rotation's magnitude register through the whole normalizer and
        # the rounder into the register file), the engine's residual-angle register through the operand select into
        # the CORDIC multiplier's unregistered DSP input (87.9 MHz, then 100.3).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=1)),
    ),
    _for_example(
        "kepler",
        FlowId.VIVADO_ARTIX7,
        150,
        # 101.0 MHz lean: the CORDIC's normalizer, entered from its back-end magnitude register and run through the
        # exponent subtract and the rounder into the held sine, the CORDIC multiplier's operand select off its phase
        # register into the unregistered DSP operand and on through the cascaded tile (130.6 MHz, then 154.0 with 0.17
        # ns to spare).
        _op_config(_F_e6m18, fcordic=FCordicOptions(stage_product=1, stage_normalize=1)),
    ),
    # iq_oscillator: the phase accumulator's float-to-integer conversion on the rounder, beside a CORDIC rotation.
    _for_example(
        "iq_oscillator",
        FlowId.YOSYS_ECP5,
        100,
        # 53.3 MHz lean: the CORDIC's back-end magnitude through the whole normalizer, the exponent subtract and the
        # rounder into the register file, the integer-to-float converter's magnitude through its whole normalizer,
        # exponent subtract and rounder into the register file (59.3 MHz), the CORDIC's multiply -- the phase register
        # through the operand select, the DSP tiles and their fabric sum (72.3 MHz), the CORDIC normalizer's second
        # half through the exponent subtract and the rounder into the register file (82.6 MHz), the CORDIC's multiply
        # again, from its operand register through the DSP tiles and the two partial-product adders, which takes the
        # split into registered tile-sized partial products (88.2 MHz, then 100.2).
        _op_config(
            _F_e6m18,
            fcordic=FCordicOptions(stage_product=2, stage_normalize=1, stage_pack=1),
            frint=FRintOptions(),
            ffromint=FFromIntOptions(stage_normalize=1),
            wmultiplier=18,
            wint_min=34,
        ),
    ),
    _for_example(
        "iq_oscillator",
        FlowId.DIAMOND_ECP5,
        100,
        # 40.1 MHz lean: the CORDIC's normalizer through the rounder into the register file, the integer-to-float
        # converter's magnitude register through its whole normalizer and rounder into the register file (56.6 MHz),
        # the CORDIC normalizer's second half through the rounder and the result into the register file's write select
        # (73.0 MHz), the same second half through the rounder into the held sine register (88.1 MHz), the engine's
        # residual-angle register through the operand select into the CORDIC multiplier's unregistered DSP input (87.0
        # MHz, then 102.6).
        _op_config(
            _F_e6m18,
            fcordic=FCordicOptions(stage_product=1, stage_normalize=1, stage_pack=1, stage_output=1),
            frint=FRintOptions(),
            ffromint=FFromIntOptions(stage_normalize=1),
            wint_min=34,
        ),
    ),
    _for_example(
        "iq_oscillator",
        FlowId.VIVADO_ARTIX7,
        150,
        # 91.1 MHz lean: the CORDIC's normalizer through the rounder into the register file, the integer-to-float
        # converter's normalizer through its rounder into the register file (113.2 MHz), the CORDIC multiplier's
        # operand select off its phase register into the unregistered DSP operand and on through the cascaded tile
        # (130.4 MHz, then 152.9 with 0.13 ns to spare). For margin, the rounder's shift register through its rounding
        # decision and increment into the register file (then 156.2).
        _op_config(
            _F_e6m18,
            fcordic=FCordicOptions(stage_product=1, stage_normalize=1),
            frint=FRintOptions(stage_round=1),
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
    # equal_temperament at the wide format, where the exp2 and log2 tables are 512 entries of 135 and 134 bits: eight
    # block RAMs, or in LUT logic four times the whole machine, which is what this row guards against. Its 36
    # multipliers outgrow the default device.
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        # 70.3 MHz lean, which here has every product wider than an 18-bit multiplier split into tile-sized partial
        # products: the log2 table's coefficient out of block RAM through the unregistered odd product and its
        # subtract into the chain product's operand capture, the same path in exp2 (70.8 MHz), the multiplier's
        # product register through its pack rounding into the register file's write select (89.7 MHz), the adder's
        # close-cancellation normalizer (95.6 MHz), the log2 final product through the rounding and the candidate
        # select into the register file (97.3 MHz), the log2 kernel's last sum through the alignment shift into the
        # final multiply's operand capture (98.4 MHz, then 103.0). The log2 output stage that the fifth path asked for
        # is redundant once the alignment stage is in; without it 101.3 MHz.
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(stage_normalize=1),
            fmul=FMulOptions(stage_product=2, stage_output=1),
            fexp2=FExp2Options(stage_product=2, stage_product_odd=1, stage_product_chain=2),
            flog2=FLog2Options(
                stage_product=2, stage_product_odd=1, stage_product_chain=2, stage_align=1, stage_product_final=2
            ),
            wmultiplier=18,
        ),
        device_class=DeviceClass.LARGE,
    ),
]

assert len({t.label for t in TARGETS}) == len(TARGETS)  # labels key build dirs and pytest ids
