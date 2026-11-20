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

The bar is 100 MHz on yosys-ecp5 and diamond-ecp5 and 150 MHz on vivado-artix7, on the slowest speed grade. Every row
was closed from the lean start (every optional stage off), one register stage per iteration at the critical path its
flow reported, and then trimmed: each stage was taken out in turn, and one the row closed without stayed out. A row
that departs from this says why.
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
    IAddsOptions,
    IShftOptions,
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
    fadd: FAddOptions = FAddOptions(stage_input=0, stage_core=FAddOptions.CoreStages.NONE, stage_output=0),
    fmul: FMulOptions = FMulOptions(stage_input=0, stage_core_mul=0, stage_output=0),
    fdivsqrt: FDivsqrtOptions = FDivsqrtOptions(
        stage_input=0, stage_core=FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
    ),
    fmul_ilog2: FMulILog2Options = FMulILog2Options(
        stage_input=0, stage_core=FMulILog2Options.CoreStages.NONE, stage_output=0
    ),
    filog2: FILog2Options = FILog2Options(stage_input=0, stage_output=0),
    fcmp: FCmpOptions = FCmpOptions(stage_input=0, stage_output=0),
    ffma: FFmaOptions | None = None,
    fexp2: FExp2Options | None = None,
    flog2: FLog2Options | None = None,
    fcordic: FCordicOptions | None = None,
    frint: FRintOptions | None = None,
    ffromint: FFromIntOptions | None = None,
    iadds: IAddsOptions = IAddsOptions(),
    ishft: IShftOptions = IShftOptions(),
    wmultiplier: int,
    wint_min: int = Options(operator=OperatorOptions(), wmultiplier=18).wint_min,
) -> Options:
    """
    The Options for fmt, every operator lean (every optional stage off) unless the row passes its own options for it.
    ffma/fexp2/flog2/fcordic/frint/ffromint are absent unless supplied, so a kernel that uses no fused multiply-add,
    transcendental, rounding or integer conversion needs no such module.
    """
    return Options(
        operator=OperatorOptions(
            fadd=fadd,
            fmul=fmul,
            fdivsqrt=fdivsqrt,
            fmul_ilog2=fmul_ilog2,
            filog2=filog2,
            fcmp=fcmp,
            ffma=ffma,
            fexp2=fexp2,
            flog2=flog2,
            fcordic=fcordic,
            frint=frint,
            ffromint=ffromint,
            iadds=iadds,
            ishft=ishft,
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
    # Off-catalogue: every bundled kernel that takes a root also divides, so nothing else builds the divider for
    # the root alone -- the build whose digit selection a resource-sharing synthesizer folds unless told to keep it.
    return _root_only


TARGETS: list[SynthTarget] = [
    # One explicit row per (example, flow), less the yosys-ecp5 rows the example scripts stand in for; duplication is
    # intentional.
    _for_example(
        "madd",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "madd",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "poly3",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "poly3",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "signal_window",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(_F_e6m18, fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0), wmultiplier=18),
    ),
    _for_example(
        "signal_window",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(_F_e6m18, fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0), wmultiplier=18),
    ),
    _for_example(
        "signal_window",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(_F_e6m18, fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0), wmultiplier=18),
    ),
    _for_example(
        "iir1_hpf",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "iir1_hpf",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "iir1_hpf",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "iir1_lpf",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "iir1_lpf",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "iir1_lpf",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "pid",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=0, stage_core=FMulILog2Options.CoreStages.DECODE, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "pid",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "pid",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example("schmitt_trigger", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("schmitt_trigger", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("schmitt_trigger", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("quadrature_encoder", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("quadrature_encoder", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("phase_frequency_detector", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("phase_frequency_detector", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("latching_fault_register", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("latching_fault_register", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("majority_voter", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("majority_voter", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("majority_voter", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example(
        "recip_newton",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "recip_newton",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "recip_newton",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "integrator",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "integrator",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "integrator",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    # Straight-line multiply-accumulate chains.
    _for_example(
        "fir",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "fir",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "fir",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "biquad",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "biquad",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "biquad",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example("uart_tx", FlowId.YOSYS_ECP5, 100, _op_config(_F_e4m8, wmultiplier=18)),
    _for_example("uart_tx", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e4m8, wmultiplier=18)),
    _for_example("uart_tx", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e4m8, wmultiplier=18)),
    _for_example("uart_rx", FlowId.YOSYS_ECP5, 100, _op_config(_F_e4m8, wmultiplier=18)),
    _for_example("uart_rx", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e4m8, wmultiplier=18)),
    _for_example("uart_rx", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e4m8, wmultiplier=18)),
    # tunable_lowpass: the one bundled kernel whose shift counts are run-time operands, hence the one that builds the
    # barrel shifter; at the 32-bit word its script ships.
    _for_example(
        "tunable_lowpass",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(_F_e6m18, ishft=IShftOptions(stage_input=1), wmultiplier=18, wint_min=32),
    ),
    _for_example(
        "tunable_lowpass",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(_F_e6m18, iadds=IAddsOptions(stage_output=1), wmultiplier=18, wint_min=32),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=1, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
    ),
    # The same EKF with a second multiplier: the allocator binds the co-issued products across the two instances.
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(instances=2, stage_input=1, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
        name="ekf1_stateless_e6m18_fmul2",
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=1, stage_core_mul=1, stage_output=1),
            wmultiplier=18,
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=1, stage_core_mul=1, stage_output=0),
            wmultiplier=18,
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    _for_example(
        "cordic_sincos",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul_ilog2=FMulILog2Options(stage_input=0, stage_core=FMulILog2Options.CoreStages.DECODE, stage_output=0),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "cordic_sincos",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "cordic_sincos",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example("octave_index", FlowId.YOSYS_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("octave_index", FlowId.DIAMOND_ECP5, 100, _op_config(_F_e6m18, wmultiplier=18)),
    _for_example("octave_index", FlowId.VIVADO_ARTIX7, 150, _op_config(_F_e6m18, wmultiplier=18)),
    # octave_index's transcendental sibling: the exp2 and log2 operators, with their lookup tables and polynomial
    # products.
    _for_example(
        "equal_temperament",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fexp2=FExp2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core=FExp2Options.CoreStages.NONE,
                stage_output=0,
                fast=True,
            ),
            flog2=FLog2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core_mul=2,
                stage_core=FLog2Options.CoreStages.ALIGN | FLog2Options.CoreStages.SELECT,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fexp2=FExp2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core=FExp2Options.CoreStages.SUM,
                stage_output=0,
                fast=True,
            ),
            flog2=FLog2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core_mul=2,
                stage_core=FLog2Options.CoreStages.NONE,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "equal_temperament",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fexp2=FExp2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core=FExp2Options.CoreStages.SUM,
                stage_output=0,
                fast=True,
            ),
            flog2=FLog2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core_mul=2,
                stage_core=FLog2Options.CoreStages.ALIGN,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "remainder",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "remainder",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "remainder",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.SUM,
                stage_output=0,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "ekf1_stateless",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=1, stage_core_mul=3, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=0, stage_core=FMulILog2Options.CoreStages.NONE, stage_output=1),
            wmultiplier=18,
        ),
    ),
    # Found from its Diamond sibling's stages rather than the lean start, whose build outruns Vivado's one-hour limit.
    _for_example(
        "ekf1_stateless",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "ekf1_stateful",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=1, stage_core_mul=3, stage_output=1),
            wmultiplier=18,
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    # Found from its Diamond sibling's stages rather than the lean start, whose build outruns Vivado's one-hour limit.
    _for_example(
        "ekf1_stateful",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
            wmultiplier=18,
        ),
        kernel=_ekf1_stateful_kernel,
    ),
    # polar: two off-catalogue 2-vector CORDIC kernels (no scalar-lane SPEC). to_polar fuses atan2+hypot into one
    # vectoring; from_polar coalesces sin+cos into one rotation. A rotation takes one product register, a kernel that
    # only vectors none. An instance only rotating or only vectoring renders its block for that alone, so only foc's
    # CORDIC carries both datapaths.
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.TAKEN
                | FCordicOptions.CoreStages.ORDERED
                | FCordicOptions.CoreStages.ANGLE
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_to_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.TAKEN
                | FCordicOptions.CoreStages.ORDERED
                | FCordicOptions.CoreStages.NORMALIZE_INPUT
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
        name="to_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
        name="from_polar_e6m18",
    ),
    SynthTarget(
        kernel=_from_polar_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=2,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
        name="from_polar_e6m18",
    ),
    # rigid_body_rates: the pivoted 3x3 Gauss-Jordan inversion -- conditional-swap select networks feeding one pooled
    # divider.
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=1),
            fdivsqrt=FDivsqrtOptions(
                stage_input=0, stage_core=FDivsqrtOptions.CoreStages.NONE, stage_output=1, fast=True
            ),
            wmultiplier=18,
        ),
        name="rigid_body_rates_e6m18",
    ),
    SynthTarget(
        kernel=_rigid_body_rates_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=0),
            fdivsqrt=FDivsqrtOptions(
                stage_input=1, stage_core=FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
            ),
            wmultiplier=18,
        ),
        name="rigid_body_rates_e6m18",
    ),
    # flux_observer: a short stateful clamped fadd/fmul/fcmp integrator feeding the same CORDIC vectoring as to_polar.
    _for_example(
        "flux_observer",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=0,
                stage_core=FCordicOptions.CoreStages.TAKEN
                | FCordicOptions.CoreStages.ORDERED
                | FCordicOptions.CoreStages.NORMALIZE_INPUT
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=0,
                stage_core=FCordicOptions.CoreStages.TAKEN
                | FCordicOptions.CoreStages.ORDERED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "flux_observer",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcmp=FCmpOptions(stage_input=0, stage_output=1),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=0,
                stage_core=FCordicOptions.CoreStages.TAKEN
                | FCordicOptions.CoreStages.ORDERED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    # foc: that observer embedded in a full current controller -- one CORDIC serving the observer's atan2 and the Park
    # rotation, the clamp on the comparator, and the divides of the limiter and the modulator, over the widest
    # microcode word in the matrix. Only the Vivado flow has a row here; a Diamond row is absent rather than guessed,
    # since closure is only ever established by running the flow.
    _for_example(
        "foc",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=1),
            fdivsqrt=FDivsqrtOptions(
                stage_input=0, stage_core=FDivsqrtOptions.CoreStages.FIRST_DIGIT, stage_output=1, fast=True
            ),
            fmul_ilog2=FMulILog2Options(stage_input=0, stage_core=FMulILog2Options.CoreStages.DECODE, stage_output=0),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.TAKEN
                | FCordicOptions.CoreStages.ORDERED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
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
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=1),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_core=FMulILog2Options.CoreStages.NONE, stage_output=0),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            wmultiplier=18,
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=0),
            fdivsqrt=FDivsqrtOptions(
                stage_input=0, stage_core=FDivsqrtOptions.CoreStages.FIRST_DIGIT, stage_output=0, fast=True
            ),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_core=FMulILog2Options.CoreStages.NONE, stage_output=0),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            iadds=IAddsOptions(stage_input=1),
            wmultiplier=18,
        ),
        kernel=_imu_fusion_kernel,
    ),
    _for_example(
        "imu_fusion",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=1),
            fdivsqrt=FDivsqrtOptions(
                stage_input=1, stage_core=FDivsqrtOptions.CoreStages.FIRST_DIGIT, stage_output=0, fast=True
            ),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_core=FMulILog2Options.CoreStages.DECODE, stage_output=0),
            filog2=FILog2Options(stage_input=1, stage_output=0),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            ffma=FFmaOptions(
                stage_input=1,
                stage_core_mul=2,
                stage_core_add=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.NORMALIZED
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            iadds=IAddsOptions(stage_output=1),
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
            fadd=FAddOptions(
                stage_input=1,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fdivsqrt=FDivsqrtOptions(
                stage_input=0, stage_core=FDivsqrtOptions.CoreStages.NONE, stage_output=1, fast=True
            ),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_core=FMulILog2Options.CoreStages.DECODE, stage_output=0),
            filog2=FILog2Options(stage_input=1, stage_output=0),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            ffma=FFmaOptions(
                stage_input=1,
                stage_core_mul=2,
                stage_core_add=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            iadds=IAddsOptions(stage_input=1),
            wmultiplier=18,
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
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=2, stage_output=1),
            fdivsqrt=FDivsqrtOptions(
                stage_input=1, stage_core=FDivsqrtOptions.CoreStages.FIRST_DIGIT, stage_output=0, fast=True
            ),
            fmul_ilog2=FMulILog2Options(stage_input=1, stage_core=FMulILog2Options.CoreStages.NONE, stage_output=0),
            fcmp=FCmpOptions(stage_input=1, stage_output=0),
            ffma=FFmaOptions(
                stage_input=0,
                stage_core_mul=2,
                stage_core_add=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            iadds=IAddsOptions(stage_input=1),
            wmultiplier=18,
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
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fdivsqrt=FDivsqrtOptions(
                stage_input=0, stage_core=FDivsqrtOptions.CoreStages.NONE, stage_output=1, fast=True
            ),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "kepler",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    _for_example(
        "kepler",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.DECODE
                | FAddOptions.CoreStages.ALIGNED
                | FAddOptions.CoreStages.NORMALIZE,
                stage_output=1,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE,
                stage_output=1,
                fast=True,
            ),
            wmultiplier=18,
        ),
    ),
    # iq_oscillator: the phase accumulator's float-to-integer conversion on the rounder, beside a CORDIC rotation.
    _for_example(
        "iq_oscillator",
        FlowId.YOSYS_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            frint=FRintOptions(stage_input=0, stage_core=FRintOptions.CoreStages.MASK, stage_output=0),
            ffromint=FFromIntOptions(
                stage_input=0,
                stage_core=FFromIntOptions.CoreStages.ABS
                | FFromIntOptions.CoreStages.NORMALIZE
                | FFromIntOptions.CoreStages.PACK,
                stage_output=0,
            ),
            wmultiplier=18,
            wint_min=34,
        ),
    ),
    _for_example(
        "iq_oscillator",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=3,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            frint=FRintOptions(stage_input=0, stage_core=FRintOptions.CoreStages.MASK, stage_output=1),
            ffromint=FFromIntOptions(
                stage_input=0,
                stage_core=FFromIntOptions.CoreStages.ABS | FFromIntOptions.CoreStages.NORMALIZE,
                stage_output=0,
            ),
            wmultiplier=18,
            wint_min=34,
        ),
    ),
    _for_example(
        "iq_oscillator",
        FlowId.VIVADO_ARTIX7,
        150,
        _op_config(
            _F_e6m18,
            fmul=FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
            fcordic=FCordicOptions(
                stage_input=0,
                stage_core_mul=2,
                stage_core=FCordicOptions.CoreStages.REDUCED
                | FCordicOptions.CoreStages.FOLDED
                | FCordicOptions.CoreStages.CORRECTIONS
                | FCordicOptions.CoreStages.NORMALIZE
                | FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
            frint=FRintOptions(stage_input=0, stage_core=FRintOptions.CoreStages.MASK, stage_output=0),
            ffromint=FFromIntOptions(stage_input=0, stage_core=FFromIntOptions.CoreStages.NORMALIZE, stage_output=0),
            wmultiplier=18,
            wint_min=34,
        ),
    ),
    # The divider built for the root alone. Diamond takes it at the wide format, where the digit selection is
    # what LSE's resource sharing folds.
    SynthTarget(
        kernel=_root_only_kernel,
        flow=FlowId.YOSYS_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(_F_e6m18, wmultiplier=18),
        name="root_only_e6m18",
    ),
    SynthTarget(
        kernel=_root_only_kernel,
        flow=FlowId.DIAMOND_ECP5,
        target_frequency_MHz=100,
        ops=_op_config(
            _F_e8m36,
            fdivsqrt=FDivsqrtOptions(
                stage_input=0, stage_core=FDivsqrtOptions.CoreStages.FIRST_DIGIT, stage_output=0, fast=True
            ),
            wmultiplier=18,
        ),
        name="root_only_e8m36",
    ),
    SynthTarget(
        kernel=_root_only_kernel,
        flow=FlowId.VIVADO_ARTIX7,
        target_frequency_MHz=150,
        ops=_op_config(_F_e6m18, wmultiplier=18),
        name="root_only_e6m18",
    ),
    # equal_temperament at the wide format, where the exp2 and log2 tables are 512 entries of 135 and 134 bits: eight
    # block RAMs, or in LUT logic four times the whole machine, which is what this row guards against. Its multipliers
    # outgrow the default device.
    _for_example(
        "equal_temperament",
        FlowId.DIAMOND_ECP5,
        100,
        _op_config(
            _F_e8m36,
            fadd=FAddOptions(
                stage_input=0,
                stage_core=FAddOptions.CoreStages.OPERANDS
                | FAddOptions.CoreStages.ALIGN
                | FAddOptions.CoreStages.SUM
                | FAddOptions.CoreStages.NORMALIZE
                | FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
            fexp2=FExp2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core=FExp2Options.CoreStages.SUM,
                stage_output=0,
                fast=True,
            ),
            flog2=FLog2Options(
                stage_input=0,
                stage_core_kernel_level=6,
                stage_core_mul=3,
                stage_core=FLog2Options.CoreStages.ALIGN,
                stage_output=0,
                fast=True,
            ),
            wmultiplier=18,
        ),
        device_class=DeviceClass.LARGE,
    ),
]

assert len({t.label for t in TARGETS}) == len(TARGETS)  # labels key build dirs and pytest ids
