"""
Golden schedule-length freeze over the WHOLE example matrix -- a committed regression guard on scheduling
efficiency, standing in for the deferred makespan/II optimization work. Every spec is frozen at every format it
declares, and the table is checked against the matrix, so a newly added example cannot escape the guard by
omission.

The differential fuzzer and the example-reference suite compare output VALUES only; they are structurally blind
to cycle count, so a schedule that still computes the right result but takes longer -- a wasted cycle, a lost
cross-block overlap, an over-pipelining congestion regression -- passes them silently. This test pins each
kernel's (min initiation interval, last microcode PC). The min II is the throughput of the shortest path; the
last PC is the out_valid boundary PC -- the end of the static schedule across every block (a zero-based PC, not a
word count) -- so the full schedule is pinned even for a data-dependent branch/loop kernel whose public max II is
reported as None (its loop body would otherwise be unguarded). Both are fixed by the scheduler before register-
allocation annealing, so they are independent of `HOLOSO_REGALLOC_EFFORT`. A deliberate schedule change is
expected to update the frozen value in the same commit.

It also folds in the chained-copy kernel shape -- a state assignment `self.a = self.b` where both sides are
slots, so one slot's live-out reads another slot's live-in (the register allocator's `tapped_by_other` path). No
committed example exercises it. Two purpose-built kernels (a float delay line and a boolean shift register) pin it in
both banks, and a behavioral check confirms the chained copy still captures each old value before it is overwritten --
which the value-blind schedule freeze cannot.
"""

import pytest

import holoso
from holoso import FloatFormat
from holoso._eel import lower as lower_frontend
from holoso._lir import WideStateSlot
from holoso._lir._ir import BoolStateSlot
from holoso._mir import lower as lower_to_mir

from ._examples import ExampleSpec, SPECS
from ._modelref import (
    build_lir,
    mir_options,
    default_mir,
    default_options,
    DEFAULT_UNROLL_MAX_TRIPS,
)

# Kernel label -> frozen (min initiation interval, last microcode PC). last_pc is the out_valid boundary PC -- the
# end of the static schedule across all blocks -- so it pins the full schedule even for data-dependent (branch/loop)
# kernels. One row per spec per declared format; a spec listing two formats is frozen at both, since the second
# exists precisely to exercise a different pipeline depth.
_FROZEN_SCHEDULE: dict[str, tuple[int, int]] = {
    "madd-e8m36": (14, 14),
    "signal_window-e8m36": (9, 9),
    "poly3-e8m36": (23, 23),
    "iir1_lpf-e8m36": (15, 15),
    "iir1_hpf-e8m36": (20, 20),
    "pid-e8m36": (36, 68),
    "schmitt_trigger-e8m36": (6, 6),
    "quadrature_encoder-e8m36": (6, 6),
    "phase_frequency_detector-e8m36": (6, 6),
    "latching_fault_register-e8m36": (5, 5),
    "majority_voter-e6m18": (15, 20),
    # The turn-native trigonometric ABI lets the phase scaling meet the cores' own conversion and cancel, so the I/Q
    # oscillator's two general multiplies collapse to one exponent add.
    "iq_oscillator-e8m36": (55, 55),
    "nco-e6m18": (12, 12),
    "pwm-e6m18": (11, 11),
    "debouncer-e6m18": (10, 10),
    "priority_encoder-e6m18": (21, 21),
    "crc32-e6m18": (45, 45),
    "lfsr16-e6m18": (12, 12),
    # Branchy kernels whose phi-arm installs source block-entry-resident values (boolean/float live-out constants, or an
    # input/state read) on the normal path -- the inline-class timing (no source-sample edge, no +1 step) lands each
    # within the work makespan rather than at the copy-pipeline boundary, shrinking every downstream block base.
    "uart_tx-e6m18": (12, 37),
    "uart_rx-e6m18": (6, 51),
    # The loop body's tail copy (y <- y_next) sources y_next, which is NOT the block's last work (delta = y_next - y
    # is), so the install fits at the work makespan instead of one past it -- shaving a cycle off every iteration.
    # The convergence test is statically true on entry (delta is seeded above the tolerance), so the first trip is
    # peeled at compile time and only trips 2..N remain residual: one body's worth of extra microcode, and a realized
    # transaction that is shorter, not longer (test_cycle_model).
    "recip_newton-e8m36": (29, 46),
    # The all-false early return and the break-terminated candidate scan survive as real branches, so the
    # frozen last PC covers the full active path with every scan trip taken.
    "finite_set_current_controller-e8m36": (160, 204),
    "remainder-e8m36": (37, 54),
    "octave_index-e6m18": (14, 36),
    "octave_index-e8m36": (14, 45),
    "equal_temperament-e8m36": (40, 40),
    "cordic_sincos-e8m36": (104, 104),
    "polar_to-e8m36": (63, 63),
    "polar_from-e8m36": (38, 38),
    # The three pivot-swap diamonds of the 3x3 Gauss-Jordan inversion if-convert into selects, so the whole
    # kernel is one straight-line block serialized on the pooled divider.
    "rigid_body_scalar-e8m36": (126, 126),
    "kepler-e8m36": (80, 155),
    "integrator-e8m36": (16, 16),
    "ekf1_stateless-e8m36": (125, 125),
    # The two graduated filter examples: both are straight-line (the FIR's static tap loop unrolls, the biquad has no
    # control flow at all), so their whole schedule is one block and min II equals last PC.
    "fir-e8m36": (20, 20),
    "biquad-e8m36": (21, 21),
    "ekf1_stateful-e8m36": (125, 125),
    # Straight-line: the clamped flux update is a short fadd/fmul/fsort dataflow serialized behind the long CORDIC
    # atan2 tail.
    "flux_observer-e8m36": (87, 87),
    # The fusion capstone: three rsqrt sites, each one native fsqrt and one fdiv (a native rsqrt operator would fold
    # the division away too), the gate and first-sample diamonds as real branches, and the clamp on the sorter.
    "imu_fusion-e8m36": (234, 412),
    "imu_fusion-e6m18": (207, 349),
}


def _label(spec: ExampleSpec, fmt: FloatFormat) -> str:
    return f"{spec.name}-e{fmt.wexp}m{fmt.wman}"


_CASES = [(spec, fmt) for spec in SPECS for fmt in spec.formats]


def test_every_example_is_frozen() -> None:
    """The guard's own coverage: an example added without a frozen row would otherwise be silently unguarded."""
    assert sorted(_FROZEN_SCHEDULE) == sorted(_label(spec, fmt) for spec, fmt in _CASES)


@pytest.mark.parametrize("spec,fmt", [pytest.param(s, f, id=_label(s, f)) for s, f in _CASES])
def test_schedule_length_is_frozen(spec: ExampleSpec, fmt: FloatFormat) -> None:
    # The min II column is the public `SynthesisResult.initiation_interval[0]`; the last PC has no public spelling
    # (the public tuple reports None past a surviving branch), so it stays pinned on the LIR. Each spec is driven at
    # its OWN configuration, since an integer-heavy kernel needs a word the shared default does not supply.
    label = _label(spec, fmt)
    frozen_ii, frozen_last_pc = _FROZEN_SCHEDULE[label]
    options = spec.options(fmt)
    result = holoso.synthesize(spec.make_kernel(), options, name=spec.name)
    assert result.initiation_interval[0] == frozen_ii, (
        f"{label}: scheduling efficiency changed -- min II {result.initiation_interval[0]} differs from the frozen "
        f"{frozen_ii}. If this is a deliberate schedule improvement, update the frozen value."
    )
    lir = build_lir(
        lower_to_mir(lower_frontend(spec.make_kernel(), DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)),
        spec.name,
    )
    assert (lir.min_initiation_interval, lir.last_pc) == (frozen_ii, frozen_last_pc), (
        f"{label}: scheduling efficiency changed -- (min II, last PC) differs from the frozen "
        f"{_FROZEN_SCHEDULE[label]}. If this is a deliberate schedule improvement, update the frozen value."
    )


class _Delay3:
    def __init__(self) -> None:
        self.x0 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0

    def __call__(self, x: float) -> float:
        out = self.x2
        self.x2 = self.x1
        self.x1 = self.x0
        self.x0 = x
        return out


class _BoolShift3:
    def __init__(self) -> None:
        self.b0 = False
        self.b1 = False
        self.b2 = False

    def __call__(self, b: bool) -> bool:
        out = self.b2
        self.b2 = self.b1
        self.b1 = self.b0
        self.b0 = b
        return out


# Chained-copy kernels and their frozen (min II, last PC). _Delay3 exercises the wide bank's tapped_by_other path,
# _BoolShift3 the boolean bank's. _FMT is the wide e8m36 datapath the example matrix uses.
_FMT = FloatFormat(8, 36)
_CHAINED_COPY: list[tuple[str, type[_Delay3] | type[_BoolShift3], tuple[int, int]]] = [
    ("delay3", _Delay3, (3, 3)),
    ("bool_shift3", _BoolShift3, (3, 3)),
]


@pytest.mark.parametrize("name,kernel_cls,frozen", _CHAINED_COPY)
def test_chained_copy_schedule_is_frozen(
    name: str, kernel_cls: type[_Delay3] | type[_BoolShift3], frozen: tuple[int, int]
) -> None:
    result = holoso.synthesize(kernel_cls().__call__, default_options(_FMT), name=name)
    assert result.initiation_interval[0] == frozen[0]
    lir = build_lir(
        lower_to_mir(lower_frontend(kernel_cls().__call__, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(_FMT)),
        name,
    )
    slots: list[WideStateSlot | BoolStateSlot] = [*lir.wide_state_slots, *lir.bool_state_slots]
    assert all(
        slot.needs_copy for slot in slots
    ), f"{name}: a chained-copy slot unexpectedly coalesced; the tapped_by_other path is no longer exercised"
    got = (lir.min_initiation_interval, lir.last_pc)
    assert got == frozen, (
        f"{name}: scheduling efficiency changed -- (min II, last PC) {got} differs from the frozen {frozen}. "
        f"If this is a deliberate schedule improvement, update the frozen value."
    )


def test_chained_copy_captures_old_values() -> None:
    """
    The chained copy must sample each slot's OLD value before the bundle overwrites it, so a 3-tap line delays its
    input by exactly three transactions. A value-blind schedule freeze cannot see a read-after-write miscompile here.
    """
    fmt = _FMT
    fmodel = holoso.synthesize(_Delay3().__call__, default_options(fmt), name="delay3").numerical_model.elaborate()
    fref = _Delay3()
    for raw in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        x = fmt.decode(fmt.encode(raw))  # quantize so the model and the reference see the identical operand
        assert float(fmodel.run(x)[0]) == fref(x)

    bmodel = holoso.synthesize(
        _BoolShift3().__call__, default_options(fmt), name="bool_shift3"
    ).numerical_model.elaborate()
    bref = _BoolShift3()
    for b in (True, False, True, True, False, False, True, False):
        assert bool(bmodel.run(b)[0]) == bref(b)
